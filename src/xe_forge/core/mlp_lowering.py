"""
Multi-layer MLP -> XeGPU-WG lowering.

A KernelBench MLP (level3) is a chain of nn.Linear layers with activations:
each layer is `C = act(x · W^T + bias)`. Torch-MLIR imports each as
  linalg.transpose(W) -> linalg.matmul(x, W^T) -> linalg.generic(bias-add)
  -> linalg.generic(activation)
repeated N times, output of layer i feeding layer i+1.

We lower the whole chain into N sequential XeGPU-WG kernels (one per layer),
chained through intermediate buffers — mirroring lighthouse's mlp_schedule
(match_and_split the matmuls, tile each layer's epilogue with the matmul +
transpose fused in as producers, outline each into its own gpu.module).

Unlike the single-op templates (Jinja2 per-config), the transform *recipe*
structure here depends on the layer count, so it's generated programmatically by
`render_mlp_recipe`. This module owns the parse (per-layer M/N/K) + recipe gen;
the executor drives the 3-stage lowering with the generated libraries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from xe_forge.core.attention_lowering import _indent_module
from xe_forge.core.linalg_lowering import DPAS_A_TILE, DPAS_B_TILE, NB_WORKITEMS


@dataclass(frozen=True)
class MlpLayer:
    """One MLP layer: C[M,N] = act(A[M,K] · W[N,K]^T + bias[N])."""

    m: int
    n: int
    k: int
    transpose_b: bool  # W stored [N,K] (physical transpose or indexing_maps)
    # chosen tile
    wg_m: int
    wg_n: int
    sg_m: int
    sg_n: int
    k_tile: int

    @property
    def sg_grid_m(self) -> int:
        return self.wg_m // self.sg_m

    @property
    def sg_grid_n(self) -> int:
        return self.wg_n // self.sg_n

    @property
    def nb_threads(self) -> int:
        return self.sg_grid_m * self.sg_grid_n * NB_WORKITEMS


def _pick_tile(m: int, n: int, k: int) -> tuple[int, int, int, int, int] | None:
    """Pick a non-degenerate (grid >= 2 in M or N), DPAS-aligned, divisible tile."""
    cands = [
        (128, 256, 32, 32, 32), (256, 256, 32, 32, 32), (128, 128, 32, 32, 32),
        (256, 128, 32, 32, 32), (64, 128, 32, 32, 32), (128, 64, 32, 32, 32),
        (64, 64, 32, 32, 32), (32, 64, 32, 32, 32), (64, 32, 32, 32, 32),
        (32, 32, 32, 32, 32),
    ]
    for require_grid in (True, False):
        for wm, wn, sm, sn, kt in cands:
            if m % wm or n % wn or k % kt:
                continue
            if require_grid and not (m // wm >= 2 or n // wn >= 2):
                continue
            # subgroup budget <= 64 (default GRF)
            if (wm // sm) * (wn // sn) > 64:
                continue
            return wm, wn, sm, sn, kt
    return None


def _standalone_layer_kernel(m: int, n: int, k: int) -> str:
    """A standalone transpose-B MLP layer C = relu(A·W^T + bias) for tile autotuning.

    Used to time a single layer's candidate tiles in isolation (the layer's own
    matmul + bias + ReLU, f16 in / f32 out), so autotune_tile can rank tiles for
    that exact shape before we bake the winner into the N-layer recipe.
    """
    return f"""#mA = affine_map<(m, n, k) -> (m, k)>
#mB = affine_map<(m, n, k) -> (n, k)>
#mC = affine_map<(m, n, k) -> (m, n)>
#e2 = affine_map<(m, n) -> (m, n)>
#eb = affine_map<(m, n) -> (n)>
func.func @layer(%A: memref<{m}x{k}xf16>, %W: memref<{n}x{k}xf16>, %bias: memref<{n}xf32>, %C: memref<{m}x{n}xf32>) {{
  %cA = bufferization.to_tensor %A restrict : memref<{m}x{k}xf16> to tensor<{m}x{k}xf16>
  %cW = bufferization.to_tensor %W restrict : memref<{n}x{k}xf16> to tensor<{n}x{k}xf16>
  %cbias = bufferization.to_tensor %bias restrict : memref<{n}xf32> to tensor<{n}xf32>
  %c0 = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<{m}x{n}xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  %mm = linalg.matmul indexing_maps = [#mA, #mB, #mC] ins(%cA, %cW : tensor<{m}x{k}xf16>, tensor<{n}x{k}xf16>) outs(%filled : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  %e1 = tensor.empty() : tensor<{m}x{n}xf32>
  %r = linalg.generic {{indexing_maps = [#e2, #eb, #e2], iterator_types = ["parallel", "parallel"]}} ins(%mm, %cbias : tensor<{m}x{n}xf32>, tensor<{n}xf32>) outs(%e1 : tensor<{m}x{n}xf32>) {{
  ^bb0(%in: f32, %b: f32, %o: f32):
    %s = arith.addf %in, %b : f32
    %cmp = arith.cmpf ugt, %s, %c0 : f32
    %sel = arith.select %cmp, %s, %c0 : f32
    linalg.yield %sel : f32
  }} -> tensor<{m}x{n}xf32>
  bufferization.materialize_in_destination %r in restrict writable %C : (tensor<{m}x{n}xf32>, memref<{m}x{n}xf32>) -> ()
  return
}}"""


def autotune_layer_tile(executor, m: int, n: int, k: int, large_grf: bool = False):
    """Return the fastest (wg_m, wg_n, sg_m, sg_n, k_tile) tile for an MLP layer of
    shape (M, N, K), by timing candidates on the GPU via executor.autotune_tile on a
    standalone single-layer kernel. Falls back to _pick_tile (first-divisible) if
    autotuning is unavailable or finds nothing runnable.

    *large_grf*: when True, autotune over the large-GRF candidates (each timed WITH
    the igc flag via run_pipeline_options, and <=32 subgroups by construction), and
    the layer's igc flag is applied module-wide at run time — see lower_mlp_to_wg.
    When False, only default-GRF tiles (64-subgroup budget) are considered.
    """
    from xe_forge.core.linalg_lowering import candidate_configs

    fallback = _pick_tile(m, n, k)
    if executor is None or not hasattr(executor, "autotune_tile"):
        return fallback
    # Large-GRF is a module-global igc flag (the xevm lowering runs once over the
    # whole module), so for an MLP it's all-or-nothing: every layer must be a valid
    # large-GRF tile (<=32 subgroups). candidate_configs already caps large-GRF tiles
    # at 32 subgroups; filter to the requested GRF mode.
    cands = [c for c in candidate_configs(m, n, k) if c.large_grf == large_grf]
    if not cands:
        return fallback
    try:
        best, _wg, _res = executor.autotune_tile(
            _standalone_layer_kernel(m, n, k), (m, n, k), cands, flop=2 * m * n * k
        )
    except Exception:
        return fallback
    if best is None:
        return fallback
    return best.wg_m, best.wg_n, best.sg_m, best.sg_n, best.k_tile


def fold_transpose_into_matmul(code: str) -> str:
    """Rewrite `t = linalg.transpose(W); matmul(A, t)` into a transpose-B matmul in
    the indexing_maps form (B map (m,n,k)->(n,k)), and drop the now-dead transpose
    (+ its tensor.empty). This keeps the transpose *in-register* during lowering
    (an in-loop vector.transpose fused before the dpas, the proven transpose-B
    path) instead of the physical form, which materializes the whole transposed
    weight up-front (extra load + a second layout that crashes WG lowering).
    Idempotent-ish: only rewrites matmuls whose 2nd operand is a transpose result.
    """
    # map: transpose-result SSA -> (source SSA, source tensor type)
    tps = {}
    for m in re.finditer(
        r"(%\w+) = linalg\.transpose ins\((%\w+)\s*:\s*(tensor<[^>]+>)\)", code
    ):
        tps[m.group(1)] = (m.group(2), m.group(3))
    if not tps:
        return code

    maps = (
        "indexing_maps = [affine_map<(m, n, k) -> (m, k)>, "
        "affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] "
    )

    def _mm(m):
        a, b, ta, _tb = m.group(1), m.group(2), m.group(3), m.group(4)
        if b not in tps:
            return m.group(0)
        src, src_ty = tps[b]
        return (
            f"linalg.matmul {maps}ins({a}, {src} : {ta}, {src_ty})"
        )

    code = re.sub(
        r"linalg\.matmul\s+ins\((%\w+), (%\w+)\s*:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)",
        _mm,
        code,
    )
    # drop dead transpose ops and their feeding tensor.empty (best-effort:
    # remove the transpose line; the empty becomes dead and is DCE'd downstream).
    code = re.sub(r"\s*%\w+ = linalg\.transpose ins\([^\n]*\n", "\n", code)
    return code


def parse_mlp(code: str, executor=None, large_grf: bool = False) -> list[MlpLayer] | None:
    """Parse an imported MLP into a list of MlpLayer, in execution order.

    Expects the transpose-folded form (call fold_transpose_into_matmul first):
    >=2 transpose-B linalg.matmul (indexing_maps B = (n,k)), each followed by
    elementwise linalg.generic (bias/activation). Returns None if it isn't this
    shape (e.g. a conv/norm slipped in, or a plain non-transpose-B matmul).

    *executor*: if given (an MlirExecutor), each layer's tile is AUTOTUNED on the
    GPU (time candidates, pick fastest) instead of first-divisible. Distinct layer
    shapes are autotuned once and cached (level3 MLPs repeat the same shape many
    times, so this keeps the tuning cost ~= number of distinct shapes).
    *large_grf*: autotune over large-GRF tiles (<=32 subgroups); the caller applies
    the igc flag module-wide (all layers share it). Requires executor.
    """
    # Must be folded first: any *real* transpose op left (ignore mentions in
    # comments) means the input wasn't run through fold_transpose_into_matmul.
    if re.search(r"^\s*%\w+ = linalg\.transpose\b", code, re.MULTILINE):
        return None
    # transpose-B matmul: ins(%A : tensor<MxK>, %W : tensor<NxK>) with indexing_maps.
    matmuls = re.findall(
        r"linalg\.matmul\s+indexing_maps.*?ins\(%\w+, %\w+\s*:\s*"
        r"tensor<(\d+)x(\d+)x[^,>]+>,\s*tensor<(\d+)x(\d+)x[^>]+>\)",
        code,
    )
    if len(matmuls) < 2:
        return None
    tile_cache: dict[tuple[int, int, int], tuple | None] = {}
    layers: list[MlpLayer] = []
    for am, ak, wn, wk in ((int(a), int(b), int(c), int(d)) for a, b, c, d in matmuls):
        # A is [M, K]; W is stored [N, K] (transpose-B). K must agree.
        m, k, n = am, ak, wn
        if wk != k:
            return None
        shape = (m, n, k)
        if shape not in tile_cache:
            tile_cache[shape] = (
                autotune_layer_tile(executor, m, n, k, large_grf=large_grf)
                if executor is not None
                else _pick_tile(m, n, k)
            )
        tile = tile_cache[shape]
        if tile is None:
            return None
        wm, wnn, sm, sn, kt = tile
        layers.append(
            MlpLayer(m=m, n=n, k=k, transpose_b=True,
                     wg_m=wm, wg_n=wnn, sg_m=sm, sg_n=sn, k_tile=kt)
        )
    return layers


def render_mlp_recipe(layers: list[MlpLayer], with_casts: bool = False) -> tuple[str, str]:
    """Generate (stage1_tile_vectorize, stage3_annotate) transform libraries for
    an N-layer MLP. The layers are matched in program order and each is tiled +
    fused + annotated with its own config.
    """
    n = len(layers)
    idx = lambda base: [f"%{base}{i}" for i in range(n)]  # noqa: E731

    # ---- stage 1: fuse-elementwise, then per-layer tile/fuse/k-tile ----
    s1 = [
        "module attributes {transform.with_named_sequence} {",
        "  transform.named_sequence @__transform_main(%root: !transform.any_op) {",
        '    %f0 = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op',
        '    %ff0 = transform.apply_registered_pass "linalg-fuse-elementwise-ops" to %f0 : (!transform.any_op) -> !transform.any_op',
    ]
    # Match the N matmuls + fills (stable counts) and split into per-layer handles.
    # The epilogue generic is found per-layer as the matmul's CONSUMER (not by
    # splitting all generics) — that's robust to extra elementwise producers such as
    # the f32->f16 operand casts (cast_matmul_operands_to_f16), whose generics would
    # otherwise inflate the generic count and break a split_handle.
    def _split(match_op, base):
        names = ", ".join(idx(base))
        s1.append(
            f'    %{base}_all = transform.structured.match ops{{["{match_op}"]}} in %root : (!transform.any_op) -> !transform.any_op'
        )
        rets = "(" + ", ".join("!transform.any_op" for _ in range(n)) + ")"
        s1.append(
            f"    {names} = transform.split_handle %{base}_all : (!transform.any_op) -> {rets}"
        )

    _split("linalg.matmul", "mm")
    _split("linalg.fill", "fill")
    for i, L in enumerate(layers):
        s1 += [
            # epilogue = matmul's consumer (the fused bias+activation generic).
            f"    %epi{i} = transform.get_consumers_of_result %mm{i}[0] : (!transform.any_op) -> !transform.any_op",
            f"    %t{i}, %fa{i} = transform.structured.tile_using_forall %epi{i} tile_sizes [{L.wg_m}, {L.wg_n}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            f"    %fm{i}, %fml{i} = transform.structured.fuse_into_containing_op %mm{i} into %fa{i} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)",
            f"    %ffl{i}, %ffll{i} = transform.structured.fuse_into_containing_op %fill{i} into %fa{i} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)",
            f"    %tk{i}, %kl{i} = transform.structured.tile_using_for %fm{i} tile_sizes [0, 0, {L.k_tile}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
        ]
        if with_casts:
            # Fuse the A/B f32->f16 cast producers into the k-loop so they become
            # in-register arith.truncf on the loaded tiles (XeGPU wants f16 A/B).
            s1 += [
                f"    %casta{i} = transform.get_producer_of_operand %tk{i}[0] : (!transform.any_op) -> !transform.any_op",
                f"    %fcasta{i}, %fcastal{i} = transform.structured.fuse_into_containing_op %casta{i} into %kl{i} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)",
                f"    %castb{i} = transform.get_producer_of_operand %tk{i}[1] : (!transform.any_op) -> !transform.any_op",
                f"    %fcastb{i}, %fcastbl{i} = transform.structured.fuse_into_containing_op %castb{i} into %kl{i} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)",
            ]
    s1 += [
        '    %fv = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op',
        "    %v = transform.structured.vectorize_children_and_apply_patterns %fv {fold_type_extensions_into_contract} : (!transform.any_op) -> !transform.any_op",
        '    %fc = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op',
        "    transform.apply_patterns to %fc {",
        "      transform.apply_patterns.vector.cast_away_vector_leading_one_dim",
        "      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast",
        "    } : !transform.any_op",
        '    %fh = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op',
        '    %h1 = transform.apply_registered_pass "loop-invariant-subset-hoisting" to %fh : (!transform.any_op) -> !transform.any_op',
        '    %h2 = transform.apply_registered_pass "canonicalize" to %h1 : (!transform.any_op) -> !transform.any_op',
        "    %bmod = transform.bufferization.one_shot_bufferize %root {function_boundary_type_conversion = 1 : i32, bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op",
        '    %fa_all = transform.structured.match ops{["scf.forall"]} in %bmod : (!transform.any_op) -> !transform.any_op',
    ]
    # per-layer forall_to_parallel (split), then gpu conversion, then per-launch threads
    fa_names = ", ".join(f"%pa{i}" for i in range(n))
    fa_rets = "(" + ", ".join("!transform.any_op" for _ in range(n)) + ")"
    s1.append(f"    {fa_names} = transform.split_handle %fa_all : (!transform.any_op) -> {fa_rets}")
    for i in range(n):
        s1.append(f"    %par{i} = transform.loop.forall_to_parallel %pa{i} : (!transform.any_op) -> !transform.any_op")
    s1 += [
        '    %f2 = transform.structured.match ops{["func.func"]} in %bmod : (!transform.any_op) -> !transform.any_op',
        '    %g1 = transform.apply_registered_pass "gpu-map-parallel-loops" to %f2 : (!transform.any_op) -> !transform.any_op',
        '    %g2 = transform.apply_registered_pass "convert-parallel-loops-to-gpu" to %g1 : (!transform.any_op) -> !transform.any_op',
        '    %g3 = transform.apply_registered_pass "gpu-launch-sink-index-computations" to %g2 : (!transform.any_op) -> !transform.any_op',
        '    %launch_all = transform.structured.match ops{["gpu.launch"]} in %bmod : (!transform.any_op) -> !transform.any_op',
    ]
    lz_names = ", ".join(f"%lz{i}" for i in range(n))
    s1.append(f"    {lz_names} = transform.split_handle %launch_all : (!transform.any_op) -> {fa_rets}")
    for i, L in enumerate(layers):
        s1.append(f"    transform.xegpu.set_gpu_launch_threads %lz{i} threads = [{L.nb_threads}, 1, 1] : !transform.any_op")
    s1 += ["    transform.yield", "  }", "}"]

    # ---- stage 3: per-gpu.func annotation ----
    s3 = [
        "module attributes {transform.with_named_sequence} {",
        "  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {",
        '    %funcs = transform.structured.match ops{["gpu.func"]} in %root : (!transform.any_op) -> !transform.any_op',
    ]
    gf_names = ", ".join(f"%gf{i}" for i in range(n))
    s3.append(f"    {gf_names} = transform.split_handle %funcs : (!transform.any_op) -> {fa_rets}")
    for i, L in enumerate(layers):
        gm, gn = L.sg_grid_m, L.sg_grid_n
        s3 += [
            f'    %d{i} = transform.structured.match ops{{["xegpu.dpas"]}} in %gf{i} : (!transform.any_op) -> !transform.any_op',
            f"    %a{i}v = transform.get_operand %d{i}[0] : (!transform.any_op) -> !transform.any_value",
            f"    %a{i} = transform.xegpu.get_load_op %a{i}v : (!transform.any_value) -> !transform.any_op",
            f"    %b{i}v = transform.get_operand %d{i}[1] : (!transform.any_op) -> !transform.any_value",
            f"    %b{i} = transform.xegpu.get_load_op %b{i}v : (!transform.any_value) -> !transform.any_op",
            f"    transform.xegpu.set_anchor_layout %a{i} sg_layout = [{gm}, 1] sg_data = [{L.sg_m}, {L.k_tile}] inst_data = [8, 16] : !transform.any_op",
            f"    transform.xegpu.set_anchor_layout %b{i} sg_layout = [{gn}, 1] sg_data = [{L.sg_n}, {L.k_tile}] inst_data = [16, 16] : !transform.any_op",
            f"    transform.xegpu.set_anchor_layout %d{i} index = 0 sg_layout = [{gm}, 1] sg_data = [{L.sg_m}, {L.k_tile}] inst_data = [8, 16] : !transform.any_op",
            f"    transform.xegpu.set_anchor_layout %d{i} index = 1 sg_layout = [1, {gn}] sg_data = [{L.k_tile}, {L.sg_n}] inst_data = [16, 16] : !transform.any_op",
            f"    transform.xegpu.set_anchor_layout %d{i} index = 2 sg_layout = [{gm}, {gn}] sg_data = [{L.sg_m}, {L.sg_n}] inst_data = [8, 16] : !transform.any_op",
            f'    %s{i} = transform.structured.match ops{{["xegpu.store_nd"]}} in %gf{i} : (!transform.any_op) -> !transform.any_op',
            f"    transform.xegpu.set_anchor_layout %s{i} sg_layout = [{gm}, {gn}] sg_data = [{L.sg_m}, {L.sg_n}] inst_data = [8, 16] : !transform.any_op",
            f'    %bc{i} = transform.structured.match ops{{["xegpu.load"]}} in %gf{i} : (!transform.any_op) -> !transform.any_op',
            f"    transform.xegpu.set_anchor_layout %bc{i} index = 0 sg_layout = [{gm}, {gn}] sg_data = [{L.sg_m}, {L.sg_n}] inst_data = [8, 16] slice_dims = [0] : !transform.any_op",
        ]
    s3 += ["    transform.yield", "  }", "}"]

    return "\n".join(s1) + "\n", "\n".join(s3) + "\n"


def synthesize_mlp_run_harness(wg_code: str, layers: list[MlpLayer]) -> str | None:
    """Wrap an N-kernel MLP WG module in a runnable single-@main harness.

    The lowered MLP has one gpu.module per layer (layer i: C_i = act(H_{i-1}·W_i^T +
    bias_i), H_{-1} = A). This emits @main that gpu.allocs the inputs (A + per-layer
    W_i/bias_i) and the intermediate/output buffers, then launches the N kernels in
    sequence, chaining H_{i-1} -> H_i. Buffers are uninitialized (a *timing/run*
    harness, matching attention_lowering.synthesize_run_harness); correctness is
    checked separately. Returns None if the kernels can't be parsed.

    Per-kernel launch bounds: threads = known_block_size; grid = 2-D (M/wg_m, N/wg_n)
    when the kernel indexes block_id y, else 1-D ((M/wg_m)*(N/wg_n)) — matching how
    the epilogue-tiled kernel maps work-groups (varies with the tile).
    Kernel ABI (from the recipe): (A_in, W, C_out, bias).
    """
    # brace-match each gpu.module
    mods, idx = [], 0
    while True:
        i = wg_code.find("gpu.module", idx)
        if i == -1:
            break
        depth, start = 0, wg_code.find("{", i)
        end = None
        for j in range(start, len(wg_code)):
            if wg_code[j] == "{":
                depth += 1
            elif wg_code[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            return None
        mods.append(wg_code[i:end])
        idx = end
    if len(mods) != len(layers):
        return None

    mod_names, inner_names, blocks, two_d = [], [], [], []
    for m in mods:
        mn = re.search(r"gpu\.module @(\w+)", m)
        fn = re.search(r"gpu\.func @(\w+)", m)
        bs = re.search(r"known_block_size = array<i32:\s*(\d+)", m)
        if not (mn and fn and bs):
            return None
        mod_names.append(mn.group(1))
        inner_names.append(fn.group(1))
        blocks.append(int(bs.group(1)))
        two_d.append("block_id y" in m)

    aliases = "\n".join(ln for ln in wg_code.splitlines() if ln.lstrip().startswith("#map"))

    # index constants needed
    consts = {0, 1}
    for L, b, td in zip(layers, blocks, two_d):
        consts.add(b)
        if td:
            consts.update({L.m // L.wg_m, L.n // L.wg_n})
        else:
            consts.add((L.m // L.wg_m) * (L.n // L.wg_n))
    cst = "\n".join(f"    %c{v} = arith.constant {v} : index" for v in sorted(consts))

    M = layers[0].m
    lines = [aliases, "module attributes {gpu.container_module} {"]
    lines += [_indent_module(m) for m in mods]
    lines.append("  func.func @main() attributes {llvm.emit_c_interface} {")
    lines.append(cst)
    K0 = layers[0].k
    lines.append(f"    %A = gpu.alloc() : memref<{M}x{K0}xf16>")
    prev, prev_k = "%A", K0
    for i, (L, mod, inner, b, td) in enumerate(
        zip(layers, mod_names, inner_names, blocks, two_d)
    ):
        last = i == len(layers) - 1
        oty = "f32" if last else "f16"
        lines.append(f"    %W{i} = gpu.alloc() : memref<{L.n}x{L.k}xf16>")
        lines.append(f"    %bias{i} = gpu.alloc() : memref<{L.n}xf32>")
        lines.append(f"    %H{i} = gpu.alloc() : memref<{L.m}x{L.n}x{oty}>")
        if td:
            gx, gy = L.m // L.wg_m, L.n // L.wg_n
            gstr = f"blocks in (%c{gx}, %c{gy}, %c1)"
        else:
            gstr = f"blocks in (%c{(L.m // L.wg_m) * (L.n // L.wg_n)}, %c1, %c1)"
        lines.append(
            f"    gpu.launch_func @{mod}::@{inner} {gstr} threads in (%c{b}, %c1, %c1) "
            f"args({prev} : memref<{L.m}x{prev_k}xf16>, %W{i} : memref<{L.n}x{L.k}xf16>, "
            f"%H{i} : memref<{L.m}x{L.n}x{oty}>, %bias{i} : memref<{L.n}xf32>)"
        )
        lines.append("    gpu.wait")
        prev, prev_k = f"%H{i}", L.n
    lines += ["    return", "  }", "}"]
    return "\n".join(lines) + "\n"


def cast_matmul_operands_to_f16(code: str) -> str:
    """Insert f32->f16 truncation on every linalg.matmul's A and B operands.

    XeGPU/DPAS requires the matmul *inputs* (A, B) to be f16/bf16; the accumulator
    (C) stays f32. Torch-MLIR imports KernelBench in pure f32 (A, B, and C all f32),
    which won't hit the XMX path. This rewrites each

        linalg.matmul ins(%a, %b : tensor<..xf32>, tensor<..xf32>) outs(%c ..f32)

    into a pair of `linalg.generic` truncf casts producing f16 A/B, then a matmul
    reading those f16 operands into the same f32 accumulator. The casts are ordinary
    elementwise producers — the MLP recipe's linalg-fuse-elementwise-ops + tile
    (fuse producers) folds them into the WG kernel (an in-kernel truncf on load).

    Idempotent: matmuls whose operands are already f16 are left unchanged. Handles
    the plain, indexing_maps (transpose-B), AND batch_matmul (3-D operand) spellings.
    """
    # Identity elementwise maps, one per operand rank (2-D matmul, 3-D batch_matmul).
    hdrs = {
        2: ("#__castmap2", "affine_map<(d0, d1) -> (d0, d1)>", '"parallel", "parallel"'),
        3: ("#__castmap3", "affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
            '"parallel", "parallel", "parallel"'),
    }
    used_ranks: set[int] = set()
    counter = [0]

    def _emit_cast(ssa: str, shape: str) -> tuple[str, str]:
        rank = shape.count("x") + 1
        mapname, _mapdef, iters = hdrs[rank]
        used_ranks.add(rank)
        i = counter[0]
        counter[0] += 1
        out = f"%__f16_{i}"
        emp = f"%__e16_{i}"
        block = (
            f"    {emp} = tensor.empty() : tensor<{shape}xf16>\n"
            f"    {out} = linalg.generic {{indexing_maps = [{mapname}, {mapname}], "
            f"iterator_types = [{iters}]}} ins({ssa} : tensor<{shape}xf32>) "
            f"outs({emp} : tensor<{shape}xf16>) {{\n"
            f"    ^bb0(%__in: f32, %__o: f16):\n"
            f"      %__t = arith.truncf %__in : f32 to f16\n"
            f"      linalg.yield %__t : f16\n"
            f"    }} -> tensor<{shape}xf16>\n"
        )
        return out, block

    # Match a matmul OR batch_matmul, capture the two operands + their f32 shapes.
    mm_re = re.compile(
        r"(?P<res>%\w+) = linalg\.(?P<op>matmul|batch_matmul)(?P<attrs>\s+indexing_maps\s*=\s*\[[^\]]*\])?"
        r"\s+ins\((?P<a>%\w+), (?P<b>%\w+)\s*:\s*tensor<(?P<as>[0-9x]+)xf32>,\s*"
        r"tensor<(?P<bs>[0-9x]+)xf32>\)\s+outs\((?P<c>%\w+)\s*:\s*(?P<cty>tensor<[0-9x]+xf32>)\)"
        r"\s*->\s*(?P<rty>tensor<[0-9x]+xf32>)"
    )
    pre_blocks: list[str] = []

    def _repl(m):
        aout, ablk = _emit_cast(m.group("a"), m.group("as"))
        bout, bblk = _emit_cast(m.group("b"), m.group("bs"))
        pre_blocks.append((m.group("res"), ablk + bblk))
        attrs = m.group("attrs") or ""
        return (
            f'{m.group("res")} = linalg.{m.group("op")}{attrs} ins({aout}, {bout} : '
            f'tensor<{m.group("as")}xf16>, tensor<{m.group("bs")}xf16>) '
            f'outs({m.group("c")} : {m.group("cty")}) -> {m.group("rty")}'
        )

    new = mm_re.sub(_repl, code)
    if new == code:
        return code  # nothing to cast (already f16, or no matching matmul)
    # Insert each matmul's cast block immediately before that matmul line.
    for res, cast_block in pre_blocks:
        idx = new.find(f"    {res} = linalg.")
        if idx == -1:
            idx = new.find(f"{res} = linalg.")
        if idx != -1:
            line_start = new.rfind("\n", 0, idx) + 1
            new = new[:line_start] + cast_block + new[line_start:]
    cast_hdr = "".join(
        f"{hdrs[r][0]} = {hdrs[r][1]}\n" for r in sorted(used_ranks)
        if hdrs[r][0] not in code
    )
    return cast_hdr + new
