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


def parse_mlp(code: str) -> list[MlpLayer] | None:
    """Parse an imported MLP into a list of MlpLayer, in execution order.

    Expects the transpose-folded form (call fold_transpose_into_matmul first):
    >=2 transpose-B linalg.matmul (indexing_maps B = (n,k)), each followed by
    elementwise linalg.generic (bias/activation). Returns None if it isn't this
    shape (e.g. a conv/norm slipped in, or a plain non-transpose-B matmul).
    """
    if "linalg.transpose" in code:
        return None  # must be folded first
    # transpose-B matmul: ins(%A : tensor<MxK>, %W : tensor<NxK>) with indexing_maps.
    matmuls = re.findall(
        r"linalg\.matmul\s+indexing_maps.*?ins\(%\w+, %\w+\s*:\s*"
        r"tensor<(\d+)x(\d+)x[^,>]+>,\s*tensor<(\d+)x(\d+)x[^>]+>\)",
        code,
    )
    if len(matmuls) < 2:
        return None
    layers: list[MlpLayer] = []
    for am, ak, wn, wk in ((int(a), int(b), int(c), int(d)) for a, b, c, d in matmuls):
        # A is [M, K]; W is stored [N, K] (transpose-B). K must agree.
        m, k, n = am, ak, wn
        if wk != k:
            return None
        tile = _pick_tile(m, n, k)
        if tile is None:
            return None
        wm, wnn, sm, sn, kt = tile
        layers.append(
            MlpLayer(m=m, n=n, k=k, transpose_b=True,
                     wg_m=wm, wg_n=wnn, sg_m=sm, sg_n=sn, k_tile=kt)
        )
    return layers


def render_mlp_recipe(layers: list[MlpLayer]) -> tuple[str, str]:
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
    # split the N epilogue generics, matmuls, fills, transposes
    def _split(match_op, base):
        names = ", ".join(idx(base))
        s1.append(
            f'    %{base}_all = transform.structured.match ops{{["{match_op}"]}} in %root : (!transform.any_op) -> !transform.any_op'
        )
        rets = "(" + ", ".join("!transform.any_op" for _ in range(n)) + ")"
        s1.append(
            f"    {names} = transform.split_handle %{base}_all : (!transform.any_op) -> {rets}"
        )

    # NOTE: fuse-elementwise coalesces bias+activation into ONE generic per layer.
    # fuse-elementwise coalesces bias+activation into ONE generic per layer;
    # transposes are already folded into the matmul indexing_maps (no transpose op).
    _split("linalg.generic", "gen")
    _split("linalg.matmul", "mm")
    _split("linalg.fill", "fill")
    for i, L in enumerate(layers):
        s1 += [
            f"    %t{i}, %fa{i} = transform.structured.tile_using_forall %gen{i} tile_sizes [{L.wg_m}, {L.wg_n}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            f"    %fm{i}, %fml{i} = transform.structured.fuse_into_containing_op %mm{i} into %fa{i} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)",
            f"    %ffl{i}, %ffll{i} = transform.structured.fuse_into_containing_op %fill{i} into %fa{i} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)",
            f"    %tk{i}, %kl{i} = transform.structured.tile_using_for %fm{i} tile_sizes [0, 0, {L.k_tile}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
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
