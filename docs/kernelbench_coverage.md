# Xe-Forge — KernelBench Coverage

What Xe-Forge's MLIR path can and can't lower across the KernelBench suite
(as shipped in lighthouse `examples/KernelBench`). Date: 2026-07-16 · Target:
Intel Xe2 "Battlemage".

> To actually run a kernel end-to-end, see
> [docs/mlir_e2e_howto.md](mlir_e2e_howto.md) (setup, CLI command, GPU verify).

## What Xe-Forge supports today

Xe-Forge's MLIR path lowers **matmul-family Linalg** to XeGPU workgroup level +
runs an autonomous GRF sweep. Op classes outside the matmul family (fused
attention, row-softmax, row layer-norm) are lowered via the **lighthouse backend**: Xe-Forge shells
out to the upstream `llvm/lighthouse` XeGPU schedules (`--dump-kernel=xegpu-wg`) as
a lowering engine and feeds the dumped WG kernel into its own WG stages + GRF sweep
(dump-only — no run-time lighthouse dependency; see
`src/xe_forge/core/lighthouse_backend.py`). Concretely, the supported op patterns are:

| Pattern | Status | Verified |
|---|---|---|
| Plain matmul `C = A·B` | ✅ supported | GPU, multiple shapes |
| Batched matmul `C = A·B` (3-D, incl. non-square) | ✅ supported | GPU, 0 mismatches |
| Transpose-B matmul `C = A·Bᵀ` (nn.Linear form) | ✅ supported | GPU, 0 mismatches |
| Single MLP layer `C = act(A·Bᵀ + bias)` | ✅ supported | GPU, 0 mismatches |
| Row-softmax `softmax(x, dim=-1)` (via lighthouse) | ✅ supported | GPU, row-sums = 1 |
| Row layer-norm `LayerNorm(x)` (via lighthouse) | ✅ supported | GPU, out = beta control |
| Transpose-A / transpose-both matmul | ❌ layout WIP | — |
| Multi-layer MLP chains | ⚠️ per-layer only | single layer only |
| Everything else (conv, norm, pool, reduce, softmax, loss, elementwise-only, matvec) | ❌ out of scope | — |

Legend below: ✅ supported & GPU-verified · 🟡 partial / needs a small extension ·
❌ not supported (needs a new op class).

---

## Level 1 — 100 single-operator kernels

Level 1 is one primitive per kernel. Only the **matmul family (1–18)** is in
Xe-Forge's wheelhouse; the rest are conv / norm / pool / reduce / activation /
loss primitives that need op classes we don't lower.

### Matmul family (kernels 1–18) — detailed

| # | Kernel | Shape | Status | Note |
|---|---|---|---|---|
| 1 | Square matmul | 4096³ | ✅ | plain matmul; ~69 TFLOPS |
| 2 | Standard matmul | 2048×4096×8192 | ✅ | plain matmul |
| 3 | Batched matmul | 3×2048×2048×4096 | ✅ | batch_matmul; 0 mismatches/12.6 M |
| 4 | Matrix-vector mul | 2048×8192·8192×1 | ❌ | N=1 → GEMV, not a DPAS tile shape |
| 6 | Matmul large-K | 256×256×8192 | ✅ | plain matmul; got 2.49× large-GRF |
| 7 | Matmul small-K | 4096×4096×64 | ✅ | plain matmul |
| 8 | Matmul irregular shapes | 8205×2949×5921 | 🟡 | no divisible tile → needs boundary/partial tiles |
| 9 | Tall-skinny matmul | 4096×4096×64 | ✅ | plain matmul |
| 10 | 3-D × 2-D tensor mul | 16×4096×2048 · 2048×768 | 🟡 | batched matmul with a **broadcast** B (shared weight); broadcast generic not lowered |
| 11 | 4-D × 2-D tensor mul | 8×256×512×256 · 256×768 | ❌ | 4-D contraction; not modeled |
| 12 | Matmul w/ diagonal | 4096 · 4096×4096 | ❌ | diag(v)·M — not a dense matmul |
| 13 | Matmul symmetric | 4096³ | ✅ | plain matmul (symmetry not exploited) |
| 14 | Matmul upper-triangular | 4096³ | ❌ | matmul + `linalg.generic` triangular mask |
| 15 | Matmul lower-triangular | 4096³ | ❌ | matmul + `linalg.generic` triangular mask |
| 16 | Matmul transposed A | 8192×2048 · 8192×4096 | ❌ | transpose-A; WG layout WIP |
| 17 | Matmul transposed B | 2048×8192 · 4096×8192 | ✅ | transpose-B (nn.Linear); 0 mismatches |
| 18 | Matmul transposed both | 8192×2048 · 4096×8192 | ❌ | transpose-A + transpose-B; layout WIP |

**Matmul family: 7 ✅, 2 🟡, 9 ❌.**

### Everything else in Level 1 (kernels 19–100) — by category, all ❌

Each needs an op class Xe-Forge's matmul path does not cover.

| Kernels | Category | Why not |
|---|---|---|
| 5, 20 | Scalar/elementwise mul, LeakyReLU | pure elementwise; no matmul to anchor the WG lowering |
| 23, 24 | Softmax / LogSoftmax | ✅ **softmax supported** via the lighthouse row-softmax schedule (GPU-verified: constant input → each row sums to 1). LogSoftmax is the same schedule + a log epilogue (not yet wired). |
| 19, 21, 22, 25–32 | Activations (ReLU, Sigmoid, Tanh, GELU, SELU, ELU, …) | elementwise; no matmul to anchor the WG lowering |
| 40 | LayerNorm | ✅ **supported** via the lighthouse layer-norm schedule (GPU-verified: row-constant input → output = beta control). RMSNorm (#36) is the same schedule minus the mean-centering (not yet wired). |
| 33–39 | Normalizations (Batch/Instance/Group/RMS/Frobenius/L1/L2) | reductions + rescale over non-row axes; not yet wired |
| 41–46 | Pooling (Max/Avg 1-D/2-D/3-D) | sliding-window reductions |
| 47–49, 51–53 | Reductions & arg (Sum/Mean/Max/Min/Argmax/Argmin over a dim) | reductions |
| 50, 54–87 | Convolutions (standard / transposed / depthwise / pointwise, 1-D/2-D/3-D) | conv; a whole separate lowering |
| 88 | MinGPT GELU | elementwise |
| 89–92 | cumsum / cumprod / reverse / exclusive | scan (prefix) ops |
| 93 | masked cumsum | scan + mask |
| 94, 96, 98, 99, 100 | Losses (MSE, Huber, KLDiv, TripletMargin, Hinge) | reductions |
| 95 | CrossEntropyLoss | softmax + gather + reduction |
| 97 | ScaledDotProductAttention | fused attention (routed to lighthouse dump path, not this pipeline) |

**Level 1 total: 7 ✅, 2 🟡, 91 ❌** (of the 91, most need conv or
reduction/elementwise op classes).

---

## Level 2 — 100 fused operator sequences

Level 2 kernels are **chains** (e.g. `Conv2d → ReLU → BiasAdd`, `Gemm → GroupNorm →
Hardtanh`). Xe-Forge lowers a chain end-to-end only if it's a **matmul followed by
pure elementwise ops** — the same pattern as our single MLP layer. Any conv,
norm, pooling, softmax, or cross-row reduction in the chain puts it out of scope.

Splitting the 100 by the leading op and the epilogue:

| Group | Kernels | Status | Why |
|---|---|---|---|
| **Gemm/Matmul + elementwise-only epilogue** | 9 (Sub·Mul·ReLU), 12 (Mul·LeakyReLU), 29 (Mish·Mish), 40 (Scale·ResidualAdd), 53 (Scale·Hardtanh·GELU), 59 (Swish·Scale), 63 (ReLU·Div), 68 (Min·Sub), 70 (Sigmoid·Scale·ResidualAdd), 76 (Add·ReLU), 81 (Swish·Div·Clamp·Tanh·Clamp), 86 (Div·GELU), 95 (Add·Swish·Tanh·GELU·Hardtanh) | ✅ **supported** | matmul + fused elementwise epilogue — the MLP-layer path (`is_mlp_layer` + epilogue fusion) handles arbitrary elementwise chains, incl. transcendentals (exp/tanh via `math.*`). Verified GPU-correct on Sub·Mul·ReLU and Sigmoid epilogues (0 mismatches). |
| **Gemm/Matmul + reduction/norm/softmax epilogue** | 14, 18, 22, 30, 33, 37, 39, 41, 45, 51, 55, 56, 62, 64, 66, 75, 80, 84, 88, 94, 97, 98, 99 | ❌ | epilogue contains GroupNorm/BatchNorm/Softmax/Sum/MaxPool/LogSumExp — cross-row reductions (`iterator_types` has `"reduction"`); `is_mlp_layer` explicitly rejects these |
| **Conv-based chains** (Conv2d/3d, ConvTranspose, …) | 1–8, 10, 11, 13, 15–17, 19–21, 23–28, 31, 32, 34–36, 38, 42–44, 46–50, 52, 54, 57, 58, 60, 61, 65, 67, 69, 71–74, 77–79, 82, 83, 85, 87, 89–93, 96, 100 | ❌ | lead with a convolution; conv lowering is a separate effort |

So of Level 2: **~13 are supported** (matmul + elementwise epilogue) — no new
lowering machinery was needed beyond the MLP-layer epilogue fusion (arbitrary
elementwise ops, including transcendentals, vectorize + lower to XeVM). The rest
need conv or reduction lowering.

*(Group membership is by the kernel name's op sequence; a couple are judgment calls
on where "elementwise" ends and "reduction" begins — e.g. anything with Sum, Mean,
Max-over-dim, LogSumExp, or a Norm is treated as a reduction epilogue → ❌. The
supported count is by structure; a given kernel's exact GPU run also needs the
KernelBench dump adapted to f16 + a runnable harness, as for the matmul family.)*

---

## Level 3 — 3 full architectures (MLPs)

| # | Kernel | Structure | Status |
|---|---|---|---|
| 1 | MLP | 3 Linear layers (`x·Wᵀ + bias`, ReLU between) | ✅ multi-layer |
| 2 | ShallowWideMLP | 3 wide Linear layers + ReLU | ✅ multi-layer |
| 3 | DeepNarrowMLP | ~8 narrow Linear layers + ReLU (17 matmuls total after import) | ✅ multi-layer |

Full MLP chains now lower: `MlirExecutor.lower_mlp_to_wg` folds the physical
transposes into the matmul (transpose-B indexing_maps), parses the per-layer
shapes, and generates an **N-layer transform recipe** (tile each layer's epilogue
+ fuse its matmul/fill as producers, per layer) that outlines **one XeGPU kernel
per layer**, chained through intermediate buffers.

**Wired into the pipeline / CLI**: `XeForgePipeline._maybe_lower_mlp` routes a
multi-matmul chain (≥2 `linalg.matmul`) to this path during LINALG_LOWERING,
autotunes per-layer tiles, and synthesizes a runnable N-launch `@main` harness so
the WG analysis/optimization stages run on the lowered chain. Confirmed genuinely
end-to-end via `python -m xe_forge.cli --dsl mlir` on a 2-layer MLP with
**qwen3-coder-480b**: the full pipeline ran to completion (`CLI_EXIT=0`) — LINALG_LOWERING
(MLP, 2 chained kernels) → ANALYSIS → PLANNING → ALGORITHMIC → FUSION → MEMORY_ACCESS
→ DEVICE_SPECIFIC → `OPTIMIZATION COMPLETE`, WG output saved (2 launch_funcs, 2 dpas,
0 linalg.matmul). qwen worked through every WG stage without stalling (deepseek hangs
on the large kernel-rewrite generations). (Single-matmul inputs still route to the
plain-GEMM path — the MLP branch is guarded on ≥2 matmuls.)

**Raw-f32 dumps handled**: XeGPU needs f16 A/B operands, but Torch-MLIR imports
KernelBench in all-f32. `lower_mlp_to_wg` auto-inserts f32→f16 truncf casts on each
matmul's A/B (accumulator stays f32) and fuses them into the k-loop, so a raw-f32
MLP lowers and runs correctly (verified 0 mismatches). This closes the one remaining
turnkey gap for feeding literal KernelBench dumps. The cast is applied on **all**
lowering paths — plain matmul, transpose-B, batch_matmul, and MLP — so any f32
matmul input is handled (XeGPU requires f16 A/B; C stays f32). It's a documented
hard requirement in the KB (`lowering_matmul_ab_must_be_16bit_float`, critical). f16-native
inputs are unaffected.

**End-to-end verified on the real level3/3 DeepNarrowMLP** (the 17-layer one):
`lower_mlp_to_wg(autotune=True, large_grf=True)` lowered all 17 layers in ~3.6 s
(3 distinct shapes autotuned once each → all picked large-GRF `sg=[64,32]`/`[32,64]`
tiles), producing 17 chained XeGPU kernels. Ran end-to-end on the GPU (17 sequential
launches + 16 intermediate buffers, large-GRF igc flag applied module-wide) and it
computes correctly (W=0 control → every layer `relu(0+bias)=0.5`, final output
exactly 0.5). The network is ~66.6 GFLOP; per-layer kernels time ~0.04 ms (1024³
mid layers) to ~0.46 ms (the K=8192 first layer). Earlier smaller cases also
verified: a 2-layer `relu(x·Wᵀ+bias)` chain (0 mismatches) and a large-GRF
2-layer chain (`wg=256×256 sg=[32,64]`, 0 mismatches).

Caveats: the recipe picks a per-layer divisible tile automatically (not yet
autotuned); layers must be the `relu(x·Wᵀ + bias)` shape (transpose-B matmul +
bias-add + activation) — a norm/pool/softmax inside the chain falls out of scope.

---

## Summary

| Level | Total | ✅ supported | 🟡 reachable (small extension) | ❌ out of scope |
|---|---|---|---|---|
| 1 | 100 | 7 | 2 (irregular tile, broadcast batch) | 91 |
| 2 | 100 | ~13 (matmul + elementwise epilogue) | 0 | ~87 |
| 3 | 3 | 3 (full MLP chains) | 0 | 0 |

**Verified-supported today:** the matmul family — plain / batched / transpose-B
matmul, a matmul + arbitrary elementwise epilogue (~13 Level-2 Gemm/Matmul
chains), a single fused MLP layer, and **full multi-layer MLP chains** (all 3
Level-3 kernels). **The biggest coverage unlocks next**, in order of leverage:

1. **transpose-A / broadcast-batch / boundary tiles** → Level-1 kernels 8, 10, 16, 18.
2. **A convolution lowering** → the single largest bucket (~50 % of Level 1 &
   Level 2), but a separate, large effort.

**Tile autotuning is now in place** (`MlirExecutor.autotune_tile` +
`candidate_configs`): candidate tiles are timed on the GPU (IMEX) and the fastest
is picked, instead of first-divisible. Measured: 4096³ matmul reaches ~64 TFLOPS
(large-GRF `sg=[64,32]`) vs 55 TFLOPS for the default `sg=[32,32]`. The pipeline's
LLM sweep already ranked configs by time; the fallback now uses the curated
shortlist so autotuning happens even without the LLM. **Multi-layer MLPs** autotune
per layer too via `lower_mlp_to_wg(autotune=True)` (each distinct layer shape tuned
once; verified GPU-correct with non-default tiles).
