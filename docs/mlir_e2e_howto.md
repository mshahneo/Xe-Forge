# Running the MLIR / XeGPU backend end-to-end (how to reproduce)

This is the reproducible flow for Xe-Forge's MLIR path: take a **Linalg** kernel
(matmul, batched matmul, transpose-B / nn.Linear, or a multi-layer MLP), lower it
to Intel **XeGPU workgroup-level**, autotune + optimize it, and run it on the GPU.

Target: Intel Xe2 "Battlemage" (BMG). Branch: `feat/mlir-xegpu-backend`.

---

## 1. Prerequisites

- An **Intel Xe GPU** (validated on Battlemage) with the Level-Zero runtime.
- An upstream **LLVM + mlir-extensions (IMEX)** build providing `imex-opt`,
  `mlir-opt`, `mlir-runner`, and the runtime libs (`libmlir_levelzero_runtime.so`,
  `libmlir_runner_utils.so`, `libmlir_c_runner_utils.so`, `libimex_runner_utils.so`).
- **Python ≥ 3.11** (the package uses `StrEnum`). The core MLIR path is torch-free.
- An **OpenAI-compatible LLM endpoint** (any litellm `openai/...` model) for the
  WG-level optimization stages. The lowering + autotune + GPU run work without it.
- *(Only if importing raw PyTorch/KernelBench kernels)* Intel oneAPI + a
  Torch-MLIR-enabled env — see §6.

---

## 2. Setup

```bash
git clone https://github.com/mshahneo/Xe-Forge && cd Xe-Forge
git checkout feat/mlir-xegpu-backend

# Python env (jinja2 + dspy + deps; no torch needed for the MLIR lowering path)
uv venv --python 3.11 .venv-xf
uv pip install --python .venv-xf/bin/python -e .
```

## 3. Environment (`.env`)

```bash
# --- LLM (litellm: provider/model). Xe-Forge passes OPENAI_API_KEY to dspy.LM. ---
LLM_MODEL=openai/<your-model>       # e.g. openai/qwen.qwen3-coder-480b-a35b-instruct
LLM_MODEL_TYPE=chat                 # "chat" for chat-completions proxies; else "responses"
OPENAI_API_KEY=<your-key>
# OPENAI_API_BASE=<endpoint>        # or set OPENAI_BASE_URL (accepted as fallback)
LLM_TIMEOUT=180                     # per-request seconds (avoids hung runs)
LLM_NUM_RETRIES=2

# --- MLIR / XeGPU backend ---
DSL=mlir
DEVICE_TYPE=xpu
KNOWLEDGE_BASE_ENABLED=true
KNOWLEDGE_DIR=./knowledge_base
# Point these at your LLVM+IMEX build:
MLIR_BIN_DIR=/path/to/llvm-project/build-with-imex/bin
MLIR_LIB_DIR=/path/to/llvm-project/build-with-imex/lib
```

> **Model note:** large kernel-rewrite generations need a model that returns big
> outputs reliably. `qwen.qwen3-coder-480b-a35b-instruct` runs the WG optimization
> stages to completion; some models stall on the large rewrites (the request
> timeout + retries above bound the damage, and the pipeline continues).

---

## 4. Run end-to-end through the CLI

The CLI auto-detects the kernel shape and routes it — **no special flags per
kernel type.** A single plain matmul goes through the tile-search path; a
multi-matmul chain (≥2 `linalg.matmul`) is recognized as an MLP and lowered to N
chained kernels; f32 operands are auto-cast to f16 (XeGPU needs 16-bit-float A/B).

```bash
set -a; source .env; set +a
export OPENAI_API_BASE="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"   # bridge if only BASE_URL is set

.venv-xf/bin/python -m xe_forge.cli \
  --input <kernel>.mlir \
  --name <run-name> \
  --dsl mlir \
  --output outputs/<run-name>.opt.mlir \
  --no-trials
```

Ready-to-run example inputs live in `pipelines/linalg_to_wg/examples/`:

| Input file | What it exercises |
|---|---|
| `matmul.mlir` | plain `linalg.matmul` (single GEMM, tile-search path) |
| `matmul_transpose_b.mlir` | transpose-B matmul (nn.Linear form) |
| `batch_matmul.mlir` | batched matmul |
| `gemm_elementwise_epilogue.mlir` | matmul + fused elementwise epilogue (level-2 Gemm chain) |
| `mlp_layer.mlir` | one fused MLP layer `relu(A·Bᵀ + bias)` |
| `mlp_2layer_bare.mlir` | 2-layer MLP chain (f16) — routes to the multi-kernel MLP path |
| `mlp_2layer_f32.mlir` | 2-layer MLP in **all-f32** — exercises the auto f32→f16 A/B cast |

Example (a 2-layer MLP, genuinely end-to-end through the pipeline):

```bash
.venv-xf/bin/python -m xe_forge.cli \
  --input pipelines/linalg_to_wg/examples/mlp_2layer_bare.mlir \
  --name mlp2 --dsl mlir --output outputs/mlp2.opt.mlir --no-trials
```

**Expected log stages:**
```
STAGE: LINALG_LOWERING — MLP chain of 2 layers (autotune, large_grf=...)
  ... autotune LoweringConfig(...): OK <ms>          # per-layer tile timed on GPU
MLP lowered to 2 chained XeGPU-WG kernels (runnable harness synthesized).
STAGE: ANALYSIS         — Detected N issues
STAGE: PLANNING
STAGE: ALGORITHMIC / FUSION / MEMORY_ACCESS / DEVICE_SPECIFIC
OPTIMIZATION COMPLETE
Optimized kernel saved to: outputs/mlp2.opt.mlir
```
For a single matmul the first stage instead reads
`LINALG_LOWERING — sweeping N configs for MxNxK` with a per-config `Average time
(ms)` and a chosen `best config`. A kernel already at XeGPU WG level (contains
`xegpu.load_nd`/`dpas`) skips LINALG_LOWERING and goes straight to the WG stages.

---

## 5. Verify / benchmark a lowered kernel on the GPU

The single-matmul path emits a self-contained, verifiable kernel:

```bash
$MLIR_BIN_DIR/imex-opt outputs/<run-name>.opt.mlir \
    --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
  | $MLIR_BIN_DIR/mlir-runner -e main --entry-point-result=void \
      --shared-libs=$MLIR_LIB_DIR/libmlir_levelzero_runtime.so \
      --shared-libs=$MLIR_LIB_DIR/libmlir_runner_utils.so \
      --shared-libs=$MLIR_LIB_DIR/libmlir_c_runner_utils.so \
      --shared-libs=$MLIR_LIB_DIR/libimex_runner_utils.so
# Expect: [ALLCLOSE: TRUE]  and  Average time (ms): <value>
```

> For the multi-kernel MLP output, add the large-GRF igc flag when the run used it:
> `--gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup igc-cmd-options=-ze-opt-large-register-file"`.

### Just the lowering (no LLM) — sanity-check the Linalg→WG bridge
```bash
# Bare linalg.matmul in, XeGPU WG-level IR out (uses the default config).
pipelines/linalg_to_wg/lower.sh pipelines/linalg_to_wg/examples/matmul.mlir /tmp/out_wg.mlir
```

---

## 6. (Optional) Import a raw PyTorch / KernelBench kernel to Linalg

KernelBench kernels are PyTorch modules; get their Linalg via lighthouse's
Torch-MLIR importer, then feed the dump to the CLI above. Torch-MLIR needs its own
env (Intel oneAPI + `torch-mlir` installed):

```bash
source /path/to/intel/oneapi/setvars.sh           # provides libsycl / libpti_view for torch
# in an env with lighthouse + torch-mlir installed:
python examples/KernelBench/test-kernel-bench.py --kernel level1/1_ --print-original-module
```

The dump is **all-f32** (A, B, and C). That's fine — the CLI auto-casts A/B to
f16 (XeGPU requires 16-bit-float inputs; C stays f32). See the KB constraint
`lowering_matmul_ab_must_be_16bit_float`.

---

## 7. Notes & scope

- **Autotuning** is on by default for the MLP path (each distinct layer shape's
  tile is timed on GPU); the single-matmul path ranks LLM-proposed configs by
  measured time. Large-GRF tiles (`sg=[64,32]`/`[32,64]`, ≤32 subgroups) are
  included and often win on compute-bound GEMM.
- **Dtypes:** A/B must be a 16-bit float (f16 or bf16); C/accumulator may be f32.
  f32 A/B are auto-cast on all paths.
- **Shapes:** must be divisible by the candidate tile sizes (256/128/64/32); the
  config's `fits_shape()` filters incompatible tiles automatically. Prime/irregular
  shapes (no boundary-tile handling yet) are skipped.
- **Target chip** defaults to `bmg` in `pipelines/linalg_to_wg/lower.sh` and the
  templates; change it for other Xe targets.
- **Currently out of scope:** transpose-A / transpose-both matmul, convolutions,
  and reduction/norm/softmax epilogues. See `docs/kernelbench_coverage.md` for the
  full per-kernel matrix.
