# MLIR / XeGPU Backend — Architecture & How It Works

**Branch:** `feat/mlir-xegpu-backend`  ·  **Target:** Intel Xe2 "Battlemage" (BMG), Xe-HPC (PVC)  ·  **Status:** working end-to-end

This document is the technical reference for the MLIR backend we added to Xe-Forge:
what it does, how it works, where every piece lives (with file/line anchors), where it
**differs** from the original Triton-oriented Xe-Forge, where it **reuses** existing
Xe-Forge machinery unchanged, and which parts of the Knowledge Base (KB) are shared vs
MLIR-specific. It exists so we can decide which path to invest in next.

> Companion docs: [mlir_backend_journey.md](mlir_backend_journey.md) (narrative/chronology),
> [mlir_e2e_howto.md](mlir_e2e_howto.md) (reproducible run steps),
> [kernelbench_coverage.md](kernelbench_coverage.md) (op coverage matrix).

---

## 1. What we built, in one paragraph

Xe-Forge was originally an LLM-driven optimizer for **Triton** kernels on Intel XPU: a
multi-stage loop (analyze → plan → optimize-with-verification → gate on
correctness+speedup) driven by a Chain-of-Verification-and-Refinement (CoVeR) agent and
an Intel-GPU Knowledge Base. We added a fifth DSL — **MLIR** — that plugs into that same
loop but operates on the MLIR/XeGPU compiler stack instead of Triton. The MLIR backend
introduces a **two-level flow** unique to it: first *lower* a high-level `linalg` kernel
to a workgroup-level (WG) XeGPU kernel (a tile-search optimization in its own right),
then run the standard analyze/optimize stages on that WG kernel. Correctness is checked
not against a PyTorch reference but against an **in-file CPU reference** embedded in the
kernel's `@main` harness (`[ALLCLOSE: TRUE]`), and a device-specific **GRF sweep** stage
exploits a correctness-free compiler knob (large register file). Ops we cannot lower
natively (attention, softmax, layer-norm) are lowered by shelling out to upstream
`llvm/lighthouse` schedules as a dump-only backend.

---

## 2. The shared optimization loop (what MLIR inherits unchanged)

The core insight of the whole effort: **the optimization loop is DSL-agnostic.** MLIR
did not fork the pipeline — it selects a different executor and swaps a few DSPy
signatures. The loop, defined in [pipeline.py](../src/xe_forge/pipeline.py) `optimize()`
(lines 656–1120), is:

```
baseline measure → [MLIR: LINALG_LOWERING] → [MLIR: GRF_SWEEP] → ANALYSIS
    → PLANNING → per-stage optimize (CoVeR verify/retry) → final measure → best-of-k
```

The following components are **reused by every DSL**, including MLIR, with only internal
branching (no MLIR-specific fork):

| Component | Location | MLIR-awareness |
|---|---|---|
| `XeForgePipeline.optimize()` — the whole loop | [pipeline.py:656-1120](../src/xe_forge/pipeline.py) | shared; 3 MLIR-only branches (§4) |
| `AnalyzerAgent` — issue detection | [analyzer_agent.py:441-510](../src/xe_forge/agents/analyzer_agent.py) | picks `MlirAnalysisSignature` (388-430) by DSL |
| `PlannerAgent` — issue→stage plan | [planner.py:138-180](../src/xe_forge/planner.py) | **fully DSL-agnostic** (no DSL refs at all) |
| `OptimizerAgent` — the optimizer | [optimizer_agent.py:590-1143](../src/xe_forge/agents/optimizer_agent.py) | picks MLIR signatures + `_verify_mlir` |
| `CoVeR` — verify-and-revise loop | [cover.py:19-221](../src/xe_forge/agents/cover.py) | **DSL-agnostic**; takes signature + verify tool |
| stage enum, issue taxonomy, configs | [models.py:11-220](../src/xe_forge/models.py) | added 1 stage value (`LINALG_LOWERING`) |
| issue→stage mapping (5-layer resolver) | [knowledge/patterns.py:251-281](../src/xe_forge/knowledge/patterns.py) | shared |
| KB framework (loader/format/scope) | [knowledge/loader.py](../src/xe_forge/knowledge/loader.py) | shared; content scoped per-DSL dir |
| DSL→supported-stage registry | [dsl_registry.py:8-74](../src/xe_forge/dsl_registry.py) | MLIR set added (57-65) |
| device/config plumbing, CLI | [config.py](../src/xe_forge/config.py), [cli.py](../src/xe_forge/cli.py) | `--dsl mlir` sets `DSL` env |

**How the loop stays generic:** the analyzer and optimizer each hold three DSPy
signatures (Triton / SYCL / MLIR) and select one by `self.dsl`; the verify *tool* handed
to CoVeR branches to `_verify_sycl` / `_verify_mlir` / the Triton path. CoVeR itself, the
planner, the stage enum, the best-of-k selection, the trial tree, and the KB machinery
never learn they're running MLIR. This is the single most important architectural fact:
**MLIR is a plug-in, not a fork.**

---

## 3. What makes MLIR different from Triton Xe-Forge

There are exactly **three** structural divergences on the MLIR path. Everything else is
shared (§2). Original Triton flow was mapped for comparison; the divergences are:

### 3.1 Baseline & correctness oracle: self-contained, in-file, no PyTorch

- **Triton/SYCL:** baseline is a PyTorch/torch reference run
  ([pipeline.py:726-764](../src/xe_forge/pipeline.py)); correctness compares kernel output
  tensors to torch.
- **MLIR:** the input kernel is **self-contained** — it carries its own host `@main`
  that launches the kernel via `gpu.launch_func`, computes a **CPU reference in-file**,
  and prints `[ALLCLOSE: TRUE/FALSE]`. Baseline is just running that file once
  ([pipeline.py:768-784](../src/xe_forge/pipeline.py)). The core module is deliberately
  **torch-free** (lazy PEP-562 imports in `core/__init__.py`), so the MLIR path never
  imports torch. The optimizer's MLIR signatures are hard-constrained to edit **only** the
  `gpu.module` body, `#xegpu.layout` attrs, and `gpu.launch_func` geometry — never the
  `@main` harness/reference, which is the fixed oracle.

### 3.2 A pre-stage that has no Triton analog: `LINALG_LOWERING`

- Triton kernels arrive already at the level Xe-Forge optimizes. MLIR kernels may arrive
  as high-level `linalg` that must first be **lowered** to a WG-level XeGPU kernel.
- This lowering is itself an optimization (tile-search over lowering configs), so it is a
  first-class **stage** (`OptimizationStage.LINALG_LOWERING`,
  [models.py:46](../src/xe_forge/models.py)) that runs **before** ANALYSIS
  ([pipeline.py:825-840](../src/xe_forge/pipeline.py)).
- `detect_mlir_level(code)` ([linalg_lowering.py:364-376](../src/xe_forge/core/linalg_lowering.py))
  gates it: input containing `xegpu.`/`gpu.launch_func` → `"xegpu_wg"` (skip lowering);
  input containing `linalg.` → `"linalg"` (lower first).

### 3.3 A device-specific, correctness-free stage: `GRF_SWEEP`

- `_maybe_sweep_grf` ([pipeline.py:605-654](../src/xe_forge/pipeline.py)) runs the same
  IR twice — with and without the IGC large-register-file flag
  (`igc-cmd-options=-ze-opt-large-register-file`) — and keeps the faster.
- It's a **compile-flag** change, not an IR edit, so correctness is unaffected; it maps
  to `OptimizationStage.DEVICE_SPECIFIC`. Triton has a conceptually similar GRF idea but
  expresses it as an autotune meta-param, not a post-lowering sweep.

Everything else — analyzer, planner, CoVeR verify/retry, stage gating with
`_MIN_IMPROVEMENT = 1.02`, best-of-k, trial recording, VTune hooks — is the **same code**
the Triton path runs.

---

## 4. The MLIR two-level flow, step by step

### Level 0 — routing (`optimize()`, MLIR branches)

```
pipeline.optimize()
 ├─ baseline: execute(kernel) → read embedded time + [ALLCLOSE]     (768-784)
 ├─ STAGE LINALG_LOWERING  if dsl==MLIR: _maybe_lower_linalg(...)    (825-840)
 ├─ STAGE GRF_SWEEP        if dsl==MLIR & level==xegpu_wg: _maybe_sweep_grf (842-859)
 ├─ STAGE ANALYSIS         analyzer.analyze(WG kernel)               (861-878)
 ├─ STAGE PLANNING         planner.plan(issues) ∩ get_stages_for_dsl (885-924)
 ├─ per-stage loop         optimizer.optimize_stage(...) via CoVeR   (936-1062)
 └─ final measure + best-of-k                                        (1064-1109)
```

### Level 1 — lowering (`_maybe_lower_linalg`, [pipeline.py:257-344](../src/xe_forge/pipeline.py))

The router picks a lowering path by structural detection:

| Input shape | Detector | Lowering path | Engine |
|---|---|---|---|
| ≥2 `linalg.matmul` | count | MLP (`_maybe_lower_mlp`, 552-603) | native tile-search + templates |
| 1 `linalg.matmul` | `extract_matmul_dims` | matmul tile-search (314-344) | native, LLM-shortlisted configs |
| batched matmul | `extract_matmul_dims` | matmul path (rank-reduce) | native |
| attention `(Z,H,n_ctx,n_head)` | `detect_attention_shape` | `_maybe_lower_attention` (424-468) | **lighthouse dump** |
| `linalg.softmax dim(1)` | shape match | `_maybe_lower_softmax` (470-509) | **lighthouse dump** |
| layer-norm (`math.rsqrt`) | structure match | `_maybe_lower_layernorm` (511-550) | **lighthouse dump** |

**Native matmul lowering** ([mlir_executor.py](../src/xe_forge/core/mlir_executor.py)
`lower_linalg_to_wg`, 486-568) is a 3-stage `imex-opt`/`mlir-opt` recipe driven by a
5-knob `LoweringConfig` `(wg_m, wg_n, sg_m, sg_n, k_tile)` +
`large_grf` ([linalg_lowering.py:45-214](../src/xe_forge/core/linalg_lowering.py),
`DEFAULT_CONFIG=(256,256,32,32,32)` at line 326):

1. tile + vectorize (Jinja templates `tile_vectorize*.mlir.j2`)
2. `gpu-kernel-outlining, xevm-attach-target{chip=bmg O=3}, gpu.module(convert-vector-to-xegpu)`
3. WG layout annotation (`wg_annotate*.mlir.j2` — derives every `#xegpu.layout` from the config)

**Tile-search is hybrid:** `LinalgLoweringAgent.propose(bare, m,n,k)`
([linalg_lowering_agent.py](../src/xe_forge/agents/linalg_lowering_agent.py)) asks the LLM
for a shortlist of configs (seeded by the `mlir/linalg` KB, §6), then
`executor.sweep_configs` ([mlir_executor.py:710-781](../src/xe_forge/core/mlir_executor.py))
**times each config's kernel** via an rtclock harness (`timing_harness.mlir.j2`) and keeps
the fastest correct one. A lighthouse **second opinion** (`_matmul_second_opinion`,
346-422) times an upstream schedule through the *same* harness and only wins if it beats
our best by > `speedup_tol`.

### Level 2 — WG-level optimization (the shared loop)

Once a WG-level kernel exists (either lowered above or provided directly as
`xegpu_wg`), the standard analyzer/optimizer stages run on it. The MLIR optimizer
signatures (`MlirOptimizationSignature` / `MlirAlgorithmicOptimizationSignature`,
[optimizer_agent.py:418-517](../src/xe_forge/agents/optimizer_agent.py)) emit
`dspy.Code["mlir"]` and are constrained to edit only the kernel body / layout / geometry.

---

## 5. Correctness gating and the no-op guard (recent hardening)

The MLIR verify path is `_verify_mlir` ([optimizer_agent.py:77-126](../src/xe_forge/agents/optimizer_agent.py)),
handed to CoVeR via `_create_verify_tool` (626-800, MLIR branch at 650). It:

1. structural pre-checks (`gpu.launch_func`, `func.func @main`, `[ALLCLOSE]`/`printAllclose`);
2. calls `executor.compare_kernels(orig, opt, flop=...)`
   ([mlir_executor.py:350-471](../src/xe_forge/core/mlir_executor.py));
3. **rejects no-op edits** — the recently added *lowered-IR-equivalence guard*: both
   variants are lowered via `lower_only` (271) with `_pipeline_options()`; if the lowered
   IR is byte-identical, the edit is dead code the compiler eliminates, so `speedup` is
   forced to 1.0 and `lowered_identical=True` is returned (dataclass field ~line 92). Both
   `_verify_mlir` and `_final_verify` (1374+) reject `lowered_identical` — the latter was
   critical because a phantom speedup had been sneaking through the `baseline_ms` recompute
   path.

This guard was added after a "1.68× verified" turned out to be a dead `vector.transfer_read`
that DCE removed (kernel lowered byte-identical). See the KB anti-pattern
`xegpu_no_dead_prefetch_reads` (§6).

---

## 6. Knowledge Base — shared vs MLIR-specific

The KB loader ([knowledge/loader.py:286-354](../src/xe_forge/knowledge/loader.py)) collects
YAML in priority order: `common/` → `<dsl>/common/` → `<dsl>/<device_type>/`. This means
**every DSL always picks up `common/` first**, then diverges by directory. Content is
scoped by directory; the *framework* (dataclasses, `format_for_stage`, stage indexing,
issue→stage resolution) is shared by all DSLs.

### 6.1 Reusable across DSLs (structurally shared — the only truly common files)

Loaded for **every** DSL, MLIR included:

- [`knowledge_base/common/algorithmic_patterns.yaml`](../knowledge_base/common/algorithmic_patterns.yaml)
  — FLOP-reduction reordering, precompute-in-`__init__`. Framed at the algorithm level,
  DSL-neutral.
- [`knowledge_base/common/correctness.yaml`](../knowledge_base/common/correctness.yaml)
  — outputs-must-match, device-placement correctness.

These are the parts of the Triton KB that **directly benefit MLIR today** with zero
porting.

### 6.2 MLIR-specific KB (two device_type scopes)

MLIR uniquely loads **two** KB scopes for its two levels:

- **WG-level** (`device_type="xpu"`, used by analyzer/optimizer):
  [`mlir/xpu/xegpu_wg_patterns.yaml`](../knowledge_base/mlir/xpu/xegpu_wg_patterns.yaml)
  (DPAS/SIMD16/f32-accum constraints; `sg_data`/`sg_layout` tuning patterns) +
  [`mlir/xpu/xegpu_memory_patterns.yaml`](../knowledge_base/mlir/xpu/xegpu_memory_patterns.yaml)
  (prefetch_nd, cache hints, coalesced block loads; plus the `xegpu_no_dead_prefetch_reads`
  anti-pattern and the precondition-gated `xegpu_add_prefetch_nd`).
- **Lowering-config** (`device_type="linalg"`, used by `LinalgLoweringAgent`, loaded via a
  separate cached KB at [pipeline.py:240-255](../src/xe_forge/pipeline.py)):
  [`mlir/linalg/lowering_config_patterns.yaml`](../knowledge_base/mlir/linalg/lowering_config_patterns.yaml)
  (the 5-tuple seed configs + divisibility/DPAS-alignment/thread-cap constraints) +
  [`mlir/linalg/lowering_scope_and_findings.yaml`](../knowledge_base/mlir/linalg/lowering_scope_and_findings.yaml)
  (engineering scope/findings — what lowers, what routes to lighthouse).

### 6.3 Conceptual overlap (NOT shared files — re-expressed per DSL)

The same **hardware truths** recur across DSLs but live in separate files because the
syntax differs: SIMD16 subgroups, f32 accumulation for f16/bf16, no-f64, DPAS tile shapes
`[8,16]`/`[16,16]`, large-GRF tradeoffs, tile/occupancy tuning, prefetch/cache-hints,
epilogue fusion. Compare `mlir/xpu/xegpu_wg_patterns.yaml` (`mlir_dpas_tile_shape_alignment`,
`xegpu_f32_accumulator_softmax_recipe`) with `sycl/xpu/xetla_patterns.yaml`
(`sycl_dpas_requires_simd16`, `sycl_fp32_accumulator_required`). **Opportunity:** these
could be refactored into a shared `common/hardware_xe.yaml` of device facts with
per-DSL pattern files carrying only the syntax — see §8.

### 6.4 KB feature added for MLIR: `precondition`

`KnowledgeEntry` gained a `precondition` field ([loader.py:87](../src/xe_forge/knowledge/loader.py),
parsed at ~441, formatted as `PRECONDITION (apply ONLY if this holds): …` at ~201). This
lets a pattern like `xegpu_add_prefetch_nd` declare it applies **only** to kernels already
in `load_nd` block-load form — preventing the LLM from fabricating dead prefetch reads on
vector-dialect gather kernels. This is a generic KB improvement usable by any DSL.

---

## 7. File map (where everything lives)

### MLIR-specific code

| File | Role |
|---|---|
| [core/mlir_executor.py](../src/xe_forge/core/mlir_executor.py) | executor: `execute`, `lower_linalg_to_wg`, `lower_mlp_to_wg`, `sweep_configs`, `sweep_grf`, `time_wg_matmul`, `autotune_tile`, `compare_kernels` (+ no-op guard: `lower_only`, `_pipeline_options`) |
| [core/linalg_lowering.py](../src/xe_forge/core/linalg_lowering.py) | `LoweringConfig` (5 knobs), `DEFAULT_CONFIG`, `candidate_configs`, `detect_mlir_level`, harness rendering |
| [core/mlp_lowering.py](../src/xe_forge/core/mlp_lowering.py) | MLP structural parse + per-layer lowering |
| [core/lighthouse_backend.py](../src/xe_forge/core/lighthouse_backend.py) | subprocess dump-only bridge to `llvm/lighthouse` |
| [core/matmul_lowering.py](../src/xe_forge/core/matmul_lowering.py), [attention_lowering.py](../src/xe_forge/core/attention_lowering.py), [softmax_lowering.py](../src/xe_forge/core/softmax_lowering.py), [layer_norm_lowering.py](../src/xe_forge/core/layer_norm_lowering.py) | per-op lowering helpers |
| [agents/linalg_lowering_agent.py](../src/xe_forge/agents/linalg_lowering_agent.py) | LLM half of the hybrid tile-search (config shortlist) |
| [pipelines/linalg_to_wg/templates/](../pipelines/linalg_to_wg/templates/) | Jinja templates: `tile_vectorize*`, `wg_annotate*`, `timing_harness`, `profiling_harness` |

### Shared code with MLIR branches

| File | MLIR touch-points |
|---|---|
| [pipeline.py](../src/xe_forge/pipeline.py) | executor select (124-145); `_maybe_lower_*` (257-603); `_maybe_sweep_grf` (605-654); 3 MLIR stage branches in `optimize()` (825-859) |
| [agents/analyzer_agent.py](../src/xe_forge/agents/analyzer_agent.py) | `MlirAnalysisSignature` (388-430); MLIR issue descriptions/skips (71-127); signature pick (441-450) |
| [agents/optimizer_agent.py](../src/xe_forge/agents/optimizer_agent.py) | (CoVeR) MLIR signatures (418-517); `_verify_mlir` (77-126); verify-tool branch (650); signature pick (916-941); `_final_verify` MLIR branch (1391) |
| [agents/react_agent.py](../src/xe_forge/agents/react_agent.py) | (ReAct) `MlirOptimizationReActSignature`; module-level `_verify_mlir`; verify-tool branch; signature pick; final-verify + `_is_valid_kernel` MLIR branches |
| [models.py](../src/xe_forge/models.py) | `DSL.MLIR` (11-24); `LINALG_LOWERING` stage (46) |
| [dsl_registry.py](../src/xe_forge/dsl_registry.py) | MLIR supported-stage set (57-65) |
| [config.py](../src/xe_forge/config.py) | `DSL=mlir` selects XPUConfig + MLIR path |
| [cli.py](../src/xe_forge/cli.py) | `--dsl mlir` flag (109-115) |

### Not touched by MLIR (Triton/SYCL only)

`core/executor.py` (Triton `KernelBenchExecutor`), `core/sycl_executor.py`, and all
`triton/**` KB are untouched. Both agent strategies now support MLIR: `"cover"`
(CoVeR, default) and `"react"` (`AGENT_STRATEGY=react`).

---

## 8. Where to invest next (decision inputs)

Concrete forward paths this architecture enables, roughly ordered by leverage:

1. **Real `xegpu.prefetch_nd` for vector-dialect kernels.** Today the FMHA vector kernel
   lowers to gathers (no `tensor_desc`), so prefetch can't attach. Needs (a) exposing the
   post-`convert-vector-to-xegpu` IR as an editable stage, and (b) making Q/K/V
   `transfer_read`s lower to contiguous `load_nd` block loads. High-value, scoped.
2. **Refactor conceptual KB overlap (§6.3)** into a shared `common/hardware_xe.yaml` of
   device facts, leaving per-DSL files with only syntax. Cuts triplicated maintenance
   across MLIR/SYCL/Triton.
3. **Broaden native lowering coverage.** Attention/softmax/layer-norm currently route to
   lighthouse dumps; making them native (Linalg→WG emitting register-vector accumulators)
   is deferred but is the path to controlling their perf. Keep
   [kernelbench_coverage.md](kernelbench_coverage.md) updated as coverage grows.
4. **~~ReAct strategy for MLIR.~~** *(done)* `react_agent.py` now has an MLIR branch
   (`MlirOptimizationReActSignature` + `_verify_mlir`), so `AGENT_STRATEGY=react DSL=mlir`
   runs stock `dspy.ReAct` with the verify tool as an alternative to CoVeR.

---

## 9. How to run it

```bash
# bare linalg.matmul → two-level flow (lowering + WG stages)
python -m xe_forge.cli --dsl mlir --input kernel.mlir --spec spec.yaml \
    --output outputs/mm.opt.mlir

# already-WG XeGPU kernel → skips lowering, straight to GRF sweep + WG stages
python -m xe_forge.cli --dsl mlir --input wg_kernel.mlir --spec spec.yaml
```

Env equivalents: `DSL=mlir`, `DEVICE_TYPE=xpu`, `KNOWLEDGE_BASE_ENABLED=true`,
`AGENT_STRATEGY=cover`. LLVM+IMEX toolchain at
`/home/gta/upstream/llvm-project/build-with-imex`. See
[mlir_e2e_howto.md](mlir_e2e_howto.md) for the full reproducible flow.
