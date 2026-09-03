# MLIR / XeGPU Backend for Xe-Forge — Overview

*A 5-slide overview of the effort. Companion to
[mlir_backend_architecture.md](mlir_backend_architecture.md).*

---

## Slide 1 — The Idea

**Xe-Forge, originally:** an LLM-driven optimizer for **Triton** kernels on Intel XPU.
A multi-stage loop — *analyze → plan → optimize (verify & retry) → gate on
correctness + speedup* — driven by the **CoVeR** agent (Chain of Verification and
Refinement) and an Intel-GPU **Knowledge Base**.

**What we added:** a fifth DSL — **MLIR / XeGPU** — targeting Intel Xe2 "Battlemage".

**The bet that paid off:** the optimization loop is **DSL-agnostic**. MLIR is a
*plug-in, not a fork* — it reuses the analyzer, planner, CoVeR, gating, trial tree, and
KB framework unchanged, and only swaps the executor and a few DSPy signatures.

> Analogy: same driver and race strategy; we swapped the engine and fuel.

---

## Slide 2 — What's the Same (reused Xe-Forge machinery)

MLIR inherits, unchanged, the entire spine of Xe-Forge:

| Reused component | Why it just works |
|---|---|
| **CoVeR** verify-and-revise loop | takes a signature + a verify tool; doesn't care about DSL |
| **PlannerAgent** (issue → stage plan) | fully DSL-agnostic, zero DSL references |
| **AnalyzerAgent / OptimizerAgent** | hold 3 signatures each; pick MLIR one by `self.dsl` |
| **Stage enum, issue taxonomy, best-of-k, gating** (`_MIN_IMPROVEMENT=1.02`) | shared code |
| **KB framework** (load / format / issue→stage resolver) | content scoped per-DSL directory |
| **`common/` KB** (algorithmic + correctness) | loaded for *every* DSL, MLIR included |

Adding MLIR to the core models was tiny: **one DSL value** (`DSL.MLIR`) and **one new
stage** (`LINALG_LOWERING`).

---

## Slide 3 — What's Different (the 3 MLIR divergences)

Only **three** structural differences from the Triton path:

1. **Self-contained correctness oracle.** No PyTorch. The kernel carries its own
   `@main` harness with an **in-file CPU reference** and prints `[ALLCLOSE: TRUE]`.
   Core stays **torch-free**. The optimizer may edit only the `gpu.module` /
   `#xegpu.layout` / launch geometry — never the harness.

2. **A pre-stage with no Triton analog — `LINALG_LOWERING`.** MLIR kernels can arrive as
   high-level `linalg` and must be *lowered* to workgroup-level XeGPU first. Lowering is
   itself an optimization (**tile-search**), so it's a first-class stage that runs
   *before* analysis.

3. **A correctness-free device knob — `GRF_SWEEP`.** Run the same IR with/without the IGC
   large-register-file flag, keep the faster. A compile flag, not an IR edit → correctness
   untouched.

Everything else in the loop is the same code the Triton path runs.

---

## Slide 4 — The Two-Level MLIR Flow

```
  linalg.matmul  ──►  LEVEL 1: LOWER  ──►  xegpu WG kernel  ──►  LEVEL 2: OPTIMIZE
   (high level)        (tile-search)        (@main + [ALLCLOSE])    (shared loop)
```

**Level 1 — Lowering** (`_maybe_lower_linalg`), routed by structure:
- **matmul / MLP** → *native* 3-stage lowering (tile+vectorize → outline+attach-target →
  WG layout annotate), driven by a 5-knob `LoweringConfig (wg_m,wg_n,sg_m,sg_n,k_tile)`.
  Tile-search is **hybrid**: LLM shortlists configs (KB-seeded) → each is **timed on GPU**
  → fastest correct wins, with a **lighthouse second opinion**.
- **attention / softmax / layer-norm** → *lighthouse* dump-only backend (shell out to
  upstream `llvm/lighthouse` schedules).

**Level 2 — Optimize:** the standard analyze → plan → CoVeR-optimize → gate loop runs on
the WG kernel.

**Correctness hardening:** a **lowered-IR-equivalence guard** rejects no-op edits — if two
variants lower to byte-identical IR, the "speedup" is dead-code noise (caught a phantom
1.68× from a DCE-eliminated prefetch read).

---

## Slide 5 — Knowledge Base & Where Next

**KB layering** (loader always reads `common/` → `<dsl>/common/` → `<dsl>/<device>/`):

- **Shared today:** `common/algorithmic_patterns.yaml`, `common/correctness.yaml` —
  benefit MLIR with zero porting.
- **MLIR-specific, two scopes:** `mlir/xpu/*` (WG DPAS/layout/memory patterns) and
  `mlir/linalg/*` (lowering-config seeds + scope/findings).
- **New KB feature for MLIR (reusable by all):** a `precondition` field so a pattern
  (e.g. `xegpu_add_prefetch_nd`) fires *only* when it actually applies.
- **Conceptual overlap** (SIMD16, f32-accum, DPAS shapes, GRF) is re-expressed per DSL
  today — a refactor opportunity.

**Forward paths (by leverage):**
1. Real `xegpu.prefetch_nd` for vector-dialect kernels (expose post-vectorization IR;
   make transfer_reads lower to `load_nd`).
2. Refactor shared hardware facts into `common/hardware_xe.yaml`.
3. Native lowering for attention/softmax/layer-norm (off lighthouse).
4. MLIR support in the ReAct strategy (currently CoVeR-only).

> **Bottom line:** we proved the Xe-Forge loop generalizes beyond Triton. MLIR runs
> end-to-end on real GPU with a two-level flow, a self-contained correctness oracle, and
> verified speedups — at the cost of one new DSL value, one new stage, a dedicated
> executor, and DSL-scoped KB content.
