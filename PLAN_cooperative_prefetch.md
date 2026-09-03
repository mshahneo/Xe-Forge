# Plan: cooperative prefetch + branchless prefetch + zero-accum cleanup

## Context / findings (from prototyping on 4K GEMM)

Current best 4K kernel: **6.57 ms (~21 TFLOPS)** via LLM `scf.if`-guarded prefetch.
Reference WG kernel target: **~83 TFLOPS median (~1.65 ms)** — a 4x gap. The
reference's recipe is a *coupled* set of choices, not independent knobs:
- 32 subgroups (`sg_layout=[8,4]`), NOT 64 — paired with **large GRF**.
- `sg_data=[32,64]` for B (heavier subgroup tiles).
- **Cooperative prefetch** with dedicated layouts distinct from compute layouts:
  `#a_prefetch=[32,1]/[8,32]`, `#b_prefetch=[4,8]/[8,32]`.
- Branchless 3-stage prefetch (prologue + single in-loop `prefetch_nd`).

### What each suggestion turned out to be
1. **`scf.if` prefetch** = LLM artifact. Upstream `transform.xegpu.insert_prefetch`
   generates the clean branchless reference form (prologue prefetches + one
   in-loop `arith.addi` offset prefetch, **0 scf.if**); `nb_prefetch` = depth.
   PROVEN working, but requires the A/B `create_nd_tdesc` hoisted out of the
   k-loop first (LICM), else "operand does not dominate" — PROVEN LICM fixes it.
2. **Cooperative prefetch** = the essential + hard part. `insert_prefetch` emits
   `layout=nullptr` (non-cooperative). `prefetch_nd` implements
   `AnchorLayoutInterface`, so cooperative layouts CAN be stamped via
   `set_anchor_layout` — but the naive match scope silently didn't land it.
   **Branchless prefetch WITHOUT cooperative layout measured 14.8 ms — 2x SLOWER
   than the LLM version.** So (1) is only worth doing together with (2).
3. **`vector.broadcast %c0` -> `arith.constant dense<0.0>`** = trivial, correct,
   matches reference. Low-risk.

## Approach (ordered by proven leverage; verify each on GPU before the next)

### Step A — zero-accum constant (suggestion 3, cheap, do first)
The accumulator-init win already exists (skip C load). Emit the zero vector as
`arith.constant dense<0.0> : vector<MxNxf32>` rather than `vector.broadcast`.
- Where: it currently comes from the LLM ALGORITHMIC stage. Make it deterministic
  in the lowering instead (the folded `linalg.fill 0` already produces a zero
  accumulator; ensure it lowers to a constant, not a broadcast). Verify the
  generated WG kernel uses `arith.constant dense<0.0>`.
- Low risk; expect neutral-to-tiny gain. Mainly removes an LLM dependency.

### Step B — branchless cooperative prefetch in the LOWERING (suggestions 1+2 together)
Add prefetch to the transform pipeline (`wg_annotate.mlir` / a new prefetch lib),
NOT the LLM MEMORY_ACCESS stage:
1. LICM to hoist A/B `create_nd_tdesc` out of the k-loop (registered pass
   `--loop-invariant-code-motion`, or `transform.apply_registered_pass`).
2. `transform.xegpu.insert_prefetch %aload nb_prefetch=<N>` and same for B.
3. Stamp cooperative layouts on the generated `prefetch_nd` ops via
   `set_anchor_layout` (A: `sg_layout=[32,1] sg_data=[8,32]`, B:
   `sg_layout=[4,8] sg_data=[8,32]`, inst_data `[8,16]`). MUST solve the match
   scope problem: match `xegpu.prefetch_nd` in the *function/module* handle, not
   the tdesc handle returned by insert_prefetch; may need to match A's and B's
   prefetches separately (e.g. by the tensor_desc shape 256x32 vs 32x256, or by
   walking from each desc op). This is the crux — prototype until the
   `sg_layout=[32,1]`/`[4,8]` actually appears on the prefetch ops.
4. Verify on GPU: correctness + time. Compare vs 6.57 ms. Only keep if faster.

### Step C — couple with the reference compute recipe (the real 4x lever)
The prefetch layouts assume the reference *compute* layout. Our configs differ.
Add/verify a config matching the reference: 32 subgroups + large GRF +
`sg_data` heavier on N. Concretely a config like `wg=256x256 sg=[32,64]`
(-> sg_layout `[8,4]` = 32 subgroups) with `large_grf=True`, and cooperative
prefetch layouts keyed to it. This is where the reference gets ~83 TFLOPS.
- Make the cooperative prefetch layouts *derive from* the compute config (like
  the existing `wg_annotate` layouts do) rather than hardcoding [32,1]/[4,8],
  so they stay consistent across configs.
- This may be the step that actually closes the gap; A/B alone likely won't.

### Step D — expose as tunable knobs + wire into the sweep
- Add `nb_prefetch` (0=off, else depth) to `LoweringConfig`; thread into the
  prefetch transform lib (templated, like tile sizes).
- Cooperative-prefetch layouts derived in the templater from the config.
- Agent/KB: propose `nb_prefetch` and the 32-sg+large-GRF+heavy-sg recipe.
- Sweep compares prefetch depths / GRF / sg tiles head-to-head by measured time.

## Files
- `pipelines/linalg_to_wg/transforms/wg_annotate.mlir(.j2)` — add LICM +
  insert_prefetch + cooperative layout stamping.
- `src/xe_forge/core/linalg_lowering.py` — `nb_prefetch` knob; derive prefetch
  layouts in the templater; render into the transform lib.
- `src/xe_forge/core/mlir_executor.py` — ensure LICM runs before prefetch in the
  lowering recipe (stage 2/3).
- `knowledge_base/mlir/linalg/lowering_config_patterns.yaml` — cooperative
  prefetch + 32-sg/large-GRF recipe patterns.
- `src/xe_forge/agents/linalg_lowering_agent.py` — nb_prefetch in ConfigProposal.

## Verification
Each step: lower -> splice timing harness -> run on 4K GPU -> require
`[ALLCLOSE: TRUE]` and record ms. Gate: keep a change only if it's faster than
the current best (6.57 ms). Target to chase: reference ~1.65 ms (~83 TFLOPS).
Honest expectation: A/B/C may only partially close the gap — the perf report
itself attributes the remaining WG-vs-lane gap to missing `array_length` support
(DPAS-sized loads under-utilizing the load engine), which is outside our scope.

## Risk / open question
The cooperative-layout stamping on generated prefetch ops is unproven (silent
no-op so far). If it can't be made to land via `set_anchor_layout`, fallback is
to emit the prefetch + layouts textually in a templated kernel prologue (less
clean, but matches the reference exactly). Decide after Step B prototyping.
