# MLIR knowledge base — layout

Not loaded by the KB loader (it globs `*.yaml`); this is the map for humans and for
whoever adds the next rule.

## Two views, three tiers

The loader walks `common/` → `<dsl>/common/` → `<dsl>/<device>/`
(`loader._collect_yaml_files`), and MLIR is the one DSL with **two** device scopes,
loaded separately and never together:

| view | who reads it | what it decides |
| --- | --- | --- |
| `dsl=mlir device=linalg` | `LinalgLoweringAgent` | the lowering config 5-tuple `(wg_m, wg_n, sg_m, sg_n, k_tile)` + `large_grf` |
| `dsl=mlir device=xpu` | analyzer + WG optimizer | the XeGPU IR itself: `#xegpu.layout` attrs, kernel body, launch geometry |

`mlir/common/` loads for **both**. The repo-root `common/` tier is host-Python
(PyTorch/KernelBench wrapper) knowledge and is excluded from the MLIR views with
`excludes_dsl: [mlir]`, since it is meaningless inside a standalone MLIR module.

## Where a new rule goes

**`mlir/common/`** — only if it is a property of the hardware or the XeGPU contract,
and therefore true in both views:

- `xegpu_hardware_contract.yaml` — the DPAS dtype contract, subgroup count / GRF
  budget, DPAS tile alignment, exact tiling. Each of these is stated **once** here;
  the per-view files cross-reference them instead of restating them.

**`mlir/linalg/`** — the lowering path:

- `lowering_config_patterns.yaml` — tunable tile/subgroup configs and the autotuner.
- `lowering_scope_and_findings.yaml` — what the Linalg→WG path actually supports
  (plain/transpose-B/batch matmul, MLP layers), what routes to lighthouse, and the
  harness/verification facts.

**`mlir/xpu/`** — XeGPU IR edits, organized **by concern, not by kernel**. A file
named after a kernel makes generic rules look kernel-specific: the register-budget
and prefetch-pipeline rules apply to any loop streaming large tiles, not only to
attention. Each entry carries its own `precondition` stating when it fires.

A `precondition` is written as an **IR-feature decision procedure** — which ops,
which operand types, which layout fields, what to count — never "applies to
attention". Where a workload is named in one, it is as an example list ("a GEMM
K-panel, an attention K/V stream, a convolution weight stream all qualify"), not as a
gate. The test: a precondition should still be actionable for someone who has never
heard of the kernel it was tuned on. The measurements stay, in a
**`MEASURED INSTANCE`** block at the end of the `description` (constraints) or `notes`
(patterns), which records every number *with the conditions it was measured under* —
kernel, shape, GRF mode, what else was already applied. A number without its
conditions is how a 1.36x gets re-applied where it measures 0.96x.

- `xegpu_dtype.yaml` — which dtype each value carries; where a cast must go.
- `xegpu_layout_and_tiling.yaml` — `sg_layout` / `sg_data` / `inst_data`, tile
  geometry, layout propagation, and the register budget for loop-carried state.
- `xegpu_memory_patterns.yaml` — loads, stores, prefetch, cache hints, 2D-block
  message limits.
- `xegpu_reductions.yaml` — cross-lane reductions (softmax max/sum, layernorm).
- `xegpu_transcendental_patterns.yaml` — `math.exp`/`exp2`, fastmath, algebraic
  folds on the softmax.

## Conventions

- `severity: critical` and `severity: warning` constraints are the **only** KB
  content the analyzer sees (`analyzer_agent._get_kb_context`), so a rule that must
  influence issue detection has to be a constraint at one of those severities.
- Declaring `stage:`/`stages:` on a constraint scopes it to those stages *and* keeps
  it visible to the analyzer (`loader.constraints_for_stage`); a constraint with no
  declared stage is visible everywhere.
- Keep measured numbers, hardware, and the measurement conditions in the entry that
  claims the speedup — a number without its conditions cannot be re-checked.
