# Linalg → XeGPU Workgroup-Level Lowering (Phase 0)

Validated on Intel Xe (Battlemage) via the upstream `build-with-imex` toolchain:
a tensor-level `linalg.matmul` is lowered all the way to a running XeGPU
workgroup-level kernel that matches its CPU reference (`[ALLCLOSE: TRUE]`).

This is the **lowering bridge** for the two-level Xe-Forge MLIR pipeline:

```
Linalg  ──(Xe-Forge optimize @ linalg level)──►  Linalg'
        ──(THIS pipeline: lower.sh)───────────►  XeGPU WG-level
        ──(Xe-Forge optimize @ WG level)──────►  optimized WG kernel  ──► run
```

## Files

| File | Role |
|------|------|
| `examples/matmul.mlir` | **Plain Linalg input** — just the kernel, no embedded transform module. This is all a user supplies. |
| `transforms/tile_vectorize.mlir` | Stage-1 transform library: tiles (WG forall + k-loop), vectorizes with contract folding, bufferizes, `forall→parallel`, GPU map/convert/sink, sets launch threads. Matches by op type — works on any `linalg.matmul` kernel. |
| `transforms/wg_annotate.mlir` | Stage-3 transform library: seeds `#xegpu.layout` on the A/B loads, the `dpas` (`layout_a`/`layout_b`/`layout_cd`), and the `store_nd`. Ported from lighthouse's `xegpu_wg_annotation_for_mlp_layer`. |
| `lower.sh` | Driver: takes a bare Linalg file, runs the 3 stages, emits WG-level XeGPU IR. |
| `matmul_runnable_reference.mlir` | Proven end-to-end kernel + `@main` harness (for regression). |

Input is a **plain Linalg `.mlir`** — the transform schedules are external
libraries that match by op type, so the same pipeline lowers any
`linalg.matmul`-based kernel with no edits to the input.

## The recipe (what `lower.sh` does)

**Stage 1 — transform-dialect** (`--transform-interpreter`):
`tile_using_forall [256,256]` → `tile_using_for [0,0,32]` →
`vectorize_children_and_apply_patterns {fold_type_extensions_into_contract}`
(→ mixed-precision `vector.contract`) → `one_shot_bufferize` →
`loop.forall_to_parallel` → `gpu-map-parallel-loops` /
`convert-parallel-loops-to-gpu` / `gpu-launch-sink-index-computations` →
`xegpu.set_gpu_launch_threads [1024,1,1]`.

**Stage 2 — normal pass pipeline:**
`gpu-kernel-outlining` → `xevm-attach-target{chip=bmg}` →
`gpu.module(convert-vector-to-xegpu)`.

**Stage 3 — WG layout seed** (`--transform-preload-library` + interpreter):
`get_load_op` on the dpas A/B operands → `set_anchor_layout` on each load, on
the dpas (a/b/cd), and on the store.

## Non-obvious gotchas (why it is shaped this way)

1. **Hybrid, not a pure `.pp`.** Tiling, vectorization, bufferize, and
   `forall→parallel` are transform-dialect ops with no registered-pass
   equivalent, so the pipeline cannot be a single `imex-opt --pass-pipeline`
   string. It is transform-dialect + registered passes.

2. **dlti multi-threading crash.** Running `gpu-kernel-outlining` /
   `xevm-attach-target` via `apply_registered_pass` *inside* the transform
   interpreter crashes: "Loading a dialect (dlti) while in a multi-threaded
   execution context". Hence stage 2 runs them as a normal (single-PM) pipeline.

3. **Scope `convert-vector-to-xegpu` to `gpu.module`.** If run module-wide it
   converts the host-side `linalg.fill` (zeroing C) into a scattered
   `xegpu.store` that never lowers ("Dialect `xegpu' not found for custom op
   'xegpu.store'" at runtime). Scoping keeps host code as plain vector/memref.

4. **WG layouts must be seeded explicitly.** No pass auto-assigns WG
   `sg_layout`. `xegpu-propagate-layout` is SIMT-oriented and its "no layout
   assigned" warnings are a red herring — the WG lowering pipeline
   (`gpu-lower-to-xevm-pipeline{xegpu-op-level=workgroup}`) is what actually
   validates the layouts. The layouts must be shape-consistent across loads,
   dpas operands, and the store (broadcast semantics: A=[8,1], B=[1,8],
   C=[8,8] sg_layout with sg_data [32,32] for a 256×256×32 tile, 8×8 subgroups).

5. **Get the accumulator layout from the dpas, not a load.** `get_load_op` on
   the dpas C operand (index 2) fails — it is an `scf.for` iter_arg, not a load.
   Set its layout via `set_anchor_layout %dpas index=2` and on the `store_nd`.

6. **Flat module, not nested.** Wrapping the payload in a nested `@payload`
   module makes `convert-vector-to-xegpu` emit scattered loads instead of
   `load_nd`. Keep the payload func at top level.

## Current scope / TODO

- Fixed for a 512×512×512 f16→f32 matmul, WG tile 256×256, k-tile 32, 8×8
  subgroups. Tile sizes and layouts are constants in `01_tile_vectorize.mlir`
  and `03_wg_annotate.mlir` — parameterize next (lighthouse derives them from
  an SMT solver; we compute them directly from tile shapes).
- matmul only; extend to fused epilogues / attention later.
- Reference `chip=bmg`; change in stage 2 for other targets.

## Run

```bash
./lower.sh 01_tile_vectorize.mlir /tmp/out_wg.mlir
# then wrap in a host @main harness and run:
#   imex-opt out_wg.mlir --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
#     | mlir-runner -e main --entry-point-result=void --shared-libs=...
```
