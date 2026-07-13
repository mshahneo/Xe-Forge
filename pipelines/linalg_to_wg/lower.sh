#!/usr/bin/env bash
# Linalg -> XeGPU workgroup-level lowering pipeline (validated on Intel Xe).
#
# INPUT: a plain Linalg .mlir file — just the kernel, NO embedded transform
#        module (see examples/matmul.mlir). The transform schedules live in
#        external libraries under transforms/ and match by op type, so the same
#        pipeline works on any linalg.matmul-based kernel.
#
# OUTPUT: XeGPU WG-level IR (xegpu.load_nd / dpas / store_nd with #xegpu.layout)
#        ready for --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup".
#
# Why the multi-stage shape (see README.md):
#   - tiling / vectorize+contract-fold / bufferize / forall->parallel are
#     transform-dialect ops with NO registered-pass equivalent -> stage 1 runs
#     a transform library via --transform-interpreter. A pure imex-opt
#     --pass-pipeline string is therefore impossible for this stage.
#   - gpu-kernel-outlining + xevm-attach-target crash the transform interpreter
#     with a dlti multi-threading error -> stage 2 runs them as a normal pipeline.
#   - convert-vector-to-xegpu is scoped to gpu.module() so the host-side fill is
#     NOT converted to a broken scattered xegpu.store.
#   - WG #xegpu.layout has no auto-seeding pass -> stage 3 applies a transform
#     library (set_anchor_layout on loads + dpas a/b/cd + store).
#
# Usage: lower.sh <input_linalg.mlir> <output_wg.mlir>
set -euo pipefail

BIN="${MLIR_BIN_DIR:-/home/gta/upstream/llvm-project/build-with-imex/bin}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="$1"; OUT="$2"
TMP="$(mktemp -d)"

# Stage 1: transform-dialect schedule (external library, bare input).
"$BIN/mlir-opt" "$IN" \
  --transform-preload-library="transform-library-paths=$HERE/transforms/tile_vectorize.mlir" \
  --transform-interpreter -o "$TMP/s1.mlir"

# Stage 2: GPU outlining + xegpu conversion (normal pipeline; dlti-safe).
"$BIN/mlir-opt" "$TMP/s1.mlir" \
  --pass-pipeline='builtin.module(gpu-kernel-outlining, xevm-attach-target{chip=bmg O=3}, gpu.module(convert-vector-to-xegpu))' \
  -o "$TMP/s2.mlir"

# Stage 3: seed WG #xegpu.layout on loads + dpas + store (external library).
"$BIN/mlir-opt" "$TMP/s2.mlir" \
  --transform-preload-library="transform-library-paths=$HERE/transforms/wg_annotate.mlir" \
  --transform-interpreter -o "$OUT"

echo "Linalg -> XeGPU WG lowering complete: $OUT"
rm -rf "$TMP"
