module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %root : (!transform.any_op) -> !transform.any_op
    %t, %fa = transform.structured.tile_using_forall %mm tile_sizes [256, 256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse the zero-fill into the workgroup tile so the zero sits adjacent to the
    // accumulator read (lets canonicalize forward it into the k-loop init).
    %ff, %nc = transform.structured.fuse_into_containing_op %fill into %fa : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tk, %kl = transform.structured.tile_using_for %t tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %v = transform.structured.vectorize_children_and_apply_patterns %f {fold_type_extensions_into_contract} : (!transform.any_op) -> !transform.any_op
    // Promote accumulator to a zero-initialized vector iter_arg: hoist the
    // transfer_read/write subset pair out of the k-loop (register-carried, single
    // store after the loop), then canonicalize to forward the fused zero into the
    // loop init (accumulator = arith.constant dense<0.0>, not a C load). Matches
    // the reference GEMM: drops a C load per workgroup + idempotent across launches.
    %fh = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %h1 = transform.apply_registered_pass "loop-invariant-subset-hoisting" to %fh : (!transform.any_op) -> !transform.any_op
    %h2 = transform.apply_registered_pass "canonicalize" to %h1 : (!transform.any_op) -> !transform.any_op
    %bmod = transform.bufferization.one_shot_bufferize %root {function_boundary_type_conversion = 1 : i32, bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op
    %fa2 = transform.structured.match ops{["scf.forall"]} in %bmod : (!transform.any_op) -> !transform.any_op
    %par = transform.loop.forall_to_parallel %fa2 : (!transform.any_op) -> !transform.any_op
    %f2 = transform.structured.match ops{["func.func"]} in %bmod : (!transform.any_op) -> !transform.any_op
    %g1 = transform.apply_registered_pass "gpu-map-parallel-loops" to %f2 : (!transform.any_op) -> !transform.any_op
    %g2 = transform.apply_registered_pass "convert-parallel-loops-to-gpu" to %g1 : (!transform.any_op) -> !transform.any_op
    %g3 = transform.apply_registered_pass "gpu-launch-sink-index-computations" to %g2 : (!transform.any_op) -> !transform.any_op
    %launch = transform.structured.match ops{["gpu.launch"]} in %g3 : (!transform.any_op) -> !transform.any_op
    transform.xegpu.set_gpu_launch_threads %launch threads = [1024, 1, 1] : !transform.any_op
    transform.yield
  }
}
