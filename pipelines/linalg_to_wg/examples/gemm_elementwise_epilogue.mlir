// GEMM + fused elementwise epilogue (KernelBench level2/9 shape):
// C = relu((A·W^T + bias - 2.0) * 1.5). A matmul followed by a chain of pure
// elementwise ops (bias-add, scalar sub/mul, ReLU). Detected by is_mlp_layer and
// lowered by the MLP-layer templates — the epilogue fuses into the WG kernel after
// the k-loop. Arbitrary elementwise (incl. transcendentals: exp/tanh) is supported.
#e2 = affine_map<(m, n) -> (m, n)>
#eb = affine_map<(m, n) -> (n)>
func.func @k9(%A: memref<128x512xf16>, %W: memref<512x512xf16>, %bias: memref<512xf32>, %C: memref<128x512xf32>) {
  %cA = bufferization.to_tensor %A restrict : memref<128x512xf16> to tensor<128x512xf16>
  %cW = bufferization.to_tensor %W restrict : memref<512x512xf16> to tensor<512x512xf16>
  %cbias = bufferization.to_tensor %bias restrict : memref<512xf32> to tensor<512xf32>
  %c0 = arith.constant 0.0 : f32
  %csub = arith.constant 2.0 : f32
  %cmul = arith.constant 1.5 : f32
  %et = tensor.empty() : tensor<512x512xf16>
  %t = linalg.transpose ins(%cW : tensor<512x512xf16>) outs(%et : tensor<512x512xf16>) permutation = [1, 0]
  %e0 = tensor.empty() : tensor<128x512xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %mm = linalg.matmul ins(%cA, %t : tensor<128x512xf16>, tensor<512x512xf16>) outs(%filled : tensor<128x512xf32>) -> tensor<128x512xf32>
  %e1 = tensor.empty() : tensor<128x512xf32>
  // epilogue: (mm + bias - 2.0) * 1.5, then relu  -- all fused into generics
  %biased = linalg.generic {indexing_maps = [#e2, #eb, #e2], iterator_types = ["parallel","parallel"]} ins(%mm, %cbias : tensor<128x512xf32>, tensor<512xf32>) outs(%e1 : tensor<128x512xf32>) {
  ^bb0(%in: f32, %b: f32, %o: f32):
    %s0 = arith.addf %in, %b : f32
    %s1 = arith.subf %s0, %csub : f32
    %s2 = arith.mulf %s1, %cmul : f32
    %cmp = arith.cmpf ugt, %s2, %c0 : f32
    %sel = arith.select %cmp, %s2, %c0 : f32
    linalg.yield %sel : f32
  } -> tensor<128x512xf32>
  bufferization.materialize_in_destination %biased in restrict writable %C : (tensor<128x512xf32>, memref<128x512xf32>) -> ()
  return
}
