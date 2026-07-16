// One MLP / nn.Linear layer with activation: C = relu(A · B^T + bias).
// transpose-B matmul (indexing_maps) + bias-add generic + ReLU generic.
// Lowered by the epilogue-fused MLP-layer templates (tile_vectorize_mlp_layer +
// wg_annotate_mlp_layer): the matmul+bias+ReLU fuse into one WG kernel.
#mA = affine_map<(m, n, k) -> (m, k)>
#mBt = affine_map<(m, n, k) -> (n, k)>
#mC = affine_map<(m, n, k) -> (m, n)>
#e2 = affine_map<(m, n) -> (m, n)>
#eb = affine_map<(m, n) -> (n)>
func.func @mlp_layer(%A: memref<128x1024xf16>, %B: memref<1024x1024xf16>, %bias: memref<1024xf32>, %C: memref<128x1024xf32>) {
  %cA = bufferization.to_tensor %A restrict : memref<128x1024xf16> to tensor<128x1024xf16>
  %cB = bufferization.to_tensor %B restrict : memref<1024x1024xf16> to tensor<1024x1024xf16>
  %cbias = bufferization.to_tensor %bias restrict : memref<1024xf32> to tensor<1024xf32>
  %c0 = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<128x1024xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %mm = linalg.matmul indexing_maps = [#mA, #mBt, #mC] ins(%cA, %cB : tensor<128x1024xf16>, tensor<1024x1024xf16>) outs(%filled : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %e1 = tensor.empty() : tensor<128x1024xf32>
  %biased = linalg.generic {indexing_maps = [#e2, #eb, #e2], iterator_types = ["parallel", "parallel"]} ins(%mm, %cbias : tensor<128x1024xf32>, tensor<1024xf32>) outs(%e1 : tensor<128x1024xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %s = arith.addf %in, %b : f32
    linalg.yield %s : f32
  } -> tensor<128x1024xf32>
  %e2t = tensor.empty() : tensor<128x1024xf32>
  %relu = linalg.generic {indexing_maps = [#e2, #e2], iterator_types = ["parallel", "parallel"]} ins(%biased : tensor<128x1024xf32>) outs(%e2t : tensor<128x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %cmp = arith.cmpf ugt, %in, %c0 : f32
    %sel = arith.select %cmp, %in, %c0 : f32
    linalg.yield %sel : f32
  } -> tensor<128x1024xf32>
  bufferization.materialize_in_destination %relu in restrict writable %C : (tensor<128x1024xf32>, memref<128x1024xf32>) -> ()
  return
}
