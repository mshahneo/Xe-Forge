// A 2-layer MLP: C = relu(relu(A·W1^T + b1) · W2^T + b2). Two nn.Linear layers
// with physical linalg.transpose weights (the Torch-MLIR import form). Lowered by
// MlirExecutor.lower_mlp_to_wg: transposes folded into the matmul, then N-layer
// recipe generation (one XeGPU-WG kernel per layer, chained via intermediate buf).
#e2 = affine_map<(m, n) -> (m, n)>
#eb = affine_map<(m, n) -> (n)>
func.func @mlp2(%A: memref<128x512xf16>, %W1: memref<512x512xf16>, %bias1: memref<512xf32>,
                %W2: memref<256x512xf16>, %bias2: memref<256xf32>, %C: memref<128x256xf32>) {
  %cA = bufferization.to_tensor %A restrict : memref<128x512xf16> to tensor<128x512xf16>
  %cW1 = bufferization.to_tensor %W1 restrict : memref<512x512xf16> to tensor<512x512xf16>
  %cb1 = bufferization.to_tensor %bias1 restrict : memref<512xf32> to tensor<512xf32>
  %cW2 = bufferization.to_tensor %W2 restrict : memref<256x512xf16> to tensor<256x512xf16>
  %cb2 = bufferization.to_tensor %bias2 restrict : memref<256xf32> to tensor<256xf32>
  %c0 = arith.constant 0.0 : f32
  %et1 = tensor.empty() : tensor<512x512xf16>
  %t1 = linalg.transpose ins(%cW1 : tensor<512x512xf16>) outs(%et1 : tensor<512x512xf16>) permutation = [1, 0]
  %e01 = tensor.empty() : tensor<128x512xf32>
  %fill1 = linalg.fill ins(%c0 : f32) outs(%e01 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %mm1 = linalg.matmul ins(%cA, %t1 : tensor<128x512xf16>, tensor<512x512xf16>) outs(%fill1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %eh = tensor.empty() : tensor<128x512xf16>
  %h = linalg.generic {indexing_maps = [#e2, #eb, #e2], iterator_types = ["parallel", "parallel"]} ins(%mm1, %cb1 : tensor<128x512xf32>, tensor<512xf32>) outs(%eh : tensor<128x512xf16>) {
  ^bb0(%in: f32, %b: f32, %out: f16):
    %s = arith.addf %in, %b : f32
    %cmp = arith.cmpf ugt, %s, %c0 : f32
    %sel = arith.select %cmp, %s, %c0 : f32
    %tr = arith.truncf %sel : f32 to f16
    linalg.yield %tr : f16
  } -> tensor<128x512xf16>
  %et2 = tensor.empty() : tensor<512x256xf16>
  %t2 = linalg.transpose ins(%cW2 : tensor<256x512xf16>) outs(%et2 : tensor<512x256xf16>) permutation = [1, 0]
  %e02 = tensor.empty() : tensor<128x256xf32>
  %fill2 = linalg.fill ins(%c0 : f32) outs(%e02 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %mm2 = linalg.matmul ins(%h, %t2 : tensor<128x512xf16>, tensor<512x256xf16>) outs(%fill2 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %ec = tensor.empty() : tensor<128x256xf32>
  %relu2 = linalg.generic {indexing_maps = [#e2, #eb, #e2], iterator_types = ["parallel", "parallel"]} ins(%mm2, %cb2 : tensor<128x256xf32>, tensor<256xf32>) outs(%ec : tensor<128x256xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %s = arith.addf %in, %b : f32
    %cmp = arith.cmpf ugt, %s, %c0 : f32
    %sel = arith.select %cmp, %s, %c0 : f32
    linalg.yield %sel : f32
  } -> tensor<128x256xf32>
  bufferization.materialize_in_destination %relu2 in restrict writable %C : (tensor<128x256xf32>, memref<128x256xf32>) -> ()
  return
}
