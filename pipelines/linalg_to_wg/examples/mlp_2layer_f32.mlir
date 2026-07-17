// A 2-layer MLP in ALL-f32 (the raw Torch-MLIR / KernelBench import form: A, W, and
// C all f32). XeGPU/DPAS needs f16 A/B operands, so lower_mlp_to_wg inserts f32->f16
// truncf casts on each matmul's A/B (cast_matmul_operands_to_f16), fused into the
// k-loop as in-register truncf; the accumulator/output stays f32. Verified GPU-correct.
#m = affine_map<(d0, d1) -> (d0, d1)>
#b = affine_map<(d0, d1) -> (d1)>
func.func @mlp2(%A: memref<128x512xf32>, %W1: memref<512x512xf32>, %bias1: memref<512xf32>, %W2: memref<256x512xf32>, %bias2: memref<256xf32>, %C: memref<128x256xf32>) {
  %cA = bufferization.to_tensor %A restrict : memref<128x512xf32> to tensor<128x512xf32>
  %cW1 = bufferization.to_tensor %W1 restrict : memref<512x512xf32> to tensor<512x512xf32>
  %cb1 = bufferization.to_tensor %bias1 restrict : memref<512xf32> to tensor<512xf32>
  %cW2 = bufferization.to_tensor %W2 restrict : memref<256x512xf32> to tensor<256x512xf32>
  %cb2 = bufferization.to_tensor %bias2 restrict : memref<256xf32> to tensor<256xf32>
  %c0 = arith.constant 0.0 : f32
  %et1 = tensor.empty() : tensor<512x512xf32>
  %t1 = linalg.transpose ins(%cW1 : tensor<512x512xf32>) outs(%et1 : tensor<512x512xf32>) permutation = [1, 0]
  %e01 = tensor.empty() : tensor<128x512xf32>
  %f1 = linalg.fill ins(%c0 : f32) outs(%e01 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %mm1 = linalg.matmul ins(%cA, %t1 : tensor<128x512xf32>, tensor<512x512xf32>) outs(%f1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %eh = tensor.empty() : tensor<128x512xf32>
  %h = linalg.generic {indexing_maps = [#m, #b, #m], iterator_types = ["parallel","parallel"]} ins(%mm1, %cb1 : tensor<128x512xf32>, tensor<512xf32>) outs(%eh : tensor<128x512xf32>) {
  ^bb0(%in: f32, %bb: f32, %o: f32): %s = arith.addf %in, %bb : f32
    %cmp = arith.cmpf ugt, %s, %c0 : f32
    %sel = arith.select %cmp, %s, %c0 : f32
    linalg.yield %sel : f32 } -> tensor<128x512xf32>
  %et2 = tensor.empty() : tensor<512x256xf32>
  %t2 = linalg.transpose ins(%cW2 : tensor<256x512xf32>) outs(%et2 : tensor<512x256xf32>) permutation = [1, 0]
  %e02 = tensor.empty() : tensor<128x256xf32>
  %f2 = linalg.fill ins(%c0 : f32) outs(%e02 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %mm2 = linalg.matmul ins(%h, %t2 : tensor<128x512xf32>, tensor<512x256xf32>) outs(%f2 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %ec = tensor.empty() : tensor<128x256xf32>
  %r2 = linalg.generic {indexing_maps = [#m, #b, #m], iterator_types = ["parallel","parallel"]} ins(%mm2, %cb2 : tensor<128x256xf32>, tensor<256xf32>) outs(%ec : tensor<128x256xf32>) {
  ^bb0(%in: f32, %bb: f32, %o: f32): %s = arith.addf %in, %bb : f32
    %cmp = arith.cmpf ugt, %s, %c0 : f32
    %sel = arith.select %cmp, %s, %c0 : f32
    linalg.yield %sel : f32 } -> tensor<128x256xf32>
  bufferization.materialize_in_destination %r2 in restrict writable %C : (tensor<128x256xf32>, memref<128x256xf32>) -> ()
  return
}
