func.func @batch_matmul(%A: memref<4x256x256xf16>, %B: memref<4x256x256xf16>, %C: memref<4x256x256xf32>) {
  %cA = bufferization.to_tensor %A restrict : memref<4x256x256xf16> to tensor<4x256x256xf16>
  %cB = bufferization.to_tensor %B restrict : memref<4x256x256xf16> to tensor<4x256x256xf16>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x256x256xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x256x256xf32>) -> tensor<4x256x256xf32>
  %res = linalg.batch_matmul ins(%cA, %cB : tensor<4x256x256xf16>, tensor<4x256x256xf16>)
                       outs(%filled : tensor<4x256x256xf32>) -> tensor<4x256x256xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<4x256x256xf32>, memref<4x256x256xf32>) -> ()
  return
}
