func.func @matmul(%A: memref<512x512xf16>, %B: memref<512x512xf16>, %C: memref<512x512xf32>) {
  %cA = bufferization.to_tensor %A restrict : memref<512x512xf16> to tensor<512x512xf16>
  %cB = bufferization.to_tensor %B restrict : memref<512x512xf16> to tensor<512x512xf16>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<512x512xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
  %res = linalg.matmul ins(%cA, %cB : tensor<512x512xf16>, tensor<512x512xf16>)
                       outs(%filled : tensor<512x512xf32>) -> tensor<512x512xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<512x512xf32>, memref<512x512xf32>) -> ()
  return
}
