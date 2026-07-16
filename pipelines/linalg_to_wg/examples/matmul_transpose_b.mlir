// C = A · B^T  (the nn.Linear form: x @ W^T). B is stored [N, K]; the transpose is
// expressed via indexing_maps and lowers to an in-register vector.transpose feeding
// the dpas. Lowered with the transpose-B stage-3 annotation (see
// templates/wg_annotate_transpose_b.mlir.j2).
#mA = affine_map<(m, n, k) -> (m, k)>
#mB = affine_map<(m, n, k) -> (n, k)>
#mC = affine_map<(m, n, k) -> (m, n)>
func.func @matmul(%A: memref<2048x8192xf16>, %B: memref<4096x8192xf16>, %C: memref<2048x4096xf32>) {
  %cA = bufferization.to_tensor %A restrict : memref<2048x8192xf16> to tensor<2048x8192xf16>
  %cB = bufferization.to_tensor %B restrict : memref<4096x8192xf16> to tensor<4096x8192xf16>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x4096xf32>
  %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<2048x4096xf32>) -> tensor<2048x4096xf32>
  %res = linalg.matmul indexing_maps = [#mA, #mB, #mC] ins(%cA, %cB : tensor<2048x8192xf16>, tensor<4096x8192xf16>) outs(%filled : tensor<2048x4096xf32>) -> tensor<2048x4096xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<2048x4096xf32>, memref<2048x4096xf32>) -> ()
  return
}
