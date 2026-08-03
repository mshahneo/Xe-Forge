// Standalone row layer-norm (nn.LayerNorm over the last dim) at Linalg level.
// There is no named linalg op for layer-norm; torch-mlir decomposes it into a mean
// reduction, a variance reduction, and a (x-mean)*inv_std*gamma + beta combine
// (note math.rsqrt = the inverse-std). Xe-Forge's matmul pipeline has no reduction
// lowering, so this routes to the lighthouse layer_norm schedule
// (examples/xegpu/layer_norm.py --dump-kernel=xegpu-wg) as the lowering engine; the
// dumped XeGPU-WG kernel then flows into the WG stages. f32 end-to-end (no DPAS) —
// the 16-bit-float A/B matmul constraint does not apply.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @layer_norm(%x: tensor<1024x512xf32>, %gamma: tensor<512xf32>, %beta: tensor<512xf32>) -> tensor<1024x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %inv_n = arith.constant 0.001953125 : f32
  %eps = arith.constant 9.99999974E-6 : f32
  %out = tensor.empty() : tensor<1024x512xf32>
  %r0 = tensor.empty() : tensor<1024xf32>
  %sum_init = linalg.fill ins(%cst : f32) outs(%r0 : tensor<1024xf32>) -> tensor<1024xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<1024x512xf32>) outs(%sum_init : tensor<1024xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %a = arith.addf %in, %acc : f32
    linalg.yield %a : f32
  } -> tensor<1024xf32>
  %v0 = tensor.empty() : tensor<1024xf32>
  %var_init = linalg.fill ins(%cst : f32) outs(%v0 : tensor<1024xf32>) -> tensor<1024xf32>
  %var = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]}
      ins(%x, %sum : tensor<1024x512xf32>, tensor<1024xf32>) outs(%var_init : tensor<1024xf32>) {
  ^bb0(%in: f32, %s: f32, %acc: f32):
    %mean = arith.mulf %s, %inv_n : f32
    %d = arith.subf %in, %mean : f32
    %sq = arith.mulf %d, %d : f32
    %a = arith.addf %sq, %acc : f32
    linalg.yield %a : f32
  } -> tensor<1024xf32>
  %ln = linalg.generic {indexing_maps = [#map, #map1, #map1, #map2, #map2, #map], iterator_types = ["parallel", "parallel"]}
      ins(%x, %sum, %var, %gamma, %beta : tensor<1024x512xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>)
      outs(%out : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %s: f32, %vv: f32, %g: f32, %b: f32, %o: f32):
    %mean = arith.mulf %s, %inv_n : f32
    %varm = arith.mulf %vv, %inv_n : f32
    %vare = arith.addf %varm, %eps : f32
    %inv_std = math.rsqrt %vare : f32
    %d = arith.subf %in, %mean : f32
    %norm = arith.mulf %d, %inv_std : f32
    %scaled = arith.mulf %norm, %g : f32
    %res = arith.addf %scaled, %b : f32
    linalg.yield %res : f32
  } -> tensor<1024x512xf32>
  return %ln : tensor<1024x512xf32>
}
