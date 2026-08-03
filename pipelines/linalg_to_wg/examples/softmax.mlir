// Standalone row-softmax (torch.softmax(x, dim=-1)) at Linalg level.
// Xe-Forge's matmul pipeline has no reduction lowering, so this routes to the
// lighthouse softmax schedule (examples/xegpu/softmax.py --dump-kernel=xegpu-wg)
// as the lowering engine; the dumped XeGPU-WG kernel then flows into the WG stages.
// f32 end-to-end (no DPAS) — the 16-bit-float A/B matmul constraint does not apply.
func.func @softmax(%arg0: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  %e = tensor.empty() : tensor<1024x512xf32>
  %0 = linalg.softmax dimension(1)
      ins(%arg0 : tensor<1024x512xf32>)
      outs(%e : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %0 : tensor<1024x512xf32>
}
