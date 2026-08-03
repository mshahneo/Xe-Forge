"""
Layer-norm Linalg -> XeGPU-WG lowering via the lighthouse schedule.

Like softmax, layer-norm is a reduction+rescale that Xe-Forge's matmul pipeline
does not lower. We reuse lighthouse's `examples/xegpu/layer_norm.py
--dump-kernel=xegpu-wg` as the lowering engine (see
:mod:`xe_forge.core.lighthouse_backend` — dump-only, no run-time dependency). The
dumped WG kernel carries known_grid/block_size, so the attention path's harness
synthesis wraps it into a runnable @main directly.

Unlike softmax there is no named `linalg.layer_norm` op: torch-mlir decomposes it
into a sequence of `linalg.generic` (a mean reduction, a variance reduction, then a
`(x-mean)*inv_std*gamma + beta` combine). Detection is therefore structural — its
signature is `math.rsqrt` (the inv-std; softmax uses `math.exp`, matmul uses
neither) over a 2-D f32 tensor with a matching 1-D f32 gamma/beta. f32 end-to-end
(no DPAS), so the 16-bit-float A/B matmul constraint does not apply. Best-effort:
any step that can't be satisfied returns None so the pipeline falls back cleanly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from xe_forge.core.lighthouse_backend import LIGHTHOUSE_ROOT, dump_wg_kernel

logger = logging.getLogger(__name__)

LAYER_NORM_REL = "examples/xegpu/layer_norm.py"


def detect_layernorm_shape(code: str) -> tuple[int, int] | None:
    """Recognize a row layer-norm Linalg graph and return its (M, N) shape.

    Structural match (there is no named linalg op for layer-norm):
      * `math.rsqrt` present — the inverse-std that distinguishes layer-norm from
        softmax (`math.exp`) and from a plain reduction (neither);
      * a 2-D f32 tensor operand `tensor<MxNxf32>` (the normalized activation);
      * a 1-D f32 tensor `tensor<Nxf32>` whose length matches the 2-D last dim
        (the gamma/beta affine params).
    Rejects graphs containing a contraction (matmul/attention own their routing)
    or a softmax. Returns None if not a layer-norm.
    """
    if "math.rsqrt" not in code:
        return None
    if "linalg.matmul" in code or "linalg.batch_matmul" in code:
        return None
    if "linalg.softmax" in code:
        return None
    two_d = re.search(r"tensor<(\d+)x(\d+)xf32>", code)
    if two_d is None:
        return None
    m, n = int(two_d.group(1)), int(two_d.group(2))
    # Confirm a matching 1-D gamma/beta of length N (the affine params over the
    # normalized dim). This guards against a bare 2-D reduction with an rsqrt.
    if not re.search(rf"tensor<{n}xf32>", code):
        return None
    return m, n


def lower_layernorm_to_wg(
    m: int,
    n: int,
    eps: float = 1e-5,
    lighthouse_root: str | Path = LIGHTHOUSE_ROOT,
    timeout: int = 300,
) -> str | None:
    """Lower an (M, N) row layer-norm to an XeGPU-WG kernel via lighthouse.

    Runs ``layer_norm.py --dump-kernel=xegpu-wg --sizes M N --eps <eps>`` and
    returns the dumped WG module (the @payload_kernel gpu.module + host funcs), or
    None if the generator is unavailable or fails.
    """
    logger.info("Lowering layer-norm via lighthouse: M=%d N=%d eps=%g", m, n, eps)
    return dump_wg_kernel(
        LAYER_NORM_REL,
        ["--sizes", str(m), str(n), "--eps", repr(eps)],
        lighthouse_root=lighthouse_root,
        timeout=timeout,
    )
