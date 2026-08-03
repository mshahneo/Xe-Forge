"""
Row-softmax Linalg -> XeGPU-WG lowering via the lighthouse schedule.

Xe-Forge's matmul-family pipeline has no reduction/softmax lowering, so a bare
`linalg.softmax` (or a torch-mlir softmax dump) is out of its scope. Rather than
build a reduction schedule from scratch, we reuse lighthouse's
`examples/xegpu/softmax.py --dump-kernel=xegpu-wg` as the lowering engine (see
:mod:`xe_forge.core.lighthouse_backend` — dump-only, no run-time dependency). The
dumped WG kernel flows into Xe-Forge's existing WG stages + GRF sweep like any
other WG input.

The softmax schedule is f32 end-to-end (no DPAS), so the "A/B must be 16-bit
float" matmul constraint does not apply. This module (a) recognizes a row-softmax
Linalg graph and its (M, N) shape, (b) shells out to the lighthouse generator, and
(c) reuses the attention path's harness synthesis to wrap the dumped gpu.module in
a runnable single-launch @main. Best-effort: any step that can't be satisfied
returns None so the pipeline falls back cleanly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from xe_forge.core.lighthouse_backend import LIGHTHOUSE_ROOT, dump_wg_kernel

logger = logging.getLogger(__name__)

SOFTMAX_REL = "examples/xegpu/softmax.py"


def detect_softmax_shape(code: str) -> tuple[int, int] | None:
    """Recognize a row-softmax Linalg graph and return its (M, N) shape.

    Matches the canonical `linalg.softmax dimension(1)` over a 2-D f32 tensor
    (the torch-mlir spelling of `torch.softmax(x, dim=-1)`). The reduction must be
    over the last dim — that is what lighthouse's softmax schedule computes. A
    non-2-D softmax, a reduction over a non-last dim, or a matmul/attention graph
    (which own their own routing) returns None.
    """
    # Only a *standalone* softmax — if the graph also contains a contraction it is
    # attention or a matmul epilogue, routed elsewhere.
    if "linalg.softmax" not in code:
        return None
    if "linalg.matmul" in code or "linalg.batch_matmul" in code:
        return None
    m = re.search(
        r"linalg\.softmax\s+dimension\((\d+)\)[\s\S]*?tensor<(\d+)x(\d+)xf32>",
        code,
    )
    if m is None:
        return None
    dim, rows, cols = int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Softmax must reduce over the last (column) dim of the 2-D tensor.
    if dim != 1:
        return None
    return rows, cols


def lower_softmax_to_wg(
    m: int,
    n: int,
    lighthouse_root: str | Path = LIGHTHOUSE_ROOT,
    timeout: int = 300,
) -> str | None:
    """Lower an (M, N) row-softmax to an XeGPU-WG kernel via lighthouse.

    Runs ``softmax.py --dump-kernel=xegpu-wg --sizes M N`` and returns the dumped
    WG module (the @payload_kernel gpu.module + host funcs), or None if the
    generator is unavailable or fails.
    """
    logger.info("Lowering softmax via lighthouse: M=%d N=%d", m, n)
    return dump_wg_kernel(
        SOFTMAX_REL,
        ["--sizes", str(m), str(n)],
        lighthouse_root=lighthouse_root,
        timeout=timeout,
    )
