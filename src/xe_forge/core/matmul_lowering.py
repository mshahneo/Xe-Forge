"""
Matmul "second-opinion" lowering via the lighthouse XeGPU schedule.

Xe-Forge's own Linalg->WG pipeline already lowers plain matmul (an LLM-proposed,
KB-grounded config sweep — see :mod:`xe_forge.core.linalg_lowering`). For matmul we
therefore use lighthouse NOT to add coverage but as a *second opinion*: lower the
same GEMM via lighthouse's tuned schedule, time it through the identical rtclock
harness the config sweep uses, and keep whichever kernel is actually faster
(correctness-gated). This is distinct from the softmax/layer-norm/attention paths,
where lighthouse is the *only* engine that lowers the op.

Dump-only, like the rest of the lighthouse backend (no run-time import). Two matmul
specifics vs the other schedules:
  * the matmul schedule needs a ``matmul_params.json`` param DB that ships only in
    the lighthouse checkout, not the pip-installed site-packages — so the dump MUST
    run via ``uv run`` (``prefer_uv=True``);
  * the dumped kernel carries its own ``known_grid_size``/``known_block_size`` (it
    chose its own tiling), so timing it needs the launch geometry read back out of
    the dump rather than computed from a Xe-Forge LoweringConfig.

The dumped ABI is ``@payload_kernel(C: MxNxf32, A: MxKxf16, B: KxNxf16)`` — the same
(C, A, B) = (f32, f16, f16) order Xe-Forge's own lowering emits, so the shared
timing harness (which fills A/B and CPU-checks via gemmF16F16F32) applies unchanged.
Best-effort: any step that can't be satisfied returns None so the caller keeps its
own (Xe-Forge) kernel.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from xe_forge.core.lighthouse_backend import LIGHTHOUSE_ROOT, dump_wg_kernel

logger = logging.getLogger(__name__)

MATMUL_REL = "examples/xegpu/matmul.py"


def lower_matmul_via_lighthouse(
    m: int,
    n: int,
    k: int,
    ab_type: str = "f16",
    lighthouse_root: str | Path = LIGHTHOUSE_ROOT,
    timeout: int = 300,
) -> str | None:
    """Lower an (M, N, K) matmul to an XeGPU-WG kernel via lighthouse.

    Runs ``matmul.py --dump-kernel=xegpu-wg --sizes M N K --ab-type <ab_type>`` and
    returns the dumped WG module (the @payload_kernel gpu.module + host funcs), or
    None if the generator is unavailable or fails. Forces ``uv run`` because the
    matmul schedule needs the checkout-only ``matmul_params.json``.
    """
    logger.info("Second opinion: lowering matmul via lighthouse M=%d N=%d K=%d", m, n, k)
    return dump_wg_kernel(
        MATMUL_REL,
        ["--sizes", str(m), str(n), str(k), "--ab-type", ab_type],
        lighthouse_root=lighthouse_root,
        timeout=timeout,
        prefer_uv=True,
    )


def extract_launch_geometry(wg_code: str) -> tuple[int, int, int] | None:
    """Read (grid_m, grid_n, nb_threads) from a dumped WG matmul kernel.

    The lighthouse gpu.func carries its chosen tiling as
    ``known_grid_size = array<i32: gx, gy, gz>`` and
    ``known_block_size = array<i32: threads, 1, 1>``. Returns
    (grid_m=gx, grid_n=gy, nb_threads=threads), or None if either attribute is
    missing (so the caller skips timing this dump rather than guessing a geometry).
    """
    grid = re.search(r"known_grid_size\s*=\s*array<i32:\s*(\d+),\s*(\d+),\s*(\d+)>", wg_code)
    block = re.search(r"known_block_size\s*=\s*array<i32:\s*(\d+),\s*(\d+),\s*(\d+)>", wg_code)
    if grid is None or block is None:
        return None
    grid_m, grid_n = int(grid.group(1)), int(grid.group(2))
    nb_threads = int(block.group(1))
    return grid_m, grid_n, nb_threads
