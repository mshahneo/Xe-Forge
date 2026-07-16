"""
Fused-attention Linalg -> XeGPU-WG lowering via the lighthouse schedule.

Our hand-written Linalg->WG pipeline (pipelines/linalg_to_wg) lowers exactly one
`linalg.matmul`. A *fused attention* graph (transpose + batch_matmul QK^T + scale
+ softmax + batch_matmul PV) is out of its scope: batch_matmul isn't supported and
there is no fused flash-attention schedule. Rather than reimplement that schedule,
we reuse lighthouse's `examples/xegpu/fused_attention.py --dump-kernel=xegpu-wg`
as the *lowering engine* only (independence-friendly: we consume the dumped .mlir,
we don't depend on lighthouse at run time). The dumped WG kernel then flows into
Xe-Forge's existing WG path, where the GRF sweep applies the large-register-file
win (~1.7x) autonomously.

This module is deliberately narrow: it (a) recognizes the fused-attention Linalg
pattern and its (Z, H, n_ctx, n_head) shape, (b) shells out to the lighthouse
generator, and (c) synthesizes a runnable single-launch @main harness around the
dumped gpu.module so the executor can time it. Everything is best-effort: any step
that can't be satisfied returns None so the pipeline falls back cleanly.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the lighthouse fused-attention generator lives (pip-installed there).
LIGHTHOUSE_ROOT = os.environ.get(
    "LIGHTHOUSE_ROOT", "/data/gta/upstream/lighthouse"
)
FUSED_ATTENTION_REL = "examples/xegpu/fused_attention.py"


def detect_attention_shape(code: str) -> tuple[int, int, int, int] | None:
    """Recognize the fused-attention Linalg graph and return (Z, H, n_ctx, n_head).

    The pattern (from lighthouse's linalg attention input) is: two
    `linalg.batch_matmul` (QK^T and PV) with a `linalg.softmax` between them. The
    4-D operand memrefs carry the shape as ``memref<ZxHxn_ctxxn_headxf16>``.
    Returns None if the graph is not this fused-attention shape.
    """
    has_softmax = "linalg.softmax" in code
    n_batch_matmul = len(re.findall(r"linalg\.batch_matmul", code))
    if not (has_softmax and n_batch_matmul >= 2):
        return None
    # Q/K/V operands are 4-D memrefs (ZxHxn_ctxxn_head). Take the first such shape;
    # in the attention graph all of Q/K/V share it.
    m = re.search(r"memref<(\d+)x(\d+)x(\d+)x(\d+)xf16>", code)
    if not m:
        return None
    z, h, n_ctx, n_head = (int(g) for g in m.groups())
    return z, h, n_ctx, n_head


def _lighthouse_python() -> str | None:
    """Pick an interpreter that can run the lighthouse generator.

    Prefers ``uv run`` from the lighthouse root (README-recommended), else a
    Python that already imports lighthouse. Returns a shell-ready command prefix
    or None if none is available.
    """
    # A direct interpreter with lighthouse importable is fastest (~0.6s).
    for py in (os.environ.get("LIGHTHOUSE_PYTHON"), "/home/gta/.venv/bin/python"):
        if py and os.path.exists(py):
            try:
                r = subprocess.run(
                    [py, "-c", "import lighthouse"],
                    capture_output=True,
                    timeout=60,
                )
                if r.returncode == 0:
                    return py
            except Exception:
                pass
    # Fall back to `uv run` (resolves the lighthouse venv itself).
    if shutil.which("uv"):
        return "uv-run"
    return None


def lower_attention_to_wg(
    z: int,
    h: int,
    n_ctx: int,
    n_head: int,
    lighthouse_root: str | Path = LIGHTHOUSE_ROOT,
    timeout: int = 300,
) -> str | None:
    """Lower a fused-attention shape to an XeGPU-WG kernel via lighthouse.

    Runs ``fused_attention.py --dump-kernel=xegpu-wg`` for the given shape and
    returns the dumped MLIR module (the @payload_kernel gpu.module + host funcs),
    or None if the generator is unavailable or fails.
    """
    root = Path(lighthouse_root)
    gen = root / FUSED_ATTENTION_REL
    if not gen.exists():
        logger.warning("Lighthouse generator not found at %s; cannot lower attention.", gen)
        return None
    py = _lighthouse_python()
    if py is None:
        logger.warning("No lighthouse-capable interpreter found; cannot lower attention.")
        return None

    dump_args = [
        str(gen),
        "--dump-kernel=xegpu-wg",
        f"--batch-size={z}",
        f"--num-heads={h}",
        f"--n-ctx={n_ctx}",
        f"--n-head={n_head}",
    ]
    cmd = (["uv", "run", *dump_args] if py == "uv-run" else [py, *dump_args])
    logger.info(
        "Lowering attention via lighthouse: Z=%d H=%d n_ctx=%d n_head=%d", z, h, n_ctx, n_head
    )
    try:
        r = subprocess.run(
            cmd, cwd=str(root), capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        logger.warning("Lighthouse attention lowering timed out.")
        return None
    if r.returncode != 0:
        logger.warning(
            "Lighthouse attention lowering failed (rc=%d): %s",
            r.returncode,
            r.stderr.decode(errors="replace")[-400:],
        )
        return None
    out = r.stdout.decode(errors="replace")
    if "gpu.container_module" not in out or "gpu.func" not in out:
        logger.warning("Lighthouse dump did not contain a WG gpu.module; skipping.")
        return None
    return out


# --- runnable-harness synthesis ------------------------------------------------

_GPU_FUNC_RE = re.compile(r"gpu\.func\s+@(\w+)\s*\(([^)]*)\)")
_GRID_RE = re.compile(r"known_grid_size\s*=\s*array<i32:\s*([\d,\s]+)>")
_BLOCK_RE = re.compile(r"known_block_size\s*=\s*array<i32:\s*([\d,\s]+)>")
_ARG_TYPE_RE = re.compile(r"%\w+\s*:\s*(memref<[^>]+>)")


def extract_gpu_module(code: str) -> str | None:
    """Return the ``gpu.module { ... }`` block from *code* (brace-matched)."""
    start = code.find("gpu.module")
    if start == -1:
        return None
    brace = code.find("{", start)
    if brace == -1:
        return None
    depth = 0
    for i in range(brace, len(code)):
        c = code[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return code[start : i + 1]
    return None


def synthesize_run_harness(
    wg_code: str,
    grid: tuple[int, int, int] | None = None,
    block: tuple[int, int, int] | None = None,
) -> str | None:
    """Wrap a WG gpu.module in a runnable single-launch @main harness.

    Parses the kernel name, grid/block sizes (from the gpu.func known_grid_size /
    known_block_size attributes) and argument memref types (from the gpu.func
    signature), then emits a self-contained module that gpu.allocs one buffer per
    kernel argument and launches the kernel ONCE. This is a *timing* harness (buffers
    left uninitialized, no reference/allclose) — used by the correctness-free GRF
    sweep and by tile autotuning. Returns None if the signature can't be parsed.

    *grid* / *block* override the launch bounds when the kernel doesn't carry
    known_grid_size (e.g. matmul/MLP kernels collapse the grid to 1-D and compute
    it at launch). Pass grid explicitly (computed from config + dims) in that case.
    """
    gpu_module = extract_gpu_module(wg_code)
    if gpu_module is None:
        return None
    fm = _GPU_FUNC_RE.search(gpu_module)
    if fm is None:
        return None
    kernel_name = fm.group(1)
    arg_types = _ARG_TYPE_RE.findall(fm.group(2))
    if not arg_types:
        return None
    if grid is None:
        gm = _GRID_RE.search(gpu_module)
        if gm is None:
            return None
        grid = [int(x) for x in gm.group(1).split(",") if x.strip()]
    else:
        grid = list(grid)
    if block is None:
        bm = _BLOCK_RE.search(gpu_module)
        if bm is None:
            return None
        block = [int(x) for x in bm.group(1).split(",") if x.strip()]
    else:
        block = list(block)
    grid = (grid + [1, 1, 1])[:3]
    block = (block + [1, 1, 1])[:3]

    # Distinct index constants needed for grid/block + the launch.
    needed = sorted({1, *grid, *block})
    const_lines = "\n".join(
        f"    %c{v} = arith.constant {v} : index" for v in needed
    )
    alloc_lines = "\n".join(
        f"    %buf{i} = gpu.alloc() : {t}" for i, t in enumerate(arg_types)
    )
    launch_args = ", ".join(
        f"%buf{i} : {t}" for i, t in enumerate(arg_types)
    )
    gx, gy, gz = grid
    bx, by, bz = block

    # Carry any top-level affine_map alias defs (#map = ...) the gpu.module uses.
    aliases = "\n".join(
        ln for ln in wg_code.splitlines() if ln.lstrip().startswith("#map")
    )

    return f"""{aliases}
module attributes {{gpu.container_module}} {{
{_indent_module(gpu_module)}
  func.func @main() attributes {{llvm.emit_c_interface}} {{
{const_lines}
{alloc_lines}
    gpu.launch_func @{kernel_name}::@{kernel_name} blocks in (%c{gx}, %c{gy}, %c{gz}) threads in (%c{bx}, %c{by}, %c{bz}) args({launch_args})
    gpu.wait
    return
  }}
}}
"""


def _indent_module(block: str) -> str:
    """Indent a top-level gpu.module block by two spaces to sit inside @main's module."""
    return "\n".join("  " + line if line else line for line in block.splitlines())
