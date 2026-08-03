"""
Shared seam for using lighthouse schedules as Xe-Forge lowering engines.

Xe-Forge's hand-written Linalg->WG pipeline (pipelines/linalg_to_wg) covers the
matmul family + MLP chains. For op classes it does *not* lower natively (fused
attention, softmax, layer-norm, whole-model schedules), we reuse the upstream
`llvm/lighthouse` project's XeGPU schedules as the lowering engine only: we shell
out to `examples/xegpu/<schedule>.py --dump-kernel=xegpu-wg`, consume the dumped
WG-level .mlir, and feed it into Xe-Forge's existing WG stages (analysis, the GRF
sweep, autotuning). We do NOT import lighthouse at run time — the dump is a plain
subprocess, keeping Xe-Forge's core torch-free and venv-isolated.

This module is the single choke point for that subprocess. Every lighthouse-backed
lowering (attention, softmax, ...) goes through :func:`dump_wg_kernel`. That is
deliberate: the "import-later" path (swap the subprocess for an in-process
`lighthouse.PipelineDriver` built on the MLIR Python bindings) is a reimplementation
of *this one function* — all callers and downstream WG stages stay untouched.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the lighthouse checkout lives (its XeGPU schedules are under examples/xegpu).
LIGHTHOUSE_ROOT = os.environ.get("LIGHTHOUSE_ROOT", "/data/gta/upstream/lighthouse")


def lighthouse_python() -> str | None:
    """Pick an interpreter that can run a lighthouse generator.

    Prefers a direct Python that already imports ``lighthouse`` (fastest, ~0.6s),
    else falls back to ``uv run`` from the lighthouse root (resolves its own venv).
    Returns a shell-ready command token: a python path, the sentinel ``"uv-run"``,
    or None if neither is available.
    """
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
    if shutil.which("uv"):
        return "uv-run"
    return None


def dump_wg_kernel(
    script_rel: str,
    extra_args: list[str],
    *,
    dump_flag: str = "--dump-kernel=xegpu-wg",
    lighthouse_root: str | Path = LIGHTHOUSE_ROOT,
    timeout: int = 300,
    require: tuple[str, ...] = ("gpu.container_module", "gpu.func"),
    prefer_uv: bool = False,
) -> str | None:
    """Run a lighthouse schedule to XeGPU-WG level and return the dumped MLIR.

    Args:
        script_rel: path to the generator relative to the lighthouse root,
            e.g. ``"examples/xegpu/softmax.py"``.
        extra_args: schedule-specific shape/option flags, e.g. ``["--sizes", "1024", "512"]``.
        dump_flag: the stop-at-WG flag. Most schedules use ``--dump-kernel=xegpu-wg``;
            nanoGPT uses ``--dump xegpu-wg`` (pass that here).
        require: substrings that must all be present in stdout for the dump to be
            considered a valid WG module (guards against a schedule that printed a
            different stage or an error banner on a zero exit).
        prefer_uv: force ``uv run`` from the lighthouse checkout instead of the
            (faster) importable interpreter. The matmul schedule needs a
            ``matmul_params.json`` param DB that ships in the checkout but NOT in the
            pip-installed site-packages, so the direct interpreter imports lighthouse
            fine yet fails at run time — such schedules must set this.

    Returns the dumped module text, or None if the generator is missing, no
    lighthouse-capable interpreter exists, the run fails/times out, or the output
    doesn't look like a WG gpu.module. Best-effort by design: callers fall back
    cleanly to keeping the original input.
    """
    root = Path(lighthouse_root)
    gen = root / script_rel
    if not gen.exists():
        logger.warning("Lighthouse generator not found at %s; cannot lower.", gen)
        return None
    if prefer_uv:
        py = "uv-run" if shutil.which("uv") else None
    else:
        py = lighthouse_python()
    if py is None:
        logger.warning("No lighthouse-capable interpreter found; cannot lower.")
        return None

    dump_args = [str(gen), dump_flag, *extra_args]
    cmd = (["uv", "run", *dump_args] if py == "uv-run" else [py, *dump_args])
    logger.info("Lowering via lighthouse: %s %s", script_rel, " ".join(extra_args))
    try:
        r = subprocess.run(cmd, cwd=str(root), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Lighthouse lowering timed out (%s).", script_rel)
        return None
    if r.returncode != 0:
        logger.warning(
            "Lighthouse lowering failed (%s, rc=%d): %s",
            script_rel,
            r.returncode,
            r.stderr.decode(errors="replace")[-400:],
        )
        return None
    out = r.stdout.decode(errors="replace")
    if not all(tok in out for tok in require):
        logger.warning(
            "Lighthouse dump (%s) did not contain a WG gpu.module; skipping.", script_rel
        )
        return None
    return out
