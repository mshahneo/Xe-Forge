"""
MLIR Kernel Executor — compiles and runs XeGPU workgroup-level MLIR kernels.

Targets self-contained WG-level XeGPU test files (the form used in
mlir-extensions/test/Integration/Dialect/XeGPU/WG): each file carries its own
``@main`` host harness that allocates + fills inputs, launches the
``gpu.module`` kernel, computes an in-file CPU reference, and prints
``[ALLCLOSE: TRUE]``. Optionally it brackets timed launches with ``rtclock()``
and prints the elapsed seconds.

The run pipeline mirrors the IMEX lit RUN line:

    imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
      | mlir-runner --shared-libs=<levelzero,runner_utils,c_runner_utils,irunner_utils> \
                    --entry-point-result=void

Because correctness and the reference are baked into the file, the optimizer
edits only the kernel body / layout attrs / launch geometry while ``@main``
stays fixed as the oracle. This executor therefore:
  - compiles+runs a whole .mlir file,
  - reads correctness from the printed ``[ALLCLOSE: TRUE]`` line,
  - reads timing from rtclock-printed seconds when present,
and exposes the same compare_kernels() contract the optimizer agent expects.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Toolchain discovery
# ---------------------------------------------------------------------------
# Allow override via env; default to the upstream build-with-imex tree.
_DEFAULT_BIN = "/home/gta/upstream/llvm-project/build-with-imex/bin"
_DEFAULT_LIB = "/home/gta/upstream/llvm-project/build-with-imex/lib"

MLIR_BIN_DIR = os.environ.get("MLIR_BIN_DIR", _DEFAULT_BIN)
MLIR_LIB_DIR = os.environ.get("MLIR_LIB_DIR", _DEFAULT_LIB)

# The WG-level XeGPU lowering pipeline (matches IMEX lit RUN lines).
WG_PIPELINE = os.environ.get(
    "MLIR_XEGPU_PIPELINE",
    "--gpu-lower-to-xevm-pipeline=xegpu-op-level=workgroup",
)

# Shared runtime libs passed to mlir-runner, in order.
_RUNTIME_LIBS = [
    "libmlir_levelzero_runtime.so",
    "libmlir_runner_utils.so",
    "libmlir_c_runner_utils.so",
    "libimex_runner_utils.so",
]

# Markers emitted by the in-file harness.
_ALLCLOSE_ANY_RE = re.compile(r"\[ALLCLOSE:\s*(TRUE|FALSE)\]", re.IGNORECASE)
_FLOAT = r"([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)"
# Preferred: an explicitly labeled timing line, e.g.
#   "Average time (ms): 8.76411"  or  "GPU time: 1.23 ms"
# We capture the value and remember whether the label says milliseconds.
_TIME_MS_RE = re.compile(r"time[^:\n]*\(ms\)\s*:?\s*" + _FLOAT, re.IGNORECASE)
_TIME_LABELED_RE = re.compile(r"\btime\b[^:\n]*:\s*" + _FLOAT, re.IGNORECASE)
# Fallback: a bare standalone float on its own line (older rtclock harnesses
# that print raw seconds). Only used when no labeled line is found.
_FLOAT_LINE_RE = re.compile(r"^\s*" + _FLOAT + r"\s*$")


@dataclass
class MlirComparisonResult:
    """Result of comparing original vs optimized MLIR kernel performance."""

    original_time_ms: float
    optimized_time_ms: float
    speedup: float
    original_tflops: float | None = None
    optimized_tflops: float | None = None
    original_correct: bool = True
    optimized_correct: bool = True
    is_slower: bool = False
    feedback_message: str = ""

    @property
    def original_time_us(self) -> float:
        return self.original_time_ms * 1000

    @property
    def optimized_time_us(self) -> float:
        return self.optimized_time_ms * 1000


class MlirExecutor:
    """Compiles and runs XeGPU WG-level MLIR kernels, measures performance."""

    def __init__(
        self,
        bin_dir: str = MLIR_BIN_DIR,
        lib_dir: str = MLIR_LIB_DIR,
        pipeline: str = WG_PIPELINE,
        compile_timeout: int = 300,
        run_timeout: int = 300,
        require_correctness: bool = True,
        speedup_tol: float = 0.03,
    ):
        self.bin_dir = bin_dir
        self.lib_dir = lib_dir
        self.pipeline = pipeline
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.require_correctness = require_correctness
        # A candidate is only flagged "slower" when it regresses by more than
        # this fraction. Timing is measured across separate processes, so
        # run-to-run noise of a few percent is expected; without a band,
        # identical code is misread as a regression.
        self.speedup_tol = speedup_tol
        self._build_dir: str | None = None

        self.imex_opt = str(Path(bin_dir) / "imex-opt")
        self.mlir_opt = str(Path(bin_dir) / "mlir-opt")
        self.mlir_runner = str(Path(bin_dir) / "mlir-runner")
        self._shared_libs = [str(Path(lib_dir) / name) for name in _RUNTIME_LIBS]

        missing = [
            p
            for p in (self.imex_opt, self.mlir_runner, *self._shared_libs)
            if not os.path.exists(p)
        ]
        if missing:
            logger.warning("MLIR toolchain paths not found: %s", missing)

    @property
    def build_dir(self) -> str:
        if self._build_dir is None:
            self._build_dir = tempfile.mkdtemp(prefix="mlir_build_")
        return self._build_dir

    # ------------------------------------------------------------------
    # Core: run one self-contained .mlir file end-to-end
    # ------------------------------------------------------------------
    def execute(
        self,
        kernel_code: str | None = None,
        kernel_path: str | None = None,
        output_name: str = "kernel_mlir",
        flop: float | None = None,
        pipeline_options: str | None = None,
    ) -> ExecutionResult:
        """Lower + run a self-contained WG-level .mlir file.

        Correctness is read from the in-file ``[ALLCLOSE: TRUE]`` marker;
        runtime (ms) from an rtclock-printed seconds value when present.

        *pipeline_options* overrides the ``--gpu-lower-to-xevm-pipeline`` options
        (e.g. to enable large-GRF via ``igc-cmd-options=-ze-opt-large-register-file``).
        The options may contain spaces; the flag is passed as a single argv token.
        """
        if kernel_code is not None:
            src_path = Path(self.build_dir) / f"{output_name}.mlir"
            src_path.write_text(kernel_code)
        elif kernel_path is not None:
            src_path = Path(kernel_path)
        else:
            return ExecutionResult(success=False, error_message="No source code or path provided")

        # Build the lowering flag. When options are given, pass the whole
        # "--gpu-lower-to-xevm-pipeline=<opts>" as ONE argv token (opts may
        # contain a space, e.g. the igc large-GRF option).
        if pipeline_options is not None:
            pipeline_args = [f"--gpu-lower-to-xevm-pipeline={pipeline_options}"]
        else:
            pipeline_args = self.pipeline.split()

        # Stage 1: lower with imex-opt.
        try:
            lowered = subprocess.run(
                [self.imex_opt, str(src_path), *pipeline_args],
                capture_output=True,
                timeout=self.compile_timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, error_message="imex-opt timed out")
        if lowered.returncode != 0:
            err = lowered.stderr.decode(errors="replace")
            return ExecutionResult(
                success=False,
                error_message=f"LOWERING FAILED (imex-opt):\n{_tail(err)}",
            )

        # Stage 2: JIT + run with mlir-runner, feeding lowered IR on stdin.
        run_cmd = [
            self.mlir_runner,
            *_interleave("--shared-libs", self._shared_libs),
            "--entry-point-result=void",
        ]
        try:
            run = subprocess.run(
                run_cmd,
                input=lowered.stdout,
                capture_output=True,
                timeout=self.run_timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, error_message="mlir-runner timed out")
        if run.returncode != 0:
            err = run.stderr.decode(errors="replace")
            return ExecutionResult(
                success=False,
                error_message=f"RUNTIME FAILED (mlir-runner):\n{_tail(err)}",
            )

        stdout = run.stdout.decode(errors="replace")
        return self._parse_output(stdout, flop=flop)

    def _parse_output(self, stdout: str, flop: float | None = None) -> ExecutionResult:
        """Parse correctness marker and rtclock timing from harness stdout."""
        allclose_match = _ALLCLOSE_ANY_RE.search(stdout)
        # If the harness emits an ALLCLOSE marker, trust it; otherwise we treat
        # a clean run as correct (some kernels only print timing).
        if allclose_match:
            correct = allclose_match.group(1).upper() == "TRUE"
        else:
            correct = True

        # Timing: prefer an explicitly labeled line (already in ms); the harness
        # computes the average over `nruns` and prints e.g. "Average time (ms): X".
        time_ms: float | None = None
        m = _TIME_MS_RE.search(stdout)
        if m:
            time_ms = float(m.group(1))
        else:
            m = _TIME_LABELED_RE.search(stdout)
            if m:
                # Labeled but no explicit (ms); assume the harness already
                # reports milliseconds (the WG convention).
                time_ms = float(m.group(1))
            else:
                # Fallback: a lone float line from a raw-seconds rtclock harness.
                seconds: float | None = None
                for line in stdout.splitlines():
                    fm = _FLOAT_LINE_RE.match(line)
                    if fm:
                        seconds = float(fm.group(1))
                time_ms = seconds * 1000.0 if seconds is not None else None

        tflops = None
        if flop and time_ms and time_ms > 0:
            tflops = (flop / 1e12) / (time_ms / 1e3)

        return ExecutionResult(
            success=True,
            output_correct=correct,
            execution_time_ms=time_ms,
            tflops=tflops,
        )

    # ------------------------------------------------------------------
    # CoVeR contract: compare original vs optimized
    # ------------------------------------------------------------------
    def compare_kernels(
        self,
        original_code: str | None = None,
        optimized_code: str | None = None,
        original_path: str | None = None,
        optimized_path: str | None = None,
        dims: dict[str, int | float] | None = None,
        flop: float | None = None,
        **_ignored,
    ) -> MlirComparisonResult:
        """Run both kernels and compare correctness + runtime.

        Each file self-verifies against its embedded CPU reference, so the
        optimized kernel's own ``[ALLCLOSE: TRUE]`` is the correctness gate.
        Speedup uses rtclock timing when the harness prints it.
        """
        orig = self.execute(
            kernel_code=original_code,
            kernel_path=original_path,
            output_name="original_mlir",
            flop=flop,
        )
        opt = self.execute(
            kernel_code=optimized_code,
            kernel_path=optimized_path,
            output_name="optimized_mlir",
            flop=flop,
        )

        if not orig.success:
            return MlirComparisonResult(
                original_time_ms=float("inf"),
                optimized_time_ms=float("inf"),
                speedup=0.0,
                original_correct=False,
                feedback_message=f"FAILURE: Original kernel failed: {orig.error_message}",
            )
        if not opt.success:
            return MlirComparisonResult(
                original_time_ms=orig.execution_time_ms or float("inf"),
                optimized_time_ms=float("inf"),
                speedup=0.0,
                optimized_correct=False,
                feedback_message=(
                    f"FAILURE: Optimized kernel failed: {opt.error_message} "
                    "Fix lowering or runtime errors."
                ),
            )

        opt_correct = bool(opt.output_correct)
        orig_ms = orig.execution_time_ms or float("inf")
        opt_ms = opt.execution_time_ms or float("inf")
        # When no timing is available, fall back to neutral 1.0x so correctness
        # alone gates acceptance.
        if orig.execution_time_ms is None or opt.execution_time_ms is None:
            speedup = 1.0
            timing_note = " (no rtclock timing; speedup not measured)"
        else:
            speedup = orig_ms / opt_ms if opt_ms > 0 else 0.0
            timing_note = f" Original: {orig_ms:.4f}ms, Optimized: {opt_ms:.4f}ms."
        is_slower = speedup < (1.0 - self.speedup_tol)

        if not opt_correct:
            msg = (
                "CORRECTNESS FAILURE: Optimized kernel does not match the in-file "
                f"reference ([ALLCLOSE: FALSE]).{timing_note} "
                "Fix numerical correctness before optimizing for speed."
            )
        elif is_slower:
            sd = 1.0 / speedup if speedup > 0 else float("inf")
            msg = (
                f"PERFORMANCE REGRESSION: {sd:.2f}x SLOWER.{timing_note} "
                "Try a different approach."
            )
        else:
            msg = f"SUCCESS: {speedup:.2f}x speedup. Correctness: PASSED.{timing_note}"

        return MlirComparisonResult(
            original_time_ms=orig_ms,
            optimized_time_ms=opt_ms,
            speedup=speedup,
            original_tflops=orig.tflops,
            optimized_tflops=opt.tflops,
            original_correct=bool(orig.output_correct),
            optimized_correct=opt_correct,
            is_slower=is_slower,
            feedback_message=msg,
        )

    # ------------------------------------------------------------------
    # Two-level flow: lower a bare Linalg kernel to XeGPU WG-level
    # ------------------------------------------------------------------
    def lower_linalg_to_wg(self, linalg_code: str, config) -> tuple[bool, str, str]:
        """Lower a bare Linalg matmul kernel to XeGPU WG-level IR under *config*.

        *config* is a LoweringConfig (xe_forge.core.linalg_lowering). Renders the
        two transform libraries for the config, then runs the 3-stage recipe:
          1. transform-interpreter with the tile/vectorize library
          2. gpu-kernel-outlining + xevm-attach-target + gpu.module(vector->xegpu)
          3. transform-interpreter with the WG layout-annotation library

        Returns (success, wg_code, error_message).
        """
        errs = config.validate()
        if errs:
            return False, "", f"invalid lowering config: {'; '.join(errs)}"

        work = tempfile.mkdtemp(prefix="mlir_lower_")
        try:
            tile_lib, anno_lib = config.render(work)
            src = Path(work) / "input.mlir"
            src.write_text(linalg_code)

            # Stage 1
            s1 = self._run_opt(
                [str(src),
                 f"--transform-preload-library=transform-library-paths={tile_lib}",
                 "--transform-interpreter"],
                use_imex=False,
            )
            if not s1.ok:
                return False, "", f"LOWERING stage1 (tile/vectorize) failed:\n{_tail(s1.err)}"

            # Stage 2 (normal pass pipeline; dlti-safe)
            s2 = self._run_opt(
                ["-",
                 "--pass-pipeline=builtin.module(gpu-kernel-outlining, "
                 "xevm-attach-target{chip=bmg O=3}, "
                 "gpu.module(convert-vector-to-xegpu))"],
                use_imex=False,
                stdin=s1.out,
            )
            if not s2.ok:
                return False, "", f"LOWERING stage2 (outline/xegpu) failed:\n{_tail(s2.err)}"

            # Stage 3
            s3 = self._run_opt(
                ["-",
                 f"--transform-preload-library=transform-library-paths={anno_lib}",
                 "--transform-interpreter"],
                use_imex=False,
                stdin=s2.out,
            )
            if not s3.ok:
                return False, "", f"LOWERING stage3 (wg-annotate) failed:\n{_tail(s3.err)}"

            return True, s3.out.decode(errors="replace"), ""
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def sweep_configs(
        self,
        linalg_code: str,
        harness: str | None,
        configs,
        flop=None,
        dims: tuple[int, int, int] | None = None,
    ):
        """Lower *linalg_code* under each config, run on GPU, rank by kernel time.

        Two modes:
          - Timed (preferred): pass ``dims=(M, N, K)`` and ``harness=None``. For each
            config a kernel-only rtclock timing harness is rendered
            (render_timing_harness) and only the lowered ``gpu.module`` is spliced in,
            so ``Average time (ms)`` reflects kernel time. Configs are ranked by that.
          - Legacy (correctness-only): pass a full ``harness`` with a ``// KERNEL``
            marker and no dims; the whole lowered body is spliced (no timing signal).

        Each candidate must pass ``[ALLCLOSE: TRUE]``. Returns
        (best_config, best_wg_code, best_result, all_results). Invalid/failing configs
        are skipped.
        """
        from xe_forge.core.linalg_lowering import render_timing_harness

        results = []
        best = None
        for cfg in configs:
            ok, wg, err = self.lower_linalg_to_wg(linalg_code, cfg)
            if not ok:
                logger.info("config %s: lowering failed (%s)", cfg, _tail(err, 2))
                results.append((cfg, None, err))
                continue
            if dims is not None:
                m, n, k = dims
                kname = _kernel_name(wg)
                timing_harness = render_timing_harness(cfg, m, n, k, kernel_name=kname)
                runnable = self._splice_kernel(timing_harness, wg, kernel_only=True)
            else:
                runnable = self._splice_kernel(harness, wg)
            # Large-GRF is a run-time (igc) lowering flag carried by the config.
            r = self.execute(
                kernel_code=runnable,
                output_name="sweep",
                flop=flop,
                pipeline_options=cfg.run_pipeline_options(),
            )
            ok_run = r.success and (r.output_correct is not False)
            ms = r.execution_time_ms if ok_run else None
            logger.info(
                "config %s: %s%s",
                cfg,
                "OK" if ok_run else f"FAIL ({_tail(r.error_message or '', 2)})",
                f" {ms:.4f}ms" if ms else "",
            )
            results.append((cfg, r, None))
            if ok_run:
                score = ms if ms is not None else float("inf")
                if best is None or score < best[0]:
                    best = (score, cfg, wg, r)
        if best is None:
            return None, None, None, results
        return best[1], best[2], best[3], results

    @staticmethod
    def _splice_kernel(harness: str, wg_code: str, kernel_only: bool = False) -> str:
        """Splice the lowered WG kernel into *harness*.

        The lowered module is:  <alias defs> module {...host @test... gpu.module...}
        The harness is:         <// ALIASES> module { // KERNEL  ...@main... }
        Alias defs (``#map = ...``) are hoisted to the ``// ALIASES`` slot.

        If *kernel_only*, only the ``gpu.module`` (the device kernel) is spliced —
        used with the timing harness, which supplies its own host launch/@main.
        Otherwise the whole module body (host @test + gpu.module) is spliced.
        """
        text = wg_code.strip()
        mod_at = text.find("module")
        aliases = text[:mod_at].strip()
        body = text[mod_at:]

        if kernel_only:
            inner = _extract_gpu_module(body)
        else:
            # Find the module's *body* opening brace, skipping the optional
            # "attributes {...}" dict.
            i = body.find("{")
            prefix = body[:i]
            if prefix.rstrip().endswith("attributes"):
                depth = 0
                j = i
                while j < len(body):
                    if body[j] == "{":
                        depth += 1
                    elif body[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                i = body.find("{", j + 1)
            inner = body[i + 1 : body.rfind("}")].strip()

        out = harness.replace("// KERNEL", inner)
        if "// ALIASES" in out:
            out = out.replace("// ALIASES", aliases)
        elif aliases:
            out = aliases + "\n" + out
        return out

    def _run_opt(self, args, use_imex=False, stdin: bytes | None = None):
        """Run mlir-opt (or imex-opt) with *args*; return a small result holder."""
        tool = self.imex_opt if use_imex else self.mlir_opt
        try:
            p = subprocess.run(
                [tool, *args],
                input=stdin,
                capture_output=True,
                timeout=self.compile_timeout,
            )
        except subprocess.TimeoutExpired:
            return _OptResult(False, b"", b"mlir-opt timed out")
        return _OptResult(p.returncode == 0, p.stdout, p.stderr)


@dataclass
class _OptResult:
    ok: bool
    out: bytes
    err: bytes

    @property
    def error(self) -> str:
        return self.err.decode(errors="replace")


def _kernel_name(wg_code: str) -> str:
    """Return the outlined gpu.func kernel name (defaults to 'test_kernel')."""
    m = re.search(r"gpu\.func @(\w+)", wg_code)
    return m.group(1) if m else "test_kernel"


def _extract_gpu_module(text: str) -> str:
    """Return the full ``gpu.module {...}`` op (brace-matched) from *text*."""
    start = text.find("gpu.module")
    if start == -1:
        return ""
    brace = text.find("{", start)
    depth = 0
    j = brace
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
        j += 1
    return text[start:]


def _interleave(flag: str, values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        out += [flag, v]
    return out


def _tail(text, n: int = 40) -> str:
    if isinstance(text, (bytes, bytearray)):
        text = text.decode(errors="replace")
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])
