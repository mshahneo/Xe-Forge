"""Tests for what counts as a stage win, and for not skipping a planned stage.

Three defects these pin, all observed on the flash-attention pipeline:

* ``MlirExecutor`` only flags ``is_slower`` below ``1 - speedup_tol``, so anything in
  the noise band passed as an improvement — a run recorded ``memory_access 0.98x ✓``
  and carried the slower kernel into the next stage.
* With ``MLIR_COMPARE_REPEATS=1`` a single process pair decided accept/reject for
  candidates whose claimed gain was the same size as the run-to-run spread.
* The plan is built from the first analysis but the analysis is redone after every
  stage, so a scheduled stage could be handed an analysis that no longer mentions its
  issue and return "No changes needed" without trying.
"""

from dataclasses import dataclass

from xe_forge.agents.react_agent import (
    SUCCESS_MESSAGE,
    _BestCandidate,
    _verify_mlir,
)
from xe_forge.models import DetectedIssue, IssueType, KernelAnalysis, OptimizationStage
from xe_forge.pipeline import XeForgePipeline

MLIR_KERNEL = """\
func.func @main() {
  gpu.launch_func @k::@k blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
  call @printAllclose(%a, %b) : (memref<*xf32>, memref<*xf32>) -> ()
  return
}
"""
CANDIDATE = MLIR_KERNEL + "// tuned\n"


@dataclass
class FakeComparison:
    speedup: float
    original_time_ms: float = 1.0
    optimized_time_ms: float = 0.5
    original_tflops: float | None = 10.0
    optimized_tflops: float | None = 20.0
    optimized_correct: bool = True
    is_slower: bool = False
    lowered_identical: bool = False
    low_confidence: bool = False
    feedback_message: str = ""

    @property
    def original_time_us(self) -> float:
        return self.original_time_ms * 1000

    @property
    def optimized_time_us(self) -> float:
        return self.optimized_time_ms * 1000


def timed(speedup: float, **kw) -> FakeComparison:
    """A comparison whose times are consistent with *speedup* (1.0ms original)."""
    return FakeComparison(
        speedup=speedup, original_time_ms=1.0, optimized_time_ms=1.0 / speedup, **kw
    )


class FakeMlirExecutor:
    """Scripted comparisons, and a record of the repeat count used for each call."""

    speedup_tol = 0.03

    def __init__(self, comparisons, compare_repeats=1):
        self._comparisons = list(comparisons)
        self.compare_repeats = compare_repeats
        self.repeats_per_call: list[int] = []

    def compare_kernels(self, **kwargs):
        c = self._comparisons[min(len(self.repeats_per_call), len(self._comparisons) - 1)]
        self.repeats_per_call.append(self.compare_repeats)
        return c


def test_noise_band_gain_is_not_a_win():
    """0.98x passed `is_slower` but is not an improvement; 1.30x is."""
    best = _BestCandidate()
    executor = FakeMlirExecutor([timed(0.98)])
    verdict = _verify_mlir(CANDIDATE, MLIR_KERNEL, executor, best=best)
    assert "NO MEASURABLE GAIN" in verdict
    assert best.code is None, "a noise-band candidate must not be kept as the stage result"

    best = _BestCandidate()
    executor = FakeMlirExecutor([timed(1.30)])
    assert _verify_mlir(CANDIDATE, MLIR_KERNEL, executor, best=best) == SUCCESS_MESSAGE
    assert best.code == CANDIDATE


def test_marginal_gain_is_retimed_before_the_verdict():
    # Claimed 1.02x -> re-timed with median-of-3, which shows a real 1.20x: accept, and
    # record the *confirmed* number, not the first noisy one.
    executor = FakeMlirExecutor([timed(1.02), timed(1.20)])
    best = _BestCandidate()
    assert _verify_mlir(CANDIDATE, MLIR_KERNEL, executor, best=best) == SUCCESS_MESSAGE
    assert executor.repeats_per_call == [1, 3]
    assert best.speedup == 1.20

    # Same claim, but the re-timing says there was nothing there: reject.
    executor = FakeMlirExecutor([timed(1.02), timed(1.00)])
    best = _BestCandidate()
    assert "NO MEASURABLE GAIN" in _verify_mlir(CANDIDATE, MLIR_KERNEL, executor, best=best)
    assert executor.repeats_per_call == [1, 3]
    assert best.code is None


def test_decisive_results_are_not_retimed():
    """Re-timing costs a full compile+run pair, so only borderline cases pay for it."""
    for comparison in (timed(1.30), timed(0.50, is_slower=True)):
        executor = FakeMlirExecutor([comparison])
        _verify_mlir(CANDIDATE, MLIR_KERNEL, executor)
        assert executor.repeats_per_call == [1]

    # Already timing with 3 repeats: nothing to gain from re-running.
    executor = FakeMlirExecutor([timed(1.02)], compare_repeats=3)
    _verify_mlir(CANDIDATE, MLIR_KERNEL, executor)
    assert executor.repeats_per_call == [3]


def test_untrustworthy_timing_keeps_correctness_only_acceptance():
    """No rtclock timing, or a sub-noise-floor kernel: correctness alone still gates."""
    no_timing = FakeComparison(
        speedup=1.0, original_time_ms=float("inf"), optimized_time_ms=float("inf")
    )
    executor = FakeMlirExecutor([no_timing])
    assert _verify_mlir(CANDIDATE, MLIR_KERNEL, executor) == SUCCESS_MESSAGE
    assert executor.repeats_per_call == [1], "an untimed comparison must not be re-timed"

    executor = FakeMlirExecutor([timed(1.01, low_confidence=True)])
    assert _verify_mlir(CANDIDATE, MLIR_KERNEL, executor) == SUCCESS_MESSAGE
    assert executor.repeats_per_call == [1]


def _analysis(*issue_types: IssueType) -> KernelAnalysis:
    return KernelAnalysis(
        kernel_name="k",
        detected_issues=[
            DetectedIssue(
                issue_type=t, severity=3, description=f"{t.value} found", suggested_fix="fix it"
            )
            for t in issue_types
        ],
    )


def test_planned_stage_issues_survive_reanalysis():
    planned = _analysis(IssueType.DTYPE_PRECISION).detected_issues
    stage = OptimizationStage.DTYPE_FIX

    # Re-analysis dropped the dtype issue -> put the planned one back, so the stage runs.
    reanalyzed = _analysis(IssueType.TRANSPOSE_IN_LOOP)
    restored = XeForgePipeline._analysis_with_planned_issues(reanalyzed, planned, stage)
    kinds = [i.issue_type for i in restored.detected_issues]
    assert IssueType.DTYPE_PRECISION in kinds
    assert IssueType.TRANSPOSE_IN_LOOP in kinds, "re-analysis findings are kept as well"
    assert reanalyzed.detected_issues == _analysis(IssueType.TRANSPOSE_IN_LOOP).detected_issues, (
        "the caller's analysis must not be mutated"
    )

    # Re-analysis still reports the stage's issue -> hand it through untouched, including
    # the fresher description the new analysis wrote.
    fresh = _analysis(IssueType.DTYPE_PRECISION)
    assert XeForgePipeline._analysis_with_planned_issues(fresh, planned, stage) is fresh

    # Nothing was planned for this stage -> nothing to restore.
    assert XeForgePipeline._analysis_with_planned_issues(fresh, [], stage) is fresh
