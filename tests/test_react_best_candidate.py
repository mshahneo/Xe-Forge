"""Tests for keeping GPU-verified intermediate candidates in ReAct stages.

A ReAct stage calls its verify tool once per iteration, and every call compiles,
runs and times a candidate on the GPU. Previously only the agent's FINAL answer
was kept, so a stage that proved a faster kernel at iteration k and then answered
with something worse (or nothing) fell back to its untouched input, discarding a
result that had already been measured on hardware.
"""

from dataclasses import dataclass

from xe_forge.agents.react_agent import (
    SUCCESS_MESSAGE,
    OptimizerReActAgent,
    _BestCandidate,
    _metrics_from_comparison,
    _verify_mlir,
)
from xe_forge.models import DSL, OptimizationStage

MLIR_KERNEL = """\
func.func @main() {
  gpu.launch_func @k::@k blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
  call @printAllclose(%a, %b) : (memref<*xf32>, memref<*xf32>) -> ()
  return
}
"""


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
    feedback_message: str = ""

    @property
    def original_time_us(self) -> float:
        return self.original_time_ms * 1000

    @property
    def optimized_time_us(self) -> float:
        return self.optimized_time_ms * 1000


class FakeMlirExecutor:
    """Returns a scripted comparison per call, in order."""

    def __init__(self, comparisons):
        self._comparisons = list(comparisons)
        self.calls = 0

    def compare_kernels(self, **kwargs):
        c = self._comparisons[min(self.calls, len(self._comparisons) - 1)]
        self.calls += 1
        return c


class FakeCode:
    """Stands in for dspy.Code, which the verify tool unwraps via .code."""

    def __init__(self, code):
        self.code = code


def test_offer_keeps_the_fastest_and_ignores_regressions():
    best = _BestCandidate()
    assert best.code is None

    best.offer("slower", 0.8)
    assert best.code is None, "a candidate slower than the input must not be kept"

    best.offer("ok", 1.2)
    assert (best.code, best.speedup) == ("ok", 1.2)

    best.offer("worse", 1.1)
    assert best.code == "ok", "a later, slower candidate must not displace the best"

    best.offer("better", 1.5)
    assert (best.code, best.speedup) == ("better", 1.5)

    best.offer("nospeedup", None)
    assert best.code == "better"


def test_verify_tool_records_verified_candidates():
    """The MLIR verify tool must feed every accepted candidate to `best`."""
    executor = FakeMlirExecutor([FakeComparison(speedup=1.30)])
    agent = OptimizerReActAgent(executor=executor, dsl=DSL.MLIR)
    best = _BestCandidate()
    verify = agent._create_verify_tool(
        original_code=MLIR_KERNEL,
        kernel_name=None,
        input_shapes=None,
        flop=None,
        best=best,
    )

    candidate = MLIR_KERNEL + "// tuned\n"
    assert verify(FakeCode(candidate)).startswith(SUCCESS_MESSAGE)
    assert best.code == candidate
    assert best.speedup == 1.30


def test_verify_tool_does_not_record_rejected_candidates():
    for comparison in (
        FakeComparison(speedup=0.5, is_slower=True),
        FakeComparison(speedup=1.4, optimized_correct=False),
        FakeComparison(speedup=1.4, lowered_identical=True),
    ):
        agent = OptimizerReActAgent(executor=FakeMlirExecutor([comparison]), dsl=DSL.MLIR)
        best = _BestCandidate()
        verify = agent._create_verify_tool(
            original_code=MLIR_KERNEL,
            kernel_name=None,
            input_shapes=None,
            flop=None,
            best=best,
        )
        assert not verify(FakeCode(MLIR_KERNEL + "// x\n")).startswith(SUCCESS_MESSAGE)
        assert best.code is None


def test_keep_best_or_fail_prefers_a_verified_candidate():
    agent = OptimizerReActAgent(executor=None, dsl=DSL.MLIR)
    stage = OptimizationStage.MEMORY_ACCESS

    empty = _BestCandidate()
    failed = agent._keep_best_or_fail(stage, MLIR_KERNEL, empty, "agent returned nothing")
    assert not failed.success
    assert failed.output_code == MLIR_KERNEL, "with nothing verified, fall back to the input"

    best = _BestCandidate()
    best.offer("tuned kernel", 1.25, FakeComparison(speedup=1.25))
    kept = agent._keep_best_or_fail(stage, MLIR_KERNEL, best, "agent returned nothing")
    assert kept.success
    assert kept.output_code == "tuned kernel"
    assert kept.speedup == 1.25
    assert kept.metrics_after == {"time_us": 500.0, "tflops": 20.0}
    assert "GPU-verified" in kept.changes_made[0]


def test_success_verdict_reports_the_best_and_the_remaining_budget():
    """A bare "Success!" is the agent's cue to finish, so say what is banked and left."""
    executor = FakeMlirExecutor([FakeComparison(speedup=1.20, optimized_time_ms=0.83)])
    agent = OptimizerReActAgent(executor=executor, dsl=DSL.MLIR)
    best = _BestCandidate(max_attempts=5)
    verify = agent._create_verify_tool(
        original_code=MLIR_KERNEL,
        kernel_name=None,
        input_shapes=None,
        flop=None,
        best=best,
    )

    verdict = verify(FakeCode(MLIR_KERNEL + "// tuned\n"))
    assert verdict.startswith(SUCCESS_MESSAGE)
    assert "1.200x" in verdict and "0.8300ms" in verdict
    assert "4 attempt(s) left" in verdict
    assert "cannot lose it" in verdict, "continuing must read as free, not risky"
    assert best.attempts == 1

    # Out of budget: stop asking for more and ask for the answer.
    best.attempts = 5
    assert "No attempts left" in best.keep_pushing_note()


def test_attempts_are_counted_even_when_the_candidate_is_rejected():
    """The budget is spent by reaching hardware, not by succeeding."""
    executor = FakeMlirExecutor([FakeComparison(speedup=0.5, is_slower=True)])
    best = _BestCandidate(max_attempts=3)
    _verify_mlir(MLIR_KERNEL + "// slow\n", MLIR_KERNEL, executor, best=best)
    assert (best.attempts, best.attempts_left()) == (1, 2)

    unbounded = _BestCandidate()
    unbounded.note_attempt()
    assert unbounded.attempts_left() is None, "no declared budget -> nothing to report"


def test_metrics_from_comparison_omits_incomplete_sides():
    before, after = _metrics_from_comparison(None)
    assert (before, after) == (None, None)

    # No FLOP count -> no TFLOPS -> that side is dropped rather than half-filled.
    before, after = _metrics_from_comparison(
        FakeComparison(speedup=1.1, original_tflops=None, optimized_tflops=None)
    )
    assert (before, after) == (None, None)
