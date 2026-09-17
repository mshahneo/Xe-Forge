"""Tests that a stage spends its whole iteration budget.

The bug: the best-of loop in `optimize_stage` began with an unconditional

    if not ok:
        logger.debug(f"Run failed ({err}), stopping best-of loop")
        break

`_final_verify` returns ``ok=False`` for every MEASURED miss too — a slower
kernel, wrong results, a no-op edit. So the first slower candidate ended the
stage with the rest of the budget unspent, and the reason was logged at DEBUG,
so the log gave no hint it had happened.

It was measured. A layer-norm 4096x4096 run (2026-09-17, `d34db4d`) stopped its
`algorithmic` stage after ONE attempt at 0.959x and reported only "no valid
result in budget". The previous run of the same input spent all 5 attempts and
reached the whole-kernel rewrite on attempt 4.

The break also contradicted the no-improvement branch 70 lines below it, which
already says a measured miss is not a reason to quit. The break sat above it and
short-circuited it, so that rule only ever applied to a candidate measuring
>= 1.0x.

One failure still stops the stage: an executor exception ("Verify failed: ...").
A broken harness will not fix itself, and every further attempt pays GPU time to
learn the same thing.
"""

from dataclasses import dataclass
from typing import ClassVar

from xe_forge.agents import optimizer_agent as oa
from xe_forge.models import DetectedIssue, IssueType, KernelAnalysis, OptimizationStage

MLIR_KERNEL = """\
func.func @main() {
  gpu.launch_func @k::@k blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
  return
}
"""

STAGE = OptimizationStage.DTYPE_FIX
ISSUE = IssueType.DTYPE_PRECISION
BUDGET = 5


@dataclass
class FakeComparison:
    """Times consistent with *speedup*, original fixed at 1.0 ms."""

    speedup: float
    optimized_correct: bool = True
    is_slower: bool = False
    lowered_identical: bool = False
    low_confidence: bool = False
    original_tflops: float = 10.0
    optimized_tflops: float = 10.0
    feedback_message: str = ""

    @property
    def original_time_us(self) -> float:
        return 1000.0

    @property
    def optimized_time_us(self) -> float:
        return 1000.0 / self.speedup


class SlowerExecutor:
    """Every candidate measures slower. Counts the compares it was asked for."""

    speedup_tol = 0.03
    compare_repeats = 1

    def __init__(self):
        self.compares = 0

    def compare_kernels(self, **kwargs):
        self.compares += 1
        return FakeComparison(speedup=0.80, is_slower=True)


class BrokenExecutor:
    """The harness itself is broken — `_final_verify` turns this into "Verify failed"."""

    speedup_tol = 0.03
    compare_repeats = 1

    def __init__(self):
        self.compares = 0

    def compare_kernels(self, **kwargs):
        self.compares += 1
        raise RuntimeError("level-zero device lost")


class FakeCoVeR:
    """Stands in for the LLM. Returns one distinct candidate per call."""

    calls = 0

    def __init__(self, **kwargs):
        FakeCoVeR.calls = 0
        self.max_iters = kwargs.get("max_iters", BUDGET)

    def __call__(self, **kwargs):
        FakeCoVeR.calls += 1

        class _Result:
            # One thought per run, so `iters_used` advances by exactly 1 —
            # the same accounting the real CoVeR produces for a single
            # thought + verify pair.
            trajectory: ClassVar[dict] = {"thought_0": "try something"}
            optimized_code = MLIR_KERNEL + f"// candidate {FakeCoVeR.calls}\n"

        return _Result()


def _agent(executor):
    agent = oa.OptimizerAgent.__new__(oa.OptimizerAgent)  # no LLM, no dspy setup
    agent.executor = executor
    agent.validator = None
    agent.max_iterations = BUDGET
    agent.knowledge_base = None
    agent.dsl = oa.DSL.MLIR
    agent.attempts_dir = None
    return agent


def _analysis():
    return KernelAnalysis(
        kernel_name="k",
        detected_issues=[
            DetectedIssue(
                issue_type=ISSUE,
                severity=4,
                description="f32 where f16 would do",
                suggested_fix="retype",
            )
        ],
    )


def _run(monkeypatch, executor):
    monkeypatch.setattr(oa, "CoVeR", FakeCoVeR)
    agent = _agent(executor)
    result = agent.optimize_stage(
        code=MLIR_KERNEL,
        stage=STAGE,
        analysis=_analysis(),
        xpu_config={},
        kernel_name="k",
    )
    return result


def test_a_slower_candidate_does_not_end_the_stage(monkeypatch):
    executor = SlowerExecutor()
    result = _run(monkeypatch, executor)

    assert FakeCoVeR.calls == BUDGET, (
        f"stage stopped after {FakeCoVeR.calls} of {BUDGET} attempts; a slower "
        "candidate must not end the stage with budget left"
    )
    assert executor.compares == BUDGET, "every attempt must be measured"
    assert not result.success, "nothing beat the baseline, so the stage still fails"
    assert result.output_code == MLIR_KERNEL, "a slower candidate must not be returned"


def test_a_harness_error_does_end_the_stage(monkeypatch):
    executor = BrokenExecutor()
    result = _run(monkeypatch, executor)

    assert FakeCoVeR.calls == 1, (
        f"stage made {FakeCoVeR.calls} attempts against a broken harness; each one "
        "pays GPU time to learn the same thing"
    )
    assert executor.compares == 1
    assert not result.success
