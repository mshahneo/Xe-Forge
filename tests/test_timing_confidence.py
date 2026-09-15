"""Tests for the one shared rule that decides whether a measured speedup is real.

The rule used to exist twice. ``react_agent`` derived its bar from the executor's
``speedup_tol`` and re-timed undecided candidates. ``optimizer_agent`` — the staged
path, which is what a normal run uses — hardcoded ``_MIN_IMPROVEMENT = 1.02`` and
never re-timed. That gap let a 0.983x prefetch bank itself as a 1.025x win on 4k
flash attention.

These tests pin the fixed rule and the two ways it could quietly come back:

* an accept bar below the executor's own tolerance, and
* a "re-time" that uses the same repeat count as the first measurement.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from xe_forge.agents.optimizer_agent import OptimizerAgent
from xe_forge.core.timing_confidence import (
    confident_gain,
    confirm_marginal,
    confirm_repeats,
    min_gain,
    needs_confirmation,
    speedup_tol,
    timing_trustworthy,
    undecided_floor,
)
from xe_forge.models import DSL

TOL = 0.03  # MlirExecutor's default speedup_tol


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


class FakeExecutor:
    """Scripted comparisons, and a record of the repeat count used for each call."""

    def __init__(self, comparisons=(), compare_repeats=1, speedup_tol=TOL, raises=False):
        self._comparisons = list(comparisons)
        self.compare_repeats = compare_repeats
        self.speedup_tol = speedup_tol
        self.raises = raises
        self.repeats_per_call: list[int] = []

    def compare_kernels(self, **kwargs):
        self.repeats_per_call.append(self.compare_repeats)
        if self.raises:
            raise RuntimeError("mlir-runner died")
        return self._comparisons[min(len(self.repeats_per_call) - 1, len(self._comparisons) - 1)]


# --- the thresholds -------------------------------------------------------------


def test_the_bar_is_the_executors_own_tolerance_not_1_02():
    """The incident in one assertion: 1.025x must not be an improvement at tol=0.03."""
    ex = FakeExecutor()
    assert min_gain(ex) == pytest.approx(1.03)
    assert 1.025 < min_gain(ex), "a 2.5% claim is below the tool's own resolution"
    assert 1.02 < min_gain(ex), "the old hardcoded 1.02 gate sat inside the noise band"


def test_thresholds_track_the_tolerance():
    ex = FakeExecutor(speedup_tol=0.10)
    assert speedup_tol(ex) == pytest.approx(0.10)
    assert undecided_floor(ex) == pytest.approx(0.90)
    assert min_gain(ex) == pytest.approx(1.10)
    assert confident_gain(ex) == pytest.approx(1.20)


def test_a_missing_or_zero_tolerance_falls_back_to_the_default():
    assert speedup_tol(SimpleNamespace()) == pytest.approx(TOL)
    assert speedup_tol(SimpleNamespace(speedup_tol=0.0)) == pytest.approx(TOL)


def test_a_backend_with_no_tolerance_keeps_the_bar_its_caller_names():
    """Only MlirExecutor publishes speedup_tol. Triton and SYCL must not silently
    inherit the XeGPU band — the staged optimizer passes default=0.02 for them."""
    triton_like = SimpleNamespace()
    assert min_gain(triton_like, default=0.02) == pytest.approx(1.02)
    assert min_gain(FakeExecutor(speedup_tol=TOL), default=0.02) == pytest.approx(1.03)


# --- which results get re-timed --------------------------------------------------


@pytest.mark.parametrize(
    ("spd", "expected"),
    [
        (0.50, False),  # clear regression: the executor already calls it slower
        (0.96, False),  # below 1 - tol
        (0.98, True),  # undecided: at low repeats this could be a real win
        (1.02, True),  # the claim from the incident
        (1.05, True),  # above the bar but still inside 2*tol
        (1.06, False),  # 1 + 2*tol: clear of the band
        (1.30, False),  # clear win
    ],
)
def test_only_the_undecided_band_is_re_timed(spd, expected):
    assert needs_confirmation(timed(spd), FakeExecutor()) is expected


def test_the_band_is_re_timed_from_below_the_bar_too():
    """0.98x is not an improvement, but at 1 repeat it is not measured either."""
    ex = FakeExecutor([timed(1.20)])  # what the re-time will report
    confirmed = confirm_marginal(timed(0.98), "opt", "orig", ex)
    assert confirmed.speedup == 1.20
    assert ex.repeats_per_call == [3]


# --- the escalation, which is where wiring this in could have been a no-op -------


def test_a_re_time_always_raises_the_repeat_count():
    """The bug guard. 3 is the staged path's default, so bailing at reps >= 3 (what the
    old ReAct helper did) would have meant never re-timing a staged candidate."""
    for reps in (1, 2, 3, 5, 8):
        assert confirm_repeats(FakeExecutor(compare_repeats=reps)) > reps


def test_the_re_time_count_is_odd_so_the_median_is_a_real_sample():
    assert confirm_repeats(FakeExecutor(compare_repeats=1)) == 3
    assert confirm_repeats(FakeExecutor(compare_repeats=3)) == 5
    assert confirm_repeats(FakeExecutor(compare_repeats=5)) == 9


def test_the_env_override_can_raise_the_count_but_not_lower_it(monkeypatch):
    ex = FakeExecutor(compare_repeats=3)
    monkeypatch.setenv("MLIR_CONFIRM_REPEATS", "11")
    assert confirm_repeats(ex) == 11
    monkeypatch.setenv("MLIR_CONFIRM_REPEATS", "2")
    assert confirm_repeats(ex) == 4, "an override at or below reps would be theatre"
    monkeypatch.setenv("MLIR_CONFIRM_REPEATS", "not-a-number")
    assert confirm_repeats(ex) == 5, "a bad value falls back to the derived count"


def test_the_repeat_count_is_restored_after_a_re_time():
    ex = FakeExecutor([timed(1.20)], compare_repeats=3)
    confirm_marginal(timed(1.02), "opt", "orig", ex)
    assert ex.repeats_per_call == [5]
    assert ex.compare_repeats == 3, "the executor must be left as it was found"


# --- what happens when the re-time cannot help ----------------------------------


def test_untrustworthy_timing_is_never_re_timed():
    ex = FakeExecutor([timed(1.20)])
    untimed = FakeComparison(
        speedup=1.0, original_time_ms=float("inf"), optimized_time_ms=float("inf")
    )
    assert timing_trustworthy(untimed, ex) is False
    assert confirm_marginal(untimed, "opt", "orig", ex) is untimed

    below_floor = timed(1.02, low_confidence=True)
    assert timing_trustworthy(below_floor, ex) is False
    assert confirm_marginal(below_floor, "opt", "orig", ex) is below_floor
    assert ex.repeats_per_call == [], "neither case should cost a GPU run"


def test_a_failed_re_time_keeps_the_first_measurement():
    """Never let a crashed re-time upgrade a candidate. The caller still applies
    min_gain to what comes back, so keeping the noisy 1.02x still rejects it."""
    ex = FakeExecutor(compare_repeats=3, raises=True)
    first = timed(1.02)
    assert confirm_marginal(first, "opt", "orig", ex) is first
    assert ex.compare_repeats == 3
    assert first.speedup < min_gain(ex)


def test_no_executor_is_a_no_op():
    first = timed(1.02)
    assert confirm_marginal(first, "opt", "orig", None) is first


# --- the staged path, which is the path the bug shipped in ----------------------


MLIR_KERNEL = "func.func @main() {\n  gpu.launch_func @k::@k\n  return\n}\n"


def _staged_verify(executor, **kw):
    """Call the staged optimizer's verify with a stub self: it only needs dsl+executor."""
    stub = SimpleNamespace(dsl=DSL.MLIR, executor=executor)
    return OptimizerAgent._final_verify(
        stub,
        MLIR_KERNEL,  # orig
        MLIR_KERNEL + "// tuned\n",  # opt
        "k",  # kernel name
        None,  # shapes
        None,  # flop
        None,  # dtype
        **kw,
    )


def test_the_staged_path_re_times_an_undecided_gain():
    """This is the regression the whole change is for: the staged optimizer accepted a
    stage result on ONE measurement. It must now confirm before it reports a number."""
    ex = FakeExecutor([timed(1.025), timed(0.983, is_slower=False)], compare_repeats=3)
    ok, spd, _mb, _ma, err = _staged_verify(ex)
    assert ex.repeats_per_call == [3, 5], "the first call, then a re-time with more"
    assert ok and err is None
    assert spd == pytest.approx(0.983), "the reported speedup is the confirmed one"
    assert spd < min_gain(ex), "so the caller's gate rejects it"


def test_the_staged_path_does_not_re_time_a_clear_win():
    ex = FakeExecutor([timed(1.30)], compare_repeats=3)
    ok, spd, _mb, _ma, err = _staged_verify(ex)
    assert ex.repeats_per_call == [3]
    assert ok and err is None and spd == pytest.approx(1.30)


def test_the_staged_path_skips_the_re_time_when_speed_does_not_matter():
    """Baseline/reference runs pass skip_speedup_check — nothing is gated, so a re-time
    would be pure cost."""
    ex = FakeExecutor([timed(1.02)], compare_repeats=3)
    _staged_verify(ex, skip_speedup_check=True)
    assert ex.repeats_per_call == [3]


def test_the_staged_path_re_times_after_the_correctness_and_no_op_checks():
    """A wrong or no-op kernel is rejected without paying for a re-time."""
    ex = FakeExecutor([timed(1.02, optimized_correct=False)], compare_repeats=3)
    ok, _spd, _mb, _ma, err = _staged_verify(ex)
    assert (ok, err) == (False, "Incorrect results")
    assert ex.repeats_per_call == [3]

    ex = FakeExecutor([timed(1.02, lowered_identical=True)], compare_repeats=3)
    ok, _spd, _mb, _ma, err = _staged_verify(ex)
    assert (ok, err) == (False, "no-op (lowers to identical IR)")
    assert ex.repeats_per_call == [3]


def test_both_strategies_share_one_rule():
    """Neither agent may hold a private copy of the rule. react_agent used to define
    all three of these itself, which is the whole reason the staged path diverged."""
    from xe_forge.agents import optimizer_agent, react_agent

    assert react_agent.confirm_marginal is confirm_marginal
    assert react_agent.min_gain is min_gain
    assert react_agent.timing_trustworthy is timing_trustworthy
    assert optimizer_agent.confirm_marginal is confirm_marginal
    assert optimizer_agent.min_gain is min_gain
    assert not hasattr(optimizer_agent, "_MIN_IMPROVEMENT"), "the hardcoded 1.02 is gone"
