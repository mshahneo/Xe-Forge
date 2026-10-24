"""A precision fix that keeps the speed must still be accepted in dtype_fix.

Why it exists: on KernelBench #36 (channel RMSNorm, bf16) the dtype_fix stage
produced two verified rewrites that moved the sum of squares to f32. They measured
1.004x and 1.005x, which is inside the noise band, so the speed-only rule rejected
both and the kernel shipped with bf16 accumulation (3.4x over a one-rounding bound).
In a memory-bound kernel f32 math is free (0.3151 vs 0.3154 ms measured), so a
precision fix will almost never clear a speed bar.
"""

from types import SimpleNamespace

from xe_forge.core.mlir_executor import MlirExecutor
from xe_forge.core.timing_confidence import ACCURACY_GAIN, accuracy_gain_at_parity, undecided_floor

# vector.print puts the label and the value on separate lines.
HARNESS_STDOUT = "MAX_ABS_ERROR:\n0.0310318\nMAX_TOL_RATIO:\n0.318261\n[ALLCLOSE: TRUE]\n"


def _executor(tol=0.03):
    return SimpleNamespace(speedup_tol=tol)


def _cmp(orig, opt):
    return SimpleNamespace(original_error_ratio=orig, optimized_error_ratio=opt)


def test_the_executor_reads_the_error_ratio():
    ex = MlirExecutor.__new__(MlirExecutor)  # no build dirs needed to parse output
    r = ex._parse_output(HARNESS_STDOUT)
    assert r.output_correct is True
    assert r.error_ratio == 0.318261


def test_no_ratio_line_means_unknown_not_zero():
    ex = MlirExecutor.__new__(MlirExecutor)
    assert ex._parse_output("[ALLCLOSE: TRUE]\n").error_ratio is None


def test_the_kernelbench_case_is_accepted():
    # bf16 sum (ratio 0.33) -> f32 sum (0.078) at 1.004x.
    assert accuracy_gain_at_parity(_cmp(0.33, 0.078), 1.004, _executor())


def test_parity_inside_the_noise_band_is_enough():
    ex = _executor()
    assert accuracy_gain_at_parity(_cmp(0.33, 0.078), undecided_floor(ex), ex)


def test_a_clear_slowdown_is_not_bought_with_accuracy():
    ex = _executor()
    assert not accuracy_gain_at_parity(_cmp(0.33, 0.078), undecided_floor(ex) - 0.01, ex)


def test_error_noise_is_not_a_gain():
    # The input is random per run, so the ratio moves a little on its own.
    assert not accuracy_gain_at_parity(_cmp(0.33, 0.33 * (ACCURACY_GAIN + 0.05)), 1.0, _executor())


def test_less_accurate_is_never_accepted_here():
    assert not accuracy_gain_at_parity(_cmp(0.08, 0.33), 1.0, _executor())


def test_missing_or_nan_ratios_fall_back_to_the_speed_rule():
    ex = _executor()
    assert not accuracy_gain_at_parity(_cmp(None, 0.08), 1.0, ex)
    assert not accuracy_gain_at_parity(_cmp(0.33, None), 1.0, ex)
    assert not accuracy_gain_at_parity(_cmp(0.33, float("nan")), 1.0, ex)
    assert not accuracy_gain_at_parity(None, 1.0, ex)
    assert not accuracy_gain_at_parity(_cmp(0.33, 0.08), None, ex)


def test_a_later_candidate_must_beat_the_current_best():
    ex = _executor()
    # Stage best already has ratio 0.08 at 1.004x: a candidate at 0.078 is not a gain.
    assert not accuracy_gain_at_parity(_cmp(0.33, 0.078), 1.0, ex, best_spd=1.004, best_ratio=0.08)
    # ...but one that halves it again, at the same speed, is.
    assert accuracy_gain_at_parity(_cmp(0.33, 0.03), 1.0, ex, best_spd=1.004, best_ratio=0.08)
    # ...and it must hold the best's speed, not the stage input's.
    assert not accuracy_gain_at_parity(_cmp(0.33, 0.03), 0.95, ex, best_spd=1.004, best_ratio=0.08)
