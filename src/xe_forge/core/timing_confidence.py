"""One place that decides whether a measured speedup is real.

Why this module exists: the accept decision used to live in two places with two
different rules.

``react_agent`` derived its threshold from the executor's own ``speedup_tol`` and
re-timed candidates that landed inside the noise band. ``optimizer_agent``
hardcoded ``_MIN_IMPROVEMENT = 1.02``, consulted no tolerance, and never re-timed
anything -- it accepted a stage result on one ``compare_kernels`` call.

That gap shipped a regression into a kernel. A ``memory_access`` retry on 4k flash
attention reported 1.025x, cleared the 1.02 gate by 0.005, and banked its kernel.
Re-timed at 500 runs x 3 interleaved reps (spread < 0.15%) the same edit measures
0.983x -- a 1.7% regression recorded as a 2.5% win. What it banked was a
prefetch whose offset equalled the load's offset; deleting it was worth 1.017x,
which made the variant the stage threw away the best kernel of the run.

The threshold was also self-contradictory. ``MlirExecutor`` only flags
``is_slower`` below ``1 - speedup_tol`` because, in its own words, "run-to-run
noise of a few percent is expected". With the default tolerance of 0.03 the tool
says +/-3% is indistinguishable from no change, and the optimizer then accepted
wins at 2% -- below its own measurement resolution. No number of repeats fixes a
threshold set inside the noise band.

The rule here, all derived from ``executor.speedup_tol`` so there is one number to
tune and no magic constants:

    spd < 1 - tol                reject. The executor already calls this slower.
    1 - tol <= spd < 1 + 2*tol   UNDECIDED. Re-time with more repeats, then hold the
                                 confirmed number to ``min_gain`` (= 1 + tol).
    spd >= 1 + 2*tol             accept. Clear of the band, so more samples will
                                 not change the verdict -- do not pay for them.

Two things follow, and both are the point:

* The accept bar is 1 + tol, never 1.02. A 2% claim at a 3% tolerance is below the
  tool's own resolution, so it can never be an accept.
* The bar is applied to a *confirmed* number. Re-timing runs across the whole
  undecided band, not just above the bar, because at low repeats the first
  measurement cannot tell 0.98x from 1.20x -- ``MlirExecutor`` records that the same
  code measured 1.53x in one process and 0.73x in the next. Rejecting the lower half
  of the band unmeasured would throw away real wins.

Outside the band nothing is re-timed. Extra samples cannot move a clear win or a
clear regression across the bar, so they are not worth the GPU time.

Measured on the real pair that caused the incident (FA_opt_full_noprefetch.mlir vs
FA_opt_full.mlir, BMG B580, large GRF, truth 0.983x): one process pair reported
1.0385x, and median-of-3 reported 1.0245x. Note what that says. Re-timing narrows the
error, it does not remove it -- 1.0245x still has the sign wrong. What rejects this
kernel is the 1.03x bar, not the extra samples. The re-time only matters for
candidates the bar would otherwise decide on a single noisy number.
"""

from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger(__name__)

_DEFAULT_TOL = 0.03


def speedup_tol(executor, default: float = _DEFAULT_TOL) -> float:
    """The executor's own noise band.

    Only ``MlirExecutor`` publishes one. Pass *default* to say what an executor that
    does not should be held to, rather than inheriting a number measured on a
    different backend.
    """
    return getattr(executor, "speedup_tol", default) or default


def min_gain(executor, default: float = _DEFAULT_TOL) -> float:
    """Smallest speedup that counts as an improvement rather than noise."""
    return 1.0 + speedup_tol(executor, default)


def confident_gain(executor) -> float:
    """Speedup above which the verdict is safe without extra repeats."""
    return 1.0 + 2 * speedup_tol(executor)


def undecided_floor(executor) -> float:
    """Speedup below which the kernel is a clear regression, so no re-timing is owed."""
    return 1.0 - speedup_tol(executor)


def timing_trustworthy(comparison, executor) -> bool:
    """True when this comparison produced a number worth holding to a threshold."""
    if getattr(comparison, "low_confidence", False):
        return False  # below the noise floor: the executor already says don't trust it
    o = getattr(comparison, "original_time_ms", None)
    p = getattr(comparison, "optimized_time_ms", None)
    return o is not None and math.isfinite(o) and p is not None and math.isfinite(p) and p > 0


def confirm_repeats(executor) -> int:
    """How many independent processes a re-time should use.

    Must EXCEED whatever the first measurement used, or the re-time is theatre. The
    old ReAct helper re-timed with a fixed 3 and bailed out at ``reps >= 3``. The
    staged path's default ``compare_repeats`` IS 3, so wiring that helper in unchanged
    would have been a no-op exactly where the bug lived. ``2 * reps - 1`` keeps the
    count odd, so the median is a real sample: 1 -> 3, 3 -> 5, 5 -> 9.
    Env ``MLIR_CONFIRM_REPEATS`` overrides, but never down to or below ``reps``.
    """
    reps = max(1, int(getattr(executor, "compare_repeats", 1) or 1))
    try:
        override = int(os.environ.get("MLIR_CONFIRM_REPEATS", "0"))
    except ValueError:
        override = 0
    if override > 0:
        return max(override, reps + 1)
    return max(3, 2 * reps - 1)


def needs_confirmation(comparison, executor) -> bool:
    """True for a result the current repeat count cannot decide."""
    if not timing_trustworthy(comparison, executor):
        return False
    spd = getattr(comparison, "speedup", None)
    if spd is None:
        return False
    return undecided_floor(executor) <= spd < confident_gain(executor)


def confirm_marginal(comparison, code, original_code, executor, flop=None):
    """Re-time a candidate whose result the current repeat count cannot decide.

    Returns the confirmed comparison, or the original one when re-timing does not
    apply or fails. Callers then apply :func:`min_gain` to whichever they get back,
    so a failed re-time never silently upgrades a candidate.
    """
    if executor is None or not needs_confirmation(comparison, executor):
        return comparison
    reps = max(1, int(getattr(executor, "compare_repeats", 1) or 1))
    target = confirm_repeats(executor)
    if target <= reps:
        return comparison
    logger.info(
        "Speedup %.3fx is inside the undecided band [%.3f, %.3f) — re-timing with "
        "median-of-%d (was %d)",
        comparison.speedup,
        undecided_floor(executor),
        confident_gain(executor),
        target,
        reps,
    )
    try:
        executor.compare_repeats = target
        confirmed = executor.compare_kernels(
            original_code=original_code, optimized_code=code, flop=flop
        )
    except Exception as e:  # keep the first measurement on any failure
        logger.warning("Re-timing failed (%s); keeping the first measurement", e)
        return comparison
    finally:
        executor.compare_repeats = reps
    logger.info("Re-timed: %.3fx -> %.3fx", comparison.speedup, confirmed.speedup)
    return confirmed
