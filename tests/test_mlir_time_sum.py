"""The module time is the sum of every kernel launch's profiler median.

The IMEX profiler prints one Min/Max/Avg/Median/Std Dev block per gpu.launch_func.
`_extract_time_ms` used to read only the first, so a 5-kernel chain was timed as
its first kernel and a candidate that fused the chain was compared against that.
"""
from xe_forge.core.mlir_executor import _extract_time_ms

BLOCK = "Cache Size: 18874368\nMin: {m}\nMax: {m}\nAvg: {m}\nMedian: {m}\nStd Dev: 0.001\n"


def test_one_launch_is_unchanged():
    assert _extract_time_ms(BLOCK.format(m="1.5")) == 1.5


def test_a_chain_of_launches_is_summed():
    text = "".join(BLOCK.format(m=v) for v in ("4.59", "1.15", "4.60", "2.72", "1.38"))
    assert abs(_extract_time_ms(text) - 14.44) < 1e-9


def test_the_sum_is_taken_from_stderr_too():
    # The runtime prints the profiler block on stderr; the parser reads it there.
    text = "noise line\n" + BLOCK.format(m="2.0") + BLOCK.format(m="3.0")
    assert _extract_time_ms(text, allow_lone_float=False) == 5.0
