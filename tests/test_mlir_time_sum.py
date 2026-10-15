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


def test_profiling_counts_default_to_steady_state(monkeypatch):
    # 10 + 10 read boost clocks (11.5% fast on an 8 ms module); 100 + 100 is within
    # 0.3% of 500 + 500.
    from xe_forge.core.mlir_executor import MlirExecutor

    monkeypatch.delenv("MLIR_PROFILING_WARMUPS", raising=False)
    monkeypatch.delenv("MLIR_PROFILING_RUNS", raising=False)
    ex = MlirExecutor()
    assert (ex.imex_warmups, ex.imex_runs) == (100, 100)


def test_profiling_counts_follow_the_environment(monkeypatch):
    from xe_forge.core.mlir_executor import MlirExecutor

    monkeypatch.setenv("MLIR_PROFILING_WARMUPS", "500")
    monkeypatch.setenv("MLIR_PROFILING_RUNS", "300")
    ex = MlirExecutor()
    assert (ex.imex_warmups, ex.imex_runs) == (500, 300)
    assert (MlirExecutor(imex_warmups=7, imex_runs=8).imex_warmups) == 7
