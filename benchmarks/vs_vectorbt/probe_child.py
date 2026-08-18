"""One measurement, one fresh process, one JSON line on stdout.

Two things cannot be measured honestly inside the main harness process:

*Cold start* - the wait between typing "run" and seeing a result. vectorbt
compiles its numba kernels on the first call, which is a real cost a user pays
in every new notebook or script, and which the steady-state benchmark
deliberately discards. Measuring it requires a process that has never imported
either engine.

*Memory* - peak resident memory attributable to the run. Once one engine has
run in a process, the allocator has already grown and the other engine's
measurement is meaningless.

The ``baseline`` mode measures the same process doing everything except calling
an engine (interpreter start, numpy and pandas import, data generation) so the
engine's own share can be read off rather than argued about.

    python probe_child.py coldstart mbt sma_cross 20000
    python probe_child.py memory    vbt sma_cross 5000000
    python probe_child.py baseline  none sma_cross 20000
"""
from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import threading
import time

START = time.perf_counter()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def _build(engine: str, workload: str, bars: int):
    import data as data_mod

    frame = data_mod.make_ohlcv(bars)
    if engine == "mbt":
        import engine_mbt

        return engine_mbt.prepare(workload, frame, tempfile.mkdtemp(prefix="mbt_probe_"))
    if engine == "vbt":
        import engine_vbt

        return engine_vbt.prepare(workload, frame, None)
    return lambda: {}


def cold_start(engine: str, workload: str, bars: int) -> dict:
    """Wall time from process start to a finished backtest, engine import included."""
    run = _build(engine, workload, bars)
    run()
    return {"seconds": time.perf_counter() - START}


def memory(engine: str, workload: str, bars: int) -> dict:
    """Resident memory the run itself adds, sampled while it runs.

    A warmup call first, so what is measured is the steady-state cost of running
    a backtest rather than the one-off growth of a cold allocator.
    """
    run = _build(engine, workload, bars)
    run()
    gc.collect()
    time.sleep(0.3)

    peak = [_rss_mb()]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], _rss_mb())
            time.sleep(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    before = _rss_mb()
    started = time.perf_counter()
    run()
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join()

    delta = peak[0] - before
    return {
        "before_mb": before,
        "peak_mb": peak[0],
        "added_mb": delta,
        "added_mb_per_million_bars": delta / (bars / 1e6),
        "seconds": elapsed,
    }


def baseline(engine: str, workload: str, bars: int) -> dict:
    """Everything except the engine: interpreter, numpy, pandas, data generation."""
    _build("none", workload, bars)
    return {"seconds": time.perf_counter() - START}


MODES = {"coldstart": cold_start, "memory": memory, "baseline": baseline}


def main() -> int:
    mode, engine, workload, bars = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    payload = MODES[mode](engine, workload, bars)
    payload.update({"mode": mode, "engine": engine, "workload": workload, "bars": bars})
    # A marker prefix: the engines print a banner on import, and the parent must
    # not have to guess which line is the result.
    sys.stdout.write("\nPROBE_RESULT " + json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
