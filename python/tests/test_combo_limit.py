"""Community sweep gating: cumulative combo budget + throughput penalty.

See docs/sweep-combo-limit-plan.md. Two mechanisms, tested here through real
sweep calls on a tiny dataset:

* the 500-combo cap is enforced on the running total **per process**, not per
  call — otherwise slicing a grid into small calls bypasses it at a measured
  +0.5% cost;
* every accepted Community call waits `SWEEP_MIN_INTERVAL`, held under a
  machine-wide file lock, so the remaining bypass (a fresh interpreter per
  slice) costs 5 s each and cannot be parallelised away.

The rate-gate tests run in subprocesses on purpose: "serialised across
processes" is only observable between processes, and an in-process test would
also depend on which test happened to run first.
"""
import subprocess
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt._native import _combo_budget

IS_PRO = bt.license_info()[0] == "Pro"
community_only = pytest.mark.skipif(
    IS_PRO, reason="Community-only; deactivate Pro/BT_UNLOCKED to test"
)
pro_only = pytest.mark.skipif(not IS_PRO, reason="requires an active Pro license")


@pytest.fixture(scope="module")
def store_paths(tmp_path_factory):
    """A minimal store on disk; returns (data_root, metadata_db, arrow_dir)."""
    root = tmp_path_factory.mktemp("combo_limit")
    idx = pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC")
    close = 100.0 + np.arange(120, dtype=float)
    df = pd.DataFrame({
        "timestamp": idx,
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(120, 1_000.0),
    })
    data_root, metadata_db = str(root / "data"), str(root / "metadata.sqlite")
    bt.import_dataframe(
        df, symbol="CL", symbol_id=1, interval="1m",
        data_root=data_root, metadata_db=metadata_db,
    )
    return data_root, metadata_db, f"{data_root}/mega"


@pytest.fixture(scope="module")
def daily_store(store_paths):
    data_root, metadata_db, arrow_dir = store_paths
    return bt.DataStore(data_root, metadata_db, "bars_1m", None, arrow_dir)


# --- shared snippet: build strategy + config, run one sweep of n combos ------
_HARNESS = '''
import sys, time
import manifoldbt as bt

store = bt.DataStore({data_root!r}, {metadata_db!r}, "bars_1m", None, {arrow_dir!r})
strat = bt.Strategy(
    name="budget_probe",
    signals={{"signal": bt.lit(1.0)}},
    position_sizing=bt.lit(1.0) * bt.param("size", default=1.0),
    parameters={{"size": bt.param("size", default=1.0)}},
)
t0, t1 = bt.time_range("2024-01-01", "2024-01-02")
cfg = bt.BacktestConfig(universe=[1], time_range_start=t0, time_range_end=t1,
                        bar_interval={{"Minutes": 1}})

def sweep(n):
    grid = {{"size": [1.0 + 0.001 * i for i in range(n)]}}
    return bt.run_sweep_lite(strat, grid, cfg, store)

def timed(n):
    # PermissionError comes from the native gate (cumulative cap), LicenseError
    # from the Python mirror (a single call larger than the cap). Both are
    # "refused", and neither should have waited out the rate limit.
    t = time.perf_counter()
    try:
        sweep(n)
        ok = True
    except (PermissionError, bt.LicenseError):
        ok = False
    return time.perf_counter() - t, ok
'''


def _run(store_paths, body):
    """Run `body` in a fresh interpreter; return its stdout floats/flags."""
    data_root, metadata_db, arrow_dir = store_paths
    code = _HARNESS.format(
        data_root=data_root, metadata_db=metadata_db, arrow_dir=arrow_dir
    ) + textwrap.dedent(body)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout.strip().splitlines()[-1].split()


def _sweep(store, n_combos):
    """One in-process run_sweep_lite call with exactly n_combos combinations."""
    strat = bt.Strategy(
        name="budget_probe",
        signals={"signal": bt.lit(1.0)},
        position_sizing=bt.lit(1.0) * bt.param("size", default=1.0),
        parameters={"size": bt.param("size", default=1.0)},
    )
    # Minutes(1) is silently coarsened to daily on Community; the runs then
    # produce zero trades, which is irrelevant here — only the gate matters.
    t0, t1 = bt.time_range("2024-01-01", "2024-01-02")
    cfg = bt.BacktestConfig(
        universe=[1], time_range_start=t0, time_range_end=t1,
        bar_interval={"Minutes": 1},
    )
    grid = {"size": [1.0 + 0.001 * i for i in range(n_combos)]}
    return bt.run_sweep_lite(strat, grid, cfg, store)


# --------------------------------------------------------------- the budget --

@community_only
def test_cumulative_budget(daily_store):
    used0, limit, is_pro = _combo_budget()
    assert not is_pro
    remaining = limit - used0
    if remaining < 8:
        pytest.skip(f"only {remaining} combos left in this process")

    # Two calls of just over half the remaining budget: the first fits,
    # the second would cross the cap even though it is individually small.
    n = int(remaining) // 2 + 1
    _sweep(daily_store, n)
    with pytest.raises(PermissionError, match="already used this session"):
        _sweep(daily_store, n)

    # The rejected call consumed nothing: what actually remains still fits.
    leftover = int(limit - _combo_budget()[0])
    assert leftover == int(remaining) - n
    if leftover >= 1:
        _sweep(daily_store, leftover)
    assert _combo_budget()[0] == limit

    # Budget now exhausted: even a single combo is refused.
    with pytest.raises(PermissionError, match="0 remaining"):
        _sweep(daily_store, 1)


# ------------------------------------------------------------ the rate gate --
#
# SWEEP_MIN_INTERVAL is 5 s. Thresholds leave generous slack: a call that
# waited is asserted above 4 s, one that did not below 2 s. Nothing here
# depends on machine speed — that is the point of a wall-clock gate.
_INTERVAL = 5.0
_WAITED = 4.0
_DID_NOT_WAIT = 2.0


@community_only
def test_rate_gate_applies_to_every_call(store_paths):
    """Each accepted call waits the interval — it is a rate limit, not a toll."""
    first, second = (
        float(x) for x in _run(store_paths, """
            t1, _ = timed(2)
            t2, _ = timed(2)
            print(t1, t2)
        """)
    )
    assert first > _WAITED, f"first call took {first:.2f}s, expected a ~5 s wait"
    assert second > _WAITED, (
        f"second call took {second:.2f}s — the gate is behaving like a one-off "
        f"charge instead of a rate limit"
    )


@community_only
def test_rate_gate_serialises_across_processes(store_paths):
    """The lock is the mechanism: concurrent waits must queue, not overlap.

    Without the file lock two processes would sleep through the same 5 s and
    both proceed — the failure mode of every sleep-based limiter. With it, two
    concurrent sweeps cost two intervals.
    """
    data_root, metadata_db, arrow_dir = store_paths
    code = _HARNESS.format(
        data_root=data_root, metadata_db=metadata_db, arrow_dir=arrow_dir
    ) + "timed(2)\n"
    t = time.perf_counter()
    procs = [
        subprocess.Popen([sys.executable, "-c", code],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(2)
    ]
    for proc in procs:
        assert proc.wait(timeout=120) == 0
    elapsed = time.perf_counter() - t
    assert elapsed > 2 * _WAITED, (
        f"two concurrent sweeps took {elapsed:.2f}s — under two intervals, so "
        f"their waits overlapped and the lock is not serialising them"
    )


@community_only
def test_refused_call_does_not_wait(store_paths):
    """Refusing is instant: no 5 s wait before being told no.

    The refusal comes first in a fresh process, so a later accepted call still
    waits — the refusal neither charged nor exempted anything.
    """
    refused, ok, accepted = _run(store_paths, """
        t_refused, ok = timed(1000)   # larger than the cap
        t_accepted, _ = timed(2)      # first *accepted* call: waits
        print(t_refused, ok, t_accepted)
    """)
    assert ok == "False", "expected the over-cap call to be refused"
    assert float(refused) < _DID_NOT_WAIT, (
        f"refused call took {float(refused):.2f}s — it should not wait"
    )
    assert float(accepted) > _WAITED, (
        "the accepted call after a refusal did not wait out the rate limit"
    )


@pro_only
def test_pro_is_not_rate_limited(store_paths):
    """Pro skips the gate entirely: no counter, no wait, on any call."""
    first, second = (
        float(x) for x in _run(store_paths, """
            t1, _ = timed(2)
            t2, _ = timed(2)
            print(t1, t2)
        """)
    )
    assert first < _DID_NOT_WAIT and second < _DID_NOT_WAIT, (
        f"Pro waited ({first:.2f}s, {second:.2f}s) — the rate gate leaked"
    )
