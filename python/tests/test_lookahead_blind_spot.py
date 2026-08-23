"""The look-ahead the detector cannot see, pinned as a characterization test.

`detect_lookahead` used to document itself as catching global look-ahead,
"e.g. np.mean(all_prices) instead of rolling". It does not, and it cannot: both
its sub-tests re-run the *same strategy* on a shorter window, so a threshold
computed in a notebook and passed in as a number is identical in every run.

This file asserts the blind spot on purpose. A test that pins a limitation is
worth more than a docstring promising the opposite, because the docstring was
wrong for as long as nobody tried it.

It also pins the method that DOES catch it, so the boundary is not just
described but demonstrated: re-derive the parameter on the truncated window and
compare the same prefix.
"""
import os

import pytest

import manifoldbt as bt

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from manifoldbt.helpers import Interval, Slippage  # noqa: E402

N_DAYS = 400
SPLIT = 260


def _mean_reverting_daily():
    """A series that pulls back to its mean, where knowing that mean pays."""
    rng = np.random.default_rng(11)
    level = np.cumsum(rng.normal(0.0, 0.018, N_DAYS))
    px = 100.0 * np.exp(level - np.linspace(0, level[-1], N_DAYS))
    o = px
    c = np.roll(px, -1)
    c[-1] = px[-1]
    amp = np.abs(rng.normal(0.0, 0.004, N_DAYS))
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=N_DAYS, freq="1D", tz="UTC"),
        "open": o,
        "high": np.maximum(o, c) * (1 + amp),
        "low": np.minimum(o, c) * (1 - amp),
        "close": c,
        "volume": rng.uniform(1_000, 5_000, N_DAYS),
    })


def _store(frame, tmp_path, tag):
    root = os.path.join(str(tmp_path), tag)
    return bt.import_dataframe(
        frame, symbol="SYNTH", symbol_id=1, interval="1d",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )


def _config(frame):
    ts = pd.DatetimeIndex(frame["timestamp"])
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=int(ts[0].value),
        time_range_end=int(ts[-1].value) + 86_400_000_000_000,
        bar_interval=Interval.days(1),
        initial_capital=10_000,
        execution=bt.ExecutionConfig(
            signal_delay=1, max_position_pct=1.0,
            allow_short=True, position_sizing_mode="FractionOfEquity",
        ),
        slippage=Slippage.fixed_bps(0),
        warmup_bars=0,
    )


def _leaky(mean_price):
    """The threshold is a number the researcher computed over everything."""
    from manifoldbt.indicators import close

    return (
        bt.Strategy.create("global_mean_leak")
        .signal("edge", close)
        .size(bt.when(close < mean_price, 1.0, -1.0))
    )


def _equity(result):
    return np.array([float(x) for x in result.equity_curve])


def test_a_parameter_baked_at_research_time_flatters_the_result(tmp_path):
    """First establish there IS a leak, otherwise the blind spot is moot."""
    from manifoldbt.indicators import close

    frame = _mean_reverting_daily()
    store = _store(frame, tmp_path, "seduction")
    global_mean = float(frame["close"].mean())

    leaked = bt.run(_leaky(global_mean), _config(frame), store)
    honest = bt.run(
        bt.Strategy.create("rolling")
        .signal("edge", close)
        .size(bt.when(close < close.rolling_mean(60), 1.0, -1.0)),
        _config(frame), store,
    )

    assert leaked.metrics["total_return"] > honest.metrics["total_return"], (
        "the global mean did not flatter the result, so this fixture no longer "
        "demonstrates a leak worth detecting"
    )


def test_the_detector_is_blind_to_it(tmp_path):
    """Pinned limitation: PASS here is the documented, expected answer.

    If this ever starts failing, the detector gained the ability to audit a
    baked parameter. That would be good news, and the warning in
    `detect_lookahead`'s docstring should be revisited rather than this test
    silenced.
    """
    from manifoldbt.diagnostics import detect_lookahead

    frame = _mean_reverting_daily()
    result = detect_lookahead(
        _leaky(float(frame["close"].mean())),
        _config(frame), _store(frame, tmp_path, "blind"), mode="all",
    )

    compared = sum(r.total_trades_overlap for r in result.reports)
    assert compared > 0, "empty verdict, the blind spot is not what is being shown"
    assert result.passed, (
        "the detector now catches a research-time constant; update the docstring "
        "warning instead of deleting this test"
    )


def test_re_deriving_the_parameter_catches_it(tmp_path):
    """The technique that works, and the reason the blind spot is acceptable.

    Same window, same strategy shape: only the threshold differs, one computed
    with the future and one without. The equity must diverge.
    """
    frame = _mean_reverting_daily()
    truncated = frame.iloc[:SPLIT + 1]

    with_future = _equity(bt.run(
        _leaky(float(frame["close"].mean())),          # knows all 400 days
        _config(truncated), _store(truncated, tmp_path, "future"),
    ))
    with_past = _equity(bt.run(
        _leaky(float(truncated["close"].mean())),      # knows only the first 261
        _config(truncated), _store(truncated, tmp_path, "past"),
    ))

    n = min(len(with_future), len(with_past))
    assert n > 100, f"only {n} bars compared, too few to conclude"
    gap = float(np.abs(with_future[:n] - with_past[:n]).max())
    assert gap > 0.0, (
        "re-deriving the threshold changed nothing, so this method would not "
        "catch the leak either"
    )
