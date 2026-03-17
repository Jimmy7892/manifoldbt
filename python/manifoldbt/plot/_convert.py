"""Convert Arrow / RecordBatch data from BacktestResult to numpy arrays."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa


def arrow_to_numpy(arr: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """Convert a PyArrow array to a numpy array, combining chunks if needed."""
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy(zero_copy_only=False)
    return np.array(arr.to_pylist())


def _ts_to_int64(arr: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """Convert a PyArrow Timestamp column to int64 nanoseconds via Arrow cast."""
    import pyarrow as pa

    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    # Cast Timestamp → int64 inside Arrow (no Python Timestamp objects)
    if pa.types.is_timestamp(arr.type):
        return arr.cast(pa.int64()).to_numpy(zero_copy_only=False)
    raw = arr.to_numpy(zero_copy_only=False)
    if raw.dtype == np.int64 or raw.dtype.kind == "i":
        return raw
    # Fallback: already datetime64
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.view(np.int64)
    return np.array(arr.to_pylist(), dtype="int64")


def timestamps_to_dates(arr: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """Convert Timestamp(ns, UTC) Arrow array to numpy datetime64[ns]."""
    ns = _ts_to_int64(arr)
    return ns.view("datetime64[ns]")


def equity_with_dates(result) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (dates, equity_values) from a BacktestResult.

    The positions RecordBatch has one row per (timestamp, symbol). Equity is
    portfolio-level (same value across symbols at a given timestamp), so we
    deduplicate on timestamp.
    """
    positions = result.positions
    ts_col = positions.column("timestamp")
    eq_col = positions.column("equity")

    eq_raw = arrow_to_numpy(eq_col)

    # Get timestamps as int64 nanoseconds for deduplication
    ts_ns = _ts_to_int64(ts_col)

    _, unique_idx = np.unique(ts_ns, return_index=True)
    unique_idx.sort()

    dates = ts_ns[unique_idx].view("datetime64[ns]")
    values = eq_raw[unique_idx].astype(np.float64)
    return dates, values


def daily_returns_array(result) -> np.ndarray:
    """Extract daily_returns as a numpy float64 array."""
    return arrow_to_numpy(result.daily_returns).astype(np.float64)


def positions_arrays(result) -> dict:
    """Extract positions RecordBatch as a dict of numpy arrays.

    Returns dict with keys: timestamp, symbol_id, position, close, capital, equity.
    """
    positions = result.positions
    out = {}
    for name in positions.schema.names:
        col = positions.column(name)
        if name == "timestamp":
            out[name] = timestamps_to_dates(col)
        else:
            out[name] = arrow_to_numpy(col)
    return out


def trades_arrays(result) -> dict:
    """Extract trades RecordBatch as a dict of numpy arrays.

    Returns dict with keys matching the trades schema.
    """
    trades = result.trades
    out = {}
    for name in trades.schema.names:
        col = trades.column(name)
        if "timestamp" in name:
            out[name] = timestamps_to_dates(col)
        else:
            out[name] = arrow_to_numpy(col)
    return out
