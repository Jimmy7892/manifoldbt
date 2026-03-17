"""SweepResult — ergonomic wrapper for parameter sweep results."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

from manifoldbt.dataframe import results_to_df
from manifoldbt.result import Result


class SweepResult:
    """Results from a parameter sweep with DataFrame and analysis shortcuts.

    Wraps a list of ``BacktestResult`` objects returned by ``run_sweep()``
    and provides easy access to metrics, comparison, and plotting.

    Example::

        sweep = bt.run_sweep(strategy, {"fast": [10, 20, 30]}, config, store)
        print(len(sweep))              # 3
        df = sweep.to_df()             # DataFrame with metrics per combo
        best = sweep.best("sharpe")    # Result with highest Sharpe
        sweep.plot_metric("sharpe")    # bar/heatmap chart
    """

    __slots__ = ("_results", "_param_grid")

    def __init__(
        self,
        results: Sequence[Any],
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        self._results = [
            r if isinstance(r, Result) else Result(r)
            for r in results
        ]
        self._param_grid = param_grid or {}

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, idx: int) -> Result:
        return self._results[idx]

    def __iter__(self) -> Iterator[Result]:
        return iter(self._results)

    def to_df(self, backend: str = "auto") -> Any:
        """All results as a DataFrame with metrics and parameter columns.

        Args:
            backend: ``"pandas"``, ``"polars"``, or ``"auto"``.

        Returns:
            DataFrame with one row per parameter combination.
            Parameter columns are prefixed with ``param_``.
        """
        return results_to_df(self._results, self._param_grid, backend=backend)

    def best(self, metric: str = "sharpe") -> Result:
        """Return the Result with the highest value for *metric*.

        Args:
            metric: Metric name (e.g. ``"sharpe"``, ``"total_return"``, ``"sortino"``).
        """
        return self._extremum(metric, maximize=True)

    def worst(self, metric: str = "sharpe") -> Result:
        """Return the Result with the lowest value for *metric*.

        Args:
            metric: Metric name.
        """
        return self._extremum(metric, maximize=False)

    def _extremum(self, metric: str, maximize: bool) -> Result:
        best_val = None
        best_result = None
        for r in self._results:
            m = r.metrics
            val = m.get(metric) if isinstance(m, dict) else None
            # Check nested trade_stats
            if val is None and isinstance(m, dict):
                ts = m.get("trade_stats")
                if isinstance(ts, dict):
                    val = ts.get(metric)
            if val is None:
                continue
            if best_val is None or (val > best_val if maximize else val < best_val):
                best_val = val
                best_result = r
        if best_result is None:
            raise ValueError(f"Metric {metric!r} not found in any result")
        return best_result

    def plot_metric(self, metric: str = "sharpe", **kwargs: Any) -> Any:
        """Plot a metric across sweep results.

        For 2-parameter sweeps, delegates to ``bt.plot.heatmap_2d``.
        For 1-parameter sweeps, produces a bar chart.

        Args:
            metric: Metric to visualize.
            **kwargs: Forwarded to the plot function.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        df = self.to_df(backend="pandas")
        param_cols = [c for c in df.columns if c.startswith("param_")]

        if len(param_cols) == 2:
            # 2D heatmap
            x_col, y_col = param_cols[0], param_cols[1]
            pivot = df.pivot_table(index=y_col, columns=x_col, values=metric)
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)))
            im = ax.imshow(pivot.values, aspect="auto", cmap=kwargs.pop("cmap", "RdYlGn"))
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel(x_col.replace("param_", ""))
            ax.set_ylabel(y_col.replace("param_", ""))
            ax.set_title(f"{metric} heatmap")
            plt.colorbar(im, ax=ax, label=metric)
            plt.tight_layout()
            if kwargs.get("show", True):
                plt.show()
            return fig
        elif len(param_cols) == 1:
            # 1D bar chart
            p_col = param_cols[0]
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 5)))
            ax.bar(range(len(df)), df[metric].values, tick_label=[str(v) for v in df[p_col].values])
            ax.set_xlabel(p_col.replace("param_", ""))
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by {p_col.replace('param_', '')}")
            plt.tight_layout()
            if kwargs.get("show", True):
                plt.show()
            return fig
        else:
            # Fallback: simple bar
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 5)))
            ax.bar(range(len(df)), df[metric].values)
            ax.set_xlabel("run")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} across sweep")
            plt.tight_layout()
            if kwargs.get("show", True):
                plt.show()
            return fig

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={len(v)} vals" for k, v in self._param_grid.items())
        return f"SweepResult({len(self)} runs, {params})"
