"""Lightweight vectorized backtesting engine and tearsheet generator for Polars."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from .core import IntoExpr, parse_expr
from .evaluation import (
    calmar_ratio,
    drawdown_series,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


@dataclass(frozen=True)
class BacktestResult:
    """Container holding vectorized backtest time-series data and performance tearsheet summary.

    :ivar data: Polars DataFrame containing ``position``, ``turnover``, ``asset_return``,
        ``gross_return``, ``cost``, ``net_return``, ``equity_curve``, and ``drawdown``.
    :ivar summary: Dictionary of key performance and risk metrics.
    """

    data: pl.DataFrame
    summary: dict[str, float]

    def summary_df(self, transpose: bool = False) -> pl.DataFrame:
        """Convert the performance summary dictionary into a Polars DataFrame.

        :param transpose: If False (default), returns a 1-row wide DataFrame with one column
            per metric; if True, returns a 2-column ``['metric', 'value']`` DataFrame.
        :return: Polars DataFrame.
        """
        if transpose:
            return pl.DataFrame(
                {
                    "metric": list(self.summary.keys()),
                    "value": [float(v) for v in self.summary.values()],
                }
            )
        return pl.DataFrame([self.summary])

    def __getitem__(self, key: str) -> Any:
        if key in self.summary:
            return self.summary[key]
        return self.data[key]


def vectorized_backtest(
    df: pl.DataFrame | pl.LazyFrame,
    signal: IntoExpr = "signal",
    close_col: str = "close",
    return_col: str | None = None,
    group_by: str | Sequence[str] | None = None,
    time_col: str | None = None,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    signal_lag: int = 1,
    initial_capital: float = 1.0,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
    compound: bool = True,
    aggregate_portfolio: bool = True,
) -> BacktestResult:
    """Run a fast, loop-free vectorized backtest over single-asset or multi-asset Polars frames.

    Automatically shifts signals by ``signal_lag`` bars to eliminate look-ahead bias,
    deducts transaction fees and slippage proportional to position turnover
    ``|position[t] - position[t-1]|``, constructs the cumulative equity & drawdown curves,
    and computes an institutional performance tearsheet.

    :param df: Input Polars DataFrame or LazyFrame.
    :param signal: Signal or portfolio weight column name or Polars expression (default ``'signal'``).
        Supports boolean flags, discrete signals ``{-1, 0, 1}``, or continuous weights.
    :param close_col: Name of close price column used to compute asset returns when
        ``return_col`` is None (default ``'close'``).
    :param return_col: Optional pre-computed asset return column.
    :param group_by: Optional symbol/ticker column(s) for multi-asset universes.
    :param time_col: Optional timestamp/date column. When both ``group_by`` and ``time_col``
        are provided and ``aggregate_portfolio=True``, aggregates portfolio returns per timestamp.
    :param fee_rate: One-way proportional commission rate (e.g., ``0.0005`` for 5 bps, default 0.0).
    :param slippage_rate: One-way proportional slippage rate (e.g., ``0.0002`` for 2 bps, default 0.0).
    :param signal_lag: Execution lag in bars (default 1, preventing look-ahead bias).
    :param initial_capital: Starting portfolio equity (default 1.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :param risk_free_rate: Annualized risk-free rate for Sharpe/Sortino calculations (default 0.0).
    :param compound: When True (default), compounds returns multiplicatively; when False,
        accumulates returns additively.
    :param aggregate_portfolio: When True and both ``group_by`` and ``time_col`` are set,
        aggregates asset-level returns into a single portfolio time series ordered by ``time_col``.
    :return: :class:`BacktestResult` containing ``.data`` DataFrame and ``.summary`` dict.
    """
    if fee_rate < 0.0 or slippage_rate < 0.0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")
    if signal_lag < 0:
        raise ValueError("signal_lag must be a non-negative integer")
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be a positive number greater than zero")
    if annualization <= 0.0:
        raise ValueError("annualization must be a positive number greater than zero")

    total_cost_rate = float(fee_rate) + float(slippage_rate)
    sig_expr = parse_expr(signal).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)

    if return_col is not None:
        ret_expr = pl.col(return_col).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    else:
        ret_expr = pl.col(close_col).cast(pl.Float64).pct_change()

    gb_cols: list[str] | None = None
    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)

    pos_expr = sig_expr.shift(signal_lag, fill_value=0.0) if signal_lag > 0 else sig_expr
    if gb_cols is not None:
        if return_col is None:
            ret_expr = pl.col(close_col).cast(pl.Float64).pct_change().over(gb_cols)
        if signal_lag > 0:
            pos_expr = sig_expr.shift(signal_lag, fill_value=0.0).over(gb_cols)

    ret_expr = ret_expr.fill_null(0.0).fill_nan(0.0)
    pos_expr = pos_expr.fill_null(0.0).fill_nan(0.0)

    prev_pos_expr = (
        pl.col("position").shift(1, fill_value=0.0).over(gb_cols)
        if gb_cols is not None
        else pl.col("position").shift(1, fill_value=0.0)
    )

    step1 = df.with_columns(
        [
            pos_expr.alias("position"),
            ret_expr.alias("asset_return"),
        ]
    )

    step2 = step1.with_columns(
        [
            (pl.col("position") - prev_pos_expr).abs().alias("turnover"),
            (pl.col("position") * pl.col("asset_return")).alias("gross_return"),
        ]
    )

    step3 = step2.with_columns((pl.col("turnover") * total_cost_rate).alias("cost")).with_columns(
        (pl.col("gross_return") - pl.col("cost")).alias("net_return")
    )

    # Portfolio aggregation across assets when group_by and time_col are both provided
    if gb_cols is not None and time_col is not None and aggregate_portfolio:
        working = (
            step3.group_by(time_col)
            .agg(
                [
                    pl.col("position").sum().alias("position"),
                    pl.col("turnover").sum().alias("turnover"),
                    pl.col("asset_return").mean().alias("asset_return"),
                    pl.col("gross_return").sum().alias("gross_return"),
                    pl.col("cost").sum().alias("cost"),
                    pl.col("net_return").sum().alias("net_return"),
                ]
            )
            .sort(time_col)
        )
    else:
        working = step3

    if compound:
        eq_expr = float(initial_capital) * (1.0 + pl.col("net_return")).cum_prod()
    else:
        eq_expr = float(initial_capital) * (1.0 + pl.col("net_return").cum_sum())

    if gb_cols is not None and not (time_col is not None and aggregate_portfolio):
        eq_expr = eq_expr.over(gb_cols)
        dd_expr = drawdown_series(pl.col("equity_curve"), is_returns=False).over(gb_cols)
    else:
        dd_expr = drawdown_series(pl.col("equity_curve"), is_returns=False)

    final_frame = working.with_columns(eq_expr.alias("equity_curve")).with_columns(
        dd_expr.alias("drawdown")
    )

    bt_df = final_frame.collect() if isinstance(final_frame, pl.LazyFrame) else final_frame
    n_bars = len(bt_df)

    if n_bars == 0:
        empty_summary = {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_turnover": 0.0,
            "total_cost": 0.0,
            "n_bars": 0.0,
        }
        return BacktestResult(data=bt_df, summary=empty_summary)

    metrics_row = bt_df.select(
        [
            pl.col("equity_curve").last().alias("final_equity"),
            pl.col("net_return").mean().alias("mean_ret"),
            pl.col("net_return").std().fill_null(0.0).alias("std_ret"),
            sharpe_ratio(
                "net_return", risk_free_rate=risk_free_rate, annualization=annualization
            ).alias("sharpe"),
            sortino_ratio(
                "net_return", risk_free_rate=risk_free_rate, annualization=annualization
            ).alias("sortino"),
            calmar_ratio("net_return", annualization=annualization).alias("calmar"),
            max_drawdown("equity_curve", is_returns=False).alias("mdd"),
            win_rate("net_return", include_zeros=False).alias("win_rate"),
            profit_factor("net_return").alias("profit_factor"),
            pl.col("turnover").sum().alias("total_turnover"),
            pl.col("cost").sum().alias("total_cost"),
        ]
    ).to_dicts()[0]

    final_eq = float(metrics_row["final_equity"])
    total_ret = (final_eq / float(initial_capital)) - 1.0
    ann_ret = float(metrics_row["mean_ret"] or 0.0) * float(annualization)
    ann_vol = float(metrics_row["std_ret"] or 0.0) * math.sqrt(float(annualization))

    summary_dict: dict[str, float] = {
        "initial_capital": float(initial_capital),
        "final_equity": final_eq,
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": float(metrics_row["sharpe"] or 0.0),
        "sortino_ratio": float(metrics_row["sortino"] or 0.0),
        "calmar_ratio": float(metrics_row["calmar"] or 0.0),
        "max_drawdown": float(metrics_row["mdd"] or 0.0),
        "win_rate": float(metrics_row["win_rate"] or 0.0),
        "profit_factor": float(metrics_row["profit_factor"] or 0.0),
        "total_turnover": float(metrics_row["total_turnover"] or 0.0),
        "total_cost": float(metrics_row["total_cost"] or 0.0),
        "n_bars": float(n_bars),
    }

    return BacktestResult(data=bt_df, summary=summary_dict)
