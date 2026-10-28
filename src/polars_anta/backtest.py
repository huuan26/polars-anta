"""Lightweight vectorized backtesting engine and tearsheet generator for Polars."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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


def _parse_expr_or_scalar(val: IntoExpr | float | int) -> pl.Expr:
    """Convert column name, Polars expression, or numeric threshold into a Polars expression."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return pl.lit(float(val))
    return parse_expr(val)


def crosses_above(
    a: IntoExpr,
    b: IntoExpr | float | int,
    over: str | Sequence[str] | None = None,
) -> pl.Expr:
    """Return a boolean expression that is True on the exact bar where ``a`` crosses above ``b``.

    Supports crossing another indicator column/expression or a constant scalar threshold
    (e.g., ``pl.col('rsi_14').anta.crosses_above(30)``).

    :param a: Fast series or indicator expression/column name.
    :param b: Slow series, indicator expression/column name, or numeric threshold.
    :param over: Optional grouping column(s) for multi-asset panels.
    :return: Boolean Polars expression.
    """
    ea = parse_expr(a).cast(pl.Float64)
    eb = _parse_expr_or_scalar(b).cast(pl.Float64)
    prev_a = ea.shift(1).over(over) if over is not None else ea.shift(1)
    prev_b = (
        eb.shift(1).over(over)
        if (over is not None and not isinstance(b, (int, float)))
        else (eb if isinstance(b, (int, float)) else eb.shift(1))
    )
    return ((ea > eb) & (prev_a <= prev_b)).fill_null(False)


def crosses_below(
    a: IntoExpr,
    b: IntoExpr | float | int,
    over: str | Sequence[str] | None = None,
) -> pl.Expr:
    """Return a boolean expression that is True on the exact bar where ``a`` crosses below ``b``.

    Supports crossing another indicator column/expression or a constant scalar threshold
    (e.g., ``pl.col('rsi_14').anta.crosses_below(70)``).

    :param a: Fast series or indicator expression/column name.
    :param b: Slow series, indicator expression/column name, or numeric threshold.
    :param over: Optional grouping column(s) for multi-asset panels.
    :return: Boolean Polars expression.
    """
    ea = parse_expr(a).cast(pl.Float64)
    eb = _parse_expr_or_scalar(b).cast(pl.Float64)
    prev_a = ea.shift(1).over(over) if over is not None else ea.shift(1)
    prev_b = (
        eb.shift(1).over(over)
        if (over is not None and not isinstance(b, (int, float)))
        else (eb if isinstance(b, (int, float)) else eb.shift(1))
    )
    return ((ea < eb) & (prev_a >= prev_b)).fill_null(False)


def signals_to_positions(
    entries: IntoExpr,
    exits: IntoExpr,
    short_entries: IntoExpr | None = None,
    short_exits: IntoExpr | None = None,
    over: str | Sequence[str] | None = None,
) -> pl.Expr:
    """Convert discrete entry/exit boolean triggers into a continuous position series.

    Uses Polars' native ``forward_fill()`` state propagation with zero Python loops:
    - Holds ``+1.0`` from ``entries=True`` until ``exits=True`` (which resets to ``0.0``).
    - When ``short_entries`` is also supplied, holds ``-1.0`` from ``short_entries=True``
      until ``short_exits=True`` (or ``entries=True``).

    :param entries: Boolean column/expression triggering a long entry (``+1.0``).
    :param exits: Boolean column/expression triggering a long exit (``0.0``).
    :param short_entries: Optional boolean column/expression triggering a short entry (``-1.0``).
    :param short_exits: Optional boolean column/expression triggering a short exit (``0.0``).
    :param over: Optional symbol/ticker grouping column(s) for multi-asset panels.
    :return: Continuous position Polars expression in ``{-1.0, 0.0, 1.0}``.
    """
    ent = parse_expr(entries).cast(pl.Boolean).fill_null(False)
    ext = parse_expr(exits).cast(pl.Boolean).fill_null(False)

    long_state = (
        pl.when(ext & ~ent).then(0.0).when(ent).then(1.0).otherwise(pl.lit(None, dtype=pl.Float64))
    )
    long_pos = (
        long_state.forward_fill().over(over).fill_null(0.0)
        if over is not None
        else long_state.forward_fill().fill_null(0.0)
    )

    if short_entries is None:
        return long_pos

    s_ent = parse_expr(short_entries).cast(pl.Boolean).fill_null(False)
    s_ext = (
        parse_expr(short_exits).cast(pl.Boolean).fill_null(False)
        if short_exits is not None
        else ent
    )
    short_state = (
        pl.when(s_ext & ~s_ent)
        .then(0.0)
        .when(s_ent)
        .then(-1.0)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    short_pos = (
        short_state.forward_fill().over(over).fill_null(0.0)
        if over is not None
        else short_state.forward_fill().fill_null(0.0)
    )
    return (long_pos + short_pos).clip(-1.0, 1.0)


@dataclass(frozen=True)
class BacktestResult:
    """Container holding vectorized backtest time-series data and performance tearsheet summary.

    :ivar data: Polars DataFrame containing ``position``, ``turnover``, ``asset_return``,
        ``gross_return``, ``cost``, ``net_return``, ``equity_curve``, ``benchmark_equity``,
        and ``drawdown``.
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

    def report(self) -> pl.DataFrame:
        """Return a human-readable formatted tearsheet DataFrame with units (%, $, ratios)."""
        s = self.summary
        rows = [
            ("Initial Capital", f"${s.get('initial_capital', 0.0):,.2f}"),
            ("Final Equity", f"${s.get('final_equity', 0.0):,.2f}"),
            ("Total Return", f"{s.get('total_return', 0.0) * 100.0:+.2f}%"),
            ("Benchmark Return", f"{s.get('benchmark_return', 0.0) * 100.0:+.2f}%"),
            ("Excess Return (vs Benchmark)", f"{s.get('excess_return', 0.0) * 100.0:+.2f}%"),
            ("Annualized Return", f"{s.get('annualized_return', 0.0) * 100.0:+.2f}%"),
            ("Annualized Volatility", f"{s.get('annualized_volatility', 0.0) * 100.0:.2f}%"),
            ("Sharpe Ratio", f"{s.get('sharpe_ratio', 0.0):.2f}"),
            ("Sortino Ratio", f"{s.get('sortino_ratio', 0.0):.2f}"),
            ("Calmar Ratio", f"{s.get('calmar_ratio', 0.0):.2f}"),
            ("Max Drawdown", f"{s.get('max_drawdown', 0.0) * 100.0:.2f}%"),
            ("Win Rate", f"{s.get('win_rate', 0.0) * 100.0:.2f}%"),
            ("Profit Factor", f"{s.get('profit_factor', 0.0):.2f}"),
            ("Exposure Time", f"{s.get('exposure', 0.0) * 100.0:.1f}%"),
            ("Total Trades (Rebalances)", f"{int(s.get('n_trades', 0.0)):,d}"),
            ("Total Turnover", f"{s.get('total_turnover', 0.0):.2f}x"),
            ("Total Cost (Fee + Slippage)", f"{s.get('total_cost', 0.0) * 100.0:.3f}%"),
            ("Bars Evaluated", f"{int(s.get('n_bars', 0.0)):,d}"),
        ]
        return pl.DataFrame(
            {
                "metric": [r[0] for r in rows],
                "value": [r[1] for r in rows],
            }
        )

    def __getitem__(self, key: str) -> Any:
        if key in self.summary:
            return self.summary[key]
        return self.data[key]


def compare_strategies(
    strategies: Mapping[str, BacktestResult],
) -> pl.DataFrame:
    """Compare multiple :class:`BacktestResult` objects side-by-side in a single DataFrame.

    :param strategies: Dictionary mapping strategy names to their :class:`BacktestResult`.
    :return: Polars DataFrame with one row per strategy and key performance columns.
    """
    if not strategies:
        raise ValueError("strategies mapping must contain at least one BacktestResult")

    rows: list[dict[str, Any]] = []
    for name, res in strategies.items():
        row: dict[str, Any] = {"strategy": name}
        row.update(res.summary)
        rows.append(row)
    return pl.DataFrame(rows)


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
    entries: IntoExpr | None = None,
    exits: IntoExpr | None = None,
    short_entries: IntoExpr | None = None,
    short_exits: IntoExpr | None = None,
    weight_scheme: Literal["raw", "dollar_neutral", "long_only", "equal_weight"] = "raw",
) -> BacktestResult:
    """Run a fast, loop-free vectorized backtest over single-asset or multi-asset Polars frames.

    Automatically shifts signals by ``signal_lag`` bars to eliminate look-ahead bias,
    deducts transaction fees and slippage proportional to position turnover
    ``|position[t] - position[t-1]|``, constructs cumulative strategy & benchmark equity curves,
    and computes an institutional performance tearsheet.

    :param df: Input Polars DataFrame or LazyFrame.
    :param signal: Signal or portfolio weight column name or Polars expression (default ``'signal'``).
        Supports boolean flags, discrete signals ``{-1, 0, 1}``, or continuous weights.
        Ignored when ``entries`` and ``exits`` are explicitly provided.
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
    :param entries: Optional boolean column/expression triggering a long entry (``+1.0``).
        Requires ``exits`` to be set.
    :param exits: Optional boolean column/expression triggering a long exit (``0.0``).
    :param short_entries: Optional boolean column/expression triggering a short entry (``-1.0``).
    :param short_exits: Optional boolean column/expression triggering a short exit (``0.0``).
    :param weight_scheme: Cross-sectional portfolio weighting scheme when ``time_col`` is provided:
        ``'raw'`` (default, uses signal directly), ``'dollar_neutral'`` (demeans per timestamp and
        scales gross exposure to 1.0), ``'long_only'`` (scales positive signals to sum to 1.0),
        or ``'equal_weight'`` (assigns equal weight ``1/N`` to assets with ``signal > 0``).
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
    if weight_scheme not in ("raw", "dollar_neutral", "long_only", "equal_weight"):
        raise ValueError(
            "weight_scheme must be one of 'raw', 'dollar_neutral', 'long_only', 'equal_weight'"
        )
    if (entries is not None) ^ (exits is not None):
        raise ValueError("both entries and exits must be provided together")

    gb_cols: list[str] | None = None
    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)

    total_cost_rate = float(fee_rate) + float(slippage_rate)

    if entries is not None and exits is not None:
        raw_sig_expr = signals_to_positions(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            over=gb_cols,
        )
    else:
        raw_sig_expr = parse_expr(signal)

    sig_expr = raw_sig_expr.cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    step0 = df.with_columns(sig_expr.alias("_anta_sig"))

    # Apply cross-sectional portfolio weight normalization when requested
    if weight_scheme != "raw":
        cs_over = [time_col] if time_col is not None else None
        s_col = pl.col("_anta_sig")
        if weight_scheme == "dollar_neutral":
            mean_s = s_col.mean().over(cs_over) if cs_over is not None else s_col.mean()
            step0 = step0.with_columns((s_col - mean_s).alias("_anta_sig"))
            denom = (
                pl.col("_anta_sig").abs().sum().over(cs_over)
                if cs_over is not None
                else pl.col("_anta_sig").abs().sum()
            )
            norm_expr = pl.when(denom > 1e-12).then(pl.col("_anta_sig") / denom).otherwise(0.0)
        elif weight_scheme == "long_only":
            pos_only = pl.when(s_col > 0.0).then(s_col).otherwise(0.0)
            step0 = step0.with_columns(pos_only.alias("_anta_sig"))
            denom = (
                pl.col("_anta_sig").sum().over(cs_over)
                if cs_over is not None
                else pl.col("_anta_sig").sum()
            )
            norm_expr = pl.when(denom > 1e-12).then(pl.col("_anta_sig") / denom).otherwise(0.0)
        else:  # equal_weight
            active = pl.when(s_col > 0.0).then(1.0).otherwise(0.0)
            step0 = step0.with_columns(active.alias("_anta_sig"))
            denom = (
                pl.col("_anta_sig").sum().over(cs_over)
                if cs_over is not None
                else pl.col("_anta_sig").sum()
            )
            norm_expr = pl.when(denom > 0.0).then(pl.col("_anta_sig") / denom).otherwise(0.0)
        step0 = step0.with_columns(norm_expr.alias("_anta_sig"))

    if return_col is not None:
        ret_expr = pl.col(return_col).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    else:
        ret_expr = pl.col(close_col).cast(pl.Float64).pct_change()

    base_sig = pl.col("_anta_sig")
    pos_expr = base_sig.shift(signal_lag, fill_value=0.0) if signal_lag > 0 else base_sig
    if gb_cols is not None:
        if return_col is None:
            ret_expr = pl.col(close_col).cast(pl.Float64).pct_change().over(gb_cols)
        if signal_lag > 0:
            pos_expr = base_sig.shift(signal_lag, fill_value=0.0).over(gb_cols)

    ret_expr = ret_expr.fill_null(0.0).fill_nan(0.0)
    pos_expr = pos_expr.fill_null(0.0).fill_nan(0.0)

    prev_pos_expr = (
        pl.col("position").shift(1, fill_value=0.0).over(gb_cols)
        if gb_cols is not None
        else pl.col("position").shift(1, fill_value=0.0)
    )

    step1 = step0.with_columns(
        [
            pos_expr.alias("position"),
            ret_expr.alias("asset_return"),
        ]
    ).drop("_anta_sig")

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
        bench_eq_expr = float(initial_capital) * (1.0 + pl.col("asset_return")).cum_prod()
    else:
        eq_expr = float(initial_capital) * (1.0 + pl.col("net_return").cum_sum())
        bench_eq_expr = float(initial_capital) * (1.0 + pl.col("asset_return").cum_sum())

    if gb_cols is not None and not (time_col is not None and aggregate_portfolio):
        eq_expr = eq_expr.over(gb_cols)
        bench_eq_expr = bench_eq_expr.over(gb_cols)
        dd_expr = drawdown_series(pl.col("equity_curve"), is_returns=False).over(gb_cols)
    else:
        dd_expr = drawdown_series(pl.col("equity_curve"), is_returns=False)

    final_frame = working.with_columns(
        [
            eq_expr.alias("equity_curve"),
            bench_eq_expr.alias("benchmark_equity"),
        ]
    ).with_columns(dd_expr.alias("drawdown"))

    bt_df = final_frame.collect() if isinstance(final_frame, pl.LazyFrame) else final_frame
    n_bars = len(bt_df)

    if n_bars == 0:
        empty_summary = {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "total_return": 0.0,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "exposure": 0.0,
            "n_trades": 0.0,
            "total_turnover": 0.0,
            "total_cost": 0.0,
            "n_bars": 0.0,
        }
        return BacktestResult(data=bt_df, summary=empty_summary)

    metrics_row = bt_df.select(
        [
            pl.col("equity_curve").last().alias("final_equity"),
            pl.col("benchmark_equity").last().alias("final_bench"),
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
            (pl.col("position").abs() > 1e-12).cast(pl.Float64).mean().alias("exposure"),
            (pl.col("turnover") > 1e-12).cast(pl.Float64).sum().alias("n_trades"),
            pl.col("turnover").sum().alias("total_turnover"),
            pl.col("cost").sum().alias("total_cost"),
        ]
    ).to_dicts()[0]

    final_eq = float(metrics_row["final_equity"])
    final_bench = float(metrics_row["final_bench"])
    total_ret = (final_eq / float(initial_capital)) - 1.0
    bench_ret = (final_bench / float(initial_capital)) - 1.0
    ann_ret = float(metrics_row["mean_ret"] or 0.0) * float(annualization)
    ann_vol = float(metrics_row["std_ret"] or 0.0) * math.sqrt(float(annualization))

    summary_dict: dict[str, float] = {
        "initial_capital": float(initial_capital),
        "final_equity": final_eq,
        "total_return": total_ret,
        "benchmark_return": bench_ret,
        "excess_return": total_ret - bench_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": float(metrics_row["sharpe"] or 0.0),
        "sortino_ratio": float(metrics_row["sortino"] or 0.0),
        "calmar_ratio": float(metrics_row["calmar"] or 0.0),
        "max_drawdown": float(metrics_row["mdd"] or 0.0),
        "win_rate": float(metrics_row["win_rate"] or 0.0),
        "profit_factor": float(metrics_row["profit_factor"] or 0.0),
        "exposure": float(metrics_row["exposure"] or 0.0),
        "n_trades": float(metrics_row["n_trades"] or 0.0),
        "total_turnover": float(metrics_row["total_turnover"] or 0.0),
        "total_cost": float(metrics_row["total_cost"] or 0.0),
        "n_bars": float(n_bars),
    }

    return BacktestResult(data=bt_df, summary=summary_dict)
