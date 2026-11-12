"""Lightweight vectorized backtesting engine and tearsheet generator for Polars."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from .core import IntoExpr, check_period, check_positive, parse_expr
from .evaluation import (
    calmar_ratio,
    deflated_sharpe_ratio,
    drawdown_series,
    max_drawdown,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

__all__ = [
    "BacktestResult",
    "vectorized_backtest",
    "extract_trades",
    "parameter_sweep",
    "crosses_above",
    "crosses_below",
    "signals_to_positions",
    "compare_strategies",
]

_EMPTY_TRADE_SCHEMA: dict[str, pl.DataType] = {
    "trade_id": pl.Int64(),
    "side": pl.String(),
    "entry_index": pl.Int64(),
    "exit_index": pl.Int64(),
    "holding_bars": pl.Int64(),
    "avg_position": pl.Float64(),
    "entry_price": pl.Float64(),
    "exit_price": pl.Float64(),
    "gross_return": pl.Float64(),
    "cost": pl.Float64(),
    "net_return": pl.Float64(),
    "mae": pl.Float64(),
    "mfe": pl.Float64(),
}


def _parse_expr_or_scalar(val: IntoExpr | float | int) -> pl.Expr:
    """Convert column name, Polars expression, or numeric threshold into a Polars expression."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return pl.lit(float(val))
    return parse_expr(val)


def _cross_pair(
    a: IntoExpr,
    b: IntoExpr | float | int,
    over: str | Sequence[str] | None,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Build current and 1-bar lagged expressions for crossing detection."""
    ea = parse_expr(a).cast(pl.Float64)
    eb = _parse_expr_or_scalar(b).cast(pl.Float64)
    prev_a = ea.shift(1).over(over) if over is not None else ea.shift(1)
    is_scalar = isinstance(b, (int, float)) and not isinstance(b, bool)
    prev_b = eb if is_scalar else (eb.shift(1).over(over) if over is not None else eb.shift(1))
    return ea, eb, prev_a, prev_b


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
    ea, eb, prev_a, prev_b = _cross_pair(a, b, over)
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
    ea, eb, prev_a, prev_b = _cross_pair(a, b, over)
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


def extract_trades(
    bt_df: pl.DataFrame,
    price_col: str | None = "close",
    time_col: str | None = None,
    group_by: str | Sequence[str] | None = None,
) -> pl.DataFrame:
    """Extract a Round-Trip Trade Ledger with MAE and MFE from a backtest DataFrame.

    Identifies contiguous non-zero position regimes in ``bt_df`` using native Polars
    state-transition expressions and aggregates trade-level metrics including holding
    duration, gross/net returns, **Maximum Adverse Excursion (MAE)**, and
    **Maximum Favorable Excursion (MFE)**.

    :param bt_df: Backtest DataFrame containing ``position``, ``gross_return``, ``cost``,
        and ``net_return`` columns.
    :param price_col: Optional price column name in ``bt_df`` for reporting ``entry_price``
        and ``exit_price`` (default ``'close'``).
    :param time_col: Optional timestamp/date column name for reporting ``entry_time`` and ``exit_time``.
    :param group_by: Optional symbol/ticker column(s) when ``bt_df`` is an unaggregated multi-asset panel.
    :return: Polars DataFrame with one row per round-trip trade.
    """
    if len(bt_df) == 0 or "position" not in bt_df.columns:
        return pl.DataFrame(schema=_EMPTY_TRADE_SCHEMA)

    gb_cols: list[str] | None = None
    if group_by is not None:
        cand = [group_by] if isinstance(group_by, str) else list(group_by)
        if all(c in bt_df.columns for c in cand):
            gb_cols = cand

    has_price = price_col is not None and price_col in bt_df.columns
    has_time = time_col is not None and time_col in bt_df.columns

    pos_sign = pl.col("position").sign().fill_null(0.0)
    prev_sign = (
        pos_sign.shift(1, fill_value=0.0).over(gb_cols)
        if gb_cols is not None
        else pos_sign.shift(1, fill_value=0.0)
    )

    prep_exprs: list[pl.Expr] = [
        pl.int_range(0, pl.len()).cast(pl.Int64).alias("_row_idx"),
        pos_sign.alias("_pos_sign"),
        (pos_sign != prev_sign).cast(pl.Int64).cum_sum().alias("_raw_seg"),
    ]
    if has_price and price_col is not None:
        p_expr = pl.col(price_col).cast(pl.Float64)
        prev_p = (
            p_expr.shift(1).over(gb_cols).fill_null(p_expr)
            if gb_cols is not None
            else p_expr.shift(1).fill_null(p_expr)
        )
        prep_exprs.extend([prev_p.alias("_entry_ref_price"), p_expr.alias("_bar_price")])
    else:
        prep_exprs.extend(
            [
                pl.lit(None, dtype=pl.Float64).alias("_entry_ref_price"),
                pl.lit(None, dtype=pl.Float64).alias("_bar_price"),
            ]
        )

    active = bt_df.with_columns(prep_exprs).filter(pl.col("_pos_sign") != 0.0)
    if len(active) == 0:
        return pl.DataFrame(schema=_EMPTY_TRADE_SCHEMA)

    seg_keys = [*gb_cols, "_raw_seg"] if gb_cols is not None else ["_raw_seg"]
    active = active.with_columns(
        [
            ((1.0 + pl.col("gross_return")).cum_prod().over(seg_keys) - 1.0).alias("_cum_gross"),
            ((1.0 + pl.col("net_return")).cum_prod().over(seg_keys) - 1.0).alias("_cum_net"),
        ]
    )

    agg_exprs: list[pl.Expr] = [
        pl.when(pl.col("_pos_sign").first() > 0.0)
        .then(pl.lit("LONG"))
        .otherwise(pl.lit("SHORT"))
        .alias("side"),
        pl.col("_row_idx").first().alias("entry_index"),
        pl.col("_row_idx").last().alias("exit_index"),
    ]
    if has_time and time_col is not None:
        agg_exprs.extend(
            [
                pl.col(time_col).first().alias("entry_time"),
                pl.col(time_col).last().alias("exit_time"),
            ]
        )
    agg_exprs.extend(
        [
            pl.len().cast(pl.Int64).alias("holding_bars"),
            pl.col("position").abs().mean().alias("avg_position"),
            pl.col("_entry_ref_price").first().alias("entry_price"),
            pl.col("_bar_price").last().alias("exit_price"),
            pl.col("_cum_gross").last().alias("gross_return"),
            pl.col("cost").sum().alias("cost"),
            pl.col("_cum_net").last().alias("net_return"),
            pl.min_horizontal(pl.col("_cum_gross").min(), pl.lit(0.0)).alias("mae"),
            pl.max_horizontal(pl.col("_cum_gross").max(), pl.lit(0.0)).alias("mfe"),
        ]
    )

    trades_df = (
        active.group_by(seg_keys)
        .agg(agg_exprs)
        .sort("entry_index")
        .drop("_raw_seg")
        .with_row_index("trade_id", offset=1)
        .with_columns(pl.col("trade_id").cast(pl.Int64))
    )
    return trades_df


@dataclass(frozen=True)
class BacktestResult:
    """Container holding vectorized backtest time-series data, trade ledger, and tearsheet summary.

    :ivar data: Polars DataFrame containing ``position``, ``turnover``, ``asset_return``,
        ``gross_return``, ``cost``, ``net_return``, ``equity_curve``, ``benchmark_equity``,
        and ``drawdown``.
    :ivar summary: Dictionary of key performance, risk, and round-trip trade metrics.
    :ivar trade_ledger: Polars DataFrame of round-trip trades with MAE and MFE analytics.
    """

    data: pl.DataFrame
    summary: dict[str, float]
    trade_ledger: pl.DataFrame = field(default_factory=pl.DataFrame)

    def trades(self) -> pl.DataFrame:
        """Return the Round-Trip Trade Ledger DataFrame (with entry/exit, net_return, MAE, MFE)."""
        if len(self.trade_ledger) > 0 or "trade_id" in self.trade_ledger.columns:
            return self.trade_ledger
        return extract_trades(self.data)

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
            ("Probabilistic Sharpe Ratio (PSR)", f"{s.get('psr', 0.0) * 100.0:.1f}%"),
            ("Sortino Ratio", f"{s.get('sortino_ratio', 0.0):.2f}"),
            ("Calmar Ratio", f"{s.get('calmar_ratio', 0.0):.2f}"),
            ("Max Drawdown", f"{s.get('max_drawdown', 0.0) * 100.0:.2f}%"),
            ("Win Rate", f"{s.get('win_rate', 0.0) * 100.0:.2f}%"),
            ("Profit Factor", f"{s.get('profit_factor', 0.0):.2f}"),
            ("Exposure Time", f"{s.get('exposure', 0.0) * 100.0:.1f}%"),
            ("Round-Trip Trades", f"{int(s.get('n_round_trips', 0.0)):,d}"),
            ("Trade Win Rate", f"{s.get('trade_win_rate', 0.0) * 100.0:.2f}%"),
            ("Avg Holding Bars", f"{s.get('avg_holding_bars', 0.0):.1f}"),
            (
                "Avg Trade MAE / MFE",
                f"{s.get('avg_mae', 0.0) * 100.0:.2f}% / {s.get('avg_mfe', 0.0) * 100.0:+.2f}%",
            ),
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


def _apply_no_trade_buffer(
    pos_expr: pl.Expr,
    buffer: float,
    over: list[str] | None = None,
) -> pl.Expr:
    """Suppress position updates smaller than ``buffer`` unless exiting to ``0.0``."""
    buf = float(buffer)

    def _buf_kernel(s: pl.Series) -> pl.Series:
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.zeros(n, dtype=np.float64)
        cur = 0.0
        for i in range(n):
            target = vals[i]
            if target == 0.0 or abs(target - cur) > buf:
                cur = target
            out[i] = cur
        return pl.Series(out, dtype=pl.Float64)

    res = pos_expr.map_batches(_buf_kernel, return_dtype=pl.Float64)
    return res.over(over) if over is not None else res


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
    execution_price: Literal["close", "open", "twap", "vwap"] = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    flat_mask: IntoExpr | None = None,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    no_trade_buffer: float = 0.0,
) -> BacktestResult:
    """Run a fast, loop-free vectorized backtest over single-asset or multi-asset Polars frames.

    Automatically shifts signals by ``signal_lag`` bars to eliminate look-ahead bias,
    supports realistic execution prices (``'close'``, ``'open'``, ``'twap'``, ``'vwap'``),
    intraday EOD session flattening (``flat_mask``), dynamic Volatility Targeting (``vol_target``),
    and no-trade turnover buffers (``no_trade_buffer``). Deducts transaction fees and slippage
    proportional to position turnover ``|position[t] - position[t-1]|``, constructs cumulative
    strategy & benchmark equity curves, and generates a Round-Trip Trade Ledger with MAE/MFE.

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
    :param execution_price: Price benchmark used to compute bar returns when ``return_col=None``:
        ``'close'`` (default), ``'open'`` (next-open execution), ``'twap'`` (OHLC average),
        or ``'vwap'`` (typical price ``(H+L+C)/3``).
    :param open_col: Open price column name when ``execution_price`` is ``'open'`` or ``'twap'``.
    :param high_col: High price column name when ``execution_price`` is ``'twap'`` or ``'vwap'``.
    :param low_col: Low price column name when ``execution_price`` is ``'twap'`` or ``'vwap'``.
    :param flat_mask: Optional boolean column/expression forcing ``position = 0.0`` on matching bars
        (e.g., EOD intraday flattening at 14:25 for VN30F1M futures).
    :param vol_target: Optional target annualized portfolio volatility (e.g., ``0.15`` for 15%).
        Scales positions dynamically by ``min(max_leverage, vol_target / realized_vol)``.
    :param vol_lookback: Rolling window length in bars for realized volatility estimation (default 20).
    :param max_leverage: Maximum position scaling multiplier when ``vol_target`` is active (default 2.0).
    :param no_trade_buffer: Minimum position change threshold ``|target - prev_pos|`` required to
        trigger a rebalance trade (default 0.0).
    :return: :class:`BacktestResult` containing ``.data`` DataFrame, ``.summary`` dict, and ``.trades()``.
    """
    if fee_rate < 0.0 or slippage_rate < 0.0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")
    if signal_lag < 0:
        raise ValueError("signal_lag must be a non-negative integer")
    check_positive(initial_capital, "initial_capital")
    check_positive(annualization, "annualization")
    if weight_scheme not in ("raw", "dollar_neutral", "long_only", "equal_weight"):
        raise ValueError(
            "weight_scheme must be one of 'raw', 'dollar_neutral', 'long_only', 'equal_weight'"
        )
    if execution_price not in ("close", "open", "twap", "vwap"):
        raise ValueError("execution_price must be one of 'close', 'open', 'twap', 'vwap'")
    if (entries is not None) ^ (exits is not None):
        raise ValueError("both entries and exits must be provided together")
    if vol_target is not None:
        check_positive(vol_target, "vol_target")
        check_period(vol_lookback, 2, "vol_lookback", inclusive=True)
        check_positive(max_leverage, "max_leverage")
    if no_trade_buffer < 0.0:
        raise ValueError("no_trade_buffer must be non-negative")

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

    if execution_price == "open":
        exec_price_expr = pl.col(open_col).cast(pl.Float64)
    elif execution_price == "twap":
        exec_price_expr = (
            pl.col(open_col).cast(pl.Float64)
            + pl.col(high_col).cast(pl.Float64)
            + pl.col(low_col).cast(pl.Float64)
            + pl.col(close_col).cast(pl.Float64)
        ) / 4.0
    elif execution_price == "vwap":
        exec_price_expr = (
            pl.col(high_col).cast(pl.Float64)
            + pl.col(low_col).cast(pl.Float64)
            + pl.col(close_col).cast(pl.Float64)
        ) / 3.0
    else:
        exec_price_expr = pl.col(close_col).cast(pl.Float64)

    if return_col is not None:
        ret_expr = pl.col(return_col).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    else:
        ret_expr = (
            exec_price_expr.pct_change().over(gb_cols)
            if gb_cols is not None
            else exec_price_expr.pct_change()
        )

    base_sig = pl.col("_anta_sig")
    pos_expr = base_sig.shift(signal_lag, fill_value=0.0) if signal_lag > 0 else base_sig
    if gb_cols is not None and signal_lag > 0:
        pos_expr = base_sig.shift(signal_lag, fill_value=0.0).over(gb_cols)

    ret_expr = ret_expr.fill_null(0.0).fill_nan(0.0)
    pos_expr = pos_expr.fill_null(0.0).fill_nan(0.0)

    if flat_mask is not None:
        f_mask = parse_expr(flat_mask).cast(pl.Boolean).fill_null(False)
        pos_expr = pl.when(f_mask).then(0.0).otherwise(pos_expr)

    step1 = step0.with_columns(
        [
            pos_expr.alias("position"),
            ret_expr.alias("asset_return"),
        ]
    ).drop("_anta_sig")

    if vol_target is not None:
        r_std = pl.col("asset_return").rolling_std(window_size=vol_lookback, min_samples=2).shift(1)
        if gb_cols is not None:
            r_std = r_std.over(gb_cols)
        ann_realized_vol = r_std * math.sqrt(float(annualization))
        vol_scalar = (
            pl.when(ann_realized_vol.is_not_null() & (ann_realized_vol > 1e-8))
            .then((float(vol_target) / ann_realized_vol).clip(0.0, float(max_leverage)))
            .otherwise(1.0)
        )
        scaled_pos = pl.col("position") * vol_scalar
        if flat_mask is not None:
            f_mask = parse_expr(flat_mask).cast(pl.Boolean).fill_null(False)
            scaled_pos = pl.when(f_mask).then(0.0).otherwise(scaled_pos)
        step1 = step1.with_columns(scaled_pos.alias("position"))

    if no_trade_buffer > 0.0:
        step1 = step1.with_columns(
            _apply_no_trade_buffer(pl.col("position"), no_trade_buffer, over=gb_cols).alias(
                "position"
            )
        )

    prev_pos_expr = (
        pl.col("position").shift(1, fill_value=0.0).over(gb_cols)
        if gb_cols is not None
        else pl.col("position").shift(1, fill_value=0.0)
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
        empty_trades = extract_trades(bt_df, price_col=close_col, time_col=time_col)
        empty_summary = {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "total_return": 0.0,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "psr": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "exposure": 0.0,
            "n_round_trips": 0.0,
            "trade_win_rate": 0.0,
            "avg_holding_bars": 0.0,
            "avg_mae": 0.0,
            "avg_mfe": 0.0,
            "n_trades": 0.0,
            "total_turnover": 0.0,
            "total_cost": 0.0,
            "n_bars": 0.0,
        }
        return BacktestResult(data=bt_df, summary=empty_summary, trade_ledger=empty_trades)

    trades_df = extract_trades(
        bt_df,
        price_col=open_col if execution_price == "open" else close_col,
        time_col=time_col,
        group_by=gb_cols
        if not (gb_cols is not None and time_col is not None and aggregate_portfolio)
        else None,
    )
    n_rt = len(trades_df)
    if n_rt > 0:
        t_win_rate = float((trades_df["net_return"] > 0.0).mean() or 0.0)
        avg_hold = float(trades_df["holding_bars"].mean() or 0.0)
        avg_mae = float(trades_df["mae"].mean() or 0.0)
        avg_mfe = float(trades_df["mfe"].mean() or 0.0)
    else:
        t_win_rate = 0.0
        avg_hold = 0.0
        avg_mae = 0.0
        avg_mfe = 0.0

    metrics_row = bt_df.select(
        [
            pl.col("equity_curve").last().alias("final_equity"),
            pl.col("benchmark_equity").last().alias("final_bench"),
            pl.col("net_return").mean().alias("mean_ret"),
            pl.col("net_return").std().fill_null(0.0).alias("std_ret"),
            sharpe_ratio(
                "net_return", risk_free_rate=risk_free_rate, annualization=annualization
            ).alias("sharpe"),
            probabilistic_sharpe_ratio(
                "net_return",
                benchmark_sr=0.0,
                risk_free_rate=risk_free_rate,
                annualization=annualization,
            ).alias("psr"),
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
        "psr": float(metrics_row["psr"] or 0.0),
        "sortino_ratio": float(metrics_row["sortino"] or 0.0),
        "calmar_ratio": float(metrics_row["calmar"] or 0.0),
        "max_drawdown": float(metrics_row["mdd"] or 0.0),
        "win_rate": float(metrics_row["win_rate"] or 0.0),
        "profit_factor": float(metrics_row["profit_factor"] or 0.0),
        "exposure": float(metrics_row["exposure"] or 0.0),
        "n_round_trips": float(n_rt),
        "trade_win_rate": t_win_rate,
        "avg_holding_bars": avg_hold,
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "n_trades": float(metrics_row["n_trades"] or 0.0),
        "total_turnover": float(metrics_row["total_turnover"] or 0.0),
        "total_cost": float(metrics_row["total_cost"] or 0.0),
        "n_bars": float(n_bars),
    }

    return BacktestResult(data=bt_df, summary=summary_dict, trade_ledger=trades_df)


def parameter_sweep(
    df: pl.DataFrame | pl.LazyFrame,
    signals: Mapping[str, IntoExpr],
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
    weight_scheme: Literal["raw", "dollar_neutral", "long_only", "equal_weight"] = "raw",
    execution_price: Literal["close", "open", "twap", "vwap"] = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    flat_mask: IntoExpr | None = None,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    no_trade_buffer: float = 0.0,
) -> pl.DataFrame:
    """Evaluate a grid of candidate signals/parameter configurations and compute Deflated Sharpe Ratios.

    Runs ``vectorized_backtest`` across all configurations in ``signals``, measures the empirical
    cross-trial Sharpe dispersion $\\sigma_{\\{SR\\}}$, and computes the multiple-testing
    **Deflated Sharpe Ratio (`deflated_sharpe_ratio`)** with ``n_trials = len(signals)`` so
    quantitative researchers can distinguish genuine parameter stability from backtest overfitting.

    :param df: Input Polars DataFrame or LazyFrame.
    :param signals: Mapping from parameter/strategy name to signal column name or Polars expression.
    :param close_col: Close price column name (default ``'close'``).
    :param return_col: Optional pre-computed return column.
    :param group_by: Optional symbol/ticker column(s).
    :param time_col: Optional timestamp/date column.
    :param fee_rate: One-way commission rate (default 0.0).
    :param slippage_rate: One-way slippage rate (default 0.0).
    :param signal_lag: Execution lag in bars (default 1).
    :param initial_capital: Starting portfolio equity (default 1.0).
    :param annualization: Trading periods per year (default 252.0).
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param weight_scheme: Portfolio weighting scheme (default ``'raw'``).
    :param execution_price: Execution price benchmark (``'close'``, ``'open'``, ``'twap'``, ``'vwap'``).
    :param open_col: Open price column name.
    :param high_col: High price column name.
    :param low_col: Low price column name.
    :param flat_mask: Optional boolean column/expression forcing ``position = 0.0``.
    :param vol_target: Optional target annualized volatility.
    :param vol_lookback: Rolling window length for volatility targeting (default 20).
    :param max_leverage: Maximum leverage multiplier (default 2.0).
    :param no_trade_buffer: Minimum position change threshold to rebalance (default 0.0).
    :return: Polars DataFrame of parameter sweep results sorted by ``sharpe_ratio`` descending.
    """
    if not signals:
        raise ValueError("signals mapping must contain at least one strategy expression")

    eager_df = df.collect() if isinstance(df, pl.LazyFrame) else df
    bt_kwargs: dict[str, Any] = {
        "close_col": close_col,
        "return_col": return_col,
        "group_by": group_by,
        "time_col": time_col,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "signal_lag": signal_lag,
        "initial_capital": initial_capital,
        "annualization": annualization,
        "risk_free_rate": risk_free_rate,
        "weight_scheme": weight_scheme,
        "execution_price": execution_price,
        "open_col": open_col,
        "high_col": high_col,
        "low_col": low_col,
        "flat_mask": flat_mask,
        "vol_target": vol_target,
        "vol_lookback": vol_lookback,
        "max_leverage": max_leverage,
        "no_trade_buffer": no_trade_buffer,
    }
    results = {
        name: vectorized_backtest(eager_df, signal=sig_expr, **bt_kwargs)
        for name, sig_expr in signals.items()
    }
    sharpes = [res.summary["sharpe_ratio"] for res in results.values()]

    n_trials = len(signals)
    sr_std = float(np.std(sharpes, ddof=1)) if n_trials > 1 else 0.5
    if sr_std <= 1e-8:
        sr_std = 0.5

    rows: list[dict[str, Any]] = []
    for name, bt_res in results.items():
        if len(bt_res.data) > 0:
            dsr_val = float(
                bt_res.data.select(
                    deflated_sharpe_ratio(
                        "net_return",
                        n_trials=n_trials,
                        trials_sr_std=sr_std,
                        risk_free_rate=risk_free_rate,
                        annualization=annualization,
                    )
                ).item()
                or 0.0
            )
        else:
            dsr_val = 0.0

        row: dict[str, Any] = {"strategy": name, "deflated_sharpe_ratio": dsr_val}
        row.update(bt_res.summary)
        rows.append(row)

    return pl.DataFrame(rows).sort("sharpe_ratio", descending=True)
