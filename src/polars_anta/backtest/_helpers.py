"""Internal execution, weighting, calendar regularization, and buffer helpers for backtesting."""

from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
import polars as pl

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)


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


def _normalize_signal_weights(
    step0: FrameType,
    weight_scheme: Literal["raw", "dollar_neutral", "long_only", "equal_weight"],
    time_col: str | None,
) -> FrameType:
    """Apply cross-sectional portfolio weight normalization to ``_anta_sig``."""
    if weight_scheme == "raw":
        return step0

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
    return step0.with_columns(norm_expr.alias("_anta_sig"))


def _regularize_calendar_frame_typed(
    step0: FrameType,
    gb_cols: list[str],
    time_col: str,
    return_col: str | None,
    exec_price_expr: pl.Expr,
) -> FrameType:
    """Reindex frame onto dense (time_col, group_by) calendar grid and forward-fill signals."""
    flagged = step0.with_columns(
        [
            pl.lit(True).alias("_anta_has_bar"),
            (
                pl.col(return_col).cast(pl.Float64) if return_col is not None else exec_price_expr
            ).alias("_anta_cal_val"),
        ]
    )
    grid = flagged.select(time_col).unique().join(flagged.select(gb_cols).unique(), how="cross")
    return (
        grid.join(flagged, on=[*gb_cols, time_col], how="left")
        .sort([*gb_cols, time_col])
        .with_columns(
            [
                pl.col("_anta_has_bar").fill_null(False),
                pl.col("_anta_sig").forward_fill().over(gb_cols).fill_null(0.0),
            ]
        )
    )


def _prepare_step0_typed(
    frame: FrameType,
    sig_expr: pl.Expr,
    weight_scheme: Literal["raw", "dollar_neutral", "long_only", "equal_weight"],
    time_col: str | None,
    regularize_calendar: bool,
    gb_cols: list[str] | None,
    return_col: str | None,
    exec_price_expr: pl.Expr,
) -> FrameType:
    """Normalize signal weights and optionally regularize calendar for a typed frame."""
    norm = _normalize_signal_weights(
        frame.with_columns(sig_expr.alias("_anta_sig")),
        weight_scheme=weight_scheme,
        time_col=time_col,
    )
    if regularize_calendar and gb_cols is not None and time_col is not None:
        return _regularize_calendar_frame_typed(
            norm, gb_cols, time_col, return_col, exec_price_expr
        )
    return norm


def _prepare_signal_and_calendar(
    df: pl.DataFrame | pl.LazyFrame,
    sig_expr: pl.Expr,
    weight_scheme: Literal["raw", "dollar_neutral", "long_only", "equal_weight"],
    time_col: str | None,
    regularize_calendar: bool,
    gb_cols: list[str] | None,
    return_col: str | None,
    exec_price_expr: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """Normalize signal weights and optionally regularize onto a dense calendar grid."""
    if isinstance(df, pl.LazyFrame):
        return _prepare_step0_typed(
            df,
            sig_expr,
            weight_scheme,
            time_col,
            regularize_calendar,
            gb_cols,
            return_col,
            exec_price_expr,
        )
    return _prepare_step0_typed(
        df,
        sig_expr,
        weight_scheme,
        time_col,
        regularize_calendar,
        gb_cols,
        return_col,
        exec_price_expr,
    )


def _resolve_exec_price_expr(
    execution_price: Literal["close", "open", "twap", "vwap"],
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pl.Expr:
    """Resolve the bar execution price benchmark expression."""
    if execution_price == "open":
        return pl.col(open_col).cast(pl.Float64)
    if execution_price == "twap":
        return (
            pl.col(open_col).cast(pl.Float64)
            + pl.col(high_col).cast(pl.Float64)
            + pl.col(low_col).cast(pl.Float64)
            + pl.col(close_col).cast(pl.Float64)
        ) / 4.0
    if execution_price == "vwap":
        return (
            pl.col(high_col).cast(pl.Float64)
            + pl.col(low_col).cast(pl.Float64)
            + pl.col(close_col).cast(pl.Float64)
        ) / 3.0
    return pl.col(close_col).cast(pl.Float64)


def _empty_summary(initial_capital: float) -> dict[str, float]:
    """Return zeroed performance summary metrics for empty backtests."""
    return {
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
