"""Core utility functions and expression helpers for polars-anta."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

IntoExpr = str | pl.Expr


def parse_expr(column: IntoExpr) -> pl.Expr:
    """Normalize a column name or Polars Expression into a pl.Expr."""
    return pl.col(column) if isinstance(column, str) else column


def check_period(
    period: int,
    min_period: int = 1,
    name: str = "period",
    *,
    inclusive: bool = False,
) -> None:
    """Validate an integer lookback window or period parameter."""
    if period < min_period:
        if min_period == 1:
            raise ValueError(f"{name} must be a positive integer greater than zero")
        if inclusive or min_period >= 4:
            raise ValueError(f"{name} must be an integer greater than or equal to {min_period}")
        raise ValueError(f"{name} must be an integer greater than {min_period - 1}")


def check_periods(*periods: int) -> None:
    """Validate that multiple integer period parameters are strictly positive."""
    if any(p <= 0 for p in periods):
        raise ValueError("periods must be positive integers greater than zero")


def check_positive(value: float, name: str) -> None:
    """Validate that a numeric parameter is strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number greater than zero")


def map_sliding_windows(
    expr: IntoExpr,
    period: int,
    reducer: Callable[[np.ndarray], np.ndarray],
    *,
    skip_nan: bool = True,
) -> pl.Expr:
    """Apply a vectorized 2-D NumPy reducer across zero-copy trailing sliding windows."""
    target = parse_expr(expr)

    def _apply(s: pl.Series) -> pl.Series:
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= period:
            windows = np.lib.stride_tricks.sliding_window_view(vals, period)
            if skip_nan:
                valid_mask = ~np.isnan(windows).any(axis=1)
                if valid_mask.any():
                    out[period - 1 :][valid_mask] = reducer(windows[valid_mask])
            else:
                out[period - 1 :] = reducer(windows)
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return target.map_batches(_apply, return_dtype=pl.Float64)


def wilder_smooth(expr: pl.Expr, period: int) -> pl.Expr:
    """Wilder's exponential smoothing (RMA).

    Uses alpha = 1 / period, mathematically identical to Wilder's Smoothing.
    """
    check_period(period)
    return expr.ewm_mean(alpha=1.0 / float(period), adjust=False)


def true_range(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """True Range (TR).

    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars expression
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    prev_close = c.shift(1)
    return pl.max_horizontal(hi - lo, (hi - prev_close).abs(), (lo - prev_close).abs())
