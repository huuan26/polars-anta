"""Trend moving-average indicators for Polars."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..core import IntoExpr, check_period, check_periods, parse_expr


def sma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Simple Moving Average (SMA).

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    check_period(period)
    return parse_expr(close).rolling_mean(window_size=period)


def ema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Exponential Moving Average (EMA).

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    check_period(period)
    return parse_expr(close).ewm_mean(span=period, adjust=False)


def wma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Weighted Moving Average (WMA).

    Weights linearly increase: [1, 2, ..., period].

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    check_period(period)
    if period == 1:
        return parse_expr(close)

    total_weight = (period * (period + 1)) // 2
    c = parse_expr(close)
    weighted_sum = sum(c.shift(period - weight) * weight for weight in range(1, period + 1))
    return weighted_sum / total_weight


def hma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Hull Moving Average (HMA).

    Formula:
        HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    check_period(period)
    c = parse_expr(close)
    diff = 2.0 * wma(c, max(1, period // 2)) - wma(c, period)
    return wma(diff, max(1, int(math.sqrt(period))))


def dema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Double Exponential Moving Average (DEMA).

    Formula:
        DEMA = 2 * EMA(close, n) - EMA(EMA(close, n), n)

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    check_period(period)
    e1 = ema(close, period=period)
    return 2.0 * e1 - ema(e1, period=period)


def tema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Triple Exponential Moving Average (TEMA).

    Formula:
        TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        where EMA1 = EMA(close), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)

    :param close: price column or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    check_period(period)
    e1 = ema(close, period=period)
    e2 = ema(e1, period=period)
    return 3.0 * e1 - 3.0 * e2 + ema(e2, period=period)


def zlema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Zero-Lag Exponential Moving Average (ZLEMA).

    Formula:
        lag = (period - 1) // 2
        ZLEMA = EMA(2 * close - close.shift(lag), period)

    :param close: price column or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    check_period(period)
    c = parse_expr(close)
    return ema(2.0 * c - c.shift((period - 1) // 2), period=period)


def t3(close: IntoExpr, period: int = 5, volume_factor: float = 0.7) -> pl.Expr:
    """Tillson T3 Moving Average.

    Smoothed moving average reducing lag via triple DEMA with volume damping.

    :param close: price column or expression
    :param period: smoothing period (default 5)
    :param volume_factor: volume factor between 0.0 and 1.0 (default 0.7)
    :return: Polars expression
    """
    check_period(period)
    v = float(volume_factor)
    c1 = -(v**3)
    c2 = 3.0 * (v**2) + 3.0 * (v**3)
    c3 = -6.0 * (v**2) - 3.0 * v - 3.0 * (v**3)
    c4 = 1.0 + 3.0 * v + (v**3) + 3.0 * (v**2)

    e1 = ema(close, period=period)
    e2 = ema(e1, period=period)
    e3 = ema(e2, period=period)
    e4 = ema(e3, period=period)
    e5 = ema(e4, period=period)
    e6 = ema(e5, period=period)

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def alma(
    close: IntoExpr,
    period: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> pl.Expr:
    """Arnaud Legoux Moving Average (ALMA).

    Gaussian distribution weighted moving average with minimal overshoot.

    :param close: price column or expression
    :param period: lookback window size (default 9)
    :param offset: trade-off between smoothness and responsiveness (default 0.85)
    :param sigma: filter sharpness parameter (default 6.0)
    :return: Polars expression
    """
    check_period(period)
    m = float(offset * (period - 1))
    s = float(period / sigma)

    weights = [math.exp(-((i - m) ** 2) / (2.0 * s * s)) for i in range(period)]
    c = parse_expr(close)
    weighted_sum = sum(c.shift(period - 1 - i) * w for i, w in enumerate(weights))
    return weighted_sum / sum(weights)


def kama(
    close: IntoExpr,
    period: int = 10,
    fast_span: int = 2,
    slow_span: int = 30,
) -> pl.Expr:
    """Kaufman's Adaptive Moving Average (KAMA).

    Moving average that automatically adapts to market volatility and noise.

    :param close: price column or expression
    :param period: efficiency ratio lookback period (default 10)
    :param fast_span: fastest EMA span (default 2)
    :param slow_span: slowest EMA span (default 30)
    :return: Polars expression
    """
    check_period(period)

    c = parse_expr(close)
    fast_sc = 2.0 / (fast_span + 1)
    slow_sc = 2.0 / (slow_span + 1)

    change = (c - c.shift(period)).abs()
    volatility = c.diff(1).abs().rolling_sum(window_size=period)

    er = pl.when(volatility > 0.0).then(change / volatility).otherwise(0.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    def _calc_kama(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        c_vals = df_chunk["close"].to_numpy()
        sc_vals = df_chunk["sc"].to_numpy()
        n = len(c_vals)

        kama_vals = np.full(n, np.nan, dtype=np.float64)
        valid_idx = np.flatnonzero((~np.isnan(c_vals)) & (~np.isnan(sc_vals)))
        if valid_idx.size == 0:
            return pl.Series(kama_vals).fill_nan(None)

        first_valid = int(valid_idx[0])
        current = c_vals[first_valid]
        kama_vals[first_valid] = current

        for i in range(first_valid + 1, n):
            c_i = c_vals[i]
            sc_i = slow_sc if np.isnan(sc_vals[i]) else sc_vals[i]
            if not np.isnan(c_i):
                current = current + sc_i * (c_i - current)
                kama_vals[i] = current

        return pl.Series(kama_vals).fill_nan(None)

    combined = pl.struct([c.alias("close"), sc.alias("sc")])
    return combined.map_batches(_calc_kama, return_dtype=pl.Float64)


def mcginley_dynamic(
    close: IntoExpr,
    period: int = 10,
    k: float = 0.6,
) -> pl.Expr:
    """McGinley Dynamic tracking average.

    Adjusts dynamically for shifts in market speed and avoids price separation.

    :param close: price column or expression
    :param period: lookback period (default 10)
    :param k: constant factor (default 0.6)
    :return: Polars expression
    """
    check_period(period)
    c = parse_expr(close)

    def _calc_md(s: pl.Series) -> pl.Series:
        vals = s.to_numpy()
        n = len(vals)
        md_vals = np.full(n, np.nan, dtype=np.float64)

        valid_idx = np.flatnonzero(~np.isnan(vals))
        if valid_idx.size == 0:
            return pl.Series(md_vals).fill_nan(None)

        first_valid = int(valid_idx[0])
        curr = vals[first_valid]
        md_vals[first_valid] = curr

        for i in range(first_valid + 1, n):
            c_i = vals[i]
            if not np.isnan(c_i) and curr > 0.0:
                ratio = c_i / curr
                curr = curr + (c_i - curr) / (k * float(period) * (ratio**4))
                md_vals[i] = curr

        return pl.Series(md_vals).fill_nan(None)

    return c.map_batches(_calc_md, return_dtype=pl.Float64)


def vidya(
    close: IntoExpr,
    period: int = 9,
    cmo_period: int = 9,
) -> pl.Expr:
    """Variable Index Dynamic Average (VIDYA).

    Exponential moving average dynamically adjusted by the Chande Momentum Oscillator (CMO).

    :param close: price column or expression
    :param period: EMA lookback period (default 9)
    :param cmo_period: CMO lookback period (default 9)
    :return: Polars expression
    """
    check_periods(period, cmo_period)

    c = parse_expr(close)
    diff = c.diff(1)
    sum_gain = pl.when(diff > 0.0).then(diff).otherwise(0.0).rolling_sum(window_size=cmo_period)
    sum_loss = pl.when(diff < 0.0).then(-diff).otherwise(0.0).rolling_sum(window_size=cmo_period)
    sum_total = sum_gain + sum_loss

    cmo = pl.when(sum_total > 0.0).then(((sum_gain - sum_loss) / sum_total).abs()).otherwise(0.0)
    alpha = (2.0 / (period + 1)) * cmo

    def _calc_vidya(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        c_vals = df_chunk["close"].to_numpy()
        a_vals = df_chunk["alpha"].to_numpy()
        n = len(c_vals)

        vidya_vals = np.full(n, np.nan, dtype=np.float64)
        valid_idx = np.flatnonzero(~np.isnan(c_vals))
        if valid_idx.size == 0:
            return pl.Series(vidya_vals).fill_nan(None)

        first_valid = int(valid_idx[0])
        curr = c_vals[first_valid]
        vidya_vals[first_valid] = curr

        for i in range(first_valid + 1, n):
            c_i = c_vals[i]
            a_i = 0.0 if np.isnan(a_vals[i]) else a_vals[i]
            if not np.isnan(c_i):
                curr = a_i * c_i + (1.0 - a_i) * curr
                vidya_vals[i] = curr

        return pl.Series(vidya_vals).fill_nan(None)

    combined = pl.struct([c.alias("close"), alpha.alias("alpha")])
    return combined.map_batches(_calc_vidya, return_dtype=pl.Float64)
