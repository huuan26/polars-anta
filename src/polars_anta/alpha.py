"""Alpha factor primitives and cross-sectional / time-series signal transforms for Polars."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .core import IntoExpr, parse_expr
from .trend import wma


def ts_rank(
    close: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rank (WorldQuant Alpha primitive).

    Computes the rank of the current element within the trailing lookback window,
    scaled between 0.0 (lowest) and 1.0 (highest).

    :param close: price column or expression
    :param period: lookback window (default 10)
    :return: Polars expression
    """
    c = parse_expr(close)

    def _calc_rank(s: pl.Series) -> pl.Series:
        vals = s.to_numpy()
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(period - 1, n):
            window = vals[i - period + 1 : i + 1]
            if np.isnan(window).any():
                continue
            last = window[-1]
            count = np.sum(window <= last)
            out[i] = count / float(period)
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_rank, return_dtype=pl.Float64)


def ts_argmax(
    close: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series ArgMax (Bars ago of maximum value).

    Returns how many bars ago the maximum value occurred within the lookback window
    (0 = today, period - 1 = oldest bar).

    :param close: price column or expression
    :param period: lookback window (default 10)
    :return: Polars expression
    """
    c = parse_expr(close)

    def _calc_argmax(s: pl.Series) -> pl.Series:
        vals = s.to_numpy()
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(period - 1, n):
            window = vals[i - period + 1 : i + 1]
            if np.isnan(window).any():
                continue
            max_val = np.max(window)
            # Find the most recent bar index with this max value
            # window[-1] is 0 bars ago, window[0] is period-1 bars ago
            idx_from_end = 0.0
            for k in range(len(window) - 1, -1, -1):
                if window[k] == max_val:
                    idx_from_end = float(len(window) - 1 - k)
                    break
            out[i] = idx_from_end
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_argmax, return_dtype=pl.Float64)


def ts_argmin(
    close: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series ArgMin (Bars ago of minimum value).

    Returns how many bars ago the minimum value occurred within the lookback window
    (0 = today, period - 1 = oldest bar).

    :param close: price column or expression
    :param period: lookback window (default 10)
    :return: Polars expression
    """
    c = parse_expr(close)

    def _calc_argmin(s: pl.Series) -> pl.Series:
        vals = s.to_numpy()
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(period - 1, n):
            window = vals[i - period + 1 : i + 1]
            if np.isnan(window).any():
                continue
            min_val = np.min(window)
            idx_from_end = 0.0
            for k in range(len(window) - 1, -1, -1):
                if window[k] == min_val:
                    idx_from_end = float(len(window) - 1 - k)
                    break
            out[i] = idx_from_end
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_argmin, return_dtype=pl.Float64)


def decay_linear(
    close: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Linear Weighted Decay (WorldQuant Alpha primitive).

    Computes weighted moving average with linearly decaying weights [1, 2, ..., period].

    :param close: price column or expression
    :param period: lookback window (default 10)
    :return: Polars expression
    """
    c = parse_expr(close)
    return wma(c, period=period)


def ts_zscore(
    close: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Time-series Rolling Z-Score.

    Standardizes a series by subtracting rolling mean and dividing by rolling standard deviation.

    :param close: price column or expression
    :param period: lookback window (default 20)
    :return: Polars expression
    """
    c = parse_expr(close)
    mean = c.rolling_mean(window_size=period)
    std = c.rolling_std(window_size=period)
    return (c - mean) / std


def returns_divergence(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Price-Volume Returns Divergence.

    Calculates the difference between normalized price returns and normalized volume changes:
    ZScore(Return) - ZScore(VolumeChange).

    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback window (default 14)
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)

    c_ret = c.pct_change()
    v_change = v.pct_change()

    c_z = ts_zscore(c_ret, period=period)
    v_z = ts_zscore(v_change, period=period)
    return c_z - v_z


def hl_ratio(
    high: IntoExpr,
    low: IntoExpr,
) -> pl.Expr:
    """Log High-to-Low Ratio.

    Measures intraday volatility / price expansion: ln(High / Low).

    :param high: high price column or expression
    :param low: low price column or expression
    :return: Polars expression
    """
    h = parse_expr(high)
    lo = parse_expr(low)
    return (h / lo).log()


def co_ratio(
    close: IntoExpr,
    open_: IntoExpr,
) -> pl.Expr:
    """Log Close-to-Open Ratio.

    Measures intraday drift / conviction: ln(Close / Open).

    :param close: close price column or expression
    :param open_: open price column or expression
    :return: Polars expression
    """
    c = parse_expr(close)
    o = parse_expr(open_)
    return (c / o).log()


def volume_weighted_momentum(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Volume-Weighted Momentum.

    Measures price rate of change amplified or dampened by relative volume:
    ((Close - Close[t-p]) / Close[t-p]) * (Volume / RollingMean(Volume, p)).

    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback window (default 10)
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)

    roc = (c - c.shift(period)) / c.shift(period)
    v_ratio = v / v.rolling_mean(window_size=period)
    return roc * v_ratio


def relative_spread(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Normalized Relative Spread.

    Calculates the normalized high-low bar range relative to close: (High - Low) / Close.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars expression
    """
    h = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    return (h - lo) / c


def twap(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Rolling Time-Weighted Average Price (TWAP).

    Rolling mean of typical price (High + Low + Close) / 3 over a specified period.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: rolling lookback window (default 20)
    :return: Polars expression
    """
    h = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    typical_price = (h + lo + c) / 3.0
    return typical_price.rolling_mean(window_size=period)


def entropy(
    close: IntoExpr,
    period: int = 20,
    bins: int = 5,
) -> pl.Expr:
    """Rolling Shannon Entropy of Price Returns.

    Quantifies market disorder and information content to identify trending vs choppy regimes:
    Low entropy indicates strong directional trending; high entropy indicates noise and chop.

    :param close: price column or expression
    :param period: lookback window (default 20)
    :param bins: number of histogram bins for return discretization (default 5)
    :return: Polars expression
    """
    c = parse_expr(close)

    def _calc_entropy(s: pl.Series) -> pl.Series:
        vals = s.to_numpy()
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(period, n):
            window = vals[i - period : i + 1]
            if np.isnan(window).any():
                continue
            rets = []
            for j in range(1, len(window)):
                p0 = window[j - 1]
                p1 = window[j]
                if p0 > 0 and p1 > 0:
                    rets.append(math.log(p1 / p0))
            if len(rets) < bins:
                continue
            min_r = min(rets)
            max_r = max(rets)
            if max_r == min_r:
                out[i] = 0.0
                continue
            bin_width = (max_r - min_r) / bins
            counts = [0] * bins
            for r in rets:
                b_idx = min(int((r - min_r) / bin_width), bins - 1)
                counts[b_idx] += 1
            total = len(rets)
            ent = 0.0
            for cnt in counts:
                if cnt > 0:
                    p = cnt / total
                    ent -= p * math.log2(p)
            out[i] = ent
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_entropy, return_dtype=pl.Float64)


def fractal_dimension(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Ehlers' Fractal Dimension (FRAMA component).

    Quantifies the fractal dimension of a price series between 1.0 (pure trend / straight line)
    and 2.0 (pure noise / plane-filling random walk).

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: lookback period (default 20, should be an even integer)
    :return: Polars expression producing fractal dimension values in [1.0, 2.0]
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")

    half_period = max(1, period // 2)
    hi = parse_expr(high)
    lo = parse_expr(low)

    n1 = (
        hi.rolling_max(window_size=half_period) - lo.rolling_min(window_size=half_period)
    ) / float(half_period)
    n2 = (
        hi.rolling_max(window_size=half_period).shift(half_period)
        - lo.rolling_min(window_size=half_period).shift(half_period)
    ) / float(half_period)
    n3 = (hi.rolling_max(window_size=period) - lo.rolling_min(window_size=period)) / float(period)

    sum_n1_n2 = n1 + n2
    log2 = math.log(2.0)

    null_mask = n3.is_null() | (n3 <= 0.0) | (sum_n1_n2 <= 0.0)
    dim = (
        pl.when(null_mask)
        .then(None)
        .otherwise(((sum_n1_n2.log() - n3.log()) / log2).clip(1.0, 2.0))
    )
    return dim


def hurst_exponent(
    close: IntoExpr,
    period: int = 50,
) -> pl.Expr:
    """Rolling Hurst Exponent (Rescaled Range Analysis).

    Quantifies the long-term memory and persistence of a time series:
        - H < 0.5: Mean-reverting / anti-persistent regime
        - H = 0.5: Random walk / Geometric Brownian motion
        - H > 0.5: Trending / persistent regime

    :param close: price column or expression
    :param period: lookback window (default 50)
    :return: Polars expression producing Hurst exponent in [0.0, 1.0]
    """
    if period < 4:
        raise ValueError("period must be an integer greater than or equal to 4")

    c = parse_expr(close)
    log_half_n = math.log(float(period) * 0.5)

    def _calc_hurst(s: pl.Series) -> pl.Series:
        vals = s.to_numpy()
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            w = vals[i - period + 1 : i + 1]
            if np.isnan(w).any():
                continue
            mean_val = np.mean(w)
            cum_dev = np.cumsum(w - mean_val)
            r_val = float(np.max(cum_dev) - np.min(cum_dev))
            s_val = float(np.std(w, ddof=1))
            if s_val > 1e-12 and r_val > 1e-12:
                h_val = math.log(r_val / s_val) / log_half_n
                out[i] = min(max(h_val, 0.0), 1.0)
            elif s_val <= 1e-12:
                out[i] = 0.5

        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_hurst, return_dtype=pl.Float64)
