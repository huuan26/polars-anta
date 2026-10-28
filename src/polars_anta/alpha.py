"""Alpha factor primitives and cross-sectional / time-series signal transforms for Polars."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from .core import IntoExpr, parse_expr
from .stats import rolling_corr, rolling_cov
from .trend import wma
from .volume import vwap as _calc_vwap


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
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= period:
            windows = np.lib.stride_tricks.sliding_window_view(vals, period)
            valid_mask = ~np.isnan(windows).any(axis=1)
            if valid_mask.any():
                valid_w = windows[valid_mask]
                counts = np.sum(valid_w <= valid_w[:, -1:], axis=1, dtype=np.float64)
                out_slice = out[period - 1 :]
                out_slice[valid_mask] = counts / float(period)
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
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= period:
            windows = np.lib.stride_tricks.sliding_window_view(vals, period)
            valid_mask = ~np.isnan(windows).any(axis=1)
            if valid_mask.any():
                valid_w = windows[valid_mask]
                idx_from_end = np.argmax(valid_w[:, ::-1], axis=1).astype(np.float64)
                out_slice = out[period - 1 :]
                out_slice[valid_mask] = idx_from_end
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
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= period:
            windows = np.lib.stride_tricks.sliding_window_view(vals, period)
            valid_mask = ~np.isnan(windows).any(axis=1)
            if valid_mask.any():
                valid_w = windows[valid_mask]
                idx_from_end = np.argmin(valid_w[:, ::-1], axis=1).astype(np.float64)
                out_slice = out[period - 1 :]
                out_slice[valid_mask] = idx_from_end
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
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)

        if n >= period:
            windows = np.lib.stride_tricks.sliding_window_view(vals, period)
            valid_mask = ~np.isnan(windows).any(axis=1)
            if valid_mask.any():
                valid_w = windows[valid_mask]
                means = np.mean(valid_w, axis=1, keepdims=True)
                cum_dev = np.cumsum(valid_w - means, axis=1)
                r_vals = np.max(cum_dev, axis=1) - np.min(cum_dev, axis=1)
                s_vals = np.std(valid_w, axis=1, ddof=1)

                h_vals = np.full(len(valid_w), np.nan, dtype=np.float64)
                flat_mask = s_vals <= 1e-12
                pos_mask = (s_vals > 1e-12) & (r_vals > 1e-12)
                h_vals[flat_mask] = 0.5
                if pos_mask.any():
                    h_vals[pos_mask] = np.clip(
                        np.log(r_vals[pos_mask] / s_vals[pos_mask]) / log_half_n,
                        0.0,
                        1.0,
                    )
                out_slice = out[period - 1 :]
                out_slice[valid_mask] = h_vals

        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_hurst, return_dtype=pl.Float64)


def cs_rank(
    expr: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
    method: Literal["average", "min", "max", "dense", "ordinal", "random"] = "average",
    percentile: bool = True,
) -> pl.Expr:
    """Cross-Sectional Rank (WorldQuant Alpha primitive 'rank(x)').

    Computes the cross-sectional rank of values across assets at each point in time.
    When `percentile=True`, scales the ranks between 1/N and 1.0 (or 0.0 to 1.0).

    :param expr: Feature column or expression to rank.
    :param over: Column name(s) identifying the cross-section window (typically timestamp/date).
    :param method: Ranking method for ties ('average', 'min', 'max', 'dense', 'ordinal', 'random').
    :param percentile: If True, returns percentile rank in (0, 1]; if False, returns integer rank.
    :return: Polars expression.
    """
    e = parse_expr(expr)
    if over is not None:
        over_cols = [over] if isinstance(over, str) else list(over)
        ranked = e.rank(method=method).over(over_cols)
        if percentile:
            count = e.count().over(over_cols)
            return ranked / count
        return ranked
    else:
        ranked = e.rank(method=method)
        if percentile:
            count = e.count()
            return ranked / count
        return ranked


def neutralize(
    expr: IntoExpr,
    by: IntoExpr | Sequence[str],
    over: IntoExpr | Sequence[str] | None = None,
    standardize: bool = False,
) -> pl.Expr:
    """Group / Industry / Sector Neutralization (WorldQuant Alpha primitive 'indneutralize(x, g)').

    Demeans (or standardizes) an alpha signal within categorical groups (e.g. industry, sector),
    ensuring the resulting signal is dollar-neutral or risk-neutral with respect to that classification.

    :param expr: Alpha signal column or expression to neutralize.
    :param by: Grouping column(s) representing industry, sector, or market-cap classification.
    :param over: Optional cross-sectional time column (e.g., 'timestamp' or 'date').
    :param standardize: When True, also divides by group standard deviation to yield z-scores.
    :return: Polars expression.
    """
    e = parse_expr(expr)
    group_cols: list[str] = []
    if over is not None:
        if isinstance(over, str):
            group_cols.append(over)
        else:
            group_cols.extend(over)
    if isinstance(by, str):
        group_cols.append(by)
    else:
        group_cols.extend(by)

    mean = e.mean().over(group_cols)
    demeaned = e - mean
    if standardize:
        std = e.std().over(group_cols)
        return pl.when(std == 0.0).then(0.0).otherwise(demeaned / std)
    return demeaned


def ts_scale(
    expr: IntoExpr,
    a: float = 1.0,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """Scale Primitive (WorldQuant Alpha primitive 'scale(x, a)').

    Rescales values such that the sum of their absolute values equals `a`:
    `a * x / sum(abs(x))`. Commonly used to size portfolio positions or L1 normalize.

    :param expr: Signal column or expression to scale.
    :param a: Target sum of absolute values (default 1.0).
    :param over: Optional cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    e = parse_expr(expr)
    if over is not None:
        over_cols = [over] if isinstance(over, str) else list(over)
        abs_sum = e.abs().sum().over(over_cols)
    else:
        abs_sum = e.abs().sum()
    return pl.when(abs_sum == 0.0).then(0.0).otherwise((float(a) * e) / abs_sum)


def ts_delta(
    expr: IntoExpr,
    period: int = 1,
) -> pl.Expr:
    """Time-series Delta (WorldQuant Alpha primitive 'delta(x, d)').

    Computes x[t] - x[t - period].

    :param expr: Feature column or expression.
    :param period: Lag distance (default 1).
    :return: Polars expression.
    """
    e = parse_expr(expr)
    return e - e.shift(period)


def ts_delay(
    expr: IntoExpr,
    period: int = 1,
) -> pl.Expr:
    """Time-series Delay (WorldQuant Alpha primitive 'delay(x, d)').

    Returns the lagged value x[t - period].

    :param expr: Feature column or expression.
    :param period: Lag distance (default 1).
    :return: Polars expression.
    """
    e = parse_expr(expr)
    return e.shift(period)


def signed_power(
    expr: IntoExpr,
    exponent: float = 2.0,
) -> pl.Expr:
    """Signed Power (WorldQuant Alpha primitive 'signed_power(x, a)').

    Computes sign(x) * (|x| ** exponent). Preserves the direction/sign of the signal
    while non-linearly expanding or contracting magnitudes.

    :param expr: Signal column or expression.
    :param exponent: Exponent power (default 2.0).
    :return: Polars expression.
    """
    e = parse_expr(expr)
    return e.sign() * (e.abs() ** float(exponent))


def ts_min(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Minimum (WorldQuant Alpha primitive 'ts_min(x, d)').

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    e = parse_expr(expr)
    return e.rolling_min(window_size=period)


def ts_max(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Maximum (WorldQuant Alpha primitive 'ts_max(x, d)').

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    e = parse_expr(expr)
    return e.rolling_max(window_size=period)


def ts_sum(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Sum (WorldQuant Alpha primitive 'ts_sum(x, d)').

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    e = parse_expr(expr)
    return e.rolling_sum(window_size=period)


def ts_std(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Standard Deviation (WorldQuant Alpha primitive 'ts_std(x, d)').

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    e = parse_expr(expr)
    return e.rolling_std(window_size=period)


def ts_correlation(
    x: IntoExpr,
    y: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Correlation (WorldQuant Alpha primitive 'correlation(x, y, d)').

    Rolling Pearson correlation between x and y over lookback window.

    :param x: First feature column or expression.
    :param y: Second feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    return rolling_corr(x, y, period=period)


def ts_covariance(
    x: IntoExpr,
    y: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Covariance (WorldQuant Alpha primitive 'covariance(x, y, d)').

    Rolling sample covariance between x and y over lookback window.

    :param x: First feature column or expression.
    :param y: Second feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    return rolling_cov(x, y, period=period)


def ts_product(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Product (WorldQuant Alpha primitive 'product(x, d)').

    Computes the rolling product of values over lookback window.

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    e = parse_expr(expr)

    def _calc_prod(s: pl.Series) -> pl.Series:
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= period:
            windows = np.lib.stride_tricks.sliding_window_view(vals, period)
            valid_mask = ~np.isnan(windows).any(axis=1)
            if valid_mask.any():
                valid_w = windows[valid_mask]
                out_slice = out[period - 1 :]
                out_slice[valid_mask] = np.prod(valid_w, axis=1)
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return e.map_batches(_calc_prod, return_dtype=pl.Float64)


def cs_zscore(
    expr: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
    clip_bound: float | None = None,
) -> pl.Expr:
    """Cross-Sectional Z-Score Standardization.

    Standardizes an alpha signal across the asset universe at each point in time
    by subtracting the cross-sectional mean and dividing by the cross-sectional standard deviation.

    :param expr: Feature column or expression to standardize.
    :param over: Column name(s) identifying the cross-section window (typically 'timestamp' or 'date').
    :param clip_bound: Optional symmetric bound (e.g. 3.0) to clip standardized z-scores into [-clip_bound, clip_bound].
    :return: Polars expression.
    """
    if clip_bound is not None and clip_bound <= 0.0:
        raise ValueError("clip_bound must be a positive number greater than zero")

    e = parse_expr(expr)
    if over is not None:
        over_cols = [over] if isinstance(over, str) else list(over)
        mean = e.mean().over(over_cols)
        std = e.std().over(over_cols)
    else:
        mean = e.mean()
        std = e.std()

    z = pl.when(std == 0.0).then(0.0).otherwise((e - mean) / std)
    if clip_bound is not None:
        b = float(clip_bound)
        z = z.clip(lower_bound=-b, upper_bound=b)
    return z


def cs_winsorize(
    expr: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
    method: Literal["quantile", "zscore", "mad"] = "quantile",
    lower: float = 0.01,
    upper: float = 0.99,
    threshold: float = 3.0,
) -> pl.Expr:
    """Cross-Sectional Winsorization (Outlier Clipping).

    Clips extreme cross-sectional outliers across the asset universe at each point in time
    using empirical quantiles, standard deviation bounds, or Median Absolute Deviation (MAD).

    :param expr: Feature column or expression to winsorize.
    :param over: Column name(s) identifying the cross-section window (typically 'timestamp' or 'date').
    :param method: Outlier detection method: 'quantile', 'zscore', or 'mad' (default 'quantile').
    :param lower: Lower quantile bound when method='quantile' (default 0.01).
    :param upper: Upper quantile bound when method='quantile' (default 0.99).
    :param threshold: Multiplier k for 'zscore' (mean +/- k*std) or 'mad' (median +/- k*1.4826*MAD) (default 3.0).
    :return: Polars expression.
    """
    if method not in ("quantile", "zscore", "mad"):
        raise ValueError("method must be one of 'quantile', 'zscore', or 'mad'")
    if method == "quantile" and not (0.0 <= lower < upper <= 1.0):
        raise ValueError("lower and upper quantiles must satisfy 0.0 <= lower < upper <= 1.0")
    if method in ("zscore", "mad") and threshold <= 0.0:
        raise ValueError("threshold must be a positive number greater than zero")

    e = parse_expr(expr)
    over_cols = ([over] if isinstance(over, str) else list(over)) if over is not None else None

    if method == "quantile":
        lo_expr = e.quantile(lower, interpolation="linear")
        hi_expr = e.quantile(upper, interpolation="linear")
        if over_cols is not None:
            lo_expr = lo_expr.over(over_cols)
            hi_expr = hi_expr.over(over_cols)
    elif method == "zscore":
        mean_expr = e.mean()
        std_expr = e.std().fill_null(0.0)
        if over_cols is not None:
            mean_expr = mean_expr.over(over_cols)
            std_expr = std_expr.over(over_cols)
        lo_expr = mean_expr - float(threshold) * std_expr
        hi_expr = mean_expr + float(threshold) * std_expr
    else:
        med_expr = e.median()
        if over_cols is not None:
            med_expr = med_expr.over(over_cols)
        abs_dev = (e - med_expr).abs()
        mad_expr = abs_dev.median()
        if over_cols is not None:
            mad_expr = mad_expr.over(over_cols)
        scaled_mad = 1.4826 * float(threshold) * mad_expr
        lo_expr = med_expr - scaled_mad
        hi_expr = med_expr + scaled_mad

    return pl.when(e < lo_expr).then(lo_expr).when(e > hi_expr).then(hi_expr).otherwise(e)


def amihud_illiquidity(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
    scale: float = 1e6,
) -> pl.Expr:
    """Amihud (2002) Illiquidity Ratio.

    Measures price impact per unit of dollar trading volume over a rolling window:
    ILLIQ = RollingMean(|Return_t| / (Close_t * Volume_t), period) * scale.

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param period: Rolling lookback window (default 20).
    :param scale: Scaling factor (default 1e6, measuring return impact per $1M traded).
    :return: Polars expression.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    c = parse_expr(close)
    v = parse_expr(volume)
    abs_ret = c.pct_change().abs()
    dollar_vol = c * v
    ratio = pl.when(dollar_vol > 0.0).then((abs_ret / dollar_vol) * float(scale)).otherwise(None)
    return ratio.rolling_mean(window_size=period)


def roll_spread(
    close: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Roll (1984) Implicit Effective Bid-Ask Spread Estimator.

    Estimates the effective proportional bid-ask spread from the first-order serial
    autocovariance of log price changes:
    Spread = 2 * sqrt(max(0, -Cov(Delta ln P_t, Delta ln P_{t-1}))).

    :param close: Close price column or expression.
    :param period: Rolling lookback window (default 20).
    :return: Polars expression producing non-negative proportional spread estimates.
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")

    c = parse_expr(close)
    log_ret = (c / c.shift(1)).log()
    lag_log_ret = log_ret.shift(1)
    serial_cov = rolling_cov(log_ret, lag_log_ret, period=period)
    return (
        pl.when(serial_cov.is_null())
        .then(None)
        .otherwise(2.0 * (-serial_cov).clip(lower_bound=0.0).sqrt())
    )


def corwin_schultz_spread(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Corwin & Schultz (2012) High-Low Effective Bid-Ask Spread Estimator.

    Estimates the proportional bid-ask spread from single-bar and two-bar High/Low price ranges,
    separating transient bid-ask bounce from true price volatility without requiring orderbook data.

    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param period: Rolling smoothing window (default 20).
    :return: Polars expression producing non-negative proportional spread estimates.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)

    hl_log_sq = (hi / lo).log() ** 2
    beta = hl_log_sq + hl_log_sq.shift(1)

    two_day_hi = pl.max_horizontal(hi, hi.shift(1))
    two_day_lo = pl.min_horizontal(lo, lo.shift(1))
    gamma = (two_day_hi / two_day_lo).log() ** 2

    denom = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = ((math.sqrt(2.0) - 1.0) * beta.sqrt() / denom) - (gamma / denom).sqrt()
    exp_alpha = alpha.exp()
    raw_spread = ((2.0 * (exp_alpha - 1.0)) / (1.0 + exp_alpha)).clip(lower_bound=0.0)
    return raw_spread.rolling_mean(window_size=period)


def kyle_lambda(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
    scale: float = 1e6,
) -> pl.Expr:
    """Kyle (1985) Price Impact Coefficient (Lambda).

    Estimates the market impact cost (slope of price change per unit of signed order flow)
    over a rolling window using signed volume as an order flow imbalance proxy.

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param period: Rolling regression window (default 20).
    :param scale: Scaling multiplier (default 1e6).
    :return: Polars expression.
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")

    c = parse_expr(close)
    v = parse_expr(volume)
    ret = c.pct_change()
    signed_vol = ret.sign() * v
    cov_rv = rolling_cov(ret, signed_vol, period=period)
    var_v = signed_vol.rolling_var(window_size=period)
    return (
        pl.when(var_v.is_null() | cov_rv.is_null())
        .then(None)
        .when(var_v > 0.0)
        .then((cov_rv / var_v) * float(scale))
        .otherwise(0.0)
    )


def alpha_001(
    close: IntoExpr,
    returns: IntoExpr | None = None,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #1.

    Formula:
    `rank(ts_argmax(signed_power(returns < 0 ? ts_std(returns, 20) : close, 2), 5)) - 0.5`

    :param close: Close price column or expression.
    :param returns: Optional returns column or expression; if None, computed as close.pct_change().
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    c = parse_expr(close)
    ret = parse_expr(returns) if returns is not None else c.pct_change()
    ret_std = ret.rolling_std(window_size=20)
    inner = pl.when(ret < 0.0).then(ret_std).otherwise(c)
    pow_inner = signed_power(inner, 2.0)
    argmax_inner = ts_argmax(pow_inner, period=5)
    return cs_rank(argmax_inner, over=over) - 0.5


def alpha_002(
    open_: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #2.

    Formula:
    `-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)`

    :param open_: Open price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    o = parse_expr(open_)
    c = parse_expr(close)
    v = parse_expr(volume)
    x = cs_rank(ts_delta(v.log(), 2), over=over)
    y = cs_rank((c - o) / o, over=over)
    return -1.0 * ts_correlation(x, y, period=6)


def alpha_003(
    open_: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #3.

    Formula:
    `-1 * correlation(rank(open), rank(volume), 10)`

    :param open_: Open price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    o = parse_expr(open_)
    v = parse_expr(volume)
    return -1.0 * ts_correlation(cs_rank(o, over=over), cs_rank(v, over=over), period=10)


def alpha_004(
    low: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #4.

    Formula:
    `-1 * ts_rank(rank(low), 9)`

    :param low: Low price column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    lo = parse_expr(low)
    return -1.0 * ts_rank(cs_rank(lo, over=over), period=9)


def alpha_005(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #5.

    Formula:
    `rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))`

    :param open_: Open price column or expression.
    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    o = parse_expr(open_)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)
    vwap_val = _calc_vwap(hi, lo, c, v)
    part1 = cs_rank(o - (ts_sum(vwap_val, 10) / 10.0), over=over)
    part2 = -1.0 * cs_rank(c - vwap_val, over=over).abs()
    return part1 * part2


def alpha_006(
    open_: IntoExpr,
    volume: IntoExpr,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #6.

    Formula:
    `-1 * correlation(open, volume, 10)`

    :param open_: Open price column or expression.
    :param volume: Volume column or expression.
    :return: Polars expression.
    """
    o = parse_expr(open_)
    v = parse_expr(volume)
    return -1.0 * ts_correlation(o, v, period=10)


def alpha_007(
    close: IntoExpr,
    volume: IntoExpr,
    adv_period: int = 20,
    rank_period: int = 60,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #7.

    Formula:
    `(adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1)`

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param adv_period: Lookback window for average daily volume (default 20).
    :param rank_period: Lookback window for ts_rank (default 60).
    :return: Polars expression.
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    adv = v.rolling_mean(window_size=adv_period)
    d7 = ts_delta(c, 7)
    active = (-1.0 * ts_rank(d7.abs(), period=rank_period)) * d7.sign()
    return pl.when(adv.is_null()).then(None).when(adv < v).then(active).otherwise(-1.0)


def alpha_008(
    open_: IntoExpr,
    close: IntoExpr,
    returns: IntoExpr | None = None,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #8.

    Formula:
    `-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))`

    :param open_: Open price column or expression.
    :param close: Close price column or expression.
    :param returns: Optional returns column or expression; if None, computed as close.pct_change().
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    o = parse_expr(open_)
    c = parse_expr(close)
    ret = parse_expr(returns) if returns is not None else c.pct_change()
    prod = ts_sum(o, 5) * ts_sum(ret, 5)
    return -1.0 * cs_rank(prod - ts_delay(prod, 10), over=over)


def alpha_009(
    close: IntoExpr,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #9.

    Formula:
    `(0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))`

    :param close: Close price column or expression.
    :return: Polars expression.
    """
    c = parse_expr(close)
    d1 = ts_delta(c, 1)
    min5 = ts_min(d1, 5)
    max5 = ts_max(d1, 5)
    return (
        pl.when(min5.is_null() | max5.is_null())
        .then(None)
        .when((min5 > 0.0) | (max5 < 0.0))
        .then(d1)
        .otherwise(-1.0 * d1)
    )


def alpha_010(
    close: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #10.

    Formula:
    `rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))`

    :param close: Close price column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    c = parse_expr(close)
    d1 = ts_delta(c, 1)
    min4 = ts_min(d1, 4)
    max4 = ts_max(d1, 4)
    inner = (
        pl.when(min4.is_null() | max4.is_null())
        .then(None)
        .when((min4 > 0.0) | (max4 < 0.0))
        .then(d1)
        .otherwise(-1.0 * d1)
    )
    return cs_rank(inner, over=over)


def alpha_011(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #11.

    Formula:
    `((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))`

    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)
    vwap_val = _calc_vwap(hi, lo, c, v)
    diff = vwap_val - c
    return (cs_rank(ts_max(diff, 3), over=over) + cs_rank(ts_min(diff, 3), over=over)) * cs_rank(
        ts_delta(v, 3), over=over
    )


def alpha_012(
    close: IntoExpr,
    volume: IntoExpr,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #12.

    Formula:
    `sign(delta(volume, 1)) * (-1 * delta(close, 1))`

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :return: Polars expression.
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    return ts_delta(v, 1).sign() * (-1.0 * ts_delta(c, 1))


def alpha_013(
    close: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #13.

    Formula:
    `-1 * rank(covariance(rank(close), rank(volume), 5))`

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    cov = ts_covariance(cs_rank(c, over=over), cs_rank(v, over=over), period=5)
    return -1.0 * cs_rank(cov, over=over)


def alpha_014(
    open_: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    returns: IntoExpr | None = None,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #14.

    Formula:
    `((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))`

    :param open_: Open price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param returns: Optional returns column or expression; if None, computed as close.pct_change().
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    o = parse_expr(open_)
    c = parse_expr(close)
    v = parse_expr(volume)
    ret = parse_expr(returns) if returns is not None else c.pct_change()
    return (-1.0 * cs_rank(ts_delta(ret, 3), over=over)) * ts_correlation(o, v, period=10)


def alpha_015(
    high: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #15.

    Formula:
    `-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)`

    :param high: High price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    hi = parse_expr(high)
    v = parse_expr(volume)
    corr = ts_correlation(cs_rank(hi, over=over), cs_rank(v, over=over), period=3)
    return -1.0 * ts_sum(cs_rank(corr, over=over), period=3)


def alpha_016(
    high: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #16.

    Formula:
    `-1 * rank(covariance(rank(high), rank(volume), 5))`

    :param high: High price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    hi = parse_expr(high)
    v = parse_expr(volume)
    cov = ts_covariance(cs_rank(hi, over=over), cs_rank(v, over=over), period=5)
    return -1.0 * cs_rank(cov, over=over)


def alpha_017(
    close: IntoExpr,
    volume: IntoExpr,
    adv_period: int = 20,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #17.

    Formula:
    `(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))`

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param adv_period: Lookback window for average daily volume (default 20).
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    adv = v.rolling_mean(window_size=adv_period)
    p1 = -1.0 * cs_rank(ts_rank(c, 10), over=over)
    p2 = cs_rank(ts_delta(ts_delta(c, 1), 1), over=over)
    p3 = cs_rank(ts_rank(v / adv, 5), over=over)
    return p1 * p2 * p3


def alpha_018(
    open_: IntoExpr,
    close: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #18.

    Formula:
    `-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))`

    :param open_: Open price column or expression.
    :param close: Close price column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    o = parse_expr(open_)
    c = parse_expr(close)
    diff = c - o
    inner = ts_std(diff.abs(), 5) + diff + ts_correlation(c, o, period=10)
    return -1.0 * cs_rank(inner, over=over)


def alpha_019(
    close: IntoExpr,
    returns: IntoExpr | None = None,
    sum_period: int = 250,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #19.

    Formula:
    `((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))`

    :param close: Close price column or expression.
    :param returns: Optional returns column or expression; if None, computed as close.pct_change().
    :param sum_period: Rolling sum window for cumulative returns (default 250).
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    c = parse_expr(close)
    ret = parse_expr(returns) if returns is not None else c.pct_change()
    d7 = ts_delta(c, 7)
    p1 = -1.0 * ((c - ts_delay(c, 7)) + d7).sign()
    p2 = 1.0 + cs_rank(1.0 + ts_sum(ret, period=sum_period), over=over)
    return p1 * p2


def alpha_020(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #20.

    Formula:
    `(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))`

    :param open_: Open price column or expression.
    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    o = parse_expr(open_)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    return (
        (-1.0 * cs_rank(o - ts_delay(hi, 1), over=over))
        * cs_rank(o - ts_delay(c, 1), over=over)
        * cs_rank(o - ts_delay(lo, 1), over=over)
    )


def alpha_021(
    close: IntoExpr,
    volume: IntoExpr,
    adv_period: int = 20,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #21.

    Formula:
    `((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 <= (volume / adv20)) ? 1 : (-1)))))`

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param adv_period: Lookback window for average daily volume (default 20).
    :return: Polars expression.
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    ma8 = ts_sum(c, 8) / 8.0
    std8 = ts_std(c, 8)
    ma2 = ts_sum(c, 2) / 2.0
    adv = v.rolling_mean(window_size=adv_period)
    return (
        pl.when(ma8.is_null() | std8.is_null() | ma2.is_null() | adv.is_null())
        .then(None)
        .when((ma8 + std8) < ma2)
        .then(-1.0)
        .when(ma2 < (ma8 - std8))
        .then(1.0)
        .when((v / adv) >= 1.0)
        .then(1.0)
        .otherwise(-1.0)
    )


def alpha_022(
    high: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #22.

    Formula:
    `-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))`

    :param high: High price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    hi = parse_expr(high)
    c = parse_expr(close)
    v = parse_expr(volume)
    return -1.0 * (ts_delta(ts_correlation(hi, v, period=5), 5) * cs_rank(ts_std(c, 20), over=over))


def alpha_023(
    high: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #23.

    Formula:
    `(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)`

    :param high: High price column or expression.
    :param period: Rolling mean window (default 20).
    :return: Polars expression.
    """
    hi = parse_expr(high)
    ma = ts_sum(hi, period) / float(period)
    d2 = ts_delta(hi, 2)
    return (
        pl.when(ma.is_null() | d2.is_null()).then(None).when(ma < hi).then(-1.0 * d2).otherwise(0.0)
    )


def alpha_024(
    close: IntoExpr,
    period: int = 100,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #24.

    Formula:
    `((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) <= 0.05) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3))))`

    :param close: Close price column or expression.
    :param period: Long-term trend window (default 100).
    :return: Polars expression.
    """
    c = parse_expr(close)
    ma = ts_sum(c, period) / float(period)
    cond_ratio = ts_delta(ma, period) / ts_delay(c, period)
    branch1 = -1.0 * (c - ts_min(c, period))
    branch2 = -1.0 * ts_delta(c, 3)
    return (
        pl.when(cond_ratio.is_null())
        .then(None)
        .when(cond_ratio <= 0.05)
        .then(branch1)
        .otherwise(branch2)
    )


def alpha_025(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    returns: IntoExpr | None = None,
    adv_period: int = 20,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #25.

    Formula:
    `rank(((((-1 * returns) * adv20) * vwap) * (high - close)))`

    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param returns: Optional returns column or expression; if None, computed as close.pct_change().
    :param adv_period: Lookback window for average daily volume (default 20).
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)
    ret = parse_expr(returns) if returns is not None else c.pct_change()
    adv = v.rolling_mean(window_size=adv_period)
    vwap_val = _calc_vwap(hi, lo, c, v)
    inner = ((-1.0 * ret) * adv * vwap_val) * (hi - c)
    return cs_rank(inner, over=over)


def alpha_026(
    high: IntoExpr,
    volume: IntoExpr,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #26.

    Formula:
    `-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)`

    :param high: High price column or expression.
    :param volume: Volume column or expression.
    :return: Polars expression.
    """
    hi = parse_expr(high)
    v = parse_expr(volume)
    corr = ts_correlation(ts_rank(v, 5), ts_rank(hi, 5), period=5)
    return -1.0 * ts_max(corr, 3)


def alpha_027(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #27.

    Formula:
    `((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)`

    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)
    vwap_val = _calc_vwap(hi, lo, c, v)
    corr = ts_correlation(cs_rank(v, over=over), cs_rank(vwap_val, over=over), period=6)
    rk = cs_rank(ts_sum(corr, 2) / 2.0, over=over)
    return pl.when(rk.is_null()).then(None).when(rk > 0.5).then(-1.0).otherwise(1.0)


def alpha_028(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    adv_period: int = 20,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #28.

    Formula:
    `scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))`

    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param adv_period: Lookback window for average daily volume (default 20).
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)
    adv = v.rolling_mean(window_size=adv_period)
    inner = (ts_correlation(adv, lo, period=5) + ((hi + lo) / 2.0)) - c
    return ts_scale(inner, a=1.0, over=over)


def alpha_029(
    close: IntoExpr,
    returns: IntoExpr | None = None,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #29.

    Formula:
    `(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))`

    :param close: Close price column or expression.
    :param returns: Optional returns column or expression; if None, computed as close.pct_change().
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    c = parse_expr(close)
    ret = parse_expr(returns) if returns is not None else c.pct_change()
    r1 = cs_rank(-1.0 * cs_rank(ts_delta(c - 1.0, 5), over=over), over=over)
    m2 = ts_min(r1, 2)
    scaled = ts_scale(m2.clip(lower_bound=1e-12).log(), a=1.0, over=over)
    part1 = ts_min(cs_rank(scaled, over=over), 5)
    part2 = ts_rank(ts_delay(-1.0 * ret, 6), 5)
    return part1 + part2


def alpha_030(
    close: IntoExpr,
    volume: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #30.

    Formula:
    `(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))`

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param over: Cross-sectional grouping column (e.g. 'timestamp').
    :return: Polars expression.
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    s1 = (c - ts_delay(c, 1)).sign()
    s2 = (ts_delay(c, 1) - ts_delay(c, 2)).sign()
    s3 = (ts_delay(c, 2) - ts_delay(c, 3)).sign()
    rk = cs_rank(s1 + s2 + s3, over=over)
    return ((1.0 - rk) * ts_sum(v, 5)) / ts_sum(v, 20)


def alpha_041(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #41.

    Formula:
    `(((high * low) ** 0.5) - vwap(high, low, close, volume))`

    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :return: Polars expression.
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)
    vwap_val = _calc_vwap(hi, lo, c, v)
    return ((hi * lo) ** 0.5) - vwap_val


def alpha_054(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #54.

    Formula:
    `-(low - close) * (open ** 5) / ((low - high) * (close ** 5) + 1e-4)`

    :param open_: Open price column or expression.
    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :return: Polars expression.
    """
    o = parse_expr(open_)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    return -(lo - c) * (o**5) / (((lo - hi) * (c**5)) + 1e-4)


def alpha_101(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """WorldQuant Formulaic Alpha #101.

    Formula:
    `(close - open) / ((high - low) + 1e-4)`

    :param open_: Open price column or expression.
    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param close: Close price column or expression.
    :return: Polars expression.
    """
    o = parse_expr(open_)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    return (c - o) / ((hi - lo) + 1e-4)
