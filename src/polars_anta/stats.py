"""Quantitative statistics and rolling regression expressions for Polars."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .core import IntoExpr, parse_expr
from .trend import wma as _wma


def zscore(
    close: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Rolling Z-Score.

    Measures standard deviations from the rolling mean: (x - mean) / std.

    :param close: price column or expression
    :param period: rolling lookback window (default 20)
    :return: Polars expression
    """
    c = parse_expr(close)
    mean = c.rolling_mean(window_size=period)
    std = c.rolling_std(window_size=period)
    return (c - mean) / std


def percentile_rank(
    close: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Rolling Percentile Rank.

    Computes the percentile rank of the current value within the trailing window,
    normalized between 0.0 and 1.0.

    :param close: price column or expression
    :param period: rolling lookback window (default 20)
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


def skewness(
    close: IntoExpr,
    period: int = 20,
    bias: bool = False,
) -> pl.Expr:
    """Rolling Skewness.

    Measures the asymmetry of the rolling return or price distribution.
    Computed via pure vectorized Polars polynomial expansion.

    Two estimators are available:

    * ``bias=False`` (default): the adjusted Fisher-Pearson sample skewness
      G1 = sqrt(n * (n - 1)) / (n - 2) * g1, which matches
      ``pandas.Series.skew()`` and ``scipy.stats.skew(..., bias=False)``.
    * ``bias=True``: the biased (population) moment ratio
      g1 = m3 / m2**1.5, which matches ``polars.Series.skew(bias=True)``
      and ``scipy.stats.skew(..., bias=True)``.

    :param close: price column or expression
    :param period: rolling lookback window (default 20)
    :param bias: return the biased population estimator instead of the
        adjusted sample estimator (default False)
    :return: Polars expression
    :raises ValueError: if ``bias`` is False and ``period < 3``
    """
    if not bias and period < 3:
        raise ValueError("skewness requires period >= 3 when bias=False")
    c = parse_expr(close)
    mu = c.rolling_mean(window_size=period)
    mean_sq = (c**2).rolling_mean(window_size=period)
    mean_cube = (c**3).rolling_mean(window_size=period)

    variance = (mean_sq - mu**2).clip(lower_bound=1e-12)
    std = variance.sqrt()
    m3 = mean_cube - 3.0 * mu * mean_sq + 2.0 * (mu**3)
    g1 = m3 / (variance * std)
    if bias:
        return g1
    n = float(period)
    return g1 * (math.sqrt(n * (n - 1.0)) / (n - 2.0))


def kurtosis(
    close: IntoExpr,
    period: int = 20,
    bias: bool = False,
) -> pl.Expr:
    """Rolling Excess Kurtosis.

    Measures the tail heaviness of the rolling distribution relative to
    normal. Computed via pure vectorized Polars polynomial expansion.

    Two estimators are available:

    * ``bias=False`` (default): the adjusted Fisher-Pearson sample excess
      kurtosis G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6),
      which matches ``pandas.Series.kurt()`` and
      ``scipy.stats.kurtosis(..., bias=False)``.
    * ``bias=True``: the biased (population) excess moment ratio
      g2 = m4 / m2**2 - 3, which matches
      ``polars.Series.kurtosis(bias=True)`` and
      ``scipy.stats.kurtosis(..., bias=True)``.

    :param close: price column or expression
    :param period: rolling lookback window (default 20)
    :param bias: return the biased population estimator instead of the
        adjusted sample estimator (default False)
    :return: Polars expression
    :raises ValueError: if ``bias`` is False and ``period < 4``
    """
    if not bias and period < 4:
        raise ValueError("kurtosis requires period >= 4 when bias=False")
    c = parse_expr(close)
    mu = c.rolling_mean(window_size=period)
    mean_sq = (c**2).rolling_mean(window_size=period)
    mean_cube = (c**3).rolling_mean(window_size=period)
    mean_quad = (c**4).rolling_mean(window_size=period)

    variance = (mean_sq - mu**2).clip(lower_bound=1e-12)
    m4 = mean_quad - 4.0 * mu * mean_cube + 6.0 * (mu**2) * mean_sq - 3.0 * (mu**4)
    g2 = (m4 / (variance**2)) - 3.0
    if bias:
        return g2
    n = float(period)
    return ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)


def rolling_cov(
    x: IntoExpr,
    y: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Rolling Sample Covariance between two series.

    :param x: first series column or expression
    :param y: second series column or expression
    :param period: rolling lookback window (default 20)
    :return: Polars expression
    """
    px = parse_expr(x)
    py = parse_expr(y)
    mean_x = px.rolling_mean(window_size=period)
    mean_y = py.rolling_mean(window_size=period)
    mean_xy = (px * py).rolling_mean(window_size=period)
    cov = (mean_xy - mean_x * mean_y) * (period / float(period - 1))
    return cov


def rolling_corr(
    x: IntoExpr,
    y: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Rolling Pearson Correlation Coefficient between two series.

    :param x: first series column or expression
    :param y: second series column or expression
    :param period: rolling lookback window (default 20)
    :return: Polars expression
    """
    px = parse_expr(x)
    py = parse_expr(y)
    cov = rolling_cov(px, py, period=period)
    std_x = px.rolling_std(window_size=period)
    std_y = py.rolling_std(window_size=period)
    return (cov / (std_x * std_y)).clip(lower_bound=-1.0, upper_bound=1.0)


def rolling_beta(
    asset_returns: IntoExpr,
    benchmark_returns: IntoExpr,
    period: int = 60,
) -> pl.Expr:
    """Rolling CAPM Beta.

    Measures the systematic risk of an asset relative to a benchmark: Cov(Ra, Rb) / Var(Rb).

    :param asset_returns: asset return column or expression
    :param benchmark_returns: market benchmark return column or expression
    :param period: rolling lookback window (default 60)
    :return: Polars expression
    """
    ra = parse_expr(asset_returns)
    rb = parse_expr(benchmark_returns)
    cov = rolling_cov(ra, rb, period=period)
    var_b = rb.rolling_var(window_size=period)
    return cov / var_b


def rolling_alpha(
    asset_returns: IntoExpr,
    benchmark_returns: IntoExpr,
    rf: float = 0.0,
    period: int = 60,
) -> pl.Expr:
    """Rolling Jensen's Alpha.

    Measures risk-adjusted excess returns: Alpha = Mean(Ra) - (Rf + Beta * (Mean(Rb) - Rf)).

    :param asset_returns: asset return column or expression
    :param benchmark_returns: market benchmark return column or expression
    :param rf: risk-free rate per period (default 0.0)
    :param period: rolling lookback window (default 60)
    :return: Polars expression
    """
    ra = parse_expr(asset_returns)
    rb = parse_expr(benchmark_returns)
    beta = rolling_beta(ra, rb, period=period)
    mean_ra = ra.rolling_mean(window_size=period)
    mean_rb = rb.rolling_mean(window_size=period)
    return mean_ra - (rf + beta * (mean_rb - rf))


def linear_regression_slope(
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Rolling Ordinary Least Squares (OLS) Regression Slope.

    Computed using vectorized WMA and SMA decomposition without looping:
    Slope = (6 / (N - 1)) * (WMA - SMA).

    :param close: price column or expression
    :param period: lookback window (default 14)
    :return: Polars expression
    """
    c = parse_expr(close)
    sma = c.rolling_mean(window_size=period)
    wma = _wma(c, period=period)
    return (wma - sma) * (6.0 / float(period - 1))


def linear_regression_intercept(
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Rolling Ordinary Least Squares (OLS) Regression Intercept.

    Intercept = 4 * SMA - 3 * WMA.

    :param close: price column or expression
    :param period: lookback window (default 14)
    :return: Polars expression
    """
    c = parse_expr(close)
    sma = c.rolling_mean(window_size=period)
    wma = _wma(c, period=period)
    return 4.0 * sma - 3.0 * wma


def linear_regression_angle(
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Rolling Linear Regression Angle (in degrees).

    Angle = arctan(Slope) * (180 / pi).

    :param close: price column or expression
    :param period: lookback window (default 14)
    :return: Polars expression
    """
    slope = linear_regression_slope(close, period=period)
    return slope.arctan() * (180.0 / math.pi)


def linear_regression_forecast(
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Rolling Linear Regression Forecast at the current bar.

    Forecast = Intercept + Slope * (N - 1) = 3 * WMA - 2 * SMA.

    :param close: price column or expression
    :param period: lookback window (default 14)
    :return: Polars expression
    """
    c = parse_expr(close)
    sma = c.rolling_mean(window_size=period)
    wma = _wma(c, period=period)
    return 3.0 * wma - 2.0 * sma


def linear_regression_r2(
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Closed-form Rolling OLS Coefficient of Determination (R-Squared).

    R-Squared = clip(Slope^2 * N(N+1) / (12 * Var_sample(y)), 0, 1)

    The slope comes from `linear_regression_slope`, which uses the closed-form
    WMA/SMA identity, so no Python loop is involved. The constant N(N+1)/12 is
    the sample variance of the regressor x = 1..N, which keeps R-Squared equal
    to 1.0 on a perfectly linear series.

    A window with zero variance (flat prices) returns 0.0: there is no trend to
    explain, and returning 1.0 there would report a perfect trend on a flat market.

    :param close: price column or expression
    :param period: lookback window, must be greater than 1 (default 14)
    :return: Polars expression in [0.0, 1.0]
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")
    c = parse_expr(close)
    slope = linear_regression_slope(c, period=period)
    var_y = c.rolling_var(window_size=period)
    n = float(period)
    const = n * (n + 1.0) / 12.0
    r2 = slope.pow(2) * (const / var_y)

    return pl.when(var_y > 0.0).then(r2.clip(0.0, 1.0)).otherwise(0.0)


def linear_regression_stderr(
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Closed-form Rolling OLS Standard Error of the Estimate.

    SE = sqrt((N-1)/(N-2) * Var_sample(y) * (1 - R-Squared))

    :param close: price column or expression
    :param period: lookback window, must be greater than 2 (default 14)
    :return: Polars expression, non-negative
    """
    if period <= 2:
        raise ValueError("period must be an integer greater than 2")
    c = parse_expr(close)
    var_y = c.rolling_var(window_size=period)
    r2 = linear_regression_r2(c, period=period)
    n = float(period)
    residual = ((n - 1.0) / (n - 2.0)) * var_y * (1.0 - r2)

    return pl.when(residual > 0.0).then(residual.sqrt()).otherwise(0.0)


def frac_diff(
    close: IntoExpr,
    d: float = 0.4,
    window: int = 20,
) -> pl.Expr:
    """Fixed-Window Fractional Differentiation (FFD — López de Prado, 2018).

    Applies the fractional differencing operator $(1 - B)^d = \\sum_{k=0}^{W-1} \\omega_k B^k$
    with binomial weights $\\omega_0 = 1, \\omega_k = -\\omega_{k-1} \\frac{d - k + 1}{k}$ over
    a fixed trailing window $W$. Achieves time-series stationarity for machine learning
    features while preserving long-range memory ($d \\in (0, 1)$).

    :param close: Price or feature column/expression to fractionally differentiate.
    :param d: Fractional differencing order (default 0.4; ``d=0`` returns original series,
        ``d=1`` returns first difference). Must be non-negative.
    :param window: Fixed lookback window length $W \\ge 2$ (default 20).
    :return: Polars expression.
    """
    if d < 0.0:
        raise ValueError("d must be a non-negative number")
    if window < 2:
        raise ValueError("window must be an integer greater than or equal to 2")

    c = parse_expr(close).cast(pl.Float64)

    k = np.arange(1, window, dtype=np.float64)
    factors = -(float(d) - k + 1.0) / k
    weights = np.empty(window, dtype=np.float64)
    weights[0] = 1.0
    weights[1:] = np.cumprod(factors)
    w_rev = weights[::-1].copy()

    def _calc_ffd(s: pl.Series) -> pl.Series:
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= window:
            windows = np.lib.stride_tricks.sliding_window_view(vals, window)
            out[window - 1 :] = windows @ w_rev
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_ffd, return_dtype=pl.Float64)
