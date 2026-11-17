"""Cross-sectional and time-series alpha primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from ..core import IntoExpr, check_positive, map_sliding_windows, parse_expr
from ..stats import percentile_rank, rolling_corr, rolling_cov, zscore
from ..trend import wma


def _over_cols(over: IntoExpr | Sequence[str] | None) -> list[str | pl.Expr] | None:
    """Normalize optional cross-sectional grouping argument to a list of columns/expressions."""
    if over is None:
        return None
    return [over] if isinstance(over, (str, pl.Expr)) else list(over)


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
    return percentile_rank(close, period=period)


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
    return map_sliding_windows(
        close, period, lambda w: np.argmax(w[:, ::-1], axis=1).astype(np.float64)
    )


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
    return map_sliding_windows(
        close, period, lambda w: np.argmin(w[:, ::-1], axis=1).astype(np.float64)
    )


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
    return wma(parse_expr(close), period=period)


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
    return zscore(close, period=period)


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
    cols = _over_cols(over)
    ranked = e.rank(method=method).over(cols) if cols is not None else e.rank(method=method)
    if not percentile:
        return ranked
    count = e.count().over(cols) if cols is not None else e.count()
    return ranked / count


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
    group_cols = [*(_over_cols(over) or []), *(_over_cols(by) or [])]
    demeaned = e - e.mean().over(group_cols)
    if standardize:
        std = e.std().over(group_cols)
        return pl.when(std == 0.0).then(0.0).otherwise(demeaned / std)
    return demeaned


def orthogonalize(
    expr: IntoExpr,
    factors: IntoExpr | Sequence[IntoExpr],
    over: IntoExpr | Sequence[str] | None = None,
    standardize: bool = True,
    ridge: float = 0.0,
) -> pl.Expr:
    """Cross-Sectional Multi-Factor OLS Orthogonalization (Frisch-Waugh-Lovell / Gram-Schmidt).

    Removes linear exposure of an alpha signal ``expr`` ($Y$) to one or more continuous
    risk factors ``factors`` ($X_1, \\dots, X_K$, such as Market Beta, Volatility, Size,
    or Amihud Illiquidity) within each cross-section ``over`` using iterated closed-form
    Gram-Schmidt projections in native Polars expressions:
        $$\\epsilon = (I - P_X)(Y - \\bar{Y}), \\quad \\text{Cov}(\\epsilon, X_j) = 0 \\; \\forall j = 1 \\dots K$$

    :param expr: Target alpha signal column or expression to orthogonalize.
    :param factors: Single risk factor column/expression or sequence of factor columns/expressions.
    :param over: Optional cross-sectional grouping column(s) (typically ``'timestamp'`` or ``'date'``).
    :param standardize: If True (default), standardizes the orthogonalized residual to unit variance
        within each cross-section.
    :param ridge: Optional L2 ridge regularization penalty $\\lambda \\ge 0$ added to factor variance
        denominators (default 0.0 for exact OLS orthogonality).
    :return: Polars expression of orthogonalized alpha residuals.
    """
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    if isinstance(factors, (str, pl.Expr)):
        factor_exprs = [parse_expr(factors).cast(pl.Float64)]
    else:
        factor_exprs = [parse_expr(f).cast(pl.Float64) for f in factors]

    if not factor_exprs:
        raise ValueError("factors must contain at least one factor column or expression")

    over_cols: list[str | pl.Expr] | None = None
    if over is not None:
        over_cols = [over] if isinstance(over, (str, pl.Expr)) else list(over)

    def _cs_mean(x: pl.Expr) -> pl.Expr:
        return x.mean().over(over_cols) if over_cols is not None else x.mean()

    def _cs_std(x: pl.Expr) -> pl.Expr:
        return x.std().over(over_cols) if over_cols is not None else x.std()

    y_res = parse_expr(expr).cast(pl.Float64)
    y_res = y_res - _cs_mean(y_res)
    u_list = [x - _cs_mean(x) for x in factor_exprs]

    k = len(u_list)
    for j in range(k):
        q_j = u_list[j]
        var_q = _cs_mean(q_j * q_j)
        denom = var_q + float(ridge)

        cov_y = _cs_mean(y_res * q_j)
        beta_y = pl.when(var_q > 1e-15).then(cov_y / denom).otherwise(0.0)
        y_res = y_res - beta_y * q_j

        for m in range(j + 1, k):
            u_m = u_list[m]
            cov_m = _cs_mean(u_m * q_j)
            beta_m = pl.when(var_q > 1e-15).then(cov_m / denom).otherwise(0.0)
            u_list[m] = u_m - beta_m * q_j

    if standardize:
        std_res = _cs_std(y_res)
        return pl.when(std_res.is_null() | (std_res <= 1e-15)).then(0.0).otherwise(y_res / std_res)
    return y_res


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
    cols = _over_cols(over)
    abs_sum = e.abs().sum().over(cols) if cols is not None else e.abs().sum()
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
    return parse_expr(expr).shift(period)


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
    return parse_expr(expr).rolling_min(window_size=period)


def ts_max(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Maximum (WorldQuant Alpha primitive 'ts_max(x, d)').

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    return parse_expr(expr).rolling_max(window_size=period)


def ts_sum(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Sum (WorldQuant Alpha primitive 'ts_sum(x, d)').

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    return parse_expr(expr).rolling_sum(window_size=period)


def ts_std(
    expr: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Time-series Rolling Standard Deviation (WorldQuant Alpha primitive 'ts_std(x, d)').

    :param expr: Feature column or expression.
    :param period: Lookback window (default 10).
    :return: Polars expression.
    """
    return parse_expr(expr).rolling_std(window_size=period)


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
    return map_sliding_windows(expr, period, lambda w: np.prod(w, axis=1))


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
    if clip_bound is not None:
        check_positive(clip_bound, "clip_bound")

    e = parse_expr(expr)
    cols = _over_cols(over)
    mean = e.mean().over(cols) if cols is not None else e.mean()
    std = e.std().over(cols) if cols is not None else e.std()

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
    if method in ("zscore", "mad"):
        check_positive(threshold, "threshold")

    e = parse_expr(expr)
    over_cols = _over_cols(over)

    def _cs(agg: pl.Expr) -> pl.Expr:
        return agg.over(over_cols) if over_cols is not None else agg

    if method == "quantile":
        lo_expr = _cs(e.quantile(lower, interpolation="linear"))
        hi_expr = _cs(e.quantile(upper, interpolation="linear"))
    elif method == "zscore":
        mean_expr = _cs(e.mean())
        band = float(threshold) * _cs(e.std().fill_null(0.0))
        lo_expr, hi_expr = mean_expr - band, mean_expr + band
    else:
        med_expr = _cs(e.median())
        scaled_mad = 1.4826 * float(threshold) * _cs((e - med_expr).abs().median())
        lo_expr, hi_expr = med_expr - scaled_mad, med_expr + scaled_mad

    return pl.when(e < lo_expr).then(lo_expr).when(e > hi_expr).then(hi_expr).otherwise(e)
