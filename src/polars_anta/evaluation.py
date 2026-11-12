"""Factor evaluation (IC, Rank IC, Quantile Analysis) and strategy performance metrics for Polars."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Literal, TypeVar

import numpy as np
import polars as pl

from .core import IntoExpr, check_period, check_positive, parse_expr
from .trend import wma as _wma

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)

__all__ = [
    "drawdown_series",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "profit_factor",
    "win_rate",
    "value_at_risk",
    "conditional_var",
    "omega_ratio",
    "tail_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "forward_returns",
    "information_coefficient",
    "ic_summary",
    "ic_decay",
    "quantile_analysis",
    "factor_turnover",
    "triple_barrier_labels",
    "trend_scanning_labels",
]


# =============================================================================
# Strategy Performance Expressions (pl.Expr)
# =============================================================================


def drawdown_series(
    returns_or_equity: IntoExpr,
    is_returns: bool = True,
) -> pl.Expr:
    """Compute the underwater drawdown series from period returns or an equity curve.

    Returns values in ``(-1.0, 0.0]`` representing the percentage drop from the
    running peak at each bar.

    :param returns_or_equity: Period returns column/expression (when ``is_returns=True``)
        or cumulative equity curve column/expression (when ``is_returns=False``).
    :param is_returns: If True, compounds ``(1 + r).cum_prod()`` first; if False,
        treats input directly as an equity/price series.
    :return: Polars expression of drawdown values ``<= 0.0``.
    """
    x = parse_expr(returns_or_equity)
    equity = (1.0 + x.fill_null(0.0)).cum_prod() if is_returns else x
    peak = equity.cum_max()
    return pl.when(peak.is_null() | (peak <= 0.0)).then(0.0).otherwise((equity - peak) / peak)


def max_drawdown(
    returns_or_equity: IntoExpr,
    is_returns: bool = True,
) -> pl.Expr:
    """Compute the Maximum Drawdown (MDD) scalar from period returns or an equity curve.

    Returns the worst peak-to-trough percentage decline (e.g., ``-0.18`` for an 18% drawdown).

    :param returns_or_equity: Period returns or equity curve column/expression.
    :param is_returns: If True, compounds ``(1 + r).cum_prod()`` first.
    :return: Scalar Polars expression.
    """
    return drawdown_series(returns_or_equity, is_returns=is_returns).min().fill_null(0.0)


def sharpe_ratio(
    returns: IntoExpr,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the annualized Sharpe Ratio scalar expression.

    :param returns: Period returns column or expression.
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression.
    """
    check_positive(annualization, "annualization")

    excess = parse_expr(returns) - float(risk_free_rate) / float(annualization)
    mean_ex = excess.mean()
    std_ex = excess.std()
    return (
        pl.when(std_ex.is_null() | (std_ex == 0.0) | mean_ex.is_null())
        .then(0.0)
        .otherwise((mean_ex / std_ex) * math.sqrt(float(annualization)))
    )


def sortino_ratio(
    returns: IntoExpr,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the annualized Sortino Ratio scalar expression using downside deviation.

    :param returns: Period returns column or expression.
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression.
    """
    check_positive(annualization, "annualization")

    excess = parse_expr(returns) - float(risk_free_rate) / float(annualization)
    mean_ex = excess.mean()
    downside_dev = pl.when(excess < 0.0).then(excess**2).otherwise(0.0).mean().sqrt()
    return (
        pl.when(downside_dev.is_null() | (downside_dev == 0.0) | mean_ex.is_null())
        .then(0.0)
        .otherwise((mean_ex / downside_dev) * math.sqrt(float(annualization)))
    )


def calmar_ratio(
    returns: IntoExpr,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the Calmar Ratio (Annualized Return / |Max Drawdown|) scalar expression.

    :param returns: Period returns column or expression.
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression.
    """
    check_positive(annualization, "annualization")

    r = parse_expr(returns)
    ann_return = r.mean() * float(annualization)
    mdd_abs = max_drawdown(r, is_returns=True).abs()
    return (
        pl.when(mdd_abs.is_null() | (mdd_abs == 0.0) | ann_return.is_null())
        .then(0.0)
        .otherwise(ann_return / mdd_abs)
    )


def profit_factor(
    returns: IntoExpr,
) -> pl.Expr:
    """Compute the Profit Factor (Gross Profit / |Gross Loss|) scalar expression.

    :param returns: Period returns or trade PnL column/expression.
    :return: Scalar Polars expression.
    """
    r = parse_expr(returns)
    gross_profit = pl.when(r > 0.0).then(r).otherwise(0.0).sum()
    gross_loss = pl.when(r < 0.0).then(r.abs()).otherwise(0.0).sum()
    return (
        pl.when(gross_loss == 0.0)
        .then(pl.when(gross_profit > 0.0).then(float("inf")).otherwise(0.0))
        .otherwise(gross_profit / gross_loss)
    )


def win_rate(
    returns: IntoExpr,
    include_zeros: bool = False,
) -> pl.Expr:
    """Compute the Win Rate (proportion of positive return periods) scalar expression.

    :param returns: Period returns or trade PnL column/expression.
    :param include_zeros: When False (default), computes win rate over active non-zero
        periods ``(r != 0)``; when True, includes zero-return periods in the denominator.
    :return: Scalar Polars expression in ``[0.0, 1.0]``.
    """
    r = parse_expr(returns)
    wins = (r > 0.0).cast(pl.Float64).sum()
    if include_zeros:
        total = r.is_not_null().cast(pl.Float64).sum()
    else:
        total = (r.is_not_null() & (r != 0.0)).cast(pl.Float64).sum()
    return pl.when(total == 0.0).then(0.0).otherwise(wins / total)


def _validate_cutoff(cutoff: float) -> float:
    """Validate a left-tail probability cutoff and return it as a float."""
    alpha = float(cutoff)
    if not 0.0 < alpha < 1.0:
        raise ValueError("cutoff must be a probability strictly between 0.0 and 1.0")
    return alpha


def value_at_risk(
    returns: IntoExpr,
    cutoff: float = 0.05,
) -> pl.Expr:
    """Compute the Historical Value at Risk (VaR) scalar expression.

    Returns the empirical ``cutoff`` quantile of the period-return distribution,
    i.e. the return level that is not exceeded in ``cutoff`` of the sample::

        VaR_alpha(r) = Quantile(r, alpha)

    The quantile uses linear interpolation, so the result is continuous in the
    sample rather than snapping to an observed order statistic. On a typical
    return series this is a negative number (the size of the loss in return
    units); it is reported in the same units as the input, not as a positive
    loss magnitude.

    :param returns: Period returns column or expression.
    :param cutoff: Left-tail probability in ``(0.0, 1.0)`` (default 0.05 for 95% VaR).
    :return: Scalar Polars expression, ``null`` when the input has no values.
    """
    alpha = _validate_cutoff(cutoff)
    return parse_expr(returns).quantile(alpha, interpolation="linear")


def conditional_var(
    returns: IntoExpr,
    cutoff: float = 0.05,
) -> pl.Expr:
    """Compute the Conditional Value at Risk / Expected Shortfall scalar expression.

    The mean of the returns that fall at or below the historical VaR level::

        CVaR_alpha(r) = E[r | r <= VaR_alpha(r)]

    This is the average loss once the tail is actually entered, so it is always
    less than or equal to ``value_at_risk`` on the same sample. Because the VaR
    level is itself a sample quantile, at least one observation satisfies the
    conditioning event whenever the input has values, so the mean is never taken
    over an empty set.

    :param returns: Period returns column or expression.
    :param cutoff: Left-tail probability in ``(0.0, 1.0)`` (default 0.05).
    :return: Scalar Polars expression, ``null`` when the input has no values.
    """
    alpha = _validate_cutoff(cutoff)
    r = parse_expr(returns)
    var = r.quantile(alpha, interpolation="linear")
    tail = pl.when(r <= var).then(r).otherwise(None)
    return tail.mean()


def omega_ratio(
    returns: IntoExpr,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the Omega Ratio scalar expression.

    Ratio of probability-weighted gains to losses around a threshold::

        Omega(theta) = sum(max(0, r_t - theta)) / sum(max(0, theta - r_t))

    The threshold is the risk-free rate de-annualized to one period
    (``risk_free_rate / annualization``), so a ``risk_free_rate`` of ``0.05``
    with daily data compares each day's return against roughly 2 basis points.
    Values above ``1.0`` mean the upside beyond the threshold outweighs the
    downside.

    A series with no observations below the threshold has zero downside mass;
    following ``profit_factor``, that reports ``inf`` when there is upside and
    ``0.0`` when the series is entirely flat, rather than ``NaN``.

    :param returns: Period returns column or expression.
    :param risk_free_rate: Annualized risk-free threshold (default 0.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression, non-negative (or ``inf``).
    """
    check_positive(annualization, "annualization")

    r = parse_expr(returns)
    theta = float(risk_free_rate) / float(annualization)
    upside = pl.when(r > theta).then(r - theta).otherwise(0.0).sum()
    downside = pl.when(r < theta).then(theta - r).otherwise(0.0).sum()
    return (
        pl.when(downside == 0.0)
        .then(pl.when(upside > 0.0).then(float("inf")).otherwise(0.0))
        .otherwise(upside / downside)
    )


def tail_ratio(
    returns: IntoExpr,
    cutoff: float = 0.05,
) -> pl.Expr:
    """Compute the Tail Ratio scalar expression.

    Compares the magnitude of the right tail to the left tail::

        TailRatio = |Quantile(r, 1 - alpha)| / |Quantile(r, alpha)|

    A value above ``1.0`` means the best ``alpha`` of periods are larger in
    magnitude than the worst ``alpha`` - a right-skewed return distribution.
    Absolute values are used so the ratio is positive for a symmetric series
    even though the left quantile itself is negative.

    A degenerate left tail (lower quantile exactly ``0.0``) has no denominator;
    following ``profit_factor``, that reports ``inf`` when there is a non-zero
    right tail and ``0.0`` otherwise, rather than ``NaN``.

    :param returns: Period returns column or expression.
    :param cutoff: Tail probability in ``(0.0, 1.0)`` (default 0.05).
    :return: Scalar Polars expression, non-negative (or ``inf``).
    """
    alpha = _validate_cutoff(cutoff)
    r = parse_expr(returns)
    upper = r.quantile(1.0 - alpha, interpolation="linear").abs()
    lower = r.quantile(alpha, interpolation="linear").abs()
    return (
        pl.when(lower == 0.0)
        .then(pl.when(upper > 0.0).then(float("inf")).otherwise(0.0))
        .otherwise(upper / lower)
    )


def _norm_cdf_expr(z: pl.Expr) -> pl.Expr:
    """Closed-form Abramowitz & Stegun (7.1.26) Standard Normal CDF $\\Phi(z)$ expression."""
    x = z.abs() / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    erf_x = 1.0 - poly * (-x * x).exp()
    return (0.5 * (1.0 + z.sign() * erf_x)).clip(0.0, 1.0)


def probabilistic_sharpe_ratio(
    returns: IntoExpr,
    benchmark_sr: float = 0.0,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the Probabilistic Sharpe Ratio (PSR — Bailey & López de Prado, 2012) scalar expression.

    Estimates the probability $\\text{PSR}(SR^*) \\in [0, 1]$ that the observed Sharpe ratio
    exceeds ``benchmark_sr`` after adjusting for sample length $T$, skewness $\\hat{\\gamma}_3$,
    and fat-tailed Pearson kurtosis $\\hat{\\gamma}_4$:
        $$\\hat{\\sigma}(\\widehat{SR}) = \\sqrt{\\frac{1 - \\hat{\\gamma}_3 \\widehat{SR} + \\frac{\\hat{\\gamma}_4 - 1}{4} \\widehat{SR}^2}{T - 1}}$$
        $$\\text{PSR}(SR^*) = \\Phi\\left(\\frac{\\widehat{SR} - SR^*}{\\hat{\\sigma}(\\widehat{SR})}\\right)$$

    :param returns: Period returns column or expression.
    :param benchmark_sr: Annualized hurdle Sharpe ratio $SR^*_{\\text{ann}}$ to beat (default 0.0).
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression in ``[0.0, 1.0]``.
    """
    check_positive(annualization, "annualization")

    excess = parse_expr(returns).cast(pl.Float64) - float(risk_free_rate) / float(annualization)
    t_obs = excess.count().cast(pl.Float64)
    mean_ex = excess.mean()
    std_ex = excess.std()
    sr_hat = mean_ex / std_ex
    sr_star = float(benchmark_sr) / math.sqrt(float(annualization))

    skew_hat = excess.skew().fill_null(0.0).fill_nan(0.0)
    kurt_pearson = excess.kurtosis(fisher=False).fill_null(3.0).fill_nan(3.0)

    var_num = (1.0 - skew_hat * sr_hat + ((kurt_pearson - 1.0) / 4.0) * (sr_hat * sr_hat)).clip(
        lower_bound=1e-12
    )
    z_stat = (sr_hat - sr_star) / (var_num / (t_obs - 1.0)).sqrt()

    return (
        pl.when(
            (t_obs < 3.0) | std_ex.is_null() | (std_ex == 0.0) | mean_ex.is_null() | z_stat.is_nan()
        )
        .then(0.0)
        .otherwise(_norm_cdf_expr(z_stat))
    )


def deflated_sharpe_ratio(
    returns: IntoExpr,
    n_trials: int = 10,
    trials_sr_std: float = 0.5,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the Deflated Sharpe Ratio (DSR — Bailey & López de Prado, 2014) scalar expression.

    Adjusts the Probabilistic Sharpe Ratio for multiple-testing selection bias across
    ``n_trials`` independent backtest configurations by computing the expected maximum
    annualized Sharpe ratio under the null hypothesis using Extreme Value Theory:
        $$SR^*_{\\text{ann}} = \\sigma_{\\{SR\\}} \\left[ (1 - \\gamma) \\Phi^{-1}\\left(1 - \\frac{1}{N}\\right) + \\gamma \\Phi^{-1}\\left(1 - \\frac{1}{N \\cdot e}\\right) \\right]$$
    where $\\gamma \\approx 0.5772156649$ is the Euler-Mascheroni constant.

    :param returns: Period returns column or expression.
    :param n_trials: Total number of independent strategy/parameter trials tested $N \\ge 1$ (default 10).
    :param trials_sr_std: Standard deviation of annualized Sharpe ratios across trials (default 0.5).
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression in ``[0.0, 1.0]``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be a positive integer greater than or equal to 1")
    if trials_sr_std < 0.0:
        raise ValueError("trials_sr_std must be non-negative")
    check_positive(annualization, "annualization")

    if n_trials == 1 or trials_sr_std == 0.0:
        hurdle_sr = 0.0
    else:
        euler_gamma = 0.5772156649015329
        std_norm = NormalDist(0.0, 1.0)
        n_f = float(n_trials)
        z1 = std_norm.inv_cdf(1.0 - 1.0 / n_f)
        z2 = std_norm.inv_cdf(1.0 - 1.0 / (n_f * math.e))
        hurdle_sr = float(trials_sr_std) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sr=hurdle_sr,
        risk_free_rate=risk_free_rate,
        annualization=annualization,
    )


# =============================================================================
# DataFrame / LazyFrame Factor Evaluation Suite
# =============================================================================


def forward_returns(
    df: FrameType,
    close_col: str = "close",
    periods: Sequence[int] = (1, 5, 10),
    group_by: str | Sequence[str] | None = None,
    prefix: str = "fwd_ret_",
) -> FrameType:
    """Compute multi-horizon forward returns ``(Close[t + p] / Close[t]) - 1`` for alpha evaluation.

    :param df: Input Polars DataFrame or LazyFrame.
    :param close_col: Name of close price column (default 'close').
    :param periods: Sequence of forward lookahead horizons (default ``(1, 5, 10)``).
    :param group_by: Optional symbol/ticker column(s) to compute shifts within each asset.
    :param prefix: Column name prefix for generated forward return columns (default ``'fwd_ret_'``).
    :return: DataFrame or LazyFrame with appended forward return columns.
    """
    if not periods:
        raise ValueError("periods must contain at least one positive horizon")

    c = pl.col(close_col)
    exprs: list[pl.Expr] = []
    for p in periods:
        if p <= 0:
            raise ValueError("each forward return period must be a positive integer")
        fwd_expr = (c.shift(-p) / c) - 1.0
        if group_by is not None:
            fwd_expr = fwd_expr.over(group_by)
        exprs.append(fwd_expr.alias(f"{prefix}{p}"))

    return df.with_columns(exprs)


def information_coefficient(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str,
    group_by: str | Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pl.DataFrame:
    """Compute the Information Coefficient (IC) or Rank IC series between an alpha factor and forward returns.

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor signal.
    :param return_col: Column name of the target forward return.
    :param group_by: Optional cross-sectional grouping column(s) (typically ``'timestamp'`` or ``'date'``).
        When provided, computes cross-sectional IC per timestamp; when None, computes a single global IC.
    :param method: Correlation method: ``'spearman'`` (Rank IC, default) or ``'pearson'`` (Normal IC).
    :return: Eager Polars DataFrame with ``ic`` and ``n_obs`` columns.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be either 'spearman' or 'pearson'")

    f_col = pl.col(factor_col).cast(pl.Float64)
    r_col = pl.col(return_col).cast(pl.Float64)

    valid_filter = f_col.is_not_null() & r_col.is_not_null() & (~f_col.is_nan()) & (~r_col.is_nan())
    cleaned = df.filter(valid_filter)

    if method == "spearman":
        ic_expr = pl.corr(f_col.rank(), r_col.rank(), method="pearson").fill_nan(0.0).alias("ic")
    else:
        ic_expr = pl.corr(f_col, r_col, method="pearson").fill_nan(0.0).alias("ic")

    count_expr = pl.len().cast(pl.Int64).alias("n_obs")

    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        res = cleaned.group_by(gb_cols).agg([ic_expr, count_expr]).sort(gb_cols)
    else:
        res = cleaned.select([ic_expr, count_expr])

    return res.collect() if isinstance(res, pl.LazyFrame) else res


def _nw_auto_lag(n_obs: int) -> int:
    """Newey–West (1994) plug-in bandwidth: max(1, floor(4 * (T/100)^(2/9))), capped at T-1."""
    if n_obs <= 1:
        return 0
    return min(max(1, int(4.0 * ((float(n_obs) / 100.0) ** (2.0 / 9.0)))), n_obs - 1)


def _newey_west_lrv(u: np.ndarray, max_lag: int) -> np.ndarray | float:
    """Bartlett-kernel long-run variance of DEMEANED series.

    :param u: Demeaned array of shape ``(T,)`` or ``(T, K)``.
    :param max_lag: Maximum Bartlett kernel lag $L \\ge 0$.
    :return: ``float`` for 1-D input, or ``np.ndarray`` of shape ``(K,)`` for 2-D input, floored at ``1e-24``.
    """
    n_obs = u.shape[0]
    effective_lag = min(max(0, int(max_lag)), max(0, n_obs - 1))
    if u.ndim == 1:
        gamma_0 = float(np.dot(u, u)) / float(n_obs)
        lrv = gamma_0
        for lag_j in range(1, effective_lag + 1):
            weight = 1.0 - float(lag_j) / float(effective_lag + 1)
            gamma_j = float(np.dot(u[lag_j:], u[:-lag_j])) / float(n_obs)
            lrv += 2.0 * weight * gamma_j
        return max(lrv, 1e-24)

    gamma_0_vec = (u * u).sum(axis=0) / float(n_obs)
    lrv_vec = gamma_0_vec.copy()
    for lag_j in range(1, effective_lag + 1):
        weight = 1.0 - float(lag_j) / float(effective_lag + 1)
        gamma_j_vec = (u[lag_j:] * u[:-lag_j]).sum(axis=0) / float(n_obs)
        lrv_vec = lrv_vec + 2.0 * weight * gamma_j_vec
    return np.maximum(lrv_vec, 1e-24)


def ic_summary(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str,
    group_by: str | Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
    nw_lags: int | None = None,
) -> dict[str, float]:
    """Compute comprehensive Information Coefficient (IC) summary statistics with Newey-West HAC t-stat.

    Returns a dictionary containing:
    - ``ic_mean``: Mean IC across cross-sections.
    - ``ic_std``: Standard deviation of IC.
    - ``icir``: Information Ratio of IC (``ic_mean / ic_std``).
    - ``t_stat``: Independent-sample t-statistic testing ``IC != 0``.
    - ``nw_t_stat``: Newey-West (1987) HAC autocorrelation-consistent t-statistic.
    - ``hit_rate``: Proportion of periods where ``IC > 0``.
    - ``n_periods``: Total number of valid periods evaluated.

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor.
    :param return_col: Column name of the target forward return.
    :param group_by: Optional cross-sectional time column(s) (e.g., ``'timestamp'`` or ``'date'``).
    :param method: ``'spearman'`` (Rank IC) or ``'pearson'`` (Normal IC).
    :param nw_lags: Optional Bartlett kernel maximum lag $L \\ge 0$ for Newey-West HAC standard error.
        When None, uses the Newey-West (1994) automatic plug-in rule $\\max(1, \\lfloor 4(T/100)^{2/9} \\rfloor)$.
    :return: Dictionary of summary metrics.
    """
    if nw_lags is not None and nw_lags < 0:
        raise ValueError("nw_lags must be a non-negative integer")

    ic_df = information_coefficient(
        df,
        factor_col=factor_col,
        return_col=return_col,
        group_by=group_by,
        method=method,
    ).filter((pl.col("n_obs") > 0) & pl.col("ic").is_not_null() & (~pl.col("ic").is_nan()))

    n = len(ic_df)
    if n == 0:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "nw_t_stat": 0.0,
            "hit_rate": 0.0,
            "n_periods": 0.0,
        }

    ic_series = ic_df["ic"]
    mean_val = float(ic_series.mean() or 0.0)
    std_val = float(ic_series.std() or 0.0) if n > 1 else 0.0
    icir_val = mean_val / std_val if std_val > 1e-12 else 0.0
    t_stat_val = (mean_val / (std_val / math.sqrt(float(n)))) if std_val > 1e-12 else 0.0
    hit_rate_val = float((ic_series > 0.0).mean() or 0.0)

    if n > 1 and std_val > 1e-12:
        u = ic_series.to_numpy().astype(np.float64, copy=False) - mean_val
        max_lag = min(int(nw_lags), n - 1) if nw_lags is not None else _nw_auto_lag(n)
        lrv = float(_newey_west_lrv(u, max_lag))
        nw_se = math.sqrt(lrv / float(n))
        nw_t_stat_val = mean_val / nw_se
    else:
        nw_t_stat_val = 0.0

    return {
        "ic_mean": mean_val,
        "ic_std": std_val,
        "icir": icir_val,
        "t_stat": t_stat_val,
        "nw_t_stat": nw_t_stat_val,
        "hit_rate": hit_rate_val,
        "n_periods": float(n),
    }


def ic_decay(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    close_col: str = "close",
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 20),
    asset_col: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pl.DataFrame:
    """Compute Multi-Horizon Information Coefficient (IC) Decay and Alpha Half-Life.

    Evaluates ``ic_summary`` across a grid of forward return horizons $h \\in \\text{horizons}$,
    automatically setting Newey-West HAC bandwidth ``nw_lags = max(1, h - 1)`` to correct for
    overlapping return autocorrelation, and estimates the exponential decay half-life (in bars).

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor signal.
    :param close_col: Column name of the close price (default 'close').
    :param horizons: Sequence of forward return horizons to evaluate (default ``(1, 2, 3, 5, 10, 20)``).
    :param asset_col: Optional asset/symbol column(s) for computing forward returns per instrument.
    :param group_by: Optional cross-sectional time column(s) (e.g., ``'timestamp'`` or ``'date'``).
    :param method: Correlation method: ``'spearman'`` (Rank IC, default) or ``'pearson'`` (Normal IC).
    :return: Eager Polars DataFrame with columns ``['horizon', 'ic_mean', 'ic_std', 'icir',
        't_stat', 'nw_t_stat', 'hit_rate', 'n_periods', 'half_life']``.
    """
    if not horizons:
        raise ValueError("horizons must contain at least one positive integer")
    for h in horizons:
        if h <= 0:
            raise ValueError("each horizon in horizons must be a positive integer")

    prefix = "_anta_decay_ret_"
    with_fwd = forward_returns(
        df,
        close_col=close_col,
        periods=horizons,
        group_by=asset_col,
        prefix=prefix,
    )
    eager_df = with_fwd.collect() if isinstance(with_fwd, pl.LazyFrame) else with_fwd

    rows: list[dict[str, float | int]] = []
    abs_ics: list[float] = []
    h_vals: list[float] = []

    for h in horizons:
        ret_col = f"{prefix}{h}"
        stats = ic_summary(
            eager_df,
            factor_col=factor_col,
            return_col=ret_col,
            group_by=group_by,
            method=method,
            nw_lags=max(1, int(h) - 1),
        )
        rows.append(
            {
                "horizon": int(h),
                "ic_mean": stats["ic_mean"],
                "ic_std": stats["ic_std"],
                "icir": stats["icir"],
                "t_stat": stats["t_stat"],
                "nw_t_stat": stats["nw_t_stat"],
                "hit_rate": stats["hit_rate"],
                "n_periods": stats["n_periods"],
            }
        )
        abs_ics.append(max(abs(stats["ic_mean"]), 1e-8))
        h_vals.append(float(h))

    if len(h_vals) >= 2:
        x_arr = np.asarray(h_vals, dtype=np.float64)
        y_arr = np.log(np.asarray(abs_ics, dtype=np.float64))
        x_centered = x_arr - x_arr.mean()
        denom = float(np.dot(x_centered, x_centered))
        decay_rate = (
            -float(np.dot(x_centered, y_arr - y_arr.mean())) / denom if denom > 0.0 else 0.0
        )
        half_life = math.log(2.0) / decay_rate if decay_rate > 1e-8 else float("inf")
    else:
        half_life = float("inf")

    for row in rows:
        row["half_life"] = half_life

    return pl.DataFrame(rows)


def quantile_analysis(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str,
    quantiles: int = 5,
    group_by: str | Sequence[str] | None = None,
    annualization: float = 252.0,
) -> pl.DataFrame:
    """Group observations into N factor quantiles and compute return & risk metrics per bucket.

    Used to verify factor monotonicity (whether higher quantile buckets systematically
    earn higher forward returns than lower buckets).

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor.
    :param return_col: Column name of the target forward return.
    :param quantiles: Number of quantile buckets ``Q >= 2`` (default 5).
    :param group_by: Optional cross-sectional time column(s) (e.g., ``'date'``) to assign
        quantiles within each period.
    :param annualization: Periods per year for Sharpe computation (default 252.0).
    :return: Polars DataFrame sorted by ``quantile`` (``1`` to ``quantiles``).
    """
    if quantiles < 2:
        raise ValueError("quantiles must be an integer greater than or equal to 2")

    f_col = pl.col(factor_col).cast(pl.Float64)
    r_col = pl.col(return_col).cast(pl.Float64)

    valid_filter = f_col.is_not_null() & r_col.is_not_null() & (~f_col.is_nan()) & (~r_col.is_nan())
    cleaned = df.filter(valid_filter)

    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        pct_rank = (f_col.rank(method="ordinal") - 1.0).over(gb_cols) / pl.len().over(gb_cols)
    else:
        pct_rank = (f_col.rank(method="ordinal") - 1.0) / pl.len()

    bucket_expr = (
        (pct_rank * float(quantiles)).floor().clip(0.0, float(quantiles - 1)).cast(pl.Int32) + 1
    ).alias("quantile")

    with_q = cleaned.with_columns(bucket_expr)
    agg_df = (
        with_q.group_by("quantile")
        .agg(
            [
                pl.len().cast(pl.Int64).alias("count"),
                f_col.mean().alias("mean_factor"),
                r_col.mean().alias("mean_return"),
                r_col.std().fill_null(0.0).alias("std_return"),
                sharpe_ratio(r_col, annualization=annualization).alias("sharpe_ratio"),
                win_rate(r_col, include_zeros=True).alias("win_rate"),
                ((1.0 + r_col).product() - 1.0).alias("cumulative_return"),
            ]
        )
        .sort("quantile")
    )

    return agg_df.collect() if isinstance(agg_df, pl.LazyFrame) else agg_df


def factor_turnover(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    asset_col: str = "ticker",
    time_col: str = "timestamp",
    lag: int = 1,
) -> pl.DataFrame:
    """Compute factor rank autocorrelation and signal turnover across time.

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor.
    :param asset_col: Column name identifying each asset/ticker (default 'ticker').
    :param time_col: Column name identifying each timestamp/date (default 'timestamp').
    :param lag: Period lag for autocorrelation and turnover calculation (default 1).
    :return: Polars DataFrame with ``[time_col, 'rank_autocorr', 'turnover']``.
    """
    if lag <= 0:
        raise ValueError("lag must be a positive integer")

    f_col = pl.col(factor_col).cast(pl.Float64)
    working = (
        df.sort([asset_col, time_col])
        .with_columns(
            (f_col.rank().over(time_col) / f_col.count().over(time_col)).alias("_cs_rank"),
            (f_col / f_col.abs().sum().over(time_col)).fill_nan(0.0).alias("_weight"),
        )
        .with_columns(
            pl.col("_cs_rank").shift(lag).over(asset_col).alias("_prev_rank"),
            pl.col("_weight").shift(lag).over(asset_col).alias("_prev_weight"),
        )
        .filter(pl.col("_prev_rank").is_not_null())
    )

    res = (
        working.group_by(time_col)
        .agg(
            [
                pl.corr("_cs_rank", "_prev_rank", method="pearson")
                .fill_nan(0.0)
                .alias("rank_autocorr"),
                (pl.col("_weight") - pl.col("_prev_weight")).abs().sum().alias("turnover"),
            ]
        )
        .sort(time_col)
    )

    return res.collect() if isinstance(res, pl.LazyFrame) else res


# =============================================================================
# Quantitative Machine Learning Labeling (AFML Primitives)
# =============================================================================


def triple_barrier_labels(
    close: IntoExpr,
    volatility: IntoExpr | float | int | None = None,
    horizon: int = 10,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    min_ret: float = 0.0,
    vol_period: int = 20,
    side: IntoExpr | None = None,
) -> pl.Expr:
    """Dynamic Triple-Barrier & Meta-Labeling Method (López de Prado, 2018).

    Evaluates the forward price trajectory over ``horizon`` bars against two dynamic
    horizontal barriers scaled by local volatility $\\sigma_t$ — an upper Profit-Taking (PT)
    barrier $+\\max(\\text{pt\\_mult} \\cdot \\sigma_t, \\text{min\\_ret})$ and a lower Stop-Loss (SL)
    barrier $-\\max(\\text{sl\\_mult} \\cdot \\sigma_t, \\text{min\\_ret})$ — plus a vertical expiration
    barrier at $t + \\text{horizon}$.

    Returns a ``pl.Struct`` with fields:
        - ``'label'``: ``+1.0`` if upper barrier touched first, ``-1.0`` if lower barrier
          touched first, or ``0.0`` if vertical horizon expired first.
        - ``'touch_bars'``: Number of bars elapsed until the first barrier touch ($1 \\dots \\text{horizon}$).
        - ``'realized_return'``: Realized return at the first touched barrier.
        - ``'meta_label'``: Binary meta-label in ``{0.0, 1.0}`` indicating whether the trade
          achieved its profit-taking target.

    :param close: Close price column or expression.
    :param volatility: Optional per-bar volatility column/expression or positive scalar target.
        When None, defaults to rolling standard deviation of returns over ``vol_period``.
    :param horizon: Vertical barrier maximum holding period in bars (default 10).
    :param pt_mult: Profit-taking volatility multiplier (default 2.0; set 0.0 to disable PT barrier).
    :param sl_mult: Stop-loss volatility multiplier (default 2.0; set 0.0 to disable SL barrier).
    :param min_ret: Minimum absolute return barrier threshold (default 0.0).
    :param vol_period: Rolling window used when ``volatility=None`` (default 20).
    :param side: Optional primary model directional signal ($\\in \\{-1, +1\\}$) for Meta-Labeling.
    :return: Polars Struct expression.
    """
    check_period(horizon, 1, "horizon")
    if pt_mult < 0.0 or sl_mult < 0.0:
        raise ValueError("pt_mult and sl_mult must be non-negative")
    if min_ret < 0.0:
        raise ValueError("min_ret must be non-negative")
    check_period(vol_period, 2, "vol_period")

    c = parse_expr(close).cast(pl.Float64)

    if volatility is None:
        vol_expr = (
            c.pct_change()
            .rolling_std(window_size=vol_period, min_samples=2)
            .fill_null(strategy="backward")
            .fill_null(0.01)
        )
    elif isinstance(volatility, (int, float)) and not isinstance(volatility, bool):
        if volatility <= 0.0:
            raise ValueError("scalar volatility must be a positive number")
        vol_expr = pl.lit(float(volatility), dtype=pl.Float64)
    else:
        vol_expr = parse_expr(volatility).cast(pl.Float64)

    side_expr = (
        parse_expr(side).cast(pl.Float64).sign().fill_null(1.0)
        if side is not None
        else pl.lit(1.0, dtype=pl.Float64)
    )

    h = int(horizon)
    pt_m = float(pt_mult)
    sl_m = float(sl_mult)
    min_r = float(min_ret)

    return_dtype = pl.Struct(
        [
            pl.Field("label", pl.Float64),
            pl.Field("touch_bars", pl.Int64),
            pl.Field("realized_return", pl.Float64),
            pl.Field("meta_label", pl.Float64),
        ]
    )

    def _calc_tb(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        c_vals = df_chunk["close"].to_numpy().astype(np.float64, copy=False)
        v_vals = df_chunk["vol"].to_numpy().astype(np.float64, copy=False)
        s_vals = df_chunk["side"].to_numpy().astype(np.float64, copy=False)
        n = len(c_vals)

        labels = np.full(n, np.nan, dtype=np.float64)
        touch_bars = np.full(n, -1, dtype=np.int64)
        realized_ret = np.full(n, np.nan, dtype=np.float64)
        meta_labels = np.full(n, np.nan, dtype=np.float64)

        if n > h:
            windows = np.lib.stride_tricks.sliding_window_view(c_vals, h + 1)
            m = len(windows)
            p0 = windows[:, :1]
            valid_mask = (
                (~np.isnan(windows).any(axis=1))
                & (p0[:, 0] > 0.0)
                & (~np.isnan(v_vals[:m]))
                & (s_vals[:m] != 0.0)
            )

            if valid_mask.any():
                valid_w = windows[valid_mask]
                valid_v = np.maximum(v_vals[:m][valid_mask], 1e-8)
                valid_s = np.sign(s_vals[:m][valid_mask])
                k = len(valid_w)

                raw_rets = (valid_w[:, 1:] - valid_w[:, :1]) / valid_w[:, :1]
                dir_rets = raw_rets * valid_s[:, None]

                if pt_m > 0.0:
                    u_thresh = np.maximum(pt_m * valid_v, min_r)[:, None]
                    hit_pt = dir_rets >= u_thresh
                    first_pt = np.where(hit_pt.any(axis=1), np.argmax(hit_pt, axis=1) + 1, h + 1)
                else:
                    first_pt = np.full(k, h + 1, dtype=np.int64)

                if sl_m > 0.0:
                    l_thresh = np.maximum(sl_m * valid_v, min_r)[:, None]
                    hit_sl = dir_rets <= -l_thresh
                    first_sl = np.where(hit_sl.any(axis=1), np.argmax(hit_sl, axis=1) + 1, h + 1)
                else:
                    first_sl = np.full(k, h + 1, dtype=np.int64)

                pt_wins = (first_pt < first_sl) & (first_pt <= h)
                sl_wins = (first_sl < first_pt) & (first_sl <= h)
                tie_hit = (first_pt == first_sl) & (first_pt <= h)

                lbl_trade = np.where(
                    pt_wins | (tie_hit & (dir_rets[:, 0] >= 0.0)),
                    1.0,
                    np.where(sl_wins | tie_hit, -1.0, 0.0),
                )
                lbl_valid = np.where(lbl_trade == 0.0, 0.0, lbl_trade * valid_s)
                t_bars_valid = np.minimum(np.minimum(first_pt, first_sl), h).astype(np.int64)
                row_idx = np.arange(k)
                r_ret_valid = raw_rets[row_idx, t_bars_valid - 1]
                meta_valid = np.where(lbl_trade > 0.0, 1.0, 0.0)

                labels_slice = labels[:m]
                t_bars_slice = touch_bars[:m]
                r_ret_slice = realized_ret[:m]
                meta_slice = meta_labels[:m]

                labels_slice[valid_mask] = lbl_valid
                t_bars_slice[valid_mask] = t_bars_valid
                r_ret_slice[valid_mask] = r_ret_valid
                meta_slice[valid_mask] = meta_valid

        tb_series = pl.Series("touch_bars", touch_bars, dtype=pl.Int64)
        tb_series = pl.select(
            pl.when(tb_series > 0).then(tb_series).otherwise(None).alias("touch_bars")
        )["touch_bars"]

        return pl.DataFrame(
            {
                "label": pl.Series(labels, dtype=pl.Float64).fill_nan(None),
                "touch_bars": tb_series,
                "realized_return": pl.Series(realized_ret, dtype=pl.Float64).fill_nan(None),
                "meta_label": pl.Series(meta_labels, dtype=pl.Float64).fill_nan(None),
            }
        ).to_struct("triple_barrier")

    combined = pl.struct(
        [
            c.alias("close"),
            vol_expr.alias("vol"),
            side_expr.alias("side"),
        ]
    )
    return combined.map_batches(_calc_tb, return_dtype=return_dtype)


def trend_scanning_labels(
    close: IntoExpr,
    min_window: int = 5,
    max_window: int = 20,
    step: int = 1,
) -> pl.Expr:
    """Trend-Scanning Multi-Horizon OLS Regression Labeling (López de Prado, 2020).

    Fits forward Ordinary Least Squares (OLS) linear regressions across all candidate
    lookahead windows $L \\in [\\text{min\\_window}, \\text{max\\_window}]$ (stepping by ``step``)
    using closed-form Polars expressions and selects the horizon $L^*$ that maximizes the
    absolute slope $t$-statistic $|\\hat{\\beta}_1(L) / \\text{SE}(\\hat{\\beta}_1(L))|$.

    Returns a ``pl.Struct`` with fields:
        - ``'label'``: Trend direction $\\text{sign}(t_{L^*}) \\in \\{-1.0, 0.0, 1.0\\}$
        - ``'t_value'``: Maximum-magnitude OLS slope $t$-statistic $t_{L^*}$
        - ``'optimal_window'``: Lookahead window length $L^*$ achieving $t_{L^*}$
        - ``'ret'``: Forward return $(P_{t + L^* - 1} / P_t) - 1$ over the optimal window

    :param close: Close price column or expression.
    :param min_window: Minimum forward regression window length $L \\ge 3$ (default 5).
    :param max_window: Maximum forward regression window length $L \\ge \\text{min\\_window}$ (default 20).
    :param step: Grid step size between candidate windows (default 1).
    :return: Polars Struct expression.
    """
    if min_window < 3:
        raise ValueError("min_window must be an integer greater than or equal to 3")
    if max_window < min_window:
        raise ValueError("max_window must be greater than or equal to min_window")
    if step <= 0:
        raise ValueError("step must be a positive integer")

    c = parse_expr(close).cast(pl.Float64)
    windows = list(range(min_window, max_window + 1, step))

    best_t: pl.Expr = pl.lit(None, dtype=pl.Float64)
    best_w: pl.Expr = pl.lit(None, dtype=pl.Int64)
    best_ret: pl.Expr = pl.lit(None, dtype=pl.Float64)

    for w in windows:
        shift_k = w - 1
        sma_w = c.rolling_mean(window_size=w)
        wma_w = _wma(c, period=w)
        slope_w = (wma_w - sma_w) * (6.0 / float(w - 1))

        s_xx = float(w * (w - 1) * (w + 1)) / 12.0
        s_yy = float(w - 1) * c.rolling_var(window_size=w)
        rss = (s_yy - (slope_w**2) * s_xx).clip(lower_bound=0.0)
        se_slope = (rss / (float(w - 2) * s_xx)).sqrt()

        t_trailing = (
            pl.when(slope_w.is_null() | s_yy.is_null())
            .then(None)
            .when(se_slope > 1e-12)
            .then((slope_w / se_slope).clip(-1e6, 1e6))
            .when(slope_w.abs() > 1e-12)
            .then(slope_w.sign() * 1e6)
            .otherwise(0.0)
        )
        t_fwd = t_trailing.shift(-shift_k)
        ret_fwd = (c.shift(-shift_k) / c) - 1.0

        take_this = t_fwd.is_not_null() & (best_t.is_null() | (t_fwd.abs() > best_t.abs()))
        best_t = pl.when(take_this).then(t_fwd).otherwise(best_t)
        best_w = pl.when(take_this).then(pl.lit(w, dtype=pl.Int64)).otherwise(best_w)
        best_ret = pl.when(take_this).then(ret_fwd).otherwise(best_ret)

    label_expr = pl.when(best_t.is_null()).then(None).otherwise(best_t.sign())

    return pl.struct(
        [
            label_expr.alias("label"),
            best_t.alias("t_value"),
            best_w.alias("optimal_window"),
            best_ret.alias("ret"),
        ]
    )
