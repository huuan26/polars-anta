"""Strategy performance, tail-risk, and Sharpe-inference expressions for Polars."""

from __future__ import annotations

import math
from statistics import NormalDist

import polars as pl

from ..core import IntoExpr, check_positive, parse_expr


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
