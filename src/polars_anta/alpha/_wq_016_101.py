"""WorldQuant 101 formulaic alphas (alpha_016 to alpha_101)."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..core import IntoExpr, parse_expr
from ..volume import vwap as _calc_vwap
from ._primitives import (
    cs_rank,
    ts_correlation,
    ts_covariance,
    ts_delay,
    ts_delta,
    ts_max,
    ts_min,
    ts_rank,
    ts_scale,
    ts_std,
    ts_sum,
)


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
