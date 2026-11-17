"""WorldQuant 101 formulaic alphas (alpha_001 to alpha_015)."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..core import IntoExpr, parse_expr
from ..volume import vwap as _calc_vwap
from ._primitives import (
    cs_rank,
    signed_power,
    ts_argmax,
    ts_correlation,
    ts_covariance,
    ts_delay,
    ts_delta,
    ts_max,
    ts_min,
    ts_rank,
    ts_sum,
)


def _persistence_or_reversal(close: IntoExpr, window: int) -> pl.Expr:
    """Shared short-horizon trend persistence vs. mean-reversion switch for Alpha #9 and #10."""
    d1 = ts_delta(parse_expr(close), 1)
    lo = ts_min(d1, window)
    hi = ts_max(d1, window)
    return (
        pl.when(lo.is_null() | hi.is_null())
        .then(None)
        .when((lo > 0.0) | (hi < 0.0))
        .then(d1)
        .otherwise(-1.0 * d1)
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
    return _persistence_or_reversal(close, 5)


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
    return cs_rank(_persistence_or_reversal(close, 4), over=over)


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
