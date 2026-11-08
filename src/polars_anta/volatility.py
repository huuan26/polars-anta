"""Volatility indicators implemented natively for Polars."""

from __future__ import annotations

import math

import polars as pl

from .core import IntoExpr, parse_expr, true_range, wilder_smooth
from .trend import ema

__all__ = [
    "true_range",
    "atr",
    "bollinger_bands",
    "keltner_channels",
    "donchian_channels",
    "stddev",
    "ulcer_index",
    "chandelier_exit",
    "natr",
    "historical_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "yang_zhang_volatility",
    "chaikin_volatility",
    "donchian_width",
]


def atr(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Average True Range (ATR) using Wilder's smoothing.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    tr = true_range(high, low, close)
    return wilder_smooth(tr, period)


def bollinger_bands(
    close: IntoExpr,
    period: int = 20,
    multiplier: float = 2.0,
    num_std: float | None = None,
) -> pl.Expr:
    """Bollinger Bands (BBANDS).

    Returns a struct with fields:
        - 'upper': middle + multiplier * std
        - 'middle': SMA of close
        - 'lower': middle - multiplier * std
        - 'bandwidth': (upper - lower) / middle
        - 'percent_b': (close - lower) / (upper - lower)

    :param close: price column or expression
    :param period: lookback period (default 20)
    :param multiplier: standard deviation multiplier (default 2.0)
    :param num_std: alias for multiplier
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    mult = float(num_std) if num_std is not None else float(multiplier)
    c = parse_expr(close)
    middle = c.rolling_mean(window_size=period)
    std = c.rolling_std(window_size=period)

    upper = middle + mult * std
    lower = middle - mult * std
    spread = upper - lower

    bandwidth = pl.when(middle > 0.0).then(spread / middle).otherwise(0.0)
    percent_b = pl.when(spread > 0.0).then((c - lower) / spread).otherwise(0.5)

    return pl.struct(
        [
            upper.alias("upper"),
            middle.alias("middle"),
            lower.alias("lower"),
            bandwidth.alias("bandwidth"),
            percent_b.alias("percent_b"),
        ]
    )


def keltner_channels(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
    ema_period: int | None = None,
) -> pl.Expr:
    """Keltner Channels (KC).

    Returns a struct with fields:
        - 'upper': EMA(close, period) + multiplier * ATR(atr_period)
        - 'middle': EMA(close, period)
        - 'lower': EMA(close, period) - multiplier * ATR(atr_period)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: EMA lookback period (default 20)
    :param atr_period: ATR lookback period (default 10)
    :param multiplier: ATR multiplier (default 2.0)
    :param ema_period: alias for period
    :return: Polars Struct expression
    """
    effective_period = ema_period if ema_period is not None else period
    if effective_period <= 0 or atr_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    c = parse_expr(close)
    middle = ema(c, period=effective_period)
    band_range = atr(high, low, close, period=atr_period) * float(multiplier)

    upper = middle + band_range
    lower = middle - band_range

    return pl.struct(
        [
            upper.alias("upper"),
            middle.alias("middle"),
            lower.alias("lower"),
        ]
    )


def donchian_channels(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Donchian Channels.

    Returns a struct with fields:
        - 'upper': highest high over lookback period
        - 'middle': average of upper and lower channel
        - 'lower': lowest low over lookback period

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: lookback period (default 20)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)

    upper = hi.rolling_max(window_size=period)
    lower = lo.rolling_min(window_size=period)
    middle = (upper + lower) / 2.0

    return pl.struct(
        [
            upper.alias("upper"),
            middle.alias("middle"),
            lower.alias("lower"),
        ]
    )


def stddev(close: IntoExpr, period: int = 20) -> pl.Expr:
    """Rolling Standard Deviation (Volatility).

    :param close: price column name or expression
    :param period: lookback period (default 20)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    return parse_expr(close).rolling_std(window_size=period)


def ulcer_index(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Ulcer Index (UI).

    Technical indicator that measures downside risk and drawdown severity over a rolling period.

    :param close: price column name or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    highest_close = c.rolling_max(window_size=period)
    drawdown_pct = (
        pl.when(highest_close > 0.0)
        .then(100.0 * (c - highest_close) / highest_close)
        .otherwise(0.0)
    )
    drawdown_sq = drawdown_pct.pow(2)
    return drawdown_sq.rolling_mean(window_size=period).sqrt()


def chandelier_exit(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 22,
    multiplier: float = 3.0,
) -> pl.Expr:
    """Chandelier Exit trailing stop.

    Returns a struct with fields:
        - 'long_stop': Highest High minus (multiplier * ATR)
        - 'short_stop': Lowest Low plus (multiplier * ATR)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 22)
    :param multiplier: ATR multiplier (default 3.0)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    atr_val = atr(high, low, close, period=period)

    highest_high = hi.rolling_max(window_size=period)
    lowest_low = lo.rolling_min(window_size=period)

    long_stop = highest_high - float(multiplier) * atr_val
    short_stop = lowest_low + float(multiplier) * atr_val

    return pl.struct(
        [
            long_stop.alias("long_stop"),
            short_stop.alias("short_stop"),
        ]
    )


def natr(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Normalized Average True Range (NATR).

    Normalizes ATR as a percentage of Close price to allow comparison across different instruments.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    atr_val = atr(high, low, close, period=period)

    return pl.when(c > 0.0).then(100.0 * atr_val / c).otherwise(0.0)


def historical_volatility(
    close: IntoExpr,
    period: int = 20,
    trading_periods: int = 252,
) -> pl.Expr:
    """Annualized Historical Volatility (HV).

    Calculates the annualized standard deviation of logarithmic price returns.

    :param close: price column or expression
    :param period: rolling lookback window (default 20)
    :param trading_periods: trading periods per year (default 252 for daily)
    :return: Polars expression in percentage [0, inf)
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")
    c = parse_expr(close)
    prev_c = c.shift(1)
    log_return = pl.when((c > 0.0) & (prev_c > 0.0)).then((c / prev_c).log()).otherwise(0.0)
    ann_factor = math.sqrt(float(trading_periods)) * 100.0

    return log_return.rolling_std(window_size=period) * ann_factor


def parkinson_volatility(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 20,
    trading_periods: int = 252,
) -> pl.Expr:
    """Parkinson Volatility (High-Low estimator).

    An extreme-value volatility estimator that uses High and Low prices, ~5x more efficient than Close-Close.

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: rolling window size (default 20)
    :param trading_periods: trading periods per year (default 252)
    :return: Polars expression in percentage
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)

    hl_ratio = pl.when((hi > 0.0) & (lo > 0.0)).then((hi / lo).log()).otherwise(0.0)
    hl_sq = hl_ratio.pow(2)
    factor = 1.0 / (4.0 * math.log(2.0))
    ann_factor = math.sqrt(float(trading_periods)) * 100.0

    mean_sq = hl_sq.rolling_mean(window_size=period)
    return (mean_sq * factor).sqrt() * ann_factor


def garman_klass_volatility(
    open: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    trading_periods: int = 252,
) -> pl.Expr:
    """Garman-Klass Volatility (OHLC estimator).

    Combines both High-Low range and Close-Open jump, ~8x more efficient than standard deviation.

    :param open: open price column or expression
    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: rolling window size (default 20)
    :param trading_periods: trading periods per year (default 252)
    :return: Polars expression in percentage
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    o = parse_expr(open)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    hl_term = pl.when((hi > 0.0) & (lo > 0.0)).then((hi / lo).log().pow(2)).otherwise(0.0)
    co_term = pl.when((c > 0.0) & (o > 0.0)).then((c / o).log().pow(2)).otherwise(0.0)

    gk_const = 2.0 * math.log(2.0) - 1.0
    val = 0.5 * hl_term - gk_const * co_term
    mean_val = val.rolling_mean(window_size=period)
    ann_factor = math.sqrt(float(trading_periods)) * 100.0

    return pl.when(mean_val > 0.0).then(mean_val.sqrt() * ann_factor).otherwise(0.0)


def rogers_satchell_volatility(
    open: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    trading_periods: int = 252,
) -> pl.Expr:
    """Rogers-Satchell Volatility (drift-independent OHLC estimator).

    Rogers & Satchell (1991). Unlike close-to-close volatility this estimator
    stays unbiased when the underlying has a non-zero trend, and unlike
    Parkinson it uses the full OHLC information rather than High-Low only.

    :param open: open price column or expression
    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: rolling window size (default 20)
    :param trading_periods: trading periods per year (default 252)
    :return: Polars expression in percentage
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    o = parse_expr(open)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    valid = (hi > 0.0) & (lo > 0.0) & (o > 0.0) & (c > 0.0)
    term = (
        pl.when(valid)
        .then((hi / c).log() * (hi / o).log() + (lo / c).log() * (lo / o).log())
        .otherwise(0.0)
    )

    mean_term = term.rolling_mean(window_size=period)
    ann_factor = math.sqrt(float(trading_periods)) * 100.0

    return pl.when(mean_term > 0.0).then(mean_term.sqrt() * ann_factor).otherwise(0.0)


def yang_zhang_volatility(
    open: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    trading_periods: int = 252,
) -> pl.Expr:
    """Yang-Zhang Volatility (overnight-jump and drift adjusted OHLC estimator).

    Yang & Zhang (2000). Combines the overnight variance, the open-to-close
    variance and the Rogers-Satchell variance with the minimum-variance weight
    ``k = 0.34 / (1.34 + (n + 1) / (n - 1))``. This is the only one of the
    canonical OHLC estimators that stays unbiased under both an opening jump
    and an intraday drift.

    :param open: open price column or expression
    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: rolling window size, must be greater than 1 (default 20)
    :param trading_periods: trading periods per year (default 252)
    :return: Polars expression in percentage
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")
    o = parse_expr(open)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    prev_c = c.shift(1)

    valid = (hi > 0.0) & (lo > 0.0) & (o > 0.0) & (c > 0.0)
    rs_term = (
        pl.when(valid)
        .then((hi / c).log() * (hi / o).log() + (lo / c).log() * (lo / o).log())
        .otherwise(0.0)
    )
    overnight = pl.when(valid & (prev_c > 0.0)).then((o / prev_c).log()).otherwise(0.0)
    intraday = pl.when(valid).then((c / o).log()).otherwise(0.0)

    var_overnight = overnight.rolling_var(window_size=period)
    var_intraday = intraday.rolling_var(window_size=period)
    var_rs = rs_term.rolling_mean(window_size=period)

    k = 0.34 / (1.34 + float(period + 1) / float(period - 1))
    combined = var_overnight + k * var_intraday + (1.0 - k) * var_rs
    ann = float(trading_periods)

    return pl.when(combined > 0.0).then((combined * ann).sqrt() * 100.0).otherwise(0.0)


def chaikin_volatility(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 10,
    roc_period: int = 10,
) -> pl.Expr:
    """Chaikin's Volatility (CHV).

    Measures percentage change in the exponential moving average of the High-Low spread.

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: EMA lookback period (default 10)
    :param roc_period: rate of change shift lookback (default 10)
    :return: Polars expression
    """
    if period <= 0 or roc_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    hl = hi - lo
    ema_hl = ema(hl, period=period)
    prev_ema = ema_hl.shift(roc_period)

    return pl.when(prev_ema > 0.0).then(100.0 * (ema_hl - prev_ema) / prev_ema).otherwise(0.0)


def donchian_width(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Donchian Channel Width (Bandwidth).

    Measures the percentage spread of the Donchian Channels relative to the midline.

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: lookback period (default 20)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    upper = hi.rolling_max(window_size=period)
    lower = lo.rolling_min(window_size=period)
    middle = (upper + lower) / 2.0

    return pl.when(middle > 0.0).then(100.0 * (upper - lower) / middle).otherwise(0.0)


def envelopes(
    close: IntoExpr,
    period: int = 20,
    percentage: float = 0.025,
) -> pl.Expr:
    """Moving Average Envelopes.

    Trading bands plotted above and below a moving average by a fixed percentage.
    Returns Struct with: `upper`, `middle`, `lower`.

    :param close: price column or expression
    :param period: moving average period (default 20)
    :param percentage: band percentage offset as a decimal (e.g. 0.025 for 2.5%)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    middle = c.rolling_mean(window_size=period)
    upper = middle * (1.0 + percentage)
    lower = middle * (1.0 - percentage)

    return pl.struct([upper.alias("upper"), middle.alias("middle"), lower.alias("lower")])


def keltner_width(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pl.Expr:
    """Normalized Keltner Channel Width (Squeeze metric).

    Calculates the percentage spread of Keltner Channels relative to the EMA midline:
    (Upper - Lower) / Middle * 100.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param ema_period: period for center EMA line (default 20)
    :param atr_period: period for ATR volatility band (default 10)
    :param multiplier: ATR multiplier (default 2.0)
    :return: Polars expression
    """
    c = parse_expr(close)
    middle = c.ewm_mean(span=ema_period, adjust=False)
    atr_val = atr(high, low, close, period=atr_period)
    spread = 2.0 * multiplier * atr_val
    return pl.when(middle > 0.0).then(100.0 * spread / middle).otherwise(0.0)


def bollinger_delta(
    close: IntoExpr,
    period: int = 20,
    num_std: float = 2.0,
) -> pl.Expr:
    """Bollinger Band Delta (Band Spread).

    Difference between Upper Band and Lower Band: 2 * num_std * stddev.

    :param close: price column or expression
    :param period: rolling lookback period (default 20)
    :param num_std: standard deviation multiplier (default 2.0)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    std = c.rolling_std(window_size=period)
    return 2.0 * num_std * std


def starc_bands(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    atr_period: int = 15,
    multiplier: float = 2.0,
) -> pl.Expr:
    """STARC Bands (Stoller Average Range Channels).

    Volatility channel combining an SMA baseline with ATR-derived boundaries.
    Returns Struct with: `upper`, `middle`, `lower`.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: SMA midline period (default 20)
    :param atr_period: ATR lookback period (default 15)
    :param multiplier: ATR band multiplier (default 2.0)
    :return: Polars Struct expression
    """
    if period <= 0 or atr_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    c = parse_expr(close)
    middle = c.rolling_mean(window_size=period)
    atr_val = atr(high, low, close, period=atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val

    return pl.struct([upper.alias("upper"), middle.alias("middle"), lower.alias("lower")])
