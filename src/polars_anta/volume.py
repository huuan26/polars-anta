"""Volume-based indicators implemented natively for Polars."""

from __future__ import annotations

import polars as pl

from .core import IntoExpr, parse_expr
from .trend import ema, sma


def obv(close: IntoExpr, volume: IntoExpr) -> pl.Expr:
    """On-Balance Volume (OBV).

    Measures buying and selling pressure as a cumulative indicator, adding
    volume on up days and subtracting it on down days.

    :param close: close price column name or expression
    :param volume: volume column name or expression
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    diff = c.diff(1)

    signed_vol = pl.when(diff > 0).then(v).when(diff < 0).then(-v).otherwise(0)
    return signed_vol.cum_sum()


def vwap(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    period: int | None = None,
) -> pl.Expr:
    """Volume Weighted Average Price (VWAP).

    Calculates the average price weighted by volume. If `period` is None,
    computes cumulative session VWAP; otherwise computes rolling VWAP.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback window size, or None for cumulative VWAP
    :return: Polars expression
    """
    if period is not None and period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    typical_price = (hi + lo + c) / 3.0
    pv = typical_price * v

    if period is None:
        return pv.cum_sum() / v.cum_sum()
    return pv.rolling_sum(window_size=period) / v.rolling_sum(window_size=period)


def vwap_bands(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    period: int | None = None,
    mult1: float = 1.0,
    mult2: float = 2.0,
    mult3: float = 3.0,
) -> pl.Expr:
    """Volume-Weighted Average Price (VWAP) with Standard Deviation Bands.

    Computes institutional VWAP along with 1, 2, and 3 standard deviation bands.
    If `period` is None, computes cumulative session VWAP; otherwise computes rolling VWAP.

    Returns a struct with fields:
        - 'vwap': Volume-Weighted Average Price
        - 'upper_1', 'lower_1': +/- mult1 * standard deviation
        - 'upper_2', 'lower_2': +/- mult2 * standard deviation
        - 'upper_3', 'lower_3': +/- mult3 * standard deviation

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback window size, or None for cumulative session VWAP
    :param mult1: first band multiplier (default 1.0)
    :param mult2: second band multiplier (default 2.0)
    :param mult3: third band multiplier (default 3.0)
    :return: Polars Struct expression
    """
    if period is not None and period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    tp = (hi + lo + c) / 3.0
    pv = tp * v
    pv2 = (tp**2) * v

    if period is None:
        cum_v = v.cum_sum()
        vwap_val = pv.cum_sum() / cum_v
        vwap2_val = pv2.cum_sum() / cum_v
    else:
        roll_v = v.rolling_sum(window_size=period)
        vwap_val = pv.rolling_sum(window_size=period) / roll_v
        vwap2_val = pv2.rolling_sum(window_size=period) / roll_v

    # Weighted variance = E[X^2] - (E[X])^2
    variance = (vwap2_val - vwap_val**2).clip(0.0)
    std_val = variance.sqrt()

    return pl.struct(
        [
            vwap_val.alias("vwap"),
            (vwap_val + float(mult1) * std_val).alias("upper_1"),
            (vwap_val - float(mult1) * std_val).alias("lower_1"),
            (vwap_val + float(mult2) * std_val).alias("upper_2"),
            (vwap_val - float(mult2) * std_val).alias("lower_2"),
            (vwap_val + float(mult3) * std_val).alias("upper_3"),
            (vwap_val - float(mult3) * std_val).alias("lower_3"),
        ]
    )


def vwma(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Volume-Weighted Moving Average (VWMA).

    Weights price by volume over a rolling window.

    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback period (default 20)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    c = parse_expr(close)
    v = parse_expr(volume)

    return (c * v).rolling_sum(window_size=period) / v.rolling_sum(window_size=period)


def adl(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
) -> pl.Expr:
    """Accumulation/Distribution Line (ADL).

    Measures cumulative flow of money into and out of a security.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :return: Polars expression
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    hl_range = hi - lo
    mfm = pl.when(hl_range != 0.0).then(((c - lo) - (hi - c)) / hl_range).otherwise(0.0)
    mfv = mfm * v
    return mfv.cum_sum()


def cmf(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Chaikin Money Flow (CMF).

    Measures money flow volume over a specified rolling period.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback period (default 20)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    hl_range = hi - lo
    mfm = pl.when(hl_range != 0.0).then(((c - lo) - (hi - c)) / hl_range).otherwise(0.0)
    mfv = mfm * v

    sum_mfv = mfv.rolling_sum(window_size=period)
    sum_vol = v.rolling_sum(window_size=period)

    return pl.when(sum_vol != 0.0).then(sum_mfv / sum_vol).otherwise(0.0)


def mfi(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Money Flow Index (MFI).

    A volume-weighted RSI oscillating between 0 and 100.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    typical_price = (hi + lo + c) / 3.0
    raw_money_flow = typical_price * v
    diff = typical_price.diff(1)

    pos_flow = pl.when(diff > 0.0).then(raw_money_flow).otherwise(0.0)
    neg_flow = pl.when(diff < 0.0).then(raw_money_flow).otherwise(0.0)

    pos_sum = pos_flow.rolling_sum(window_size=period)
    neg_sum = neg_flow.rolling_sum(window_size=period)

    money_ratio = pos_sum / neg_sum

    return (
        pl.when(neg_sum == 0.0)
        .then(100.0)
        .when(pos_sum == 0.0)
        .then(0.0)
        .otherwise(100.0 - (100.0 / (1.0 + money_ratio)))
    )


def eom(
    high: IntoExpr,
    low: IntoExpr,
    volume: IntoExpr,
    period: int = 14,
    divisor: float = 100_000_000.0,
) -> pl.Expr:
    """Ease of Movement (EOM / EMV).

    Relates price change to trading volume over a lookback window.

    :param high: high price column or expression
    :param low: low price column or expression
    :param volume: volume column or expression
    :param period: moving average smoothing window (default 14)
    :param divisor: volume scaling divisor (default 100,000,000)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    v = parse_expr(volume)

    hl_mid = (hi + lo) / 2.0
    dist_moved = hl_mid - hl_mid.shift(1)
    hl_range = hi - lo

    box_ratio = (v / float(divisor)) / hl_range
    emv_1 = (
        pl.when((hl_range > 0.0) & (box_ratio > 0.0)).then(dist_moved / box_ratio).otherwise(0.0)
    )

    return emv_1.rolling_mean(window_size=period)


def klinger(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    fast_period: int = 34,
    slow_period: int = 55,
    signal_period: int = 13,
) -> pl.Expr:
    """Klinger Volume Oscillator (KVO).

    Determines long-term trend of money flow while remaining sensitive to short-term fluctuations.

    Returns a struct with fields:
        - 'klinger': KVO oscillator line
        - 'signal': EMA of KVO line

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param fast_period: fast EMA period (default 34)
    :param slow_period: slow EMA period (default 55)
    :param signal_period: signal EMA period (default 13)
    :return: Polars Struct expression
    """
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    tp = (hi + lo + c) / 3.0
    direction = pl.when(tp > tp.shift(1)).then(1.0).when(tp < tp.shift(1)).then(-1.0).otherwise(0.0)
    sv = v * direction

    fast_ema = ema(sv, period=fast_period)
    slow_ema = ema(sv, period=slow_period)
    klinger_line = fast_ema - slow_ema
    signal_line = ema(klinger_line, period=signal_period)

    return pl.struct(
        [
            klinger_line.alias("klinger"),
            signal_line.alias("signal"),
        ]
    )


def nvi(
    close: IntoExpr,
    volume: IntoExpr,
    initial: float = 1000.0,
) -> pl.Expr:
    """Negative Volume Index (NVI).

    Cumulative indicator that tracks price trend on days where volume decreased.

    :param close: price column or expression
    :param volume: volume column or expression
    :param initial: starting index level (default 1000.0)
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    diff_v = v.diff(1)
    ret = pl.when(diff_v < 0).then(c.pct_change()).otherwise(0.0).fill_null(0.0)

    return (1.0 + ret).cum_prod() * float(initial)


def pvi(
    close: IntoExpr,
    volume: IntoExpr,
    initial: float = 1000.0,
) -> pl.Expr:
    """Positive Volume Index (PVI).

    Cumulative indicator that tracks price trend on days where volume increased.

    :param close: price column or expression
    :param volume: volume column or expression
    :param initial: starting index level (default 1000.0)
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    diff_v = v.diff(1)
    ret = pl.when(diff_v > 0).then(c.pct_change()).otherwise(0.0).fill_null(0.0)

    return (1.0 + ret).cum_prod() * float(initial)


def pvt(close: IntoExpr, volume: IntoExpr) -> pl.Expr:
    """Price Volume Trend (PVT).

    Cumulative volume momentum indicator adjusted by percentage price changes.

    :param close: price column or expression
    :param volume: volume column or expression
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    ret = c.pct_change().fill_null(0.0)
    return (ret * v).cum_sum()


def force_index(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 13,
) -> pl.Expr:
    """Elder's Force Index (EFI).

    Illustrates the force and conviction behind price movements by combining price change and volume.

    :param close: price column or expression
    :param volume: volume column or expression
    :param period: EMA smoothing window (default 13)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    v = parse_expr(volume)
    raw_fi = c.diff(1).fill_null(0.0) * v

    return ema(raw_fi, period=period)


def vpci(
    close: IntoExpr,
    volume: IntoExpr,
    period_short: int = 5,
    period_long: int = 25,
) -> pl.Expr:
    """Volume Price Confirmation Indicator (VPCI).

    Measures the intrinsic relationship between volume and price changes.

    :param close: price column or expression
    :param volume: volume column or expression
    :param period_short: short window (default 5)
    :param period_long: long window (default 25)
    :return: Polars expression
    """
    if period_short <= 0 or period_long <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    c = parse_expr(close)
    v = parse_expr(volume)

    vpc = vwma(c, v, period=period_short) - sma(c, period=period_short)
    vpr = vwma(c, v, period=period_long) / sma(c, period=period_long)
    vm = sma(v, period=period_short) / sma(v, period=period_long)

    return vpc * vpr * vm


def vr(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 26,
) -> pl.Expr:
    """Volume Ratio (VR).

    Quantifies the ratio between up-day and down-day volume over a lookback window.

    :param close: price column or expression
    :param volume: volume column or expression
    :param period: lookback period (default 26)
    :return: Polars expression in percentage [0, inf)
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    v = parse_expr(volume)
    change = c.diff(1)

    up_vol = pl.when(change > 0).then(v).otherwise(0)
    down_vol = pl.when(change < 0).then(v).otherwise(0)
    flat_vol = pl.when(change == 0).then(v).otherwise(0)

    sum_up = up_vol.rolling_sum(window_size=period)
    sum_down = down_vol.rolling_sum(window_size=period)
    sum_flat = flat_vol.rolling_sum(window_size=period)

    den = sum_down + 0.5 * sum_flat
    return pl.when(den > 0.0).then(100.0 * (sum_up + 0.5 * sum_flat) / den).otherwise(100.0)
