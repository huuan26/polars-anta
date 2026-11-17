"""Trend-momentum and composite cycle oscillators implemented natively for Polars."""

from __future__ import annotations

import polars as pl

from ..core import IntoExpr, check_period, check_periods, parse_expr, true_range
from ..trend import ema, wma
from ._oscillators import roc


def macd(
    close: IntoExpr,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.Expr:
    """Moving Average Convergence Divergence (MACD).

    Returns a struct with fields:
        - 'macd': difference between fast and slow EMA
        - 'signal': EMA of the MACD line
        - 'histogram': MACD line minus signal line

    :param close: price column name or expression
    :param fast_period: fast EMA period (default 12)
    :param slow_period: slow EMA period (default 26)
    :param signal_period: signal line EMA period (default 9)
    :return: Polars Struct expression
    """
    check_periods(fast_period, slow_period, signal_period)
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    c = parse_expr(close)
    macd_line = c.ewm_mean(span=fast_period, adjust=False) - c.ewm_mean(
        span=slow_period, adjust=False
    )
    signal_line = macd_line.ewm_mean(span=signal_period, adjust=False)

    return pl.struct(
        [
            macd_line.alias("macd"),
            signal_line.alias("signal"),
            (macd_line - signal_line).alias("histogram"),
        ]
    )


def awesome_oscillator(
    high: IntoExpr,
    low: IntoExpr,
    fast_period: int = 5,
    slow_period: int = 34,
) -> pl.Expr:
    """Awesome Oscillator (AO).

    Difference between a fast (default 5) and slow (default 34) SMA of the median price (High + Low) / 2.

    :param high: high price column or expression
    :param low: low price column or expression
    :param fast_period: fast SMA period (default 5)
    :param slow_period: slow SMA period (default 34)
    :return: Polars expression
    """
    check_periods(fast_period, slow_period)
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    median_price = (parse_expr(high) + parse_expr(low)) / 2.0
    return median_price.rolling_mean(window_size=fast_period) - median_price.rolling_mean(
        window_size=slow_period
    )


def tsi(
    close: IntoExpr,
    long_period: int = 25,
    short_period: int = 13,
) -> pl.Expr:
    """True Strength Index (TSI).

    Momentum oscillator using double-smoothed exponential moving averages of price changes.
    Values oscillate between -100 and +100.

    :param close: price column or expression
    :param long_period: first smoothing span (default 25)
    :param short_period: second smoothing span (default 13)
    :return: Polars expression
    """
    check_periods(long_period, short_period)

    diff = parse_expr(close).diff(1)
    smooth_2 = diff.ewm_mean(span=long_period, adjust=False).ewm_mean(
        span=short_period, adjust=False
    )
    abs_smooth_2 = (
        diff.abs()
        .ewm_mean(span=long_period, adjust=False)
        .ewm_mean(span=short_period, adjust=False)
    )
    return pl.when(abs_smooth_2 > 0.0).then(100.0 * smooth_2 / abs_smooth_2).otherwise(0.0)


def ppo(
    close: IntoExpr,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.Expr:
    """Percentage Price Oscillator (PPO).

    Returns a struct with fields:
        - 'ppo': Percentage Price Oscillator line
        - 'signal': EMA of PPO line
        - 'histogram': PPO line minus signal line

    :param close: price column or expression
    :param fast_period: fast EMA period (default 12)
    :param slow_period: slow EMA period (default 26)
    :param signal_period: signal line EMA period (default 9)
    :return: Polars Struct expression
    """
    check_periods(fast_period, slow_period, signal_period)
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    c = parse_expr(close)
    fast_ema = ema(c, period=fast_period)
    slow_ema = ema(c, period=slow_period)

    ppo_line = (
        pl.when(slow_ema != 0.0).then(100.0 * (fast_ema - slow_ema) / slow_ema).otherwise(0.0)
    )
    signal_line = ema(ppo_line, period=signal_period)

    return pl.struct(
        [
            ppo_line.alias("ppo"),
            signal_line.alias("signal"),
            (ppo_line - signal_line).alias("histogram"),
        ]
    )


def trix(
    close: IntoExpr,
    period: int = 15,
    signal_period: int = 9,
) -> pl.Expr:
    """Triple Exponential Average (TRIX) Oscillator.

    Returns a Polars Struct with fields:
        - 'trix': 1-bar percentage ROC of triple EMA
        - 'signal': EMA of trix over signal_period

    :param close: price column name or expression
    :param period: EMA lookback applied three times (default 15)
    :param signal_period: signal line EMA period (default 9)
    :return: Polars Struct expression
    """
    check_periods(period, signal_period)

    ema_3 = ema(ema(ema(parse_expr(close), period=period), period=period), period=period)
    prev = ema_3.shift(1)
    # A missing prior triple EMA (leading bar, or a null smooth) stays null.
    # A zero prior smooth has an undefined percentage change and is reported as 0.
    trix_line = (
        pl.when(prev.is_not_null() & (prev != 0.0))
        .then(100.0 * (ema_3 - prev) / prev)
        .when(prev.is_not_null())
        .then(0.0)
    )
    return pl.struct(
        [
            trix_line.alias("trix"),
            ema(trix_line, period=signal_period).alias("signal"),
        ]
    )


def coppock(
    close: IntoExpr,
    wma_period: int = 10,
    fast_roc: int = 11,
    slow_roc: int = 14,
) -> pl.Expr:
    """Coppock Curve.

    Long-term price momentum indicator calculated as WMA of the sum of two ROCs.

    :param close: price column or expression
    :param wma_period: WMA smoothing period (default 10)
    :param fast_roc: fast ROC period (default 11)
    :param slow_roc: slow ROC period (default 14)
    :return: Polars expression
    """
    check_periods(wma_period, fast_roc, slow_roc)
    c = parse_expr(close)
    return wma(roc(c, period=fast_roc) + roc(c, period=slow_roc), period=wma_period)


def fisher_transform(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 9,
) -> pl.Expr:
    """Fisher Transform.

    Converts price distribution to a Gaussian normal distribution to pinpoint turning points.

    Returns a struct with fields:
        - 'fisher': Fisher Transform line
        - 'signal': Trigger signal line (Fisher shifted by 1)

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: lookback window (default 9)
    :return: Polars Struct expression
    """
    check_period(period)

    hi = parse_expr(high)
    lo = parse_expr(low)
    min_l = lo.rolling_min(window_size=period)
    hl_diff = hi.rolling_max(window_size=period) - min_l
    mid = (hi + lo) / 2.0

    raw_x = pl.when(hl_diff > 0.0).then(2.0 * ((mid - min_l) / hl_diff - 0.5)).otherwise(0.0)
    smoothed_x = raw_x.ewm_mean(span=3, adjust=False).clip(-0.999, 0.999)
    fisher = 0.5 * ((1.0 + smoothed_x) / (1.0 - smoothed_x)).log()

    return pl.struct(
        [
            fisher.alias("fisher"),
            fisher.shift(1).alias("signal"),
        ]
    )


def rvi(
    open: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Relative Vigor Index (RVI).

    Measures conviction of price action using close-to-open and high-to-low ranges.

    Returns a struct with fields:
        - 'rvi': RVI oscillator line
        - 'signal': 4-period triangular smoothed signal line

    :param open: open price column or expression
    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars Struct expression
    """
    check_period(period)

    def _tri4(x: pl.Expr) -> pl.Expr:
        return (x + 2.0 * x.shift(1) + 2.0 * x.shift(2) + x.shift(3)) / 6.0

    sum_num = _tri4(parse_expr(close) - parse_expr(open)).rolling_sum(window_size=period)
    sum_den = _tri4(parse_expr(high) - parse_expr(low)).rolling_sum(window_size=period)
    rvi_line = pl.when(sum_den > 0.0).then(sum_num / sum_den).otherwise(0.0)

    return pl.struct(
        [
            rvi_line.alias("rvi"),
            _tri4(rvi_line).alias("signal"),
        ]
    )


def kst(
    close: IntoExpr,
    r1: int = 10,
    r2: int = 15,
    r3: int = 20,
    r4: int = 30,
    s1: int = 10,
    s2: int = 10,
    s3: int = 10,
    s4: int = 15,
    signal_period: int = 9,
) -> pl.Expr:
    """Know Sure Thing (KST).

    Martin Pring's multi-timeframe ROC momentum oscillator.

    Returns a struct with fields:
        - 'kst': KST oscillator line
        - 'signal': SMA of KST line

    :param close: price column or expression
    :param r1: ROC 1 period (default 10)
    :param r2: ROC 2 period (default 15)
    :param r3: ROC 3 period (default 20)
    :param r4: ROC 4 period (default 30)
    :param s1: SMA 1 period (default 10)
    :param s2: SMA 2 period (default 10)
    :param s3: SMA 3 period (default 10)
    :param s4: SMA 4 period (default 15)
    :param signal_period: Signal SMA period (default 9)
    :return: Polars Struct expression
    """
    c = parse_expr(close)
    roc1_sma = roc(c, period=r1).rolling_mean(window_size=s1)
    roc2_sma = roc(c, period=r2).rolling_mean(window_size=s2)
    roc3_sma = roc(c, period=r3).rolling_mean(window_size=s3)
    roc4_sma = roc(c, period=r4).rolling_mean(window_size=s4)

    kst_line = roc1_sma * 1.0 + roc2_sma * 2.0 + roc3_sma * 3.0 + roc4_sma * 4.0
    signal = kst_line.rolling_mean(window_size=signal_period)

    return pl.struct(
        [
            kst_line.alias("kst"),
            signal.alias("signal"),
        ]
    )


def dpo(close: IntoExpr, period: int = 20) -> pl.Expr:
    """Detrended Price Oscillator (DPO).

    Eliminates trend from price to identify cycles and overbought/oversold levels.

    :param close: price column or expression
    :param period: lookback period (default 20)
    :return: Polars expression
    """
    check_period(period)
    c = parse_expr(close)
    return c - c.rolling_mean(window_size=period).shift(period // 2 + 1)


def squeeze_momentum(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    mom_period: int | None = None,
) -> pl.Expr:
    """Squeeze Momentum Indicator (John Carter / LazyBear).

    Combines Bollinger Bands and Keltner Channels to identify periods of consolidation
    (the squeeze) followed by volatility expansion breakouts, accompanied by a momentum histogram.

    Returns a struct with fields:
        - 'squeeze_on': True when Bollinger Bands are inside Keltner Channels (consolidation)
        - 'squeeze_off': True when Bollinger Bands have expanded outside Keltner Channels (breakout)
        - 'no_squeeze': True when neither squeeze_on nor squeeze_off
        - 'momentum': Linear regression endpoint of price relative to the average of Donchian midline and SMA

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param bb_period: Bollinger Bands lookback period (default 20)
    :param bb_mult: Bollinger Bands standard deviation multiplier (default 2.0)
    :param kc_period: Keltner Channels lookback period (default 20)
    :param kc_mult: Keltner Channels range multiplier (default 1.5)
    :param mom_period: Momentum linear regression lookback period (default None, uses kc_period)
    :return: Polars Struct expression
    """
    check_periods(bb_period, kc_period)
    effective_mom = mom_period if mom_period is not None else kc_period
    check_period(effective_mom, name="mom_period")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    basis_bb = c.rolling_mean(window_size=bb_period)
    dev_bb = c.rolling_std(window_size=bb_period) * float(bb_mult)
    upper_bb = basis_bb + dev_bb
    lower_bb = basis_bb - dev_bb

    basis_kc = c.rolling_mean(window_size=kc_period)
    range_kc = true_range(hi, lo, c).rolling_mean(window_size=kc_period) * float(kc_mult)
    upper_kc = basis_kc + range_kc
    lower_kc = basis_kc - range_kc

    null_mask = upper_bb.is_null() | upper_kc.is_null()
    sqz_on = pl.when(null_mask).then(None).otherwise((lower_bb > lower_kc) & (upper_bb < upper_kc))
    sqz_off = pl.when(null_mask).then(None).otherwise((lower_bb < lower_kc) & (upper_bb > upper_kc))
    no_sqz = pl.when(null_mask).then(None).otherwise((~sqz_on) & (~sqz_off))

    avg_hl = (hi.rolling_max(window_size=kc_period) + lo.rolling_min(window_size=kc_period)) / 2.0
    delta = c - (avg_hl + basis_kc) / 2.0
    momentum = 3.0 * wma(delta, period=effective_mom) - 2.0 * delta.rolling_mean(
        window_size=effective_mom
    )

    return pl.struct(
        [
            sqz_on.alias("squeeze_on"),
            sqz_off.alias("squeeze_off"),
            no_sqz.alias("no_squeeze"),
            momentum.alias("momentum"),
        ]
    )


def stc(
    close: IntoExpr,
    fast_period: int = 23,
    slow_period: int = 50,
    cycle_period: int = 10,
    factor: float = 0.5,
) -> pl.Expr:
    """Schaff Trend Cycle (STC) Oscillator.

    Applies a double-smoothed stochastic calculation to a MACD line to identify
    market cycles with reduced lag and fewer false signals.

    Formula (per common implementations matching the project issue specification):
    1. MACD = EMA(close, fast) - EMA(close, slow)
    2. %K1 = 100 * (MACD - min(MACD, cycle)) / (max(MACD, cycle) - min(MACD, cycle))
    3. %D1 = EMA(%K1, span=factor * cycle)
    4. %K2 = 100 * (%D1 - min(%D1, cycle)) / (max(%D1, cycle) - min(%D1, cycle))
    5. STC = EMA(%K2, span=factor * cycle)

    :param close: price column name or expression
    :param fast_period: fast EMA period for MACD (default 23)
    :param slow_period: slow EMA period for MACD (default 50)
    :param cycle_period: stochastic cycle lookback (default 10)
    :param factor: smoothing factor applied as EMA span = factor * cycle_period (default 0.5)
    :return: Polars expression producing STC values typically in [0, 100]
    """
    check_periods(fast_period, slow_period, cycle_period)
    if not (0.0 < factor <= 1.0):
        raise ValueError("factor must be in (0, 1]")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    c = parse_expr(close)
    macd_line = c.ewm_mean(span=fast_period, adjust=False) - c.ewm_mean(
        span=slow_period, adjust=False
    )

    lowest_macd = macd_line.rolling_min(window_size=cycle_period)
    highest_macd = macd_line.rolling_max(window_size=cycle_period)
    spread1 = highest_macd - lowest_macd
    k1 = pl.when(spread1 > 0.0).then(100.0 * (macd_line - lowest_macd) / spread1).otherwise(50.0)

    smooth_span = max(1, int(round(factor * cycle_period)))
    d1 = k1.ewm_mean(span=smooth_span, adjust=False)

    lowest_d1 = d1.rolling_min(window_size=cycle_period)
    highest_d1 = d1.rolling_max(window_size=cycle_period)
    spread2 = highest_d1 - lowest_d1
    k2 = pl.when(spread2 > 0.0).then(100.0 * (d1 - lowest_d1) / spread2).otherwise(50.0)

    return k2.ewm_mean(span=smooth_span, adjust=False)
