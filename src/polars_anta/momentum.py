"""Momentum indicators implemented natively for Polars."""

from __future__ import annotations

import polars as pl

from .core import IntoExpr, parse_expr, true_range, wilder_smooth
from .trend import ema, wma


def rsi(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Relative Strength Index (RSI) using Wilder's smoothing.

    :param close: price column name or expression
    :param period: lookback period (default 14)
    :return: Polars expression producing RSI values in [0, 100]
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    c = parse_expr(close)
    diff = c.diff(1)
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)

    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)

    rs = avg_gain / avg_loss
    return (
        pl.when(avg_loss == 0.0)
        .then(100.0)
        .when(avg_gain == 0.0)
        .then(0.0)
        .otherwise(100.0 - (100.0 / (1.0 + rs)))
    )


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
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    c = parse_expr(close)
    macd_line = c.ewm_mean(span=fast_period, adjust=False) - c.ewm_mean(
        span=slow_period, adjust=False
    )
    signal_line = macd_line.ewm_mean(span=signal_period, adjust=False)
    histogram = macd_line - signal_line

    return pl.struct(
        [
            macd_line.alias("macd"),
            signal_line.alias("signal"),
            histogram.alias("histogram"),
        ]
    )


def stochastic(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pl.Expr:
    """Stochastic Oscillator (%K, %D).

    Returns a struct with fields:
        - 'k': smoothed %K line
        - 'd': %D signal line (SMA of %K)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param k_period: lookback window (default 14)
    :param d_period: %D smoothing window (default 3)
    :param smooth_k: %K smoothing window (default 3)
    :return: Polars Struct expression
    """
    if k_period <= 0 or d_period <= 0 or smooth_k <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    lowest_low = lo.rolling_min(window_size=k_period)
    highest_high = hi.rolling_max(window_size=k_period)
    spread = highest_high - lowest_low

    fast_k = pl.when(spread > 0.0).then(100.0 * (c - lowest_low) / spread).otherwise(50.0)
    k_line = fast_k.rolling_mean(window_size=smooth_k)
    d_line = k_line.rolling_mean(window_size=d_period)

    return pl.struct(
        [
            k_line.alias("k"),
            d_line.alias("d"),
        ]
    )


def williams_r(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Williams %R momentum indicator.

    Values oscillate in [-100, 0].

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    highest_high = hi.rolling_max(window_size=period)
    lowest_low = lo.rolling_min(window_size=period)
    spread = highest_high - lowest_low

    return pl.when(spread > 0.0).then(-100.0 * (highest_high - c) / spread).otherwise(-50.0)


def cci(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    constant: float = 0.015,
) -> pl.Expr:
    """Commodity Channel Index (CCI).

    Measures current price level relative to an average price level over a given period.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 20)
    :param constant: scaling factor (default 0.015)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    tp = (hi + lo + c) / 3.0
    sma_tp = tp.rolling_mean(window_size=period)
    mad = sum((tp.shift(i) - sma_tp).abs() for i in range(period)) / float(period)

    return pl.when(mad > 0.0).then((tp - sma_tp) / (constant * mad)).otherwise(0.0)


def roc(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Rate of Change (ROC).

    Calculates the percentage change between current price and price n periods ago.

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    prev_c = c.shift(period)
    return pl.when(prev_c != 0.0).then(100.0 * (c - prev_c) / prev_c).otherwise(0.0)


def stoch_rsi(
    close: IntoExpr,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> pl.Expr:
    """Stochastic RSI (StochRSI).

    Applies the Stochastic Oscillator formula to RSI values rather than raw price.

    Returns a struct with fields:
        - 'k': smoothed %K line in [0, 100]
        - 'd': %D signal line in [0, 100]

    :param close: price column or expression
    :param rsi_period: RSI lookback period (default 14)
    :param stoch_period: Stochastic lookback period (default 14)
    :param k_period: %K smoothing window (default 3)
    :param d_period: %D smoothing window (default 3)
    :return: Polars Struct expression
    """
    if rsi_period <= 0 or stoch_period <= 0 or k_period <= 0 or d_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    rsi_val = rsi(close, period=rsi_period)
    lowest_rsi = rsi_val.rolling_min(window_size=stoch_period)
    highest_rsi = rsi_val.rolling_max(window_size=stoch_period)
    spread = highest_rsi - lowest_rsi

    fast_k = pl.when(spread > 0.0).then(100.0 * (rsi_val - lowest_rsi) / spread).otherwise(50.0)
    k_line = fast_k.rolling_mean(window_size=k_period)
    d_line = k_line.rolling_mean(window_size=d_period)

    return pl.struct(
        [
            k_line.alias("k"),
            d_line.alias("d"),
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
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    hi = parse_expr(high)
    lo = parse_expr(low)
    median_price = (hi + lo) / 2.0

    fast_sma = median_price.rolling_mean(window_size=fast_period)
    slow_sma = median_price.rolling_mean(window_size=slow_period)

    return fast_sma - slow_sma


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
    if long_period <= 0 or short_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    c = parse_expr(close)
    diff = c.diff(1)
    abs_diff = diff.abs()

    smooth_1 = diff.ewm_mean(span=long_period, adjust=False)
    smooth_2 = smooth_1.ewm_mean(span=short_period, adjust=False)

    abs_smooth_1 = abs_diff.ewm_mean(span=long_period, adjust=False)
    abs_smooth_2 = abs_smooth_1.ewm_mean(span=short_period, adjust=False)

    return pl.when(abs_smooth_2 > 0.0).then(100.0 * smooth_2 / abs_smooth_2).otherwise(0.0)


def cmo(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Chande Momentum Oscillator (CMO).

    Oscillates between -100 and +100 based on price momentum.

    :param close: price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    diff = c.diff(1)
    gain = pl.when(diff > 0.0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0.0).then(-diff).otherwise(0.0)

    sum_gain = gain.rolling_sum(window_size=period)
    sum_loss = loss.rolling_sum(window_size=period)
    total = sum_gain + sum_loss

    return pl.when(total > 0.0).then(100.0 * (sum_gain - sum_loss) / total).otherwise(0.0)


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
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    c = parse_expr(close)
    fast_ema = ema(c, period=fast_period)
    slow_ema = ema(c, period=slow_period)

    ppo_line = (
        pl.when(slow_ema != 0.0).then(100.0 * (fast_ema - slow_ema) / slow_ema).otherwise(0.0)
    )
    signal_line = ema(ppo_line, period=signal_period)
    histogram = ppo_line - signal_line

    return pl.struct(
        [
            ppo_line.alias("ppo"),
            signal_line.alias("signal"),
            histogram.alias("histogram"),
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
    if wma_period <= 0 or fast_roc <= 0 or slow_roc <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    c = parse_expr(close)
    roc_fast = roc(c, period=fast_roc)
    roc_slow = roc(c, period=slow_roc)
    total_roc = roc_fast + roc_slow

    return wma(total_roc, period=wma_period)


def ultimate_oscillator(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pl.Expr:
    """Ultimate Oscillator (UO).

    Multi-timeframe momentum oscillator bounded between 0 and 100.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period1: short lookback period (default 7)
    :param period2: medium lookback period (default 14)
    :param period3: long lookback period (default 28)
    :return: Polars expression
    """
    if period1 <= 0 or period2 <= 0 or period3 <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    prev_close = c.shift(1)

    min_l_c = pl.min_horizontal(lo, prev_close)
    max_h_c = pl.max_horizontal(hi, prev_close)

    bp = c - min_l_c
    tr = max_h_c - min_l_c

    sum_bp1 = bp.rolling_sum(window_size=period1)
    sum_tr1 = tr.rolling_sum(window_size=period1)
    avg1 = pl.when(sum_tr1 > 0.0).then(sum_bp1 / sum_tr1).otherwise(0.0)

    sum_bp2 = bp.rolling_sum(window_size=period2)
    sum_tr2 = tr.rolling_sum(window_size=period2)
    avg2 = pl.when(sum_tr2 > 0.0).then(sum_bp2 / sum_tr2).otherwise(0.0)

    sum_bp3 = bp.rolling_sum(window_size=period3)
    sum_tr3 = tr.rolling_sum(window_size=period3)
    avg3 = pl.when(sum_tr3 > 0.0).then(sum_bp3 / sum_tr3).otherwise(0.0)

    return 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0


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
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    max_h = hi.rolling_max(window_size=period)
    min_l = lo.rolling_min(window_size=period)
    hl_diff = max_h - min_l
    mid = (hi + lo) / 2.0

    raw_x = pl.when(hl_diff > 0.0).then(2.0 * ((mid - min_l) / hl_diff - 0.5)).otherwise(0.0)
    smoothed_x = raw_x.ewm_mean(span=3, adjust=False).clip(-0.999, 0.999)

    fisher = 0.5 * ((1.0 + smoothed_x) / (1.0 - smoothed_x)).log()
    signal = fisher.shift(1)

    return pl.struct(
        [
            fisher.alias("fisher"),
            signal.alias("signal"),
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
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    o = parse_expr(open)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    co = c - o
    hl = hi - lo

    co_num = (co + 2.0 * co.shift(1) + 2.0 * co.shift(2) + co.shift(3)) / 6.0
    hl_den = (hl + 2.0 * hl.shift(1) + 2.0 * hl.shift(2) + hl.shift(3)) / 6.0

    sum_num = co_num.rolling_sum(window_size=period)
    sum_den = hl_den.rolling_sum(window_size=period)

    rvi_line = pl.when(sum_den > 0.0).then(sum_num / sum_den).otherwise(0.0)
    signal = (
        rvi_line + 2.0 * rvi_line.shift(1) + 2.0 * rvi_line.shift(2) + rvi_line.shift(3)
    ) / 6.0

    return pl.struct(
        [
            rvi_line.alias("rvi"),
            signal.alias("signal"),
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
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    shift_k = period // 2 + 1
    return c - c.rolling_mean(window_size=period).shift(shift_k)


def bop(
    open: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Balance of Power (BOP).

    Evaluates the strength of buyers and sellers by comparing close-open to high-low.

    :param open: open price column or expression
    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars expression in [-1, 1]
    """
    o = parse_expr(open)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    spread = hi - lo

    return pl.when(spread > 0.0).then((c - o) / spread).otherwise(0.0)


def cfo(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Chande Forecast Oscillator (CFO).

    Computes percentage difference between actual price and linear regression line.

    :param close: price column or expression
    :param period: linear regression period (default 14)
    :return: Polars expression
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")

    n = float(period)
    mean_x = (n - 1.0) / 2.0
    sum_x_sq = n * (n * n - 1.0) / 12.0

    c = parse_expr(close)
    # Vectorized covariance sum( (i - mean_x) * close.shift(period - 1 - i) )
    cov_terms = sum(c.shift(period - 1 - i) * (float(i) - mean_x) for i in range(period))
    slope = cov_terms / sum_x_sq
    forecast = c.rolling_mean(window_size=period) + slope * (n - 1.0 - mean_x)

    return pl.when(c != 0.0).then(100.0 * (c - forecast) / c).otherwise(0.0)


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
    if bb_period <= 0 or kc_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    effective_mom = mom_period if mom_period is not None else kc_period
    if effective_mom <= 0:
        raise ValueError("mom_period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    # 1. Bollinger Bands
    basis_bb = c.rolling_mean(window_size=bb_period)
    dev_bb = c.rolling_std(window_size=bb_period) * float(bb_mult)
    upper_bb = basis_bb + dev_bb
    lower_bb = basis_bb - dev_bb

    # 2. Keltner Channels
    basis_kc = c.rolling_mean(window_size=kc_period)
    tr = true_range(hi, lo, c)
    range_kc = tr.rolling_mean(window_size=kc_period) * float(kc_mult)
    upper_kc = basis_kc + range_kc
    lower_kc = basis_kc - range_kc

    # Squeeze states
    null_mask = upper_bb.is_null() | upper_kc.is_null()
    sqz_on = pl.when(null_mask).then(None).otherwise((lower_bb > lower_kc) & (upper_bb < upper_kc))
    sqz_off = pl.when(null_mask).then(None).otherwise((lower_bb < lower_kc) & (upper_bb > upper_kc))
    no_sqz = pl.when(null_mask).then(None).otherwise((~sqz_on) & (~sqz_off))

    # 3. Momentum Histogram
    highest_h = hi.rolling_max(window_size=kc_period)
    lowest_l = lo.rolling_min(window_size=kc_period)
    avg_hl = (highest_h + lowest_l) / 2.0
    avg_mid = (avg_hl + basis_kc) / 2.0
    delta = c - avg_mid

    sma_delta = delta.rolling_mean(window_size=effective_mom)
    wma_delta = wma(delta, period=effective_mom)
    momentum = 3.0 * wma_delta - 2.0 * sma_delta

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
    if fast_period <= 0 or slow_period <= 0 or cycle_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    if not (0.0 < factor <= 1.0):
        raise ValueError("factor must be in (0, 1]")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    c = parse_expr(close)
    macd_line = c.ewm_mean(span=fast_period, adjust=False) - c.ewm_mean(
        span=slow_period, adjust=False
    )

    # First stochastic cycle on MACD
    lowest_macd = macd_line.rolling_min(window_size=cycle_period)
    highest_macd = macd_line.rolling_max(window_size=cycle_period)
    spread1 = highest_macd - lowest_macd
    k1 = pl.when(spread1 > 0.0).then(100.0 * (macd_line - lowest_macd) / spread1).otherwise(50.0)

    # Smooth with EMA whose span is factor * cycle_period
    smooth_span = max(1, int(round(factor * cycle_period)))
    d1 = k1.ewm_mean(span=smooth_span, adjust=False)

    # Second stochastic cycle on the smoothed %D1
    lowest_d1 = d1.rolling_min(window_size=cycle_period)
    highest_d1 = d1.rolling_max(window_size=cycle_period)
    spread2 = highest_d1 - lowest_d1
    k2 = pl.when(spread2 > 0.0).then(100.0 * (d1 - lowest_d1) / spread2).otherwise(50.0)

    stc_line = k2.ewm_mean(span=smooth_span, adjust=False)
    return stc_line
