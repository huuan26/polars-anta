"""Trend indicators implemented natively for Polars."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .core import IntoExpr, parse_expr, true_range, wilder_smooth


def sma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Simple Moving Average (SMA).

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    return parse_expr(close).rolling_mean(window_size=period)


def ema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Exponential Moving Average (EMA).

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    return parse_expr(close).ewm_mean(span=period, adjust=False)


def wma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Weighted Moving Average (WMA).

    Weights linearly increase: [1, 2, ..., period].

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    if period == 1:
        return parse_expr(close)

    total_weight = (period * (period + 1)) // 2
    c = parse_expr(close)
    weighted_sum = sum(c.shift(period - weight) * weight for weight in range(1, period + 1))
    return weighted_sum / total_weight


def hma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Hull Moving Average (HMA).

    Formula:
        HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(math.sqrt(period)))

    c = parse_expr(close)
    diff = 2.0 * wma(c, half_period) - wma(c, period)
    return wma(diff, sqrt_period)


def dema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Double Exponential Moving Average (DEMA).

    Formula:
        DEMA = 2 * EMA(close, n) - EMA(EMA(close, n), n)

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    e1 = ema(close, period=period)
    e2 = ema(e1, period=period)
    return 2.0 * e1 - e2


def tema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Triple Exponential Moving Average (TEMA).

    Formula:
        TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        where EMA1 = EMA(close), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    e1 = ema(close, period=period)
    e2 = ema(e1, period=period)
    e3 = ema(e2, period=period)
    return 3.0 * e1 - 3.0 * e2 + e3


def zlema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Zero-Lag Exponential Moving Average (ZLEMA).

    Formula:
        lag = (period - 1) // 2
        ZLEMA = EMA(2 * close - close.shift(lag), period)

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    lag = (period - 1) // 2
    c = parse_expr(close)
    adjusted = 2.0 * c - c.shift(lag)
    return ema(adjusted, period=period)


def vortex(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Vortex Indicator (VI).

    Returns a struct with fields:
        - 'vi_plus': positive vortex line
        - 'vi_minus': negative vortex line

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    tr = true_range(hi, lo, close)

    vm_plus = (hi - lo.shift(1)).abs()
    vm_minus = (lo - hi.shift(1)).abs()

    sum_tr = tr.rolling_sum(window_size=period)
    sum_vm_plus = vm_plus.rolling_sum(window_size=period)
    sum_vm_minus = vm_minus.rolling_sum(window_size=period)

    vi_plus = pl.when(sum_tr > 0.0).then(sum_vm_plus / sum_tr).otherwise(1.0)
    vi_minus = pl.when(sum_tr > 0.0).then(sum_vm_minus / sum_tr).otherwise(1.0)

    return pl.struct(
        [
            vi_plus.alias("vi_plus"),
            vi_minus.alias("vi_minus"),
        ]
    )


def adx(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Average Directional Index (ADX) with Directional Movement (+DI, -DI).

    Returns a struct with fields:
        - 'adx': Average Directional Index [0, 100]
        - 'plus_di': Plus Directional Indicator (+DI)
        - 'minus_di': Minus Directional Indicator (-DI)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback smoothing period (default 14)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)

    up_move = hi - hi.shift(1)
    down_move = lo.shift(1) - lo

    plus_dm = pl.when((up_move > down_move) & (up_move > 0.0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0.0)).then(down_move).otherwise(0.0)

    tr = true_range(hi, lo, close)
    atr_smooth = wilder_smooth(tr, period)
    smooth_plus_dm = wilder_smooth(plus_dm, period)
    smooth_minus_dm = wilder_smooth(minus_dm, period)

    plus_di = pl.when(atr_smooth > 0.0).then(100.0 * smooth_plus_dm / atr_smooth).otherwise(0.0)
    minus_di = pl.when(atr_smooth > 0.0).then(100.0 * smooth_minus_dm / atr_smooth).otherwise(0.0)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    dx = pl.when(di_sum > 0.0).then(100.0 * di_diff / di_sum).otherwise(0.0)
    adx_line = wilder_smooth(dx, period)

    return pl.struct(
        [
            adx_line.alias("adx"),
            plus_di.alias("plus_di"),
            minus_di.alias("minus_di"),
        ]
    )


def aroon(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 25,
) -> pl.Expr:
    """Aroon Indicator and Oscillator.

    Returns a struct with fields:
        - 'aroon_up': Aroon Up [0, 100]
        - 'aroon_down': Aroon Down [0, 100]
        - 'oscillator': Aroon Oscillator (Aroon Up - Aroon Down) [-100, 100]

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: lookback period (default 25)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)

    max_h = hi.rolling_max(window_size=period + 1)
    min_l = lo.rolling_min(window_size=period + 1)

    up_conds = [pl.when(hi.shift(k) == max_h).then(float(period - k)) for k in range(period + 1)]
    down_conds = [pl.when(lo.shift(k) == min_l).then(float(period - k)) for k in range(period + 1)]

    aroon_up = 100.0 * pl.coalesce(up_conds) / float(period)
    aroon_down = 100.0 * pl.coalesce(down_conds) / float(period)
    oscillator = aroon_up - aroon_down

    return pl.struct(
        [
            aroon_up.alias("aroon_up"),
            aroon_down.alias("aroon_down"),
            oscillator.alias("oscillator"),
        ]
    )


def chop(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Choppiness Index (CHOP).

    Determines whether the market is consolidating (> 61.8) or trending (< 38.2).

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")
    hi = parse_expr(high)
    lo = parse_expr(low)
    tr = true_range(hi, lo, close)

    sum_tr = tr.rolling_sum(window_size=period)
    max_hi = hi.rolling_max(window_size=period)
    min_lo = lo.rolling_min(window_size=period)
    diff = max_hi - min_lo

    log_period = math.log10(float(period))
    return (
        pl.when((diff > 0.0) & (sum_tr > 0.0))
        .then(100.0 * (sum_tr / diff).log10() / log_period)
        .otherwise(50.0)
    )


def t3(close: IntoExpr, period: int = 5, volume_factor: float = 0.7) -> pl.Expr:
    """Tillson T3 Moving Average.

    Smoothed moving average reducing lag via triple DEMA with volume damping.

    :param close: price column or expression
    :param period: smoothing period (default 5)
    :param volume_factor: volume factor between 0.0 and 1.0 (default 0.7)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    v = float(volume_factor)
    c1 = -(v**3)
    c2 = 3.0 * (v**2) + 3.0 * (v**3)
    c3 = -6.0 * (v**2) - 3.0 * v - 3.0 * (v**3)
    c4 = 1.0 + 3.0 * v + (v**3) + 3.0 * (v**2)

    e1 = ema(close, period=period)
    e2 = ema(e1, period=period)
    e3 = ema(e2, period=period)
    e4 = ema(e3, period=period)
    e5 = ema(e4, period=period)
    e6 = ema(e5, period=period)

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def alma(
    close: IntoExpr,
    period: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> pl.Expr:
    """Arnaud Legoux Moving Average (ALMA).

    Gaussian distribution weighted moving average with minimal overshoot.

    :param close: price column or expression
    :param period: lookback window size (default 9)
    :param offset: trade-off between smoothness and responsiveness (default 0.85)
    :param sigma: filter sharpness parameter (default 6.0)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    m = float(offset * (period - 1))
    s = float(period / sigma)

    weights = [math.exp(-((i - m) ** 2) / (2.0 * s * s)) for i in range(period)]
    sum_w = sum(weights)

    c = parse_expr(close)
    weighted_sum = sum(c.shift(period - 1 - i) * w for i, w in enumerate(weights))
    return weighted_sum / sum_w


def mass_index(
    high: IntoExpr,
    low: IntoExpr,
    fast_period: int = 9,
    slow_period: int = 25,
) -> pl.Expr:
    """Mass Index.

    Measures range expansion between High and Low to predict trend reversals.

    :param high: high price column or expression
    :param low: low price column or expression
    :param fast_period: EMA lookback period (default 9)
    :param slow_period: rolling sum period (default 25)
    :return: Polars expression
    """
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    hl_diff = hi - lo

    ema1 = ema(hl_diff, period=fast_period)
    ema2 = ema(ema1, period=fast_period)
    ratio = pl.when(ema2 > 0.0).then(ema1 / ema2).otherwise(1.0)

    return ratio.rolling_sum(window_size=slow_period)


def supertrend(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 10,
    multiplier: float = 3.0,
) -> pl.Expr:
    """SuperTrend indicator.

    Returns a struct with fields:
        - 'supertrend': dynamic trailing support/resistance level
        - 'direction': trend direction (1 for uptrend, -1 for downtrend)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: ATR period (default 10)
    :param multiplier: ATR multiplier (default 3.0)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    tr = true_range(hi, lo, c)
    atr_val = wilder_smooth(tr, period)

    hl2 = (hi + lo) / 2.0
    upper_basic = hl2 + float(multiplier) * atr_val
    lower_basic = hl2 - float(multiplier) * atr_val

    def _calc_supertrend(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        c_vals = df_chunk["close"].to_numpy()
        ub_vals = df_chunk["upper_basic"].to_numpy()
        lb_vals = df_chunk["lower_basic"].to_numpy()
        n = len(c_vals)

        upper_final = np.zeros(n, dtype=np.float64)
        lower_final = np.zeros(n, dtype=np.float64)
        st_vals = np.zeros(n, dtype=np.float64)
        direction = np.ones(n, dtype=np.int64)

        for i in range(n):
            c_i = c_vals[i]
            ub_i = ub_vals[i]
            lb_i = lb_vals[i]
            if np.isnan(c_i) or np.isnan(ub_i) or np.isnan(lb_i):
                continue

            if i == 0:
                upper_final[i] = ub_i
                lower_final[i] = lb_i
                st_vals[i] = ub_i
                direction[i] = 1
                continue

            prev_c = c_vals[i - 1]
            if np.isnan(prev_c):
                prev_c = c_i
            prev_uf = upper_final[i - 1]
            prev_lf = lower_final[i - 1]

            if lb_i > prev_lf or prev_c < prev_lf:
                lower_final[i] = lb_i
            else:
                lower_final[i] = prev_lf

            if ub_i < prev_uf or prev_c > prev_uf:
                upper_final[i] = ub_i
            else:
                upper_final[i] = prev_uf

            prev_st = st_vals[i - 1]
            if prev_st == prev_uf:
                if c_i > upper_final[i]:
                    st_vals[i] = lower_final[i]
                    direction[i] = 1
                else:
                    st_vals[i] = upper_final[i]
                    direction[i] = -1
            else:
                if c_i < lower_final[i]:
                    st_vals[i] = upper_final[i]
                    direction[i] = -1
                else:
                    st_vals[i] = lower_final[i]
                    direction[i] = 1

        return pl.DataFrame({"supertrend": st_vals, "direction": direction}).to_struct("st")

    combined = pl.struct(
        [
            c.alias("close"),
            upper_basic.alias("upper_basic"),
            lower_basic.alias("lower_basic"),
        ]
    )
    return combined.map_batches(
        _calc_supertrend,
        return_dtype=pl.Struct(
            [pl.Field("supertrend", pl.Float64), pl.Field("direction", pl.Int64)]
        ),
    )


def psar(
    high: IntoExpr,
    low: IntoExpr,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> pl.Expr:
    """Parabolic SAR (Stop and Reverse).

    Returns a Polars expression representing the trailing stop curve.

    :param high: high price column or expression
    :param low: low price column or expression
    :param af_step: acceleration factor step (default 0.02)
    :param af_max: maximum acceleration factor (default 0.2)
    :return: Polars expression
    """
    hi = parse_expr(high)
    lo = parse_expr(low)

    def _calc_psar(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        h_vals = df_chunk["high"].to_numpy()
        l_vals = df_chunk["low"].to_numpy()
        n = len(h_vals)
        if n == 0:
            return pl.Series([], dtype=pl.Float64)

        sar_vals = np.zeros(n, dtype=np.float64)
        is_long = True
        ep = h_vals[0]
        af = af_step
        sar = l_vals[0]
        sar_vals[0] = sar

        for i in range(1, n):
            h_i = h_vals[i]
            l_i = l_vals[i]
            h_prev = h_vals[i - 1]
            l_prev = l_vals[i - 1]

            if is_long:
                sar = sar + af * (ep - sar)
                sar = min(sar, l_prev, l_vals[max(0, i - 2)])
                if l_i < sar:
                    is_long = False
                    sar = ep
                    ep = l_i
                    af = af_step
                else:
                    if h_i > ep:
                        ep = h_i
                        af = min(af + af_step, af_max)
            else:
                sar = sar + af * (ep - sar)
                sar = max(sar, h_prev, h_vals[max(0, i - 2)])
                if h_i > sar:
                    is_long = True
                    sar = ep
                    ep = h_i
                    af = af_step
                else:
                    if l_i < ep:
                        ep = l_i
                        af = min(af + af_step, af_max)

            sar_vals[i] = sar

        return pl.Series(sar_vals)

    combined = pl.struct([hi.alias("high"), lo.alias("low")])
    return combined.map_batches(_calc_psar, return_dtype=pl.Float64)


def kama(
    close: IntoExpr,
    period: int = 10,
    fast_span: int = 2,
    slow_span: int = 30,
) -> pl.Expr:
    """Kaufman's Adaptive Moving Average (KAMA).

    Moving average that automatically adapts to market volatility and noise.

    :param close: price column or expression
    :param period: efficiency ratio lookback period (default 10)
    :param fast_span: fastest EMA span (default 2)
    :param slow_span: slowest EMA span (default 30)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    c = parse_expr(close)
    fast_sc = 2.0 / (fast_span + 1)
    slow_sc = 2.0 / (slow_span + 1)

    change = (c - c.shift(period)).abs()
    volatility = c.diff(1).abs().rolling_sum(window_size=period)

    er = pl.when(volatility > 0.0).then(change / volatility).otherwise(0.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    def _calc_kama(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        c_vals = df_chunk["close"].to_numpy()
        sc_vals = df_chunk["sc"].to_numpy()
        n = len(c_vals)

        kama_vals = np.full(n, np.nan, dtype=np.float64)
        first_valid = -1
        for i in range(n):
            if not np.isnan(c_vals[i]) and not np.isnan(sc_vals[i]):
                first_valid = i
                break

        if first_valid == -1:
            return pl.Series(kama_vals).fill_nan(None)

        current = c_vals[first_valid]
        kama_vals[first_valid] = current

        for i in range(first_valid + 1, n):
            c_i = c_vals[i]
            sc_i = sc_vals[i]
            if np.isnan(sc_i):
                sc_i = slow_sc
            if not np.isnan(c_i):
                current = current + sc_i * (c_i - current)
                kama_vals[i] = current

        return pl.Series(kama_vals).fill_nan(None)

    combined = pl.struct([c.alias("close"), sc.alias("sc")])
    return combined.map_batches(_calc_kama, return_dtype=pl.Float64)


def mcginley_dynamic(
    close: IntoExpr,
    period: int = 10,
    k: float = 0.6,
) -> pl.Expr:
    """McGinley Dynamic tracking average.

    Adjusts dynamically for shifts in market speed and avoids price separation.

    :param close: price column or expression
    :param period: lookback period (default 10)
    :param k: constant factor (default 0.6)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    c = parse_expr(close)

    def _calc_md(s: pl.Series) -> pl.Series:
        vals = s.to_numpy()
        n = len(vals)
        md_vals = np.full(n, np.nan, dtype=np.float64)

        first_valid = -1
        for i in range(n):
            if not np.isnan(vals[i]):
                first_valid = i
                break

        if first_valid == -1:
            return pl.Series(md_vals).fill_nan(None)

        curr = vals[first_valid]
        md_vals[first_valid] = curr

        for i in range(first_valid + 1, n):
            c_i = vals[i]
            if not np.isnan(c_i) and curr > 0.0:
                ratio = c_i / curr
                curr = curr + (c_i - curr) / (k * float(period) * (ratio**4))
                md_vals[i] = curr

        return pl.Series(md_vals).fill_nan(None)

    return c.map_batches(_calc_md, return_dtype=pl.Float64)


def vidya(
    close: IntoExpr,
    period: int = 9,
    cmo_period: int = 9,
) -> pl.Expr:
    """Variable Index Dynamic Average (VIDYA).

    Exponential moving average dynamically adjusted by the Chande Momentum Oscillator (CMO).

    :param close: price column or expression
    :param period: EMA lookback period (default 9)
    :param cmo_period: CMO lookback period (default 9)
    :return: Polars expression
    """
    if period <= 0 or cmo_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    c = parse_expr(close)
    diff = c.diff(1)
    gain = pl.when(diff > 0.0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0.0).then(-diff).otherwise(0.0)

    sum_gain = gain.rolling_sum(window_size=cmo_period)
    sum_loss = loss.rolling_sum(window_size=cmo_period)
    sum_total = sum_gain + sum_loss

    cmo = pl.when(sum_total > 0.0).then(((sum_gain - sum_loss) / sum_total).abs()).otherwise(0.0)
    alpha = (2.0 / (period + 1)) * cmo

    def _calc_vidya(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        c_vals = df_chunk["close"].to_numpy()
        a_vals = df_chunk["alpha"].to_numpy()
        n = len(c_vals)

        vidya_vals = np.full(n, np.nan, dtype=np.float64)
        first_valid = -1
        for i in range(n):
            if not np.isnan(c_vals[i]):
                first_valid = i
                break

        if first_valid == -1:
            return pl.Series(vidya_vals).fill_nan(None)

        curr = c_vals[first_valid]
        vidya_vals[first_valid] = curr

        for i in range(first_valid + 1, n):
            c_i = c_vals[i]
            a_i = a_vals[i]
            if np.isnan(a_i):
                a_i = 0.0
            if not np.isnan(c_i):
                curr = a_i * c_i + (1.0 - a_i) * curr
                vidya_vals[i] = curr

        return pl.Series(vidya_vals).fill_nan(None)

    combined = pl.struct([c.alias("close"), alpha.alias("alpha")])
    return combined.map_batches(_calc_vidya, return_dtype=pl.Float64)


def center_of_gravity(
    close: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Center of Gravity (CG) Oscillator by John Ehlers.

    Measures the center of mass within a price window to identify cycle turns with zero lag:
    CG = - Sum((i + 1) * Price[t-i]) / Sum(Price[t-i]).

    :param close: price column or expression
    :param period: lookback window (default 10)
    :return: Polars expression
    """
    c = parse_expr(close)
    weights: list[float] = [float(x) for x in range(period, 0, -1)]
    weight_sum = period * (period + 1) / 2.0
    weighted_sum = c.rolling_mean(window_size=period, weights=weights) * weight_sum
    total_sum = c.rolling_sum(window_size=period)
    return -(weighted_sum / total_sum)


def ichimoku(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pl.Expr:
    """Ichimoku Kinko Hyo Cloud.

    Comprehensive trend and support/resistance framework.
    Returns Struct with:
      - `tenkan_sen`: Conversion Line ((High9 + Low9) / 2)
      - `kijun_sen`: Base Line ((High26 + Low26) / 2)
      - `senkou_span_a`: Leading Span A ((Tenkan + Kijun) / 2 shifted forward)
      - `senkou_span_b`: Leading Span B ((High52 + Low52) / 2 shifted forward)
      - `chikou_span`: Lagging Span (Close shifted backward)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param tenkan_period: Tenkan-sen period (default 9)
    :param kijun_period: Kijun-sen period (default 26)
    :param senkou_b_period: Senkou Span B period (default 52)
    :param displacement: cloud displacement bars (default 26)
    :return: Polars Struct expression
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    tenkan = (
        hi.rolling_max(window_size=tenkan_period) + lo.rolling_min(window_size=tenkan_period)
    ) / 2.0
    kijun = (
        hi.rolling_max(window_size=kijun_period) + lo.rolling_min(window_size=kijun_period)
    ) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(displacement)
    span_b = (
        (hi.rolling_max(window_size=senkou_b_period) + lo.rolling_min(window_size=senkou_b_period))
        / 2.0
    ).shift(displacement)
    chikou = c.shift(-displacement)

    return pl.struct(
        [
            tenkan.alias("tenkan_sen"),
            kijun.alias("kijun_sen"),
            span_a.alias("senkou_span_a"),
            span_b.alias("senkou_span_b"),
            chikou.alias("chikou_span"),
        ]
    )


def pivot_points(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Standard Floor-Trader Pivot Points.

    Calculates support and resistance levels based on previous bar high, low, close.
    Returns Struct with: `pivot`, `r1`, `s1`, `r2`, `s2`, `r3`, `s3`.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars Struct expression
    """
    h = parse_expr(high).shift(1)
    l_ = parse_expr(low).shift(1)
    c = parse_expr(close).shift(1)

    pivot = (h + l_ + c) / 3.0
    r1 = 2.0 * pivot - l_
    s1 = 2.0 * pivot - h
    r2 = pivot + (h - l_)
    s2 = pivot - (h - l_)
    r3 = h + 2.0 * (pivot - l_)
    s3 = l_ - 2.0 * (h - pivot)

    return pl.struct(
        [
            pivot.alias("pivot"),
            r1.alias("r1"),
            s1.alias("s1"),
            r2.alias("r2"),
            s2.alias("s2"),
            r3.alias("r3"),
            s3.alias("s3"),
        ]
    )


def fibonacci_pivots(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Fibonacci Pivot Points.

    Calculates support and resistance levels based on 38.2%, 61.8%, and 100% Fibonacci ratios.
    Returns Struct with: `pivot`, `r1`, `s1`, `r2`, `s2`, `r3`, `s3`.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars Struct expression
    """
    h = parse_expr(high).shift(1)
    l_ = parse_expr(low).shift(1)
    c = parse_expr(close).shift(1)

    pivot = (h + l_ + c) / 3.0
    rng = h - l_
    r1 = pivot + 0.382 * rng
    s1 = pivot - 0.382 * rng
    r2 = pivot + 0.618 * rng
    s2 = pivot - 0.618 * rng
    r3 = pivot + 1.000 * rng
    s3 = pivot - 1.000 * rng

    return pl.struct(
        [
            pivot.alias("pivot"),
            r1.alias("r1"),
            s1.alias("s1"),
            r2.alias("r2"),
            s2.alias("s2"),
            r3.alias("r3"),
            s3.alias("s3"),
        ]
    )


def camarilla_pivots(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Camarilla Pivot Points.

    Calculates 8 levels of tight intraday support and resistance.
    Returns Struct with: `r4`, `r3`, `r2`, `r1`, `s1`, `s2`, `s3`, `s4`.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars Struct expression
    """
    h = parse_expr(high).shift(1)
    l_ = parse_expr(low).shift(1)
    c = parse_expr(close).shift(1)
    rng = h - l_

    r4 = c + rng * (1.1 / 2.0)
    r3 = c + rng * (1.1 / 4.0)
    r2 = c + rng * (1.1 / 6.0)
    r1 = c + rng * (1.1 / 12.0)
    s1 = c - rng * (1.1 / 12.0)
    s2 = c - rng * (1.1 / 6.0)
    s3 = c - rng * (1.1 / 4.0)
    s4 = c - rng * (1.1 / 2.0)

    return pl.struct(
        [
            r4.alias("r4"),
            r3.alias("r3"),
            r2.alias("r2"),
            r1.alias("r1"),
            s1.alias("s1"),
            s2.alias("s2"),
            s3.alias("s3"),
            s4.alias("s4"),
        ]
    )


def woodie_pivots(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Woodie's Pivot Points.

    Calculates pivot levels weighting closing price more heavily: Pivot = (H + L + 2 * C) / 4.
    Returns Struct with: `pivot`, `r1`, `s1`, `r2`, `s2`.

    :param open_: open price column or expression
    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars Struct expression
    """
    h = parse_expr(high).shift(1)
    l_ = parse_expr(low).shift(1)
    c = parse_expr(close).shift(1)

    pivot = (h + l_ + 2.0 * c) / 4.0
    r1 = 2.0 * pivot - l_
    s1 = 2.0 * pivot - h
    r2 = pivot + (h - l_)
    s2 = pivot - (h - l_)

    return pl.struct(
        [
            pivot.alias("pivot"),
            r1.alias("r1"),
            s1.alias("s1"),
            r2.alias("r2"),
            s2.alias("s2"),
        ]
    )


def ssl_channel(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """SSL Channel (Securement Stop Loss Channel).

    Returns a struct with fields:
        - 'ssl_up': trailing upper channel line
        - 'ssl_down': trailing lower channel line
        - 'hlv': trend direction signal (1 for bullish / uptrend, -1 for bearish / downtrend)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period for SMA (default 10)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    sma_high = hi.rolling_mean(window_size=period)
    sma_low = lo.rolling_mean(window_size=period)

    hlv = (
        pl.when(c > sma_high)
        .then(1)
        .when(c < sma_low)
        .then(-1)
        .otherwise(None)
        .forward_fill()
        .fill_null(1)
    )

    null_mask = sma_high.is_null() | sma_low.is_null()
    ssl_up = pl.when(null_mask).then(None).when(hlv < 0).then(sma_low).otherwise(sma_high)
    ssl_down = pl.when(null_mask).then(None).when(hlv < 0).then(sma_high).otherwise(sma_low)
    hlv_out = pl.when(null_mask).then(None).otherwise(hlv)

    return pl.struct(
        [
            ssl_up.alias("ssl_up"),
            ssl_down.alias("ssl_down"),
            hlv_out.alias("hlv"),
        ]
    )


def trend_intensity_index(
    close: IntoExpr,
    period: int = 30,
    signal_period: int | None = None,
) -> pl.Expr:
    """Trend Intensity Index (TII) by M.H. Pee.

    Measures the strength of a trend by calculating the percentage of price deviations
    that lie above the moving average versus total deviations.

    Returns a struct with fields:
        - 'tii': Trend Intensity Index value [0, 100] (values > 80 indicate strong uptrend, < 20 strong downtrend)
        - 'signal': EMA of TII (default 9-period EMA)

    :param close: price column or expression
    :param period: lookback period for the base SMA (default 30)
    :param signal_period: lookback period for deviation sum (default None, uses period // 2)
    :return: Polars Struct expression
    """
    if period <= 1:
        raise ValueError("period must be an integer greater than 1")
    dev_period = signal_period if signal_period is not None else max(1, period // 2)
    if dev_period <= 0:
        raise ValueError("signal_period must be a positive integer greater than zero")

    c = parse_expr(close)
    base_sma = c.rolling_mean(window_size=period)
    dev = c - base_sma

    pos_dev = pl.when(dev > 0.0).then(dev).otherwise(0.0)
    neg_dev = pl.when(dev < 0.0).then(-dev).otherwise(0.0)

    sum_pos = pos_dev.rolling_sum(window_size=dev_period)
    sum_neg = neg_dev.rolling_sum(window_size=dev_period)
    sum_total = sum_pos + sum_neg

    null_mask = base_sma.is_null() | sum_pos.is_null()
    tii = (
        pl.when(null_mask)
        .then(None)
        .when(sum_total > 0.0)
        .then(100.0 * (sum_pos / sum_total))
        .otherwise(50.0)
    )
    sig = ema(tii, period=9)

    return pl.struct(
        [
            tii.alias("tii"),
            sig.alias("signal"),
        ]
    )
