"""Directional trend strength, trailing stop, and channel indicators for Polars."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..core import IntoExpr, check_period, check_periods, parse_expr, true_range, wilder_smooth
from ._moving_averages import ema


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
    check_period(period)
    hi = parse_expr(high)
    lo = parse_expr(low)
    sum_tr = true_range(hi, lo, close).rolling_sum(window_size=period)
    sum_vm_plus = (hi - lo.shift(1)).abs().rolling_sum(window_size=period)
    sum_vm_minus = (lo - hi.shift(1)).abs().rolling_sum(window_size=period)

    return pl.struct(
        [
            pl.when(sum_tr > 0.0).then(sum_vm_plus / sum_tr).otherwise(1.0).alias("vi_plus"),
            pl.when(sum_tr > 0.0).then(sum_vm_minus / sum_tr).otherwise(1.0).alias("vi_minus"),
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
    check_period(period)
    hi = parse_expr(high)
    lo = parse_expr(low)

    up_move = hi - hi.shift(1)
    down_move = lo.shift(1) - lo

    plus_dm = pl.when((up_move > down_move) & (up_move > 0.0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0.0)).then(down_move).otherwise(0.0)

    atr_smooth = wilder_smooth(true_range(hi, lo, close), period)
    plus_di = (
        pl.when(atr_smooth > 0.0)
        .then(100.0 * wilder_smooth(plus_dm, period) / atr_smooth)
        .otherwise(0.0)
    )
    minus_di = (
        pl.when(atr_smooth > 0.0)
        .then(100.0 * wilder_smooth(minus_dm, period) / atr_smooth)
        .otherwise(0.0)
    )

    di_sum = plus_di + minus_di
    dx = pl.when(di_sum > 0.0).then(100.0 * (plus_di - minus_di).abs() / di_sum).otherwise(0.0)

    return pl.struct(
        [
            wilder_smooth(dx, period).alias("adx"),
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
    check_period(period)
    hi = parse_expr(high)
    lo = parse_expr(low)

    max_h = hi.rolling_max(window_size=period + 1)
    min_l = lo.rolling_min(window_size=period + 1)

    up_conds = [pl.when(hi.shift(k) == max_h).then(float(period - k)) for k in range(period + 1)]
    down_conds = [pl.when(lo.shift(k) == min_l).then(float(period - k)) for k in range(period + 1)]

    aroon_up = 100.0 * pl.coalesce(up_conds) / float(period)
    aroon_down = 100.0 * pl.coalesce(down_conds) / float(period)

    return pl.struct(
        [
            aroon_up.alias("aroon_up"),
            aroon_down.alias("aroon_down"),
            (aroon_up - aroon_down).alias("oscillator"),
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
    check_period(period, 2)
    hi = parse_expr(high)
    lo = parse_expr(low)
    sum_tr = true_range(hi, lo, close).rolling_sum(window_size=period)
    diff = hi.rolling_max(window_size=period) - lo.rolling_min(window_size=period)

    log_period = math.log10(float(period))
    return (
        pl.when((diff > 0.0) & (sum_tr > 0.0))
        .then(100.0 * (sum_tr / diff).log10() / log_period)
        .otherwise(50.0)
    )


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
    check_periods(fast_period, slow_period)
    ema1 = ema(parse_expr(high) - parse_expr(low), period=fast_period)
    ema2 = ema(ema1, period=fast_period)
    return pl.when(ema2 > 0.0).then(ema1 / ema2).otherwise(1.0).rolling_sum(window_size=slow_period)


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
    check_period(period)

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
    check_period(period)

    sma_high = parse_expr(high).rolling_mean(window_size=period)
    sma_low = parse_expr(low).rolling_mean(window_size=period)
    c = parse_expr(close)

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
    return pl.struct(
        [
            pl.when(null_mask)
            .then(None)
            .when(hlv < 0)
            .then(sma_low)
            .otherwise(sma_high)
            .alias("ssl_up"),
            pl.when(null_mask)
            .then(None)
            .when(hlv < 0)
            .then(sma_high)
            .otherwise(sma_low)
            .alias("ssl_down"),
            pl.when(null_mask).then(None).otherwise(hlv).alias("hlv"),
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
    check_period(period, 2)
    dev_period = signal_period if signal_period is not None else max(1, period // 2)
    check_period(dev_period, name="signal_period")

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
