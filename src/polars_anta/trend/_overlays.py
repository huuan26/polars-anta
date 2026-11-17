"""Price overlays, Ichimoku cloud, pivot points, and Heikin-Ashi transforms for Polars."""

from __future__ import annotations

import polars as pl

from ..core import IntoExpr, parse_expr


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


def _shifted_hlc(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Return 1-bar lagged high, low, and close expressions for pivot-point levels."""
    return parse_expr(high).shift(1), parse_expr(low).shift(1), parse_expr(close).shift(1)


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
    h, l_, c = _shifted_hlc(high, low, close)
    pivot = (h + l_ + c) / 3.0
    rng = h - l_
    return pl.struct(
        [
            pivot.alias("pivot"),
            (2.0 * pivot - l_).alias("r1"),
            (2.0 * pivot - h).alias("s1"),
            (pivot + rng).alias("r2"),
            (pivot - rng).alias("s2"),
            (h + 2.0 * (pivot - l_)).alias("r3"),
            (l_ - 2.0 * (h - pivot)).alias("s3"),
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
    h, l_, c = _shifted_hlc(high, low, close)
    pivot = (h + l_ + c) / 3.0
    rng = h - l_
    return pl.struct(
        [
            pivot.alias("pivot"),
            (pivot + 0.382 * rng).alias("r1"),
            (pivot - 0.382 * rng).alias("s1"),
            (pivot + 0.618 * rng).alias("r2"),
            (pivot - 0.618 * rng).alias("s2"),
            (pivot + 1.000 * rng).alias("r3"),
            (pivot - 1.000 * rng).alias("s3"),
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
    h, l_, c = _shifted_hlc(high, low, close)
    rng = h - l_
    return pl.struct(
        [
            (c + rng * (1.1 / 2.0)).alias("r4"),
            (c + rng * (1.1 / 4.0)).alias("r3"),
            (c + rng * (1.1 / 6.0)).alias("r2"),
            (c + rng * (1.1 / 12.0)).alias("r1"),
            (c - rng * (1.1 / 12.0)).alias("s1"),
            (c - rng * (1.1 / 6.0)).alias("s2"),
            (c - rng * (1.1 / 4.0)).alias("s3"),
            (c - rng * (1.1 / 2.0)).alias("s4"),
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
    h, l_, c = _shifted_hlc(high, low, close)
    pivot = (h + l_ + 2.0 * c) / 4.0
    rng = h - l_
    return pl.struct(
        [
            pivot.alias("pivot"),
            (2.0 * pivot - l_).alias("r1"),
            (2.0 * pivot - h).alias("s1"),
            (pivot + rng).alias("r2"),
            (pivot - rng).alias("s2"),
        ]
    )


def heikin_ashi(
    open: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Heikin-Ashi Candlestick Transform.

    Computes synthetic smoothed OHLC bars to filter out market noise and highlight trend direction.
    Formulated natively without Python loops by reducing the HA_Open recurrence
    HA_Open[t] = 0.5 * HA_Open[t-1] + 0.5 * HA_Close[t-1] to an exact EMA(alpha=0.5).

    Returns a struct with fields:
        - 'ha_open': synthetic Heikin-Ashi open price
        - 'ha_high': synthetic Heikin-Ashi high price
        - 'ha_low': synthetic Heikin-Ashi low price
        - 'ha_close': synthetic Heikin-Ashi close price

    :param open: open price column or expression
    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars Struct expression
    """
    o = parse_expr(open)
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    ha_close = (o + hi + lo + c) / 4.0
    prev_ha_close = ha_close.shift(1)
    ha_open_seed = pl.when(prev_ha_close.is_null()).then((o + c) / 2.0).otherwise(prev_ha_close)
    ha_open = ha_open_seed.ewm_mean(alpha=0.5, adjust=False)
    ha_high = pl.max_horizontal(hi, ha_open, ha_close)
    ha_low = pl.min_horizontal(lo, ha_open, ha_close)

    return pl.struct(
        [
            ha_open.alias("ha_open"),
            ha_high.alias("ha_high"),
            ha_low.alias("ha_low"),
            ha_close.alias("ha_close"),
        ]
    )
