"""Oscillator-style momentum indicators implemented natively for Polars."""

from __future__ import annotations

import polars as pl

from ..core import IntoExpr, check_period, check_periods, parse_expr, wilder_smooth
from ..stats import percentile_rank


def rsi(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Relative Strength Index (RSI) using Wilder's smoothing.

    :param close: price column name or expression
    :param period: lookback period (default 14)
    :return: Polars expression producing RSI values in [0, 100]
    """
    check_period(period)

    diff = parse_expr(close).diff(1)
    avg_gain = wilder_smooth(pl.when(diff > 0).then(diff).otherwise(0.0), period)
    avg_loss = wilder_smooth(pl.when(diff < 0).then(-diff).otherwise(0.0), period)

    return (
        pl.when(avg_loss == 0.0)
        .then(100.0)
        .when(avg_gain == 0.0)
        .then(0.0)
        .otherwise(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)))
    )


def _streak_lengths(close: pl.Expr) -> pl.Expr:
    """Signed length of the current run of consecutive up (+) or down (-) closes.

    Flat bars (close == previous close) count as 0 and start a new run.
    """
    diff = close.diff(1)
    sign = pl.when(diff > 0).then(1).when(diff < 0).then(-1).otherwise(0)
    new_run = sign != sign.shift(1)
    idx = pl.int_range(pl.len())
    run_start = pl.when(new_run).then(idx).otherwise(None).forward_fill()
    return (sign * (idx - run_start + 1)).fill_null(0)


def connors_rsi(
    close: IntoExpr,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
) -> pl.Expr:
    """Larry Connors' Composite RSI (CRSI) oscillator in [0, 100].

    Averages three momentum dimensions:

    1. Wilder RSI of price over rsi_period bars
    2. Wilder RSI of the signed streak of consecutive up/down closes over streak_period bars
    3. Rolling percentile rank of 1-bar returns, scaled to [0, 100], over rank_period bars

    The streak series is computed without Python loops and without window
    expressions nested inside an outer window: the start index of the current
    run is forward-filled, so connors_rsi(...).over("ticker") stays legal.

    :param close: price column or expression
    :param rsi_period: RSI period applied to price (default 3)
    :param streak_period: RSI period applied to the streak series (default 2)
    :param rank_period: rolling window for the return percentile rank (default 100)
    :return: Polars expression producing values in [0, 100]
    """
    c = parse_expr(close)
    check_periods(rsi_period, streak_period, rank_period)

    price_rsi = rsi(c, period=rsi_period)
    streak_rsi = rsi(_streak_lengths(c), period=streak_period)
    rank = percentile_rank(c.pct_change(1), period=rank_period) * 100.0
    return (price_rsi + streak_rsi + rank) / 3.0


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
    check_periods(k_period, d_period, smooth_k)

    lowest_low = parse_expr(low).rolling_min(window_size=k_period)
    highest_high = parse_expr(high).rolling_max(window_size=k_period)
    spread = highest_high - lowest_low

    fast_k = (
        pl.when(spread > 0.0)
        .then(100.0 * (parse_expr(close) - lowest_low) / spread)
        .otherwise(50.0)
    )
    k_line = fast_k.rolling_mean(window_size=smooth_k)

    return pl.struct(
        [
            k_line.alias("k"),
            k_line.rolling_mean(window_size=d_period).alias("d"),
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
    check_period(period)
    highest_high = parse_expr(high).rolling_max(window_size=period)
    lowest_low = parse_expr(low).rolling_min(window_size=period)
    spread = highest_high - lowest_low
    return (
        pl.when(spread > 0.0)
        .then(-100.0 * (highest_high - parse_expr(close)) / spread)
        .otherwise(-50.0)
    )


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
    check_period(period)
    tp = (parse_expr(high) + parse_expr(low) + parse_expr(close)) / 3.0
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
    check_period(period)
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
    check_periods(rsi_period, stoch_period, k_period, d_period)

    rsi_val = rsi(close, period=rsi_period)
    lowest_rsi = rsi_val.rolling_min(window_size=stoch_period)
    spread = rsi_val.rolling_max(window_size=stoch_period) - lowest_rsi

    fast_k = pl.when(spread > 0.0).then(100.0 * (rsi_val - lowest_rsi) / spread).otherwise(50.0)
    k_line = fast_k.rolling_mean(window_size=k_period)

    return pl.struct(
        [
            k_line.alias("k"),
            k_line.rolling_mean(window_size=d_period).alias("d"),
        ]
    )


def cmo(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Chande Momentum Oscillator (CMO).

    Oscillates between -100 and +100 based on price momentum.

    :param close: price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    check_period(period)
    diff = parse_expr(close).diff(1)
    sum_gain = pl.when(diff > 0.0).then(diff).otherwise(0.0).rolling_sum(window_size=period)
    sum_loss = pl.when(diff < 0.0).then(-diff).otherwise(0.0).rolling_sum(window_size=period)
    total = sum_gain + sum_loss
    return pl.when(total > 0.0).then(100.0 * (sum_gain - sum_loss) / total).otherwise(0.0)


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
    check_periods(period1, period2, period3)

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    prev_close = c.shift(1)

    min_l_c = pl.min_horizontal(lo, prev_close)
    bp = c - min_l_c
    tr = pl.max_horizontal(hi, prev_close) - min_l_c

    def _avg(p: int) -> pl.Expr:
        s_tr = tr.rolling_sum(window_size=p)
        return pl.when(s_tr > 0.0).then(bp.rolling_sum(window_size=p) / s_tr).otherwise(0.0)

    return 100.0 * (4.0 * _avg(period1) + 2.0 * _avg(period2) + _avg(period3)) / 7.0


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
    spread = parse_expr(high) - parse_expr(low)
    return (
        pl.when(spread > 0.0).then((parse_expr(close) - parse_expr(open)) / spread).otherwise(0.0)
    )


def cfo(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Chande Forecast Oscillator (CFO).

    Computes percentage difference between actual price and linear regression line.

    :param close: price column or expression
    :param period: linear regression period (default 14)
    :return: Polars expression
    """
    check_period(period, 2)

    n = float(period)
    mean_x = (n - 1.0) / 2.0
    sum_x_sq = n * (n * n - 1.0) / 12.0

    c = parse_expr(close)
    # Vectorized covariance sum( (i - mean_x) * close.shift(period - 1 - i) )
    cov_terms = sum(c.shift(period - 1 - i) * (float(i) - mean_x) for i in range(period))
    slope = cov_terms / sum_x_sq
    forecast = c.rolling_mean(window_size=period) + slope * (n - 1.0 - mean_x)

    return pl.when(c != 0.0).then(100.0 * (c - forecast) / c).otherwise(0.0)
