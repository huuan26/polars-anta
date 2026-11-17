"""Market microstructure, futures positioning, and information-theoretic alphas."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..core import IntoExpr, check_period, check_positive, map_sliding_windows, parse_expr
from ..stats import rolling_cov
from ._primitives import ts_zscore


def returns_divergence(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Price-Volume Returns Divergence.

    Calculates the difference between normalized price returns and normalized volume changes:
    ZScore(Return) - ZScore(VolumeChange).

    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback window (default 14)
    :return: Polars expression
    """
    return ts_zscore(parse_expr(close).pct_change(), period=period) - ts_zscore(
        parse_expr(volume).pct_change(), period=period
    )


def hl_ratio(
    high: IntoExpr,
    low: IntoExpr,
) -> pl.Expr:
    """Log High-to-Low Ratio.

    Measures intraday volatility / price expansion: ln(High / Low).

    :param high: high price column or expression
    :param low: low price column or expression
    :return: Polars expression
    """
    h = parse_expr(high)
    lo = parse_expr(low)
    return (h / lo).log()


def co_ratio(
    close: IntoExpr,
    open_: IntoExpr,
) -> pl.Expr:
    """Log Close-to-Open Ratio.

    Measures intraday drift / conviction: ln(Close / Open).

    :param close: close price column or expression
    :param open_: open price column or expression
    :return: Polars expression
    """
    c = parse_expr(close)
    o = parse_expr(open_)
    return (c / o).log()


def volume_weighted_momentum(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 10,
) -> pl.Expr:
    """Volume-Weighted Momentum.

    Measures price rate of change amplified or dampened by relative volume:
    ((Close - Close[t-p]) / Close[t-p]) * (Volume / RollingMean(Volume, p)).

    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback window (default 10)
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)

    roc = (c - c.shift(period)) / c.shift(period)
    v_ratio = v / v.rolling_mean(window_size=period)
    return roc * v_ratio


def relative_spread(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """Normalized Relative Spread.

    Calculates the normalized high-low bar range relative to close: (High - Low) / Close.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars expression
    """
    h = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    return (h - lo) / c


def twap(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Rolling Time-Weighted Average Price (TWAP).

    Rolling mean of typical price (High + Low + Close) / 3 over a specified period.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: rolling lookback window (default 20)
    :return: Polars expression
    """
    h = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    typical_price = (h + lo + c) / 3.0
    return typical_price.rolling_mean(window_size=period)


def entropy(
    close: IntoExpr,
    period: int = 20,
    bins: int = 5,
) -> pl.Expr:
    """Rolling Shannon Entropy of Price Returns.

    Quantifies market disorder and information content to identify trending vs choppy regimes:
    Low entropy indicates strong directional trending; high entropy indicates noise and chop.

    :param close: price column or expression
    :param period: lookback window (default 20)
    :param bins: number of histogram bins for return discretization (default 5)
    :return: Polars expression
    """
    check_period(period, 1)
    check_period(bins, 1, "bins")
    c = parse_expr(close).cast(pl.Float64)
    p_win = int(period)
    n_bins = int(bins)

    def _calc_entropy(s: pl.Series) -> pl.Series:
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        out = np.full(n, np.nan, dtype=np.float64)
        if n <= p_win or p_win < n_bins:
            return pl.Series(out, dtype=pl.Float64).fill_nan(None)

        p0 = vals[:-1]
        p1 = vals[1:]
        pos_pair = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
        safe_ratio = np.where(pos_pair, p1 / np.where(pos_pair, p0, 1.0), 1.0)
        pair_ret = np.where(pos_pair, np.log(safe_ratio), np.nan)

        win_vals = np.lib.stride_tricks.sliding_window_view(vals, p_win + 1)
        win_rets = np.lib.stride_tricks.sliding_window_view(pair_ret, p_win)

        no_nan_vals = ~np.isnan(win_vals).any(axis=1)
        fin_rets = np.isfinite(win_rets)
        cnt_rets = np.sum(fin_rets, axis=1)
        eligible = no_nan_vals & (cnt_rets >= n_bins)
        if not eligible.any():
            return pl.Series(out, dtype=pl.Float64).fill_nan(None)

        idx_elig = np.flatnonzero(eligible)
        r_elig = win_rets[idx_elig]
        f_elig = fin_rets[idx_elig]
        c_elig = cnt_rets[idx_elig].astype(np.float64)

        r_min = np.min(np.where(f_elig, r_elig, np.inf), axis=1, keepdims=True)
        r_max = np.max(np.where(f_elig, r_elig, -np.inf), axis=1, keepdims=True)
        span = r_max[:, 0] - r_min[:, 0]

        ent_elig = np.zeros(len(idx_elig), dtype=np.float64)
        non_flat = span > 0.0
        if non_flat.any():
            nf_pos = np.flatnonzero(non_flat)
            r_nf = r_elig[nf_pos]
            f_nf = f_elig[nf_pos]
            min_nf = r_min[nf_pos]
            bw_nf = span[nf_pos, None] / float(n_bins)
            tot_nf = c_elig[nf_pos]

            chunk = 16384
            stride_bins = n_bins + 1
            for start in range(0, len(nf_pos), chunk):
                stop = min(start + chunk, len(nf_pos))
                c_len = stop - start
                r_blk = r_nf[start:stop]
                f_blk = f_nf[start:stop]
                bw_blk = bw_nf[start:stop]
                min_blk = min_nf[start:stop]

                scaled = np.where(f_blk, (r_blk - min_blk) / bw_blk, 0.0)
                b_idx = np.where(
                    f_blk,
                    np.clip(scaled.astype(np.int64), 0, n_bins - 1),
                    n_bins,
                )
                row_off = (np.arange(c_len, dtype=np.int64) * stride_bins)[:, None]
                counts = np.bincount(
                    (b_idx + row_off).ravel(), minlength=c_len * stride_bins
                ).reshape(c_len, stride_bins)[:, :n_bins]
                probs = counts.astype(np.float64) / tot_nf[start:stop, None]
                safe_p = np.where(counts > 0, probs, 1.0)
                ent_blk = -np.sum(np.where(counts > 0, probs * np.log2(safe_p), 0.0), axis=1)
                ent_elig[nf_pos[start:stop]] = ent_blk

        out[p_win + idx_elig] = ent_elig
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return c.map_batches(_calc_entropy, return_dtype=pl.Float64)


def fractal_dimension(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Ehlers' Fractal Dimension (FRAMA component).

    Quantifies the fractal dimension of a price series between 1.0 (pure trend / straight line)
    and 2.0 (pure noise / plane-filling random walk).

    :param high: high price column or expression
    :param low: low price column or expression
    :param period: lookback period (default 20, should be an even integer)
    :return: Polars expression producing fractal dimension values in [1.0, 2.0]
    """
    check_period(period, 2)

    half_period = max(1, period // 2)
    hi = parse_expr(high)
    lo = parse_expr(low)

    n1 = (
        hi.rolling_max(window_size=half_period) - lo.rolling_min(window_size=half_period)
    ) / float(half_period)
    n2 = n1.shift(half_period)
    n3 = (hi.rolling_max(window_size=period) - lo.rolling_min(window_size=period)) / float(period)

    sum_n1_n2 = n1 + n2
    null_mask = n3.is_null() | (n3 <= 0.0) | (sum_n1_n2 <= 0.0)
    return (
        pl.when(null_mask)
        .then(None)
        .otherwise(((sum_n1_n2.log() - n3.log()) / math.log(2.0)).clip(1.0, 2.0))
    )


def hurst_exponent(
    close: IntoExpr,
    period: int = 50,
) -> pl.Expr:
    """Rolling Hurst Exponent (Rescaled Range Analysis).

    Quantifies the long-term memory and persistence of a time series:
        - H < 0.5: Mean-reverting / anti-persistent regime
        - H = 0.5: Random walk / Geometric Brownian motion
        - H > 0.5: Trending / persistent regime

    :param close: price column or expression
    :param period: lookback window (default 50)
    :return: Polars expression producing Hurst exponent in [0.0, 1.0]
    """
    check_period(period, 4)
    log_half_n = math.log(float(period) * 0.5)

    def _hurst_reducer(valid_w: np.ndarray) -> np.ndarray:
        cum_dev = np.cumsum(valid_w - np.mean(valid_w, axis=1, keepdims=True), axis=1)
        r_vals = np.max(cum_dev, axis=1) - np.min(cum_dev, axis=1)
        s_vals = np.std(valid_w, axis=1, ddof=1)

        h_vals = np.full(len(valid_w), np.nan, dtype=np.float64)
        h_vals[s_vals <= 1e-12] = 0.5
        pos_mask = (s_vals > 1e-12) & (r_vals > 1e-12)
        if pos_mask.any():
            h_vals[pos_mask] = np.clip(
                np.log(r_vals[pos_mask] / s_vals[pos_mask]) / log_half_n, 0.0, 1.0
            )
        return h_vals

    return map_sliding_windows(close, period, _hurst_reducer)


def amihud_illiquidity(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
    scale: float = 1e6,
) -> pl.Expr:
    """Amihud (2002) Illiquidity Ratio.

    Measures price impact per unit of dollar trading volume over a rolling window:
    ILLIQ = RollingMean(|Return_t| / (Close_t * Volume_t), period) * scale.

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param period: Rolling lookback window (default 20).
    :param scale: Scaling factor (default 1e6, measuring return impact per $1M traded).
    :return: Polars expression.
    """
    check_period(period)
    c = parse_expr(close)
    dollar_vol = c * parse_expr(volume)
    ratio = (
        pl.when(dollar_vol > 0.0)
        .then((c.pct_change().abs() / dollar_vol) * float(scale))
        .otherwise(None)
    )
    return ratio.rolling_mean(window_size=period)


def roll_spread(
    close: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Roll (1984) Implicit Effective Bid-Ask Spread Estimator.

    Estimates the effective proportional bid-ask spread from the first-order serial
    autocovariance of log price changes:
    Spread = 2 * sqrt(max(0, -Cov(Delta ln P_t, Delta ln P_{t-1}))).

    :param close: Close price column or expression.
    :param period: Rolling lookback window (default 20).
    :return: Polars expression producing non-negative proportional spread estimates.
    """
    check_period(period, 2)
    c = parse_expr(close)
    log_ret = (c / c.shift(1)).log()
    serial_cov = rolling_cov(log_ret, log_ret.shift(1), period=period)
    return (
        pl.when(serial_cov.is_null())
        .then(None)
        .otherwise(2.0 * (-serial_cov).clip(lower_bound=0.0).sqrt())
    )


def corwin_schultz_spread(
    high: IntoExpr,
    low: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Corwin & Schultz (2012) High-Low Effective Bid-Ask Spread Estimator.

    Estimates the proportional bid-ask spread from single-bar and two-bar High/Low price ranges,
    separating transient bid-ask bounce from true price volatility without requiring orderbook data.

    :param high: High price column or expression.
    :param low: Low price column or expression.
    :param period: Rolling smoothing window (default 20).
    :return: Polars expression producing non-negative proportional spread estimates.
    """
    check_period(period)
    hi = parse_expr(high)
    lo = parse_expr(low)

    hl_log_sq = (hi / lo).log() ** 2
    beta = hl_log_sq + hl_log_sq.shift(1)
    gamma = (pl.max_horizontal(hi, hi.shift(1)) / pl.min_horizontal(lo, lo.shift(1))).log() ** 2

    denom = 3.0 - 2.0 * math.sqrt(2.0)
    exp_alpha = (((math.sqrt(2.0) - 1.0) * beta.sqrt() / denom) - (gamma / denom).sqrt()).exp()
    return (
        ((2.0 * (exp_alpha - 1.0)) / (1.0 + exp_alpha))
        .clip(lower_bound=0.0)
        .rolling_mean(window_size=period)
    )


def kyle_lambda(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
    scale: float = 1e6,
) -> pl.Expr:
    """Kyle (1985) Price Impact Coefficient (Lambda).

    Estimates the market impact cost (slope of price change per unit of signed order flow)
    over a rolling window using signed volume as an order flow imbalance proxy.

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param period: Rolling regression window (default 20).
    :param scale: Scaling multiplier (default 1e6).
    :return: Polars expression.
    """
    check_period(period, 2)
    ret = parse_expr(close).pct_change()
    signed_vol = ret.sign() * parse_expr(volume)
    cov_rv = rolling_cov(ret, signed_vol, period=period)
    var_v = signed_vol.rolling_var(window_size=period)
    return (
        pl.when(var_v.is_null() | cov_rv.is_null())
        .then(None)
        .when(var_v > 0.0)
        .then((cov_rv / var_v) * float(scale))
        .otherwise(0.0)
    )


def vpin(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Volume-Synchronized Probability of Informed Trading (VPIN).

    Estimates order flow toxicity using Bulk Volume Classification (Easley, López de Prado,
    & O'Hara, 2012). Standardizes bar price changes $\\Delta P_t / \\sigma_{\\Delta P, t}$,
    applies a high-precision closed-form Gaussian CDF approximation $\\Phi(Z_t)$ to partition
    volume into buy-initiated ($V_t^B = V_t \\Phi(Z_t)$) and sell-initiated
    ($V_t^S = V_t (1 - \\Phi(Z_t))$) flow, and computes the rolling ratio of absolute order
    imbalance to total volume in $[0.0, 1.0]$.

    :param close: Close price column or expression.
    :param volume: Volume column or expression.
    :param period: Rolling lookback window for standard deviation and flow aggregation (default 20).
    :return: Polars expression producing VPIN values in ``[0.0, 1.0]``.
    """
    check_period(period, 2)
    dp = parse_expr(close).cast(pl.Float64).diff(1)
    v = parse_expr(volume).cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0)

    std_dp = dp.rolling_std(window_size=period)
    z = pl.when(std_dp.is_null()).then(None).when(std_dp > 1e-12).then(dp / std_dp).otherwise(0.0)

    # Bowling et al. (2009) closed-form Normal CDF approximation (max error < 1.4e-4)
    phi_z = 1.0 / (1.0 + (-(0.07056 * (z**3) + 1.5976 * z)).clip(-40.0, 40.0).exp())

    sum_imb = (v * (2.0 * phi_z - 1.0).abs()).rolling_sum(window_size=period)
    sum_vol = v.rolling_sum(window_size=period)

    return (
        pl.when(sum_imb.is_null() | sum_vol.is_null())
        .then(None)
        .when(sum_vol > 0.0)
        .then((sum_imb / sum_vol).clip(0.0, 1.0))
        .otherwise(0.0)
    )


def ofi(
    bid_price: IntoExpr,
    bid_size: IntoExpr,
    ask_price: IntoExpr,
    ask_size: IntoExpr,
    period: int = 1,
) -> pl.Expr:
    """Level-1 Order Flow Imbalance (OFI — Cont, Kukanov & Stoikov, 2014).

    Measures net supply and demand queue changes at the best bid and best ask:
    $e_t = \\mathbb{I}_{\\{P_t^B \\ge P_{t-1}^B\\}} q_t^B - \\mathbb{I}_{\\{P_t^B \\le P_{t-1}^B\\}} q_{t-1}^B - \\mathbb{I}_{\\{P_t^A \\le P_{t-1}^A\\}} q_t^A + \\mathbb{I}_{\\{P_t^A \\ge P_{t-1}^A\\}} q_{t-1}^A$.
    When ``period > 1``, returns the rolling sum of single-bar OFI over ``period`` bars.

    :param bid_price: Best bid price column or expression.
    :param bid_size: Best bid quantity/size column or expression.
    :param ask_price: Best ask price column or expression.
    :param ask_size: Best ask quantity/size column or expression.
    :param period: Rolling aggregation window (default 1 for single-bar OFI).
    :return: Polars expression.
    """
    check_period(period)
    bp = parse_expr(bid_price).cast(pl.Float64)
    bs = parse_expr(bid_size).cast(pl.Float64)
    ap = parse_expr(ask_price).cast(pl.Float64)
    as_ = parse_expr(ask_size).cast(pl.Float64)

    prev_bp, prev_bs, prev_ap, prev_as = bp.shift(1), bs.shift(1), ap.shift(1), as_.shift(1)
    bid_flow = pl.when(bp >= prev_bp).then(bs).otherwise(0.0) - pl.when(bp <= prev_bp).then(
        prev_bs
    ).otherwise(0.0)
    ask_flow = pl.when(ap <= prev_ap).then(as_).otherwise(0.0) - pl.when(ap >= prev_ap).then(
        prev_as
    ).otherwise(0.0)

    valid = (
        prev_bp.is_not_null()
        & prev_bs.is_not_null()
        & prev_ap.is_not_null()
        & prev_as.is_not_null()
    )
    raw_ofi = pl.when(valid).then(bid_flow - ask_flow).otherwise(None)
    return raw_ofi if period == 1 else raw_ofi.rolling_sum(window_size=period)


def futures_basis(
    futures_close: IntoExpr,
    spot_close: IntoExpr,
    days_to_expiry: IntoExpr | float | int | None = None,
    period: int = 20,
    annualization: float = 365.0,
) -> pl.Expr:
    """Index Futures vs. Underlying Spot Basis & Implied Carry Analytics.

    Designed for index futures arbitrage and derivatives research (e.g., VN30F1M vs. VN30-Index,
    CME E-mini vs. SPX). Returns a ``pl.Struct`` containing:
        - ``'basis'``: Raw points spread $F_t - S_t$
        - ``'basis_pct'``: Proportional basis $(F_t - S_t) / S_t$
        - ``'annualized_basis'``: Annualized roll yield / implied carry rate
        - ``'basis_zscore'``: Rolling Z-Score of proportional basis over ``period`` bars

    :param futures_close: Futures price column or expression.
    :param spot_close: Underlying spot index price column or expression.
    :param days_to_expiry: Optional column/expression or positive scalar representing days to maturity.
    :param period: Rolling lookback window for ``basis_zscore`` standardization (default 20).
    :param annualization: Days per year used for carry annualization (default 365.0).
    :return: Polars Struct expression.
    """
    check_period(period, 2)
    check_positive(annualization, "annualization")

    f = parse_expr(futures_close).cast(pl.Float64)
    s = parse_expr(spot_close).cast(pl.Float64)

    raw_basis = f - s
    basis_pct = pl.when(s > 0.0).then(raw_basis / s).otherwise(None)

    if days_to_expiry is None:
        ann_basis = basis_pct
    elif isinstance(days_to_expiry, (int, float)) and not isinstance(days_to_expiry, bool):
        ann_basis = basis_pct * (float(annualization) / max(float(days_to_expiry), 1.0))
    else:
        ann_basis = basis_pct * (
            float(annualization) / parse_expr(days_to_expiry).cast(pl.Float64).clip(lower_bound=1.0)
        )

    mean_bp = basis_pct.rolling_mean(window_size=period)
    std_bp = basis_pct.rolling_std(window_size=period)
    z_bp = (
        pl.when(mean_bp.is_null() | std_bp.is_null())
        .then(None)
        .when(std_bp > 1e-12)
        .then((basis_pct - mean_bp) / std_bp)
        .otherwise(0.0)
    )

    return pl.struct(
        [
            raw_basis.alias("basis"),
            basis_pct.alias("basis_pct"),
            ann_basis.alias("annualized_basis"),
            z_bp.alias("basis_zscore"),
        ]
    )


def oi_regime(
    close: IntoExpr,
    open_interest: IntoExpr,
    period: int = 1,
) -> pl.Expr:
    """Futures Open Interest (OI) & Price Action Regime Classifier.

    Decomposes concurrent price changes $\\Delta P_t$ and open interest changes $\\Delta OI_t$
    over ``period`` bars into the four canonical institutional positioning regimes:
        - ``+2.0`` (**Long Buildup**): $\\Delta P_t > 0$ and $\\Delta OI_t > 0$ (new long entries driving price up)
        - ``+1.0`` (**Short Covering**): $\\Delta P_t > 0$ and $\\Delta OI_t < 0$ (shorts buying back to exit)
        - ``-2.0`` (**Short Buildup**): $\\Delta P_t < 0$ and $\\Delta OI_t > 0$ (new short entries driving price down)
        - ``-1.0`` (**Long Liquidation**): $\\Delta P_t < 0$ and $\\Delta OI_t < 0$ (longs selling to exit)
        - ``0.0`` (**Neutral / Unchanged**): $\\Delta P_t = 0$ or $\\Delta OI_t = 0$

    Returns a ``pl.Struct`` containing:
        - ``'regime'``: Numeric regime code in ``{-2.0, -1.0, 0.0, 1.0, 2.0}``
        - ``'oi_change'``: Raw change in open interest $OI_t - OI_{t - \\text{period}}$
        - ``'oi_pct_change'``: Percentage change in open interest
        - ``'conviction'``: Directional OI-weighted price conviction $\\text{sign}(\\Delta P_t) \\cdot \\text{oi\\_pct\\_change}$

    :param close: Futures close price column or expression.
    :param open_interest: Open interest column or expression.
    :param period: Lookback lag for price and OI difference (default 1).
    :return: Polars Struct expression.
    """
    check_period(period)

    c = parse_expr(close).cast(pl.Float64)
    oi = parse_expr(open_interest).cast(pl.Float64)
    prev_oi = oi.shift(period)

    dp = c.diff(period)
    doi = oi.diff(period)
    oi_pct = pl.when(prev_oi > 0.0).then(doi / prev_oi).otherwise(None)

    regime_code = (
        pl.when(dp.is_null() | doi.is_null())
        .then(None)
        .when((dp > 0.0) & (doi > 0.0))
        .then(2.0)
        .when((dp > 0.0) & (doi < 0.0))
        .then(1.0)
        .when((dp < 0.0) & (doi > 0.0))
        .then(-2.0)
        .when((dp < 0.0) & (doi < 0.0))
        .then(-1.0)
        .otherwise(0.0)
    )

    conviction = pl.when(dp.is_null() | oi_pct.is_null()).then(None).otherwise(dp.sign() * oi_pct)

    return pl.struct(
        [
            regime_code.alias("regime"),
            doi.alias("oi_change"),
            oi_pct.alias("oi_pct_change"),
            conviction.alias("conviction"),
        ]
    )
