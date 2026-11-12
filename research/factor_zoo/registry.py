"""Pre-registered signal registry for the Vietnamese Equity Factor Zoo study.

Covers all 150 daily-OHLCV-compatible signals in ``polars_anta`` across seven families
(``trend``, ``momentum``, ``volatility``, ``volume``, ``stats``, ``microstructure``, ``alpha``)
using default parameters and mandatory scale-free transformations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

import polars_anta as pa


@dataclass(frozen=True)
class SignalSpec:
    """Immutable specification of a candidate factor in the Factor Zoo registry."""

    name: str
    family: str
    builder: Callable[[], pl.Expr]
    transform: str
    params_repr: str = "default"


# Explicit documentation of polars_anta exports excluded from the daily OHLCV signal registry
EXCLUDED_SIGNALS: tuple[tuple[str, str, str], ...] = (
    (
        "ofi",
        "microstructure",
        "Requires L1 order-book bid_price, bid_size, ask_price, ask_size.",
    ),
    (
        "futures_basis",
        "microstructure",
        "Requires derivatives futures_close, spot_close, and days_to_expiry.",
    ),
    (
        "oi_regime",
        "microstructure",
        "Requires futures open_interest series not present in cash equities.",
    ),
    (
        "session_vwap",
        "volume",
        "Intraday session-anchored VWAP requiring sub-daily bars.",
    ),
    (
        "rolling_beta",
        "stats",
        "Bivariate rolling regression requiring an external benchmark return column.",
    ),
    (
        "rolling_alpha",
        "stats",
        "Bivariate CAPM intercept requiring an external benchmark return column.",
    ),
    (
        "cs_rank / cs_zscore / cs_winsorize / neutralize / orthogonalize",
        "alpha",
        "Cross-sectional conditioning operators (not standalone time-series signals).",
    ),
    (
        "ts_scale / ts_delta / ts_delay / signed_power / ts_min / ts_max / ts_sum / ts_std / ts_product / ts_correlation / ts_covariance",
        "alpha",
        "Low-level algebraic DSL building blocks used inside alpha_001..alpha_101.",
    ),
    (
        "resample_* / stitch_futures_roll / validate_ohlcv",
        "resample",
        "Data engineering, bar aggregation, and validation utilities.",
    ),
    (
        "evaluation.* / backtest.* / inference.*",
        "evaluation/backtest/inference",
        "Target return generators, evaluation metrics, backtesting engine, and inference tests.",
    ),
)


def build_registry() -> tuple[SignalSpec, ...]:
    """Return the ordered tuple of all pre-registered daily OHLCV signals."""
    o = pl.col("open")
    h = pl.col("high")
    lo = pl.col("low")
    c = pl.col("close")
    v = pl.col("volume")
    ret = c.pct_change()
    dvol = c * v
    adv20 = v.rolling_mean(window_size=20)

    specs: list[SignalSpec] = [
        # ------------------------------------------------------------------
        # 1. Trend & Overlays (28 signals)
        # ------------------------------------------------------------------
        SignalSpec(
            "sma_10", "trend", lambda: c / pa.sma(c, 10) - 1.0, "close / sma(close, 10) - 1"
        ),
        SignalSpec(
            "ema_10", "trend", lambda: c / pa.ema(c, 10) - 1.0, "close / ema(close, 10) - 1"
        ),
        SignalSpec(
            "wma_10", "trend", lambda: c / pa.wma(c, 10) - 1.0, "close / wma(close, 10) - 1"
        ),
        SignalSpec(
            "hma_10", "trend", lambda: c / pa.hma(c, 10) - 1.0, "close / hma(close, 10) - 1"
        ),
        SignalSpec(
            "dema_10", "trend", lambda: c / pa.dema(c, 10) - 1.0, "close / dema(close, 10) - 1"
        ),
        SignalSpec(
            "tema_10", "trend", lambda: c / pa.tema(c, 10) - 1.0, "close / tema(close, 10) - 1"
        ),
        SignalSpec(
            "zlema_10", "trend", lambda: c / pa.zlema(c, 10) - 1.0, "close / zlema(close, 10) - 1"
        ),
        SignalSpec("t3_5", "trend", lambda: c / pa.t3(c, 5) - 1.0, "close / t3(close, 5) - 1"),
        SignalSpec(
            "alma_9", "trend", lambda: c / pa.alma(c, 9) - 1.0, "close / alma(close, 9) - 1"
        ),
        SignalSpec(
            "kama_10", "trend", lambda: c / pa.kama(c, 10) - 1.0, "close / kama(close, 10) - 1"
        ),
        SignalSpec(
            "mcginley_10",
            "trend",
            lambda: c / pa.mcginley_dynamic(c, 10) - 1.0,
            "close / mcginley_dynamic(close, 10) - 1",
        ),
        SignalSpec(
            "vidya_9", "trend", lambda: c / pa.vidya(c, 9) - 1.0, "close / vidya(close, 9) - 1"
        ),
        SignalSpec(
            "ssl_channel_10",
            "trend",
            lambda: (
                (
                    pa.ssl_channel(h, lo, c, 10).struct.field("ssl_up")
                    - pa.ssl_channel(h, lo, c, 10).struct.field("ssl_down")
                )
                / c
            ),
            "(ssl_up - ssl_down) / close",
        ),
        SignalSpec(
            "tii_30",
            "trend",
            lambda: pa.trend_intensity_index(c, 30).struct.field("tii"),
            "trend_intensity_index(close, 30).tii",
        ),
        SignalSpec(
            "heikin_ashi_body",
            "trend",
            lambda: (
                (
                    pa.heikin_ashi(o, h, lo, c).struct.field("ha_close")
                    - pa.heikin_ashi(o, h, lo, c).struct.field("ha_open")
                )
                / c
            ),
            "(ha_close - ha_open) / close",
        ),
        SignalSpec(
            "vortex_14",
            "trend",
            lambda: (
                pa.vortex(h, lo, c, 14).struct.field("vi_plus")
                - pa.vortex(h, lo, c, 14).struct.field("vi_minus")
            ),
            "vi_plus - vi_minus",
        ),
        SignalSpec(
            "adx_14",
            "trend",
            lambda: (
                pa.adx(h, lo, c, 14).struct.field("adx")
                * (
                    pa.adx(h, lo, c, 14).struct.field("plus_di")
                    - pa.adx(h, lo, c, 14).struct.field("minus_di")
                ).sign()
            ),
            "adx * sign(plus_di - minus_di)",
        ),
        SignalSpec(
            "aroon_25",
            "trend",
            lambda: pa.aroon(h, lo, 25).struct.field("oscillator"),
            "aroon(25).oscillator",
        ),
        SignalSpec("chop_14", "trend", lambda: pa.chop(h, lo, c, 14), "chop(14)"),
        SignalSpec(
            "mass_index_25", "trend", lambda: pa.mass_index(h, lo, 9, 25), "mass_index(9,25)"
        ),
        SignalSpec(
            "supertrend_10",
            "trend",
            lambda: (c - pa.supertrend(h, lo, c, 10, 3.0).struct.field("supertrend")) / c,
            "(close - supertrend) / close",
        ),
        SignalSpec("psar", "trend", lambda: (c - pa.psar(h, lo)) / c, "(close - psar) / close"),
        SignalSpec("cog_10", "trend", lambda: pa.center_of_gravity(c, 10), "center_of_gravity(10)"),
        SignalSpec(
            "ichimoku_tk",
            "trend",
            lambda: (
                (
                    pa.ichimoku(h, lo, c).struct.field("tenkan_sen")
                    - pa.ichimoku(h, lo, c).struct.field("kijun_sen")
                )
                / c
            ),
            "(tenkan_sen - kijun_sen) / close",
        ),
        SignalSpec(
            "pivot_classic",
            "trend",
            lambda: c / pa.pivot_points(h, lo, c).struct.field("pivot") - 1.0,
            "close / pivot - 1",
        ),
        SignalSpec(
            "pivot_fibonacci",
            "trend",
            lambda: c / pa.fibonacci_pivots(h, lo, c).struct.field("pivot") - 1.0,
            "close / fibonacci_pivot - 1",
        ),
        SignalSpec(
            "pivot_camarilla",
            "trend",
            lambda: (
                (c - pa.camarilla_pivots(h, lo, c).struct.field("s3"))
                / (
                    pa.camarilla_pivots(h, lo, c).struct.field("r3")
                    - pa.camarilla_pivots(h, lo, c).struct.field("s3")
                    + 1e-9
                )
            ),
            "(close - camarilla.s3) / (camarilla.r3 - camarilla.s3)",
        ),
        SignalSpec(
            "pivot_woodie",
            "trend",
            lambda: c / pa.woodie_pivots(o, h, lo, c).struct.field("pivot") - 1.0,
            "close / woodie_pivot - 1",
        ),
        # ------------------------------------------------------------------
        # 2. Momentum Oscillators (22 signals)
        # ------------------------------------------------------------------
        SignalSpec("rsi_14", "momentum", lambda: pa.rsi(c, 14), "rsi(close, 14)"),
        SignalSpec(
            "macd_hist",
            "momentum",
            lambda: pa.macd(c, 12, 26, 9).struct.field("histogram") / c,
            "macd.histogram / close",
        ),
        SignalSpec(
            "stochastic_k",
            "momentum",
            lambda: pa.stochastic(h, lo, c, 14, 3, 3).struct.field("k"),
            "stochastic.k",
        ),
        SignalSpec(
            "squeeze_mom",
            "momentum",
            lambda: pa.squeeze_momentum(h, lo, c).struct.field("momentum") / c,
            "squeeze_momentum.momentum / close",
        ),
        SignalSpec("stc", "momentum", lambda: pa.stc(c), "stc(close)"),
        SignalSpec(
            "williams_r_14", "momentum", lambda: pa.williams_r(h, lo, c, 14), "williams_r(14)"
        ),
        SignalSpec("cci_20", "momentum", lambda: pa.cci(h, lo, c, 20), "cci(20)"),
        SignalSpec("roc_10", "momentum", lambda: pa.roc(c, 10), "roc(close, 10)"),
        SignalSpec(
            "stoch_rsi_k", "momentum", lambda: pa.stoch_rsi(c).struct.field("k"), "stoch_rsi.k"
        ),
        SignalSpec(
            "awesome_osc",
            "momentum",
            lambda: pa.awesome_oscillator(h, lo) / c,
            "awesome_oscillator / close",
        ),
        SignalSpec("tsi", "momentum", lambda: pa.tsi(c), "tsi(close)"),
        SignalSpec("trix_15", "momentum", lambda: pa.trix(c, 15, 9).struct.field("trix"), "trix"),
        SignalSpec("cmo_14", "momentum", lambda: pa.cmo(c, 14), "cmo(close, 14)"),
        SignalSpec(
            "ppo_hist",
            "momentum",
            lambda: pa.ppo(c, 12, 26, 9).struct.field("histogram"),
            "ppo.histogram",
        ),
        SignalSpec("coppock", "momentum", lambda: pa.coppock(c), "coppock(close)"),
        SignalSpec(
            "ultimate_osc",
            "momentum",
            lambda: pa.ultimate_oscillator(h, lo, c),
            "ultimate_oscillator(7,14,28)",
        ),
        SignalSpec(
            "fisher_9",
            "momentum",
            lambda: pa.fisher_transform(h, lo, 9).struct.field("fisher"),
            "fisher_transform(9).fisher",
        ),
        SignalSpec(
            "rvi_14", "momentum", lambda: pa.rvi(o, h, lo, c, 14).struct.field("rvi"), "rvi(14).rvi"
        ),
        SignalSpec("kst", "momentum", lambda: pa.kst(c).struct.field("kst"), "kst.kst"),
        SignalSpec("dpo_20", "momentum", lambda: pa.dpo(c, 20) / c, "dpo(close, 20) / close"),
        SignalSpec("bop", "momentum", lambda: pa.bop(o, h, lo, c), "bop"),
        SignalSpec("cfo_14", "momentum", lambda: pa.cfo(c, 14), "cfo(close, 14)"),
        # ------------------------------------------------------------------
        # 3. Volatility & Channels (20 signals)
        # ------------------------------------------------------------------
        SignalSpec(
            "true_range_norm",
            "volatility",
            lambda: pa.true_range(h, lo, c) / c,
            "true_range / close",
        ),
        SignalSpec(
            "atr_14_norm", "volatility", lambda: pa.atr(h, lo, c, 14) / c, "atr(14) / close"
        ),
        SignalSpec("natr_14", "volatility", lambda: pa.natr(h, lo, c, 14), "natr(14)"),
        SignalSpec(
            "bb_percent_b",
            "volatility",
            lambda: pa.bollinger_bands(c, 20, 2.0).struct.field("percent_b"),
            "bollinger_bands(20, 2).percent_b",
        ),
        SignalSpec(
            "bb_delta_20",
            "volatility",
            lambda: pa.bollinger_delta(c, 20, 2.0),
            "bollinger_delta(20, 2)",
        ),
        SignalSpec(
            "keltner_pos",
            "volatility",
            lambda: (
                (c - pa.keltner_channels(h, lo, c).struct.field("middle"))
                / (
                    pa.keltner_channels(h, lo, c).struct.field("upper")
                    - pa.keltner_channels(h, lo, c).struct.field("lower")
                    + 1e-9
                )
            ),
            "(close - keltner.middle) / (keltner.upper - keltner.lower)",
        ),
        SignalSpec(
            "keltner_width_20",
            "volatility",
            lambda: pa.keltner_width(h, lo, c),
            "keltner_width(20, 10, 2)",
        ),
        SignalSpec(
            "donchian_pos",
            "volatility",
            lambda: (
                (c - pa.donchian_channels(h, lo, 20).struct.field("middle"))
                / (
                    pa.donchian_channels(h, lo, 20).struct.field("upper")
                    - pa.donchian_channels(h, lo, 20).struct.field("lower")
                    + 1e-9
                )
            ),
            "(close - donchian.middle) / (donchian.upper - donchian.lower)",
        ),
        SignalSpec(
            "donchian_width_20",
            "volatility",
            lambda: pa.donchian_width(h, lo, 20),
            "donchian_width(20)",
        ),
        SignalSpec(
            "stddev_20_norm",
            "volatility",
            lambda: pa.stddev(c, 20) / c,
            "stddev(close, 20) / close",
        ),
        SignalSpec(
            "ulcer_index_14", "volatility", lambda: pa.ulcer_index(c, 14), "ulcer_index(14)"
        ),
        SignalSpec(
            "chandelier_22",
            "volatility",
            lambda: (c - pa.chandelier_exit(h, lo, c, 22, 3.0).struct.field("long_stop")) / c,
            "(close - chandelier.long_stop) / close",
        ),
        SignalSpec(
            "hist_vol_20",
            "volatility",
            lambda: pa.historical_volatility(c, 20),
            "historical_volatility(20)",
        ),
        SignalSpec(
            "parkinson_vol_20",
            "volatility",
            lambda: pa.parkinson_volatility(h, lo, 20),
            "parkinson_volatility(20)",
        ),
        SignalSpec(
            "garman_klass_vol_20",
            "volatility",
            lambda: pa.garman_klass_volatility(o, h, lo, c, 20),
            "garman_klass_volatility(20)",
        ),
        SignalSpec(
            "rogers_satchell_vol_20",
            "volatility",
            lambda: pa.rogers_satchell_volatility(o, h, lo, c, 20),
            "rogers_satchell_volatility(20)",
        ),
        SignalSpec(
            "yang_zhang_vol_20",
            "volatility",
            lambda: pa.yang_zhang_volatility(o, h, lo, c, 20),
            "yang_zhang_volatility(20)",
        ),
        SignalSpec(
            "chaikin_vol_10",
            "volatility",
            lambda: pa.chaikin_volatility(h, lo, 10, 10),
            "chaikin_volatility(10, 10)",
        ),
        SignalSpec(
            "envelopes_20",
            "volatility",
            lambda: (c - pa.envelopes(c, 20, 0.025).struct.field("middle")) / c,
            "(close - envelopes.middle) / close",
        ),
        SignalSpec(
            "starc_pos",
            "volatility",
            lambda: (
                (c - pa.starc_bands(h, lo, c).struct.field("middle"))
                / (
                    pa.starc_bands(h, lo, c).struct.field("upper")
                    - pa.starc_bands(h, lo, c).struct.field("lower")
                    + 1e-9
                )
            ),
            "(close - starc.middle) / (starc.upper - starc.lower)",
        ),
        # ------------------------------------------------------------------
        # 4. Volume & Money Flow (15 signals)
        # ------------------------------------------------------------------
        SignalSpec(
            "obv_roc_20",
            "volume",
            lambda: pa.obv(c, v).diff(20) / (v.rolling_sum(20) + 1e-9),
            "diff(obv, 20) / sum(volume, 20)",
        ),
        SignalSpec(
            "vwap_20",
            "volume",
            lambda: c / pa.vwap(h, lo, c, v, period=20) - 1.0,
            "close / vwap(20) - 1",
        ),
        SignalSpec(
            "vwap_bands_20",
            "volume",
            lambda: (
                (c - pa.vwap_bands(h, lo, c, v, period=20).struct.field("vwap"))
                / (
                    pa.vwap_bands(h, lo, c, v, period=20).struct.field("upper_1")
                    - pa.vwap_bands(h, lo, c, v, period=20).struct.field("lower_1")
                    + 1e-9
                )
            ),
            "(close - vwap) / (upper_1 - lower_1)",
        ),
        SignalSpec(
            "vwma_20",
            "volume",
            lambda: c / pa.vwma(c, v, 20) - 1.0,
            "close / vwma(20) - 1",
        ),
        SignalSpec("cmf_20", "volume", lambda: pa.cmf(h, lo, c, v, 20), "cmf(20)"),
        SignalSpec(
            "adl_roc_20",
            "volume",
            lambda: pa.adl(h, lo, c, v).diff(20) / (v.rolling_sum(20) + 1e-9),
            "diff(adl, 20) / sum(volume, 20)",
        ),
        SignalSpec("mfi_14", "volume", lambda: pa.mfi(h, lo, c, v, 14), "mfi(14)"),
        SignalSpec(
            "eom_14", "volume", lambda: pa.eom(h, lo, v, 14) / (c + 1e-9), "eom(14) / close"
        ),
        SignalSpec(
            "klinger_osc",
            "volume",
            lambda: (
                (
                    pa.klinger(h, lo, c, v).struct.field("klinger")
                    - pa.klinger(h, lo, c, v).struct.field("signal")
                )
                / (adv20 + 1e-9)
            ),
            "(klinger - signal) / adv20",
        ),
        SignalSpec(
            "nvi_roc_20", "volume", lambda: pa.nvi(c, v).pct_change(20), "pct_change(nvi, 20)"
        ),
        SignalSpec(
            "pvi_roc_20", "volume", lambda: pa.pvi(c, v).pct_change(20), "pct_change(pvi, 20)"
        ),
        SignalSpec(
            "pvt_roc_20",
            "volume",
            lambda: pa.pvt(c, v).diff(20) / (adv20 + 1e-9),
            "diff(pvt, 20) / adv20",
        ),
        SignalSpec(
            "force_index_13",
            "volume",
            lambda: pa.force_index(c, v, 13) / (c * adv20 + 1e-9),
            "force_index(13) / (close * adv20)",
        ),
        SignalSpec("vpci", "volume", lambda: pa.vpci(c, v, 5, 25), "vpci(5, 25)"),
        SignalSpec("vr_26", "volume", lambda: pa.vr(c, v, 26), "vr(26)"),
        # ------------------------------------------------------------------
        # 5. Statistics, OLS & Fractional Differentiation (13 signals)
        # ------------------------------------------------------------------
        SignalSpec("zscore_20", "stats", lambda: pa.zscore(c, 20), "zscore(close, 20)"),
        SignalSpec(
            "percentile_rank_20",
            "stats",
            lambda: pa.percentile_rank(c, 20),
            "percentile_rank(close, 20)",
        ),
        SignalSpec("skewness_20", "stats", lambda: pa.skewness(ret, 20), "skewness(returns, 20)"),
        SignalSpec("kurtosis_20", "stats", lambda: pa.kurtosis(ret, 20), "kurtosis(returns, 20)"),
        SignalSpec(
            "rolling_corr_cv_20",
            "stats",
            lambda: pa.rolling_corr(ret, v.pct_change(), 20),
            "rolling_corr(ret, vol_chg, 20)",
        ),
        SignalSpec(
            "rolling_cov_cv_20",
            "stats",
            lambda: pa.rolling_cov(ret, v / (adv20 + 1e-9), 20),
            "rolling_cov(ret, v / adv20, 20)",
        ),
        SignalSpec(
            "linreg_slope_14",
            "stats",
            lambda: pa.linear_regression_slope(c, 14) / c,
            "linear_regression_slope(14) / close",
        ),
        SignalSpec(
            "linreg_intercept_14",
            "stats",
            lambda: c / pa.linear_regression_intercept(c, 14) - 1.0,
            "close / linear_regression_intercept(14) - 1",
        ),
        SignalSpec(
            "linreg_angle_14",
            "stats",
            lambda: pa.linear_regression_angle(c / c.rolling_mean(14), 14),
            "linear_regression_angle(close / sma14, 14)",
        ),
        SignalSpec(
            "linreg_forecast_14",
            "stats",
            lambda: pa.linear_regression_forecast(c, 14) / c - 1.0,
            "linear_regression_forecast(14) / close - 1",
        ),
        SignalSpec(
            "linreg_r2_14",
            "stats",
            lambda: pa.linear_regression_r2(c, 14),
            "linear_regression_r2(close, 14)",
        ),
        SignalSpec(
            "linreg_stderr_14",
            "stats",
            lambda: pa.linear_regression_stderr(c, 14) / c,
            "linear_regression_stderr(14) / close",
        ),
        SignalSpec(
            "frac_diff_04",
            "stats",
            lambda: pa.frac_diff(c.log(), d=0.4, window=20),
            "frac_diff(log(close), d=0.4, window=20)",
        ),
        # ------------------------------------------------------------------
        # 6. Market Microstructure & Liquidity (5 signals)
        # ------------------------------------------------------------------
        SignalSpec(
            "amihud_20",
            "microstructure",
            lambda: pa.amihud_illiquidity(c, dvol, 20),
            "amihud_illiquidity(close, close*volume, 20)",
        ),
        SignalSpec(
            "roll_spread_20",
            "microstructure",
            lambda: pa.roll_spread(c, 20),
            "roll_spread(close, 20)",
        ),
        SignalSpec(
            "corwin_schultz_20",
            "microstructure",
            lambda: pa.corwin_schultz_spread(h, lo, 20),
            "corwin_schultz_spread(high, low, 20)",
        ),
        SignalSpec(
            "kyle_lambda_20",
            "microstructure",
            lambda: pa.kyle_lambda(c, dvol, 20),
            "kyle_lambda(close, close*volume, 20)",
        ),
        SignalSpec(
            "vpin_20", "microstructure", lambda: pa.vpin(c, v, 20), "vpin(close, volume, 20)"
        ),
        # ------------------------------------------------------------------
        # 7. Alpha Primitives, Regime Complexity & 33 Formulaic Alphas (47 signals)
        # ------------------------------------------------------------------
        SignalSpec("ts_rank_10", "alpha", lambda: pa.ts_rank(c, 10), "ts_rank(close, 10)"),
        SignalSpec("ts_argmax_10", "alpha", lambda: pa.ts_argmax(c, 10), "ts_argmax(close, 10)"),
        SignalSpec("ts_argmin_10", "alpha", lambda: pa.ts_argmin(c, 10), "ts_argmin(close, 10)"),
        SignalSpec(
            "decay_linear_10",
            "alpha",
            lambda: c / pa.decay_linear(c, 10) - 1.0,
            "close / decay_linear(close, 10) - 1",
        ),
        SignalSpec("ts_zscore_20", "alpha", lambda: pa.ts_zscore(c, 20), "ts_zscore(close, 20)"),
        SignalSpec(
            "returns_divergence_14",
            "alpha",
            lambda: pa.returns_divergence(c, v, 14),
            "returns_divergence(close, volume, 14)",
        ),
        SignalSpec("hl_ratio", "alpha", lambda: pa.hl_ratio(h, lo), "hl_ratio(high, low)"),
        SignalSpec("co_ratio", "alpha", lambda: pa.co_ratio(c, o), "co_ratio(close, open)"),
        SignalSpec(
            "vol_weighted_mom_10",
            "alpha",
            lambda: pa.volume_weighted_momentum(c, v, 10),
            "volume_weighted_momentum(close, volume, 10)",
        ),
        SignalSpec(
            "relative_spread",
            "alpha",
            lambda: pa.relative_spread(h, lo, c),
            "relative_spread(high, low, close)",
        ),
        SignalSpec(
            "twap_20", "alpha", lambda: c / pa.twap(h, lo, c, 20) - 1.0, "close / twap(20) - 1"
        ),
        SignalSpec(
            "entropy_20", "alpha", lambda: pa.entropy(ret, 20, 5), "entropy(returns, 20, 5)"
        ),
        SignalSpec(
            "fractal_dimension_20",
            "alpha",
            lambda: pa.fractal_dimension(h, lo, 20),
            "fractal_dimension(high, low, 20)",
        ),
        SignalSpec("hurst_50", "alpha", lambda: pa.hurst_exponent(c, 50), "hurst_exponent(50)"),
        # 33 WorldQuant 101 Formulaic Alphas
        SignalSpec("alpha_001", "alpha", lambda: pa.alpha_001(c), "alpha_001(close)"),
        SignalSpec(
            "alpha_002", "alpha", lambda: pa.alpha_002(o, c, v), "alpha_002(open,close,vol)"
        ),
        SignalSpec("alpha_003", "alpha", lambda: pa.alpha_003(o, v), "alpha_003(open,vol)"),
        SignalSpec("alpha_004", "alpha", lambda: pa.alpha_004(lo), "alpha_004(low)"),
        SignalSpec(
            "alpha_005", "alpha", lambda: pa.alpha_005(o, h, lo, c, v), "alpha_005(o,h,l,c,v)"
        ),
        SignalSpec("alpha_006", "alpha", lambda: pa.alpha_006(o, v), "alpha_006(open,vol)"),
        SignalSpec("alpha_007", "alpha", lambda: pa.alpha_007(c, v), "alpha_007(close,vol)"),
        SignalSpec("alpha_008", "alpha", lambda: pa.alpha_008(o, c), "alpha_008(open,close)"),
        SignalSpec("alpha_009", "alpha", lambda: pa.alpha_009(c) / c, "alpha_009(close) / close"),
        SignalSpec("alpha_010", "alpha", lambda: pa.alpha_010(c), "alpha_010(close)"),
        SignalSpec("alpha_011", "alpha", lambda: pa.alpha_011(h, lo, c, v), "alpha_011(h,l,c,v)"),
        SignalSpec(
            "alpha_012", "alpha", lambda: pa.alpha_012(c, v) / c, "alpha_012(close,vol) / close"
        ),
        SignalSpec("alpha_013", "alpha", lambda: pa.alpha_013(c, v), "alpha_013(close,vol)"),
        SignalSpec(
            "alpha_014", "alpha", lambda: pa.alpha_014(o, c, v), "alpha_014(open,close,vol)"
        ),
        SignalSpec("alpha_015", "alpha", lambda: pa.alpha_015(h, v), "alpha_015(high,vol)"),
        SignalSpec("alpha_016", "alpha", lambda: pa.alpha_016(h, v), "alpha_016(high,vol)"),
        SignalSpec("alpha_017", "alpha", lambda: pa.alpha_017(c, v), "alpha_017(close,vol)"),
        SignalSpec("alpha_018", "alpha", lambda: pa.alpha_018(o, c), "alpha_018(open,close)"),
        SignalSpec("alpha_019", "alpha", lambda: pa.alpha_019(c), "alpha_019(close)"),
        SignalSpec(
            "alpha_020",
            "alpha",
            lambda: pa.alpha_020(o, h, lo, c),
            "alpha_020(open,high,low,close)",
        ),
        SignalSpec("alpha_021", "alpha", lambda: pa.alpha_021(c, v), "alpha_021(close,vol)"),
        SignalSpec(
            "alpha_022", "alpha", lambda: pa.alpha_022(h, c, v), "alpha_022(high,close,vol)"
        ),
        SignalSpec("alpha_023", "alpha", lambda: pa.alpha_023(h) / c, "alpha_023(high) / close"),
        SignalSpec("alpha_024", "alpha", lambda: pa.alpha_024(c) / c, "alpha_024(close) / close"),
        SignalSpec("alpha_025", "alpha", lambda: pa.alpha_025(h, lo, c, v), "alpha_025(h,l,c,v)"),
        SignalSpec("alpha_026", "alpha", lambda: pa.alpha_026(h, v), "alpha_026(high,vol)"),
        SignalSpec("alpha_027", "alpha", lambda: pa.alpha_027(h, lo, c, v), "alpha_027(h,l,c,v)"),
        SignalSpec("alpha_028", "alpha", lambda: pa.alpha_028(h, lo, c, v), "alpha_028(h,l,c,v)"),
        SignalSpec("alpha_029", "alpha", lambda: pa.alpha_029(c), "alpha_029(close)"),
        SignalSpec("alpha_030", "alpha", lambda: pa.alpha_030(c, v), "alpha_030(close,vol)"),
        SignalSpec(
            "alpha_041",
            "alpha",
            lambda: pa.alpha_041(h, lo, c, v) / c,
            "alpha_041(h,l,c,v) / close",
        ),
        SignalSpec("alpha_054", "alpha", lambda: pa.alpha_054(o, h, lo, c), "alpha_054(o,h,l,c)"),
        SignalSpec("alpha_101", "alpha", lambda: pa.alpha_101(o, h, lo, c), "alpha_101(o,h,l,c)"),
    ]
    return tuple(specs)


def registry_sha256(specs: tuple[SignalSpec, ...] | None = None) -> str:
    """Compute the canonical SHA-256 hash of the sorted signal registry specification."""
    reg = build_registry() if specs is None else specs
    canonical = sorted((s.name, s.family, s.transform, s.params_repr) for s in reg)
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def family_counts(specs: tuple[SignalSpec, ...] | None = None) -> dict[str, int]:
    """Return signal count per family."""
    reg = build_registry() if specs is None else specs
    counts: dict[str, int] = {}
    for s in reg:
        counts[s.family] = counts.get(s.family, 0) + 1
    return counts


def compute_signals_panel(
    df: pl.DataFrame,
    specs: tuple[SignalSpec, ...] | None = None,
    group_by_ticker: str = "ticker",
    group_by_date: str = "date",
    universe_col: str | None = "in_universe",
) -> pl.DataFrame:
    """Evaluate all registered signals per ticker and apply daily cross-sectional percentile rank.

    Follows Section 2 rules:
    1. Each signal is evaluated over ``group_by_ticker`` on the continuous cleaned history.
    2. If ``universe_col`` is present, rows are filtered to ``in_universe == True`` before
       cross-sectional ranking on ``group_by_date`` via ``pa.cs_rank(..., over=group_by_date)``.
    """
    reg = build_registry() if specs is None else specs
    ts_exprs = [s.builder().over(group_by_ticker).cast(pl.Float64).alias(s.name) for s in reg]
    with_raw = df.sort([group_by_ticker, group_by_date]).with_columns(ts_exprs)
    if universe_col is not None and universe_col in with_raw.columns:
        with_raw = with_raw.filter(pl.col(universe_col))

    cs_exprs = [
        pa.cs_rank(
            pl.when(pl.col(s.name).is_finite()).then(pl.col(s.name)).otherwise(None),
            over=group_by_date,
        ).alias(s.name)
        for s in reg
    ]
    return with_raw.sort([group_by_date, group_by_ticker]).with_columns(cs_exprs)


def check_registry() -> dict[str, object]:
    """Self-test that all registered signals evaluate cleanly on a synthetic multi-ticker panel."""
    rng = np.random.default_rng(12345)
    n_bars = 320
    n_tickers = 6
    n_rows = n_bars * n_tickers

    dates = np.tile(np.arange(n_bars, dtype=np.int32), n_tickers)
    tickers = np.repeat([f"T{i:02d}" for i in range(n_tickers)], n_bars)
    close = np.exp(np.cumsum(rng.normal(0.0005, 0.018, size=n_rows))) * 20_000.0
    open_ = close * (1.0 + rng.normal(0.0, 0.005, size=n_rows))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.008, size=n_rows)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.008, size=n_rows)))
    volume = rng.uniform(50_000.0, 2_000_000.0, size=n_rows)

    df = pl.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "in_universe": np.tile(np.arange(n_bars) >= 260, n_tickers),
        }
    )

    specs = build_registry()
    out = compute_signals_panel(df, specs=specs)
    for spec in specs:
        s = out[spec.name]
        non_null = s.drop_nulls()
        if non_null.len() == 0:
            raise AssertionError(f"Signal {spec.name} produced 100% null values!")
    return {
        "n_signals": len(specs),
        "family_counts": family_counts(specs),
        "sha256": registry_sha256(specs),
        "excluded_groups": len(EXCLUDED_SIGNALS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor Zoo Signal Registry")
    parser.add_argument("--check", action="store_true", help="Run synthetic panel self-check")
    args = parser.parse_args()
    specs = build_registry()
    info = {
        "n_signals": len(specs),
        "family_counts": family_counts(specs),
        "sha256": registry_sha256(specs),
    }
    if args.check:
        info = check_registry()
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
