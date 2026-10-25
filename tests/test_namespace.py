"""Tests for Polars .anta and .ta namespace integration."""

from __future__ import annotations

import polars as pl

# Importing polars_anta registers both `.anta` and `.ta` namespaces on pl.Expr
import polars_anta  # noqa: F401


def test_namespace_trend(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.sma(5).alias("sma_5"),
            pl.col("close").ta.ema(5).alias("ema_5"),
            pl.col("close").ta.wma(5).alias("wma_5"),
            pl.col("close").ta.hma(9).alias("hma_9"),
            pl.col("close").ta.dema(5).alias("dema_5"),
            pl.col("close").ta.tema(5).alias("tema_5"),
            pl.col("close").ta.zlema(5).alias("zlema_5"),
            pl.col("close").ta.vortex(high="high", low="low", period=10).alias("vortex"),
            pl.col("close").ta.adx(high="high", low="low", period=10).alias("adx"),
            pl.col("close").ta.aroon(high="high", low="low", period=10).alias("aroon"),
            pl.col("close").ta.chop(high="high", low="low", period=10).alias("chop"),
            pl.col("close").ta.t3(period=5).alias("t3"),
            pl.col("close").ta.alma(period=5).alias("alma"),
            pl.col("close")
            .ta.mass_index(high="high", low="low", fast_period=3, slow_period=10)
            .alias("mi"),
            pl.col("close").ta.supertrend(high="high", low="low", period=5).alias("st"),
            pl.col("close").ta.psar(high="high", low="low").alias("psar"),
            pl.col("close").ta.kama(period=5).alias("kama"),
            pl.col("close").ta.mcginley_dynamic(period=5).alias("md"),
            pl.col("close").ta.vidya(period=5, cmo_period=5).alias("vidya"),
            pl.col("close").ta.center_of_gravity(period=5).alias("cg"),
            pl.col("close")
            .ta.ichimoku(high="high", low="low", tenkan_period=3, kijun_period=5, senkou_b_period=8)
            .alias("ichi"),
            pl.col("close").ta.pivot_points(high="high", low="low").alias("pivots"),
            pl.col("close").ta.fibonacci_pivots(high="high", low="low").alias("fib_pivots"),
            pl.col("close").ta.camarilla_pivots(high="high", low="low").alias("cam_pivots"),
            pl.col("close").ta.woodie_pivots(open_="open", high="high", low="low").alias("woodie"),
        ]
    )

    assert "sma_5" in df_res.columns
    assert "aroon" in df_res.columns
    assert "chop" in df_res.columns
    assert "t3" in df_res.columns
    assert "st" in df_res.columns
    assert "kama" in df_res.columns
    assert "cg" in df_res.columns
    assert "ichi" in df_res.columns
    assert "pivots" in df_res.columns


def test_namespace_momentum(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.rsi(14).alias("rsi_14"),
            pl.col("close").ta.macd().alias("macd_struct"),
            pl.col("close").ta.stochastic(high="high", low="low").alias("stoch_struct"),
            pl.col("close").ta.williams_r(high="high", low="low").alias("wr_14"),
            pl.col("close").ta.cci(high="high", low="low", period=10).alias("cci"),
            pl.col("close").ta.roc(period=5).alias("roc"),
            pl.col("close").ta.stoch_rsi().alias("srsi"),
            pl.col("close").ta.awesome_oscillator(high="high", low="low").alias("ao"),
            pl.col("close").ta.tsi().alias("tsi"),
            pl.col("close").ta.cmo(period=10).alias("cmo"),
            pl.col("close").ta.ppo(fast_period=5, slow_period=10).alias("ppo"),
            pl.col("close").ta.coppock().alias("coppock"),
            pl.col("close")
            .ta.ultimate_oscillator(high="high", low="low", period1=3, period2=5, period3=8)
            .alias("uo"),
            pl.col("close").ta.fisher_transform(high="high", low="low", period=5).alias("ft"),
            pl.col("close").ta.rvi(open="open", high="high", low="low", period=5).alias("rvi"),
            pl.col("close").ta.kst().alias("kst"),
            pl.col("close").ta.dpo(period=10).alias("dpo"),
            pl.col("close").ta.bop(open="open", high="high", low="low").alias("bop"),
            pl.col("close").ta.cfo(period=5).alias("cfo"),
        ]
    )

    assert "rsi_14" in df_res.columns
    assert "cmo" in df_res.columns
    assert "ppo" in df_res.columns
    assert "uo" in df_res.columns
    assert "bop" in df_res.columns
    assert "cfo" in df_res.columns


def test_namespace_volatility(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.true_range(high="high", low="low").alias("tr"),
            pl.col("close").ta.atr(high="high", low="low", period=14).alias("atr_14"),
            pl.col("close").ta.bollinger_bands(period=20).alias("bb"),
            pl.col("close").ta.keltner_channels(high="high", low="low").alias("kc"),
            pl.col("close").ta.donchian_channels(high="high", low="low", period=10).alias("dc"),
            pl.col("close").ta.stddev(period=10).alias("std"),
            pl.col("close").ta.ulcer_index(period=10).alias("ui"),
            pl.col("close").ta.chandelier_exit(high="high", low="low", period=10).alias("ce"),
            pl.col("close").ta.natr(high="high", low="low", period=10).alias("natr"),
            pl.col("close").ta.historical_volatility(period=10).alias("hv"),
            pl.col("close").ta.parkinson_volatility(high="high", low="low", period=10).alias("pv"),
            pl.col("close")
            .ta.garman_klass_volatility(open="open", high="high", low="low", period=10)
            .alias("gk"),
            pl.col("close")
            .ta.chaikin_volatility(high="high", low="low", period=5, roc_period=5)
            .alias("cv"),
            pl.col("close").ta.donchian_width(high="high", low="low", period=10).alias("dw"),
            pl.col("close").ta.envelopes(period=10, percentage=0.02).alias("env"),
            pl.col("close")
            .ta.keltner_width(high="high", low="low", ema_period=10, atr_period=5)
            .alias("kw"),
            pl.col("close").ta.bollinger_delta(period=10).alias("b_delta"),
            pl.col("close")
            .ta.starc_bands(high="high", low="low", period=10, atr_period=5)
            .alias("starc"),
        ]
    )

    assert "tr" in df_res.columns
    assert "ce" in df_res.columns
    assert "natr" in df_res.columns
    assert "pv" in df_res.columns
    assert "gk" in df_res.columns
    assert "dw" in df_res.columns
    assert "env" in df_res.columns
    assert "kw" in df_res.columns
    assert "b_delta" in df_res.columns
    assert "starc" in df_res.columns


def test_namespace_volume(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.obv(volume_col="volume").alias("obv"),
            pl.col("close").ta.vwap(high="high", low="low", volume_col="volume").alias("vwap"),
            pl.col("close").ta.vwma(volume_col="volume", period=10).alias("vwma_10"),
            pl.col("close")
            .ta.cmf(high="high", low="low", volume_col="volume", period=10)
            .alias("cmf"),
            pl.col("close").ta.adl(high="high", low="low", volume_col="volume").alias("adl"),
            pl.col("close")
            .ta.mfi(high="high", low="low", volume_col="volume", period=10)
            .alias("mfi"),
            pl.col("close")
            .ta.eom(high="high", low="low", volume_col="volume", period=10)
            .alias("eom"),
            pl.col("close").ta.klinger(high="high", low="low", volume_col="volume").alias("kvo"),
            pl.col("close").ta.nvi(volume_col="volume").alias("nvi"),
            pl.col("close").ta.pvi(volume_col="volume").alias("pvi"),
            pl.col("close").ta.pvt(volume_col="volume").alias("pvt"),
            pl.col("close").ta.force_index(volume_col="volume", period=5).alias("fi"),
            pl.col("close")
            .ta.vpci(volume_col="volume", period_short=3, period_long=6)
            .alias("vpci"),
            pl.col("close").ta.vr(volume_col="volume", period=10).alias("vr"),
        ]
    )

    assert "obv" in df_res.columns
    assert "kvo" in df_res.columns
    assert "nvi" in df_res.columns
    assert "fi" in df_res.columns
    assert "vpci" in df_res.columns
    assert "vr" in df_res.columns


def test_namespace_multi_asset_over(multi_asset_ohlcv_df: pl.DataFrame):
    """The killer feature: parallel execution across assets using .over('ticker')."""
    res = multi_asset_ohlcv_df.with_columns(
        [
            pl.col("close").ta.sma(5).over("ticker").alias("sma_5"),
            pl.col("close").ta.rsi(10).over("ticker").alias("rsi_10"),
            pl.col("close").ta.obv("volume").over("ticker").alias("obv"),
            pl.col("close").ta.dema(5).over("ticker").alias("dema_5"),
            pl.col("close").ta.roc(5).over("ticker").alias("roc_5"),
        ]
    )

    assert "sma_5" in res.columns
    assert "dema_5" in res.columns
    assert "roc_5" in res.columns


def test_namespace_stats(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.zscore(5).alias("zscore"),
            pl.col("close").ta.percentile_rank(5).alias("pr"),
            pl.col("close").ta.skewness(5).alias("skew"),
            pl.col("close").ta.kurtosis(5).alias("kurt"),
            pl.col("close").ta.rolling_cov("open", period=5).alias("cov"),
            pl.col("close").ta.rolling_corr("open", period=5).alias("corr"),
            pl.col("close").ta.rolling_beta("open", period=5).alias("beta"),
            pl.col("close").ta.rolling_alpha("open", rf=0.0, period=5).alias("alpha"),
            pl.col("close").ta.linear_regression_slope(5).alias("slope"),
            pl.col("close").ta.linear_regression_intercept(5).alias("intercept"),
            pl.col("close").ta.linear_regression_angle(5).alias("angle"),
            pl.col("close").ta.linear_regression_forecast(5).alias("forecast"),
        ]
    )

    assert "zscore" in df_res.columns
    assert "pr" in df_res.columns
    assert "skew" in df_res.columns
    assert "slope" in df_res.columns
    assert "forecast" in df_res.columns


def test_namespace_alpha(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.ts_rank(5).alias("ts_rank"),
            pl.col("close").ta.ts_argmax(5).alias("ts_argmax"),
            pl.col("close").ta.ts_argmin(5).alias("ts_argmin"),
            pl.col("close").ta.decay_linear(5).alias("decay"),
            pl.col("close").ta.ts_zscore(5).alias("ts_zscore"),
            pl.col("close").ta.returns_divergence(volume="volume", period=5).alias("ret_div"),
            pl.col("high").ta.hl_ratio(low="low").alias("hl_r"),
            pl.col("close").ta.co_ratio(open_="open").alias("co_r"),
            pl.col("close").ta.volume_weighted_momentum(volume="volume", period=5).alias("vwm"),
            pl.col("high").ta.relative_spread(low="low", close="close").alias("rel_spread"),
            pl.col("high").ta.twap(low="low", close="close", period=5).alias("twap"),
            pl.col("close").ta.entropy(period=10, bins=3).alias("entropy"),
            pl.col("close").ta.cs_rank().alias("cs_rank"),
            pl.col("close").ta.neutralize(by="open").alias("neut"),
            pl.col("close").ta.ts_scale(a=1.0).alias("scale"),
            pl.col("close").ta.ts_delta(period=2).alias("delta"),
            pl.col("close").ta.ts_delay(period=2).alias("delay"),
            pl.col("close").ta.signed_power(exponent=2.0).alias("sp"),
            pl.col("close").ta.ts_min(5).alias("min5"),
            pl.col("close").ta.ts_max(5).alias("max5"),
            pl.col("close").ta.ts_sum(5).alias("sum5"),
            pl.col("close").ta.ts_std(5).alias("std5"),
            pl.col("close").ta.ts_correlation("volume", period=5).alias("corr5"),
            pl.col("close").ta.ts_covariance("volume", period=5).alias("cov5"),
            pl.col("close").ta.ts_product(3).alias("prod3"),
            pl.col("close").ta.alpha_001().alias("a001"),
            pl.col("high").ta.alpha_041(low="low", close="close", volume="volume").alias("a041"),
            pl.col("open").ta.alpha_054(high="high", low="low", close="close").alias("a054"),
            pl.col("open").ta.alpha_101(high="high", low="low", close="close").alias("a101"),
        ]
    )

    assert "ts_rank" in df_res.columns
    assert "ts_argmax" in df_res.columns
    assert "decay" in df_res.columns
    assert "hl_r" in df_res.columns
    assert "twap" in df_res.columns
    assert "entropy" in df_res.columns
    assert "cs_rank" in df_res.columns
    assert "neut" in df_res.columns
    assert "scale" in df_res.columns
    assert "delta" in df_res.columns
    assert "sp" in df_res.columns
    assert "min5" in df_res.columns
    assert "corr5" in df_res.columns
    assert "prod3" in df_res.columns
    assert "a001" in df_res.columns
    assert "a101" in df_res.columns


def test_namespace_anta_prefix(sample_ohlcv_df: pl.DataFrame):
    """Verify that .anta prefix works identically to .ta prefix."""
    df_anta = sample_ohlcv_df.select(
        [
            pl.col("close").anta.sma(5).alias("sma_anta"),
            pl.col("close").anta.rsi(14).alias("rsi_anta"),
            pl.col("close").anta.bollinger_bands(20).alias("bb_anta"),
            pl.col("close").anta.obv("volume").alias("obv_anta"),
            pl.col("close").anta.zscore(5).alias("zscore_anta"),
            pl.col("close").anta.ts_rank(5).alias("ts_rank_anta"),
        ]
    )
    df_ta = sample_ohlcv_df.select(
        [
            pl.col("close").ta.sma(5).alias("sma_ta"),
            pl.col("close").ta.rsi(14).alias("rsi_ta"),
            pl.col("close").ta.bollinger_bands(20).alias("bb_ta"),
            pl.col("close").ta.obv("volume").alias("obv_ta"),
            pl.col("close").ta.zscore(5).alias("zscore_ta"),
            pl.col("close").ta.ts_rank(5).alias("ts_rank_ta"),
        ]
    )

    assert df_anta["sma_anta"].equals(df_ta["sma_ta"])
    assert df_anta["rsi_anta"].equals(df_ta["rsi_ta"])
    assert df_anta["obv_anta"].equals(df_ta["obv_ta"])
    assert df_anta["zscore_anta"].equals(df_ta["zscore_ta"])
    assert df_anta["ts_rank_anta"].equals(df_ta["ts_rank_ta"])
