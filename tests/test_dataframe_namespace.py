"""Unit tests for Polars DataFrame and LazyFrame .anta and .ta namespace integrations."""

from __future__ import annotations

import polars as pl

import polars_anta as pa


def test_df_namespace_resample_and_validate(sample_time_ohlcv_df: pl.DataFrame):
    """Test DataFrame namespace resample and validate methods."""
    validated = sample_time_ohlcv_df.anta.validate_ohlcv()
    assert len(validated) == len(sample_time_ohlcv_df)

    res_anta = sample_time_ohlcv_df.anta.resample("5m")
    res_ta = sample_time_ohlcv_df.ta.resample("5m")

    assert len(res_anta) == 6
    assert res_anta.equals(res_ta)


def test_df_namespace_method_chaining(sample_time_ohlcv_df: pl.DataFrame):
    """Test multi-method fluent chaining on DataFrame."""
    res = (
        sample_time_ohlcv_df.anta.resample("5m")
        .anta.sma(period=2)
        .anta.ema(period=2)
        .anta.rsi(period=2)
        .anta.macd(unnest=True)
        .anta.bollinger_bands(period=2, unnest=True)
    )

    expected_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_2",
        "ema_2",
        "rsi_2",
        "macd",
        "signal",
        "histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_bandwidth",
        "bb_percent_b",
    ]
    for col in expected_cols:
        assert col in res.columns


def test_df_namespace_trend_indicators(sample_time_ohlcv_df: pl.DataFrame):
    """Test trend indicators on DataFrame namespace."""
    df = (
        sample_time_ohlcv_df.anta.wma(5)
        .anta.hma(5)
        .anta.dema(5)
        .anta.tema(5)
        .anta.zlema(5)
        .anta.supertrend(period=5, unnest=True, prefix="st_")
        .anta.ssl_channel(period=5, unnest=True, prefix="ssl_")
        .anta.adx(period=5, unnest=True, prefix="adx_")
        .anta.aroon(period=5, unnest=True, prefix="aroon_")
        .anta.trend_intensity_index(period=10, unnest=True, prefix="tii_")
    )

    assert "wma_5" in df.columns
    assert "hma_5" in df.columns
    assert "dema_5" in df.columns
    assert "tema_5" in df.columns
    assert "zlema_5" in df.columns
    assert "st_supertrend" in df.columns
    assert "ssl_ssl_up" in df.columns
    assert "adx_adx" in df.columns
    assert "aroon_aroon_up" in df.columns
    assert "tii_tii" in df.columns


def test_df_namespace_momentum_indicators(sample_time_ohlcv_df: pl.DataFrame):
    """Test momentum indicators on DataFrame namespace."""
    df = (
        sample_time_ohlcv_df.anta.stochastic(k_period=5, unnest=True, prefix="stoch_")
        .anta.stoch_rsi(rsi_period=5, unnest=True, prefix="srsi_")
        .anta.squeeze_momentum(bb_period=5, kc_period=5, unnest=True, prefix="sqz_")
        .anta.williams_r(period=5)
        .anta.cci(period=5)
        .anta.roc(period=5)
    )

    assert "stoch_k" in df.columns
    assert "srsi_k" in df.columns
    assert "sqz_squeeze_on" in df.columns
    assert "williams_r_5" in df.columns
    assert "cci_5" in df.columns
    assert "roc_5" in df.columns


def test_df_namespace_volatility_indicators(sample_time_ohlcv_df: pl.DataFrame):
    """Test volatility indicators on DataFrame namespace."""
    df = (
        sample_time_ohlcv_df.anta.keltner_channels(ema_period=5, atr_period=3, unnest=True)
        .anta.donchian_channels(period=5, unnest=True)
        .anta.atr(period=5)
        .anta.true_range()
        .anta.stddev(period=5)
    )

    assert "kc_upper" in df.columns
    assert "dc_upper" in df.columns
    assert "atr_5" in df.columns
    assert "true_range" in df.columns
    assert "stddev_5" in df.columns


def test_df_namespace_volume_indicators(sample_time_ohlcv_df: pl.DataFrame):
    """Test volume indicators on DataFrame namespace."""
    df = (
        sample_time_ohlcv_df.anta.obv()
        .anta.vwap()
        .anta.vwap_bands(period=5, unnest=True)
        .anta.cmf(period=5)
        .anta.mfi(period=5)
    )

    assert "obv" in df.columns
    assert "vwap" in df.columns
    assert "vwap_upper_1" in df.columns
    assert "cmf_5" in df.columns
    assert "mfi_5" in df.columns


def test_df_namespace_stats_and_alpha(sample_time_ohlcv_df: pl.DataFrame):
    """Test quantitative stats and alpha factor indicators on DataFrame namespace."""
    df = (
        sample_time_ohlcv_df.anta.zscore(period=5)
        .anta.percentile_rank(period=5)
        .anta.ts_rank(period=5)
        .anta.fractal_dimension(period=5)
        .anta.hurst_exponent(period=10)
    )

    assert "zscore_5" in df.columns
    assert "percentile_rank_5" in df.columns
    assert "ts_rank_5" in df.columns
    assert "fractal_dimension_5" in df.columns
    assert "hurst_10" in df.columns


def test_df_namespace_struct_without_unnest(sample_time_ohlcv_df: pl.DataFrame):
    """Test indicators with unnest=False returning struct columns."""
    df = sample_time_ohlcv_df.anta.bollinger_bands(period=5, alias="bb_struct", unnest=False)
    assert "bb_struct" in df.columns
    assert isinstance(df.schema["bb_struct"], pl.Struct)
    assert df["bb_struct"].struct.field("upper").is_not_null().sum() > 0


def test_df_namespace_add_indicators(sample_time_ohlcv_df: pl.DataFrame):
    """Test add_indicators method with individual and list of expressions."""
    res = sample_time_ohlcv_df.anta.add_indicators(
        pa.sma("close", 5).alias("sma_5"),
        [pa.rsi("close", 5).alias("rsi_5"), pa.roc("close", 5).alias("roc_5")],
    )

    assert "sma_5" in res.columns
    assert "rsi_5" in res.columns
    assert "roc_5" in res.columns


def test_df_namespace_multi_ticker_group_by(multi_asset_time_ohlcv_df: pl.DataFrame):
    """Test indicators evaluated with group_by on multi-ticker universe."""
    res = multi_asset_time_ohlcv_df.anta.sma(5, group_by="ticker").anta.rsi(5, group_by="ticker")

    assert "sma_5" in res.columns
    assert "rsi_5" in res.columns

    # Verify SMA begins null for each ticker's first 4 rows
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        sub = res.filter(pl.col("ticker") == ticker)
        assert sub["sma_5"][:4].is_null().all()
        assert sub["sma_5"][4] is not None


def test_df_namespace_add_all(
    sample_time_ohlcv_df: pl.DataFrame, multi_asset_time_ohlcv_df: pl.DataFrame
):
    """Test add_all batch indicator generator."""
    # Single asset
    single_res = sample_time_ohlcv_df.anta.add_all()
    assert "sma_10" in single_res.columns
    assert "rsi_14" in single_res.columns
    assert "macd" in single_res.columns
    assert "bollinger_bands" in single_res.columns
    assert "obv" in single_res.columns

    # Multi-asset
    multi_res = multi_asset_time_ohlcv_df.anta.add_all(group_by="ticker")
    assert "sma_10" in multi_res.columns
    assert "rsi_14" in multi_res.columns


def test_lazyframe_namespace_fluent(sample_time_ohlcv_df: pl.DataFrame):
    """Test LazyFrame namespace chaining and resolution."""
    lf = sample_time_ohlcv_df.lazy().anta.resample("5m").anta.sma(2).anta.rsi(2).anta.add_all()

    assert isinstance(lf, pl.LazyFrame)
    collected = lf.collect()
    assert len(collected) == 6
    assert "sma_2" in collected.columns
    assert "rsi_2" in collected.columns


def test_namespace_ta_vs_anta_equivalence(sample_time_ohlcv_df: pl.DataFrame):
    """Verify that .anta and .ta DataFrame namespaces produce identical outputs."""
    df_anta = sample_time_ohlcv_df.anta.sma(5).anta.rsi(10).anta.obv()
    df_ta = sample_time_ohlcv_df.ta.sma(5).ta.rsi(10).ta.obv()

    assert df_anta.equals(df_ta)
