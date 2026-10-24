"""Tests for volume indicators."""

from __future__ import annotations

import polars as pl
import pytest

from polars_anta.volume import (
    adl,
    cmf,
    eom,
    force_index,
    klinger,
    mfi,
    nvi,
    obv,
    pvi,
    pvt,
    vpci,
    vr,
    vwap,
    vwap_bands,
    vwma,
)


def test_obv(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(obv("close", "volume").alias("obv"))
    obv_vals = result["obv"].to_list()

    assert len(obv_vals) == len(sample_ohlcv_df)
    assert all(x is not None for x in obv_vals)


def test_vwap_cumulative(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vwap("high", "low", "close", "volume").alias("vwap"))
    vals = result["vwap"].to_list()

    assert len(vals) == len(sample_ohlcv_df)
    for v in vals:
        assert 90.0 < v < 140.0


def test_vwap_rolling(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        vwap("high", "low", "close", "volume", period=5).alias("vwap_5")
    )
    vals = result["vwap_5"].to_list()
    assert all(x is None for x in vals[:4])
    assert all(x is not None for x in vals[4:])


def test_vwma(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vwma("close", "volume", period=10).alias("vwma"))
    vals = result["vwma"].to_list()

    assert all(x is None for x in vals[:9])
    assert all(x is not None for x in vals[9:])


def test_cmf(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(cmf("high", "low", "close", "volume", period=10).alias("cmf"))
    vals = [x for x in result["cmf"].to_list() if x is not None]

    assert len(vals) > 0
    for v in vals:
        assert -1.0 <= v <= 1.0


def test_adl(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(adl("high", "low", "close", "volume").alias("adl"))
    vals = result["adl"].to_list()

    assert len(vals) == len(sample_ohlcv_df)
    assert all(x is not None for x in vals)


def test_mfi(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(mfi("high", "low", "close", "volume", period=10).alias("mfi"))
    vals = [x for x in result["mfi"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert 0.0 <= v <= 100.0


def test_eom(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(eom("high", "low", "volume", period=10).alias("eom"))
    vals = [x for x in result["eom"].to_list() if x is not None]
    assert len(vals) > 0


def test_klinger(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        klinger(
            "high", "low", "close", "volume", fast_period=5, slow_period=10, signal_period=3
        ).alias("kvo")
    ).unnest("kvo")
    assert "klinger" in result.columns
    assert "signal" in result.columns


def test_nvi(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(nvi("close", "volume").alias("nvi"))
    vals = [x for x in result["nvi"].to_list() if x is not None]
    assert len(vals) > 0


def test_pvi(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(pvi("close", "volume").alias("pvi"))
    vals = [x for x in result["pvi"].to_list() if x is not None]
    assert len(vals) > 0


def test_pvt(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(pvt("close", "volume").alias("pvt"))
    vals = [x for x in result["pvt"].to_list() if x is not None]
    assert len(vals) > 0


def test_force_index(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(force_index("close", "volume", period=5).alias("fi"))
    vals = [x for x in result["fi"].to_list() if x is not None]
    assert len(vals) > 0


def test_vpci(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        vpci("close", "volume", period_short=3, period_long=6).alias("vpci")
    )
    vals = [x for x in result["vpci"].to_list() if x is not None]
    assert len(vals) > 0


def test_vr(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vr("close", "volume", period=10).alias("vr"))
    vals = [x for x in result["vr"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_vwap_bands(sample_ohlcv_df: pl.DataFrame):
    # Cumulative mode
    res_cum = sample_ohlcv_df.select(
        vwap_bands("high", "low", "close", "volume").alias("vb")
    ).unnest("vb")

    expected_cols = ["vwap", "upper_1", "lower_1", "upper_2", "lower_2", "upper_3", "lower_3"]
    for col in expected_cols:
        assert col in res_cum.columns

    # Verify band ordering
    for row in res_cum.to_dicts():
        assert (
            row["lower_3"]
            <= row["lower_2"]
            <= row["lower_1"]
            <= row["vwap"]
            <= row["upper_1"]
            <= row["upper_2"]
            <= row["upper_3"]
        )

    # Rolling mode
    res_roll = sample_ohlcv_df.select(
        vwap_bands("high", "low", "close", "volume", period=10).alias("vb")
    ).unnest("vb")
    assert "vwap" in res_roll.columns
    assert res_roll["vwap"][0] is None

    # Namespace expressions (.anta and .ta)
    anta_res = sample_ohlcv_df.select(
        pl.col("close").anta.vwap_bands("high", "low", "volume", period=10).alias("vb")
    ).unnest("vb")
    assert "upper_1" in anta_res.columns

    ta_res = sample_ohlcv_df.select(
        pl.col("close").ta.vwap_bands("high", "low", "volume", period=10).alias("vb")
    ).unnest("vb")
    assert "lower_1" in ta_res.columns

    # Invalid period
    with pytest.raises(ValueError, match="period must be a positive integer"):
        sample_ohlcv_df.select(vwap_bands("high", "low", "close", "volume", period=0))
