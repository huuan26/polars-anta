"""Tests for volume indicators."""

from __future__ import annotations

import polars as pl
import pytest

from polars_anta.volume import (
    adl,
    bw_mfi,
    cmf,
    eom,
    force_index,
    klinger,
    mfi,
    nvi,
    obv,
    pvi,
    pvt,
    session_vwap,
    twiggs_money_flow,
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


def test_session_vwap():
    """Test session-anchored VWAP resetting across trading dates and symbols."""
    df = pl.DataFrame(
        {
            "symbol": ["VN30F1M"] * 4,
            "date": ["2026-10-01", "2026-10-01", "2026-10-02", "2026-10-02"],
            "high": [100.0, 106.0, 200.0, 206.0],
            "low": [100.0, 106.0, 200.0, 206.0],
            "close": [100.0, 106.0, 200.0, 206.0],
            "volume": [100.0, 100.0, 50.0, 50.0],
        }
    )
    # Day 1 bar 0: vwap = 100.0, zscore = 0.0
    # Day 1 bar 1: vwap = 103.0, std = 3.0, zscore = +1.0
    # Day 2 bar 0: resets! vwap = 200.0, zscore = 0.0
    # Day 2 bar 1: vwap = 203.0, std = 3.0, zscore = +1.0
    res = df.anta.session_vwap(session_by=["symbol", "date"], unnest=True)
    assert res["session_vwap"].to_list() == [100.0, 103.0, 200.0, 203.0]
    assert res["session_zscore"].to_list() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert res["session_upper_1"][1] == pytest.approx(106.0)
    assert res["session_lower_1"][1] == pytest.approx(100.0)

    # Expression namespace (.anta and .ta) with .over() when session_by=None
    expr_res = df.select(
        pl.col("close")
        .ta.session_vwap("high", "low", "volume")
        .over(["symbol", "date"])
        .alias("sv")
    ).unnest("sv")
    assert expr_res["vwap"].to_list() == [100.0, 103.0, 200.0, 203.0]

    # Single string session_by
    single_res = df.select(
        pl.col("close").anta.session_vwap("high", "low", "volume", session_by="date").alias("sv")
    ).unnest("sv")
    assert single_res["vwap"].to_list() == [100.0, 103.0, 200.0, 203.0]

    with pytest.raises(ValueError, match="band multipliers"):
        session_vwap("high", "low", "close", "volume", mult1=0.0)


def test_twiggs_money_flow_matches_manual(sample_ohlcv_df: pl.DataFrame):
    from polars_anta.core import wilder_smooth

    period = 21
    result = sample_ohlcv_df.select(
        twiggs_money_flow("high", "low", "close", "volume", period=period).alias("tmf")
    )["tmf"].to_list()

    prev_close = pl.col("close").shift(1)
    true_high = pl.max_horizontal(pl.col("high"), prev_close)
    true_low = pl.min_horizontal(pl.col("low"), prev_close)
    spread = true_high - true_low
    adv = (
        pl.when(spread == 0)
        .then(0.0)
        .otherwise(pl.col("volume") * (2 * pl.col("close") - true_low - true_high) / spread)
    )
    expected = sample_ohlcv_df.select(
        (wilder_smooth(adv, period) / wilder_smooth(pl.col("volume"), period)).alias("tmf")
    )["tmf"].to_list()

    assert result == pytest.approx(expected)
    assert all(x is None or -1.0 <= x <= 1.0 for x in result)


def test_twiggs_money_flow_zero_true_range():
    df = pl.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )
    vals = df.select(twiggs_money_flow("high", "low", "close", "volume", period=3).alias("t"))[
        "t"
    ].to_list()
    assert all(x == 0.0 for x in vals)


def test_bw_mfi_formula(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(bw_mfi("high", "low", "volume").alias("bw"))["bw"].to_list()
    expected = sample_ohlcv_df.select(
        ((pl.col("high") - pl.col("low")) / pl.col("volume") * 1e6).alias("bw")
    )["bw"].to_list()
    assert result == pytest.approx(expected)

    scaled = sample_ohlcv_df.select(bw_mfi("high", "low", "volume", scale=1.0).alias("bw"))[
        "bw"
    ].to_list()
    assert scaled == pytest.approx([x / 1e6 for x in expected])


def test_bw_mfi_zero_volume_is_zero():
    df = pl.DataFrame({"high": [10.0, 11.0], "low": [9.0, 10.0], "volume": [0.0, 100.0]})
    vals = df.select(bw_mfi("high", "low", "volume").alias("bw"))["bw"].to_list()
    assert vals[0] == 0.0
    assert vals[1] == pytest.approx(10_000.0)


def test_twiggs_money_flow_rejects_non_positive_period(sample_ohlcv_df: pl.DataFrame):
    with pytest.raises(ValueError):
        twiggs_money_flow("high", "low", "close", "volume", period=0)


def test_twiggs_and_bw_mfi_namespaces(sample_ohlcv_df: pl.DataFrame):
    direct_tmf = sample_ohlcv_df.select(
        twiggs_money_flow("high", "low", "close", "volume").alias("tmf")
    )["tmf"].to_list()
    expr_tmf = sample_ohlcv_df.select(
        pl.col("close").anta.twiggs_money_flow(high="high", low="low", volume="volume").alias("tmf")
    )["tmf"].to_list()
    assert direct_tmf == pytest.approx(expr_tmf)

    direct_bw = sample_ohlcv_df.select(bw_mfi("high", "low", "volume").alias("bw"))["bw"].to_list()
    expr_bw = sample_ohlcv_df.select(
        pl.col("high").anta.bw_mfi(low="low", volume="volume").alias("bw")
    )["bw"].to_list()
    assert direct_bw == pytest.approx(expr_bw)

    assert "twiggs_money_flow_21" in sample_ohlcv_df.anta.twiggs_money_flow().columns
    assert "bw_mfi" in sample_ohlcv_df.ta.bw_mfi().columns
