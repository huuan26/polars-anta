"""Tests for momentum indicators."""

from __future__ import annotations

import polars as pl
import pytest

from polars_anta.momentum import (
    awesome_oscillator,
    bop,
    cci,
    cfo,
    cmo,
    coppock,
    dpo,
    fisher_transform,
    kst,
    macd,
    ppo,
    roc,
    rsi,
    rvi,
    squeeze_momentum,
    stc,
    stoch_rsi,
    stochastic,
    tsi,
    ultimate_oscillator,
    williams_r,
)


def test_rsi(sample_ohlcv_df: pl.DataFrame):
    period = 14
    result = sample_ohlcv_df.select(rsi("close", period=period).alias("rsi"))
    vals = [v for v in result["rsi"].to_list() if v is not None]

    assert len(vals) > 0
    for v in vals:
        assert 0.0 <= v <= 100.0


def test_rsi_monotonic_up():
    df = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]})
    res = df.select(rsi("close", period=3).alias("rsi"))
    assert pytest.approx(res["rsi"][-1], rel=1e-3) == 100.0


def test_rsi_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        rsi("close", period=0)


def test_macd(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(macd("close", 12, 26, 9).alias("macd_struct")).unnest(
        "macd_struct"
    )

    assert "macd" in result.columns
    assert "signal" in result.columns
    assert "histogram" in result.columns

    macd_vals = result["macd"].to_list()
    signal_vals = result["signal"].to_list()
    hist_vals = result["histogram"].to_list()

    for m, s, h in zip(macd_vals[-5:], signal_vals[-5:], hist_vals[-5:], strict=True):
        assert pytest.approx(h, rel=1e-5) == (m - s)


def test_macd_invalid_periods():
    with pytest.raises(ValueError):
        macd("close", fast_period=0)
    with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
        macd("close", fast_period=26, slow_period=12)


def test_stochastic(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        stochastic("high", "low", "close", k_period=14, d_period=3).alias("stoch")
    ).unnest("stoch")

    assert "k" in result.columns
    assert "d" in result.columns

    valid_k = [x for x in result["k"].to_list() if x is not None]
    valid_d = [x for x in result["d"].to_list() if x is not None]

    assert len(valid_k) > 0
    assert len(valid_d) > 0

    for val in valid_k:
        assert 0.0 <= val <= 100.0
    for val in valid_d:
        assert 0.0 <= val <= 100.0


def test_williams_r(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(williams_r("high", "low", "close", period=14).alias("wr"))
    valid_wr = [x for x in result["wr"].to_list() if x is not None]

    assert len(valid_wr) > 0
    for val in valid_wr:
        assert -100.0 <= val <= 0.0


def test_cci(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(cci("high", "low", "close", period=10).alias("cci_10"))
    vals = [x for x in result["cci_10"].to_list() if x is not None]
    assert len(vals) > 0
    # CCI is bounded in practical range
    assert any(v != 0.0 for v in vals)


def test_roc(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(roc("close", period=5).alias("roc_5"))
    vals = [x for x in result["roc_5"].to_list() if x is not None]
    assert len(vals) > 0
    # Price is trending up, so last ROC is positive
    assert vals[-1] > 0.0


def test_stoch_rsi(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        stoch_rsi("close", rsi_period=10, stoch_period=10, k_period=3, d_period=3).alias("srsi")
    ).unnest("srsi")

    assert "k" in result.columns
    assert "d" in result.columns
    vals_k = [x for x in result["k"].to_list() if x is not None]
    assert len(vals_k) > 0
    for k in vals_k:
        assert 0.0 <= k <= 100.0


def test_awesome_oscillator(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        awesome_oscillator("high", "low", fast_period=3, slow_period=10).alias("ao")
    )
    vals = [x for x in result["ao"].to_list() if x is not None]
    assert len(vals) > 0


def test_tsi(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(tsi("close", long_period=13, short_period=7).alias("tsi"))
    vals = [x for x in result["tsi"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert -100.0 <= v <= 100.0


def test_cmo(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(cmo("close", period=10).alias("cmo"))
    vals = [x for x in result["cmo"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert -100.0 <= v <= 100.0


def test_ppo(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        ppo("close", fast_period=5, slow_period=10).alias("ppo")
    ).unnest("ppo")
    assert "ppo" in result.columns
    assert "signal" in result.columns
    assert "histogram" in result.columns


def test_coppock(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        coppock("close", wma_period=5, fast_roc=3, slow_roc=6).alias("coppock")
    )
    vals = [x for x in result["coppock"].to_list() if x is not None]
    assert len(vals) > 0


def test_ultimate_oscillator(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        ultimate_oscillator("high", "low", "close", 3, 5, 8).alias("uo")
    )
    vals = [x for x in result["uo"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert 0.0 <= v <= 100.0


def test_fisher_transform(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(fisher_transform("high", "low", period=5).alias("ft")).unnest(
        "ft"
    )
    assert "fisher" in result.columns
    assert "signal" in result.columns


def test_rvi(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        rvi("open", "high", "low", "close", period=5).alias("rvi")
    ).unnest("rvi")
    assert "rvi" in result.columns
    assert "signal" in result.columns


def test_kst(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        kst("close", r1=3, r2=4, r3=5, r4=6, s1=3, s2=3, s3=3, s4=4).alias("kst")
    ).unnest("kst")
    assert "kst" in result.columns
    assert "signal" in result.columns


def test_dpo(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(dpo("close", period=10).alias("dpo"))
    vals = [x for x in result["dpo"].to_list() if x is not None]
    assert len(vals) > 0


def test_bop(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(bop("open", "high", "low", "close").alias("bop"))
    vals = [x for x in result["bop"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert -1.0 <= v <= 1.0


def test_cfo(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(cfo("close", period=5).alias("cfo"))
    vals = [x for x in result["cfo"].to_list() if x is not None]
    assert len(vals) > 0


def test_squeeze_momentum(sample_ohlcv_df: pl.DataFrame):
    # Direct function call
    result = sample_ohlcv_df.select(
        squeeze_momentum(
            "high",
            "low",
            "close",
            bb_period=10,
            bb_mult=2.0,
            kc_period=10,
            kc_mult=1.5,
            mom_period=10,
        ).alias("sqz")
    ).unnest("sqz")

    assert "squeeze_on" in result.columns
    assert "squeeze_off" in result.columns
    assert "no_squeeze" in result.columns
    assert "momentum" in result.columns

    sqz_on = [x for x in result["squeeze_on"].to_list() if x is not None]
    sqz_off = [x for x in result["squeeze_off"].to_list() if x is not None]
    no_sqz = [x for x in result["no_squeeze"].to_list() if x is not None]
    mom = [x for x in result["momentum"].to_list() if x is not None]

    assert len(sqz_on) > 0
    assert len(sqz_off) > 0
    assert len(no_sqz) > 0
    assert len(mom) > 0

    # Mutual consistency: on and off cannot both be true simultaneously
    for on, off in zip(sqz_on, sqz_off, strict=False):
        assert not (on and off)

    # Namespace expression test (.anta and .ta)
    res_anta = sample_ohlcv_df.select(
        pl.col("close")
        .anta.squeeze_momentum("high", "low", bb_period=10, kc_period=10)
        .alias("sqz")
    ).unnest("sqz")
    assert "momentum" in res_anta.columns

    res_ta = sample_ohlcv_df.select(
        pl.col("close").ta.squeeze_momentum("high", "low", bb_period=10, kc_period=10).alias("sqz")
    ).unnest("sqz")
    assert "squeeze_on" in res_ta.columns

    # Invalid arguments
    with pytest.raises(ValueError, match="periods must be positive integers"):
        sample_ohlcv_df.select(squeeze_momentum("high", "low", "close", bb_period=0))
    with pytest.raises(ValueError, match="mom_period must be a positive integer"):
        sample_ohlcv_df.select(
            squeeze_momentum("high", "low", "close", bb_period=10, kc_period=10, mom_period=-1)
        )


def test_stc(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(stc("close").alias("stc"))
    vals = [v for v in result["stc"].to_list() if v is not None]
    assert len(vals) > 0
    for v in vals:
        # STC is typically bounded in [0, 100]; allow small numerical tolerance
        assert -1.0 <= v <= 101.0


def test_stc_flat_series():
    df = pl.DataFrame({"close": [50.0] * 100})
    res = df.select(stc("close").alias("stc"))
    # On a flat series the stochastic spreads are zero, so %K falls back to 50 and STC should approach 50
    last = res["stc"][-1]
    assert last is not None
    assert pytest.approx(last, abs=5.0) == 50.0


def test_stc_invalid_params():
    with pytest.raises(ValueError, match="periods must be positive"):
        stc("close", fast_period=0)
    with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
        stc("close", fast_period=50, slow_period=23)
    with pytest.raises(ValueError, match="factor must be in"):
        stc("close", factor=1.5)


def test_stc_namespace(sample_ohlcv_df: pl.DataFrame):
    expr_res = sample_ohlcv_df.select(pl.col("close").ta.stc().alias("stc"))
    df_res = sample_ohlcv_df.ta.stc()
    assert "stc" in df_res.columns
    # Values should match
    assert expr_res["stc"].equals(df_res["stc"])
