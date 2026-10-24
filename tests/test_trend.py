"""Tests for trend indicators."""

from __future__ import annotations

import polars as pl
import pytest

from polars_anta.trend import (
    adx,
    alma,
    aroon,
    camarilla_pivots,
    center_of_gravity,
    chop,
    dema,
    ema,
    fibonacci_pivots,
    hma,
    ichimoku,
    kama,
    mass_index,
    mcginley_dynamic,
    pivot_points,
    psar,
    sma,
    ssl_channel,
    supertrend,
    t3,
    tema,
    trend_intensity_index,
    vidya,
    vortex,
    wma,
    woodie_pivots,
    zlema,
)


def test_sma(sample_ohlcv_df: pl.DataFrame):
    period = 5
    result = sample_ohlcv_df.select(sma("close", period=period).alias("sma_5"))
    col = result["sma_5"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    for i in range(period - 1):
        assert col[i] is None

    expected_sma_5 = sum(sample_ohlcv_df["close"][:5]) / 5
    assert pytest.approx(col[4], rel=1e-5) == expected_sma_5


def test_sma_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        sma("close", period=0)


def test_ema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(ema("close", period=period).alias("ema_10"))
    col = result["ema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[0] is not None
    assert col[-1] > col[0]


def test_ema_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        ema("close", period=-2)


def test_wma(sample_ohlcv_df: pl.DataFrame):
    period = 3
    result = sample_ohlcv_df.select(wma("close", period=period).alias("wma_3"))
    col = result["wma_3"].to_list()

    p = sample_ohlcv_df["close"].to_list()
    expected = (p[0] * 1 + p[1] * 2 + p[2] * 3) / 6.0
    assert pytest.approx(col[2], rel=1e-5) == expected


def test_wma_period_1(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(wma("close", period=1).alias("wma_1"))
    assert result["wma_1"].to_list() == sample_ohlcv_df["close"].to_list()


def test_wma_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        wma("close", period=0)


def test_hma(sample_ohlcv_df: pl.DataFrame):
    period = 9
    result = sample_ohlcv_df.select(hma("close", period=period).alias("hma_9"))
    col = result["hma_9"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_hma_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        hma("close", period=-1)


def test_dema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(dema("close", period=period).alias("dema_10"))
    col = result["dema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_tema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(tema("close", period=period).alias("tema_10"))
    col = result["tema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_zlema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(zlema("close", period=period).alias("zlema_10"))
    col = result["zlema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_vortex(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vortex("high", "low", "close", period=14).alias("vi")).unnest(
        "vi"
    )

    assert "vi_plus" in result.columns
    assert "vi_minus" in result.columns

    vi_plus = [x for x in result["vi_plus"].to_list() if x is not None]
    vi_minus = [x for x in result["vi_minus"].to_list() if x is not None]

    assert len(vi_plus) > 0
    assert len(vi_minus) > 0
    for v in vi_plus:
        assert v > 0.0
    for v in vi_minus:
        assert v > 0.0


def test_adx(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        adx("high", "low", "close", period=14).alias("adx_struct")
    ).unnest("adx_struct")

    assert "adx" in result.columns
    assert "plus_di" in result.columns
    assert "minus_di" in result.columns

    adx_vals = [x for x in result["adx"].to_list() if x is not None]
    assert len(adx_vals) > 0
    for a in adx_vals:
        assert 0.0 <= a <= 100.0


def test_aroon(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(aroon("high", "low", period=10).alias("aroon")).unnest("aroon")
    assert "aroon_up" in result.columns
    assert "aroon_down" in result.columns
    assert "oscillator" in result.columns

    up = [x for x in result["aroon_up"].to_list() if x is not None]
    down = [x for x in result["aroon_down"].to_list() if x is not None]
    assert len(up) > 0
    for u in up:
        assert 0.0 <= u <= 100.0
    for d in down:
        assert 0.0 <= d <= 100.0


def test_chop(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(chop("high", "low", "close", period=10).alias("chop"))
    vals = [x for x in result["chop"].to_list() if x is not None]
    assert len(vals) > 0
    for c in vals:
        assert 0.0 <= c <= 100.0


def test_t3(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(t3("close", period=5).alias("t3"))
    vals = [x for x in result["t3"].to_list() if x is not None]
    assert len(vals) > 0


def test_alma(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(alma("close", period=9).alias("alma"))
    vals = [x for x in result["alma"].to_list() if x is not None]
    assert len(vals) > 0
    assert 100.0 < vals[-1] < 150.0


def test_mass_index(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        mass_index("high", "low", fast_period=3, slow_period=10).alias("mi")
    )
    vals = [x for x in result["mi"].to_list() if x is not None]
    assert len(vals) > 0
    for m in vals:
        assert m > 0.0


def test_supertrend(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        supertrend("high", "low", "close", period=5, multiplier=2.0).alias("st")
    ).unnest("st")
    assert "supertrend" in result.columns
    assert "direction" in result.columns
    dirs = [x for x in result["direction"].to_list() if x is not None]
    assert len(dirs) > 0
    for d in dirs:
        assert d in (1, -1)


def test_psar(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(psar("high", "low").alias("psar"))
    vals = [x for x in result["psar"].to_list() if x is not None]
    assert len(vals) > 0


def test_kama(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(kama("close", period=10).alias("kama"))
    vals = [x for x in result["kama"].to_list() if x is not None]
    assert len(vals) > 0


def test_mcginley_dynamic(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(mcginley_dynamic("close", period=10).alias("md"))
    vals = [x for x in result["md"].to_list() if x is not None]
    assert len(vals) > 0


def test_vidya(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vidya("close", period=5, cmo_period=5).alias("vidya"))
    vals = [x for x in result["vidya"].to_list() if x is not None]
    assert len(vals) > 0


def test_center_of_gravity(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(center_of_gravity("close", period=5).alias("cg"))
    vals = [x for x in result["cg"].to_list() if x is not None]
    assert len(vals) > 0


def test_ichimoku(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        ichimoku(
            "high",
            "low",
            "close",
            tenkan_period=3,
            kijun_period=5,
            senkou_b_period=8,
            displacement=3,
        ).alias("ichi")
    ).unnest("ichi")
    assert "tenkan_sen" in result.columns
    assert "kijun_sen" in result.columns
    assert "senkou_span_a" in result.columns
    assert "senkou_span_b" in result.columns
    assert "chikou_span" in result.columns


def test_pivots(sample_ohlcv_df: pl.DataFrame):
    res_std = sample_ohlcv_df.select(pivot_points("high", "low", "close").alias("p")).unnest("p")
    assert "pivot" in res_std.columns
    assert "r1" in res_std.columns
    assert "s1" in res_std.columns

    res_fib = sample_ohlcv_df.select(fibonacci_pivots("high", "low", "close").alias("p")).unnest(
        "p"
    )
    assert "pivot" in res_fib.columns
    assert "r3" in res_fib.columns

    res_cam = sample_ohlcv_df.select(camarilla_pivots("high", "low", "close").alias("p")).unnest(
        "p"
    )
    assert "r4" in res_cam.columns
    assert "s4" in res_cam.columns

    res_wd = sample_ohlcv_df.select(
        woodie_pivots("open", "high", "low", "close").alias("p")
    ).unnest("p")
    assert "pivot" in res_wd.columns
    assert "r2" in res_wd.columns


def test_ssl_channel(sample_ohlcv_df: pl.DataFrame):
    # Direct function call
    result = sample_ohlcv_df.select(
        ssl_channel("high", "low", "close", period=10).alias("ssl")
    ).unnest("ssl")

    assert "ssl_up" in result.columns
    assert "ssl_down" in result.columns
    assert "hlv" in result.columns

    up_vals = [x for x in result["ssl_up"].to_list() if x is not None]
    down_vals = [x for x in result["ssl_down"].to_list() if x is not None]
    hlv_vals = [x for x in result["hlv"].to_list() if x is not None]

    assert len(up_vals) > 0
    assert len(down_vals) > 0
    assert len(hlv_vals) > 0

    for h in hlv_vals:
        assert h in (1, -1)
    for u, d in zip(up_vals, down_vals, strict=False):
        assert u > 0.0
        assert d > 0.0

    # Namespace expression test (.anta and .ta)
    res_anta = sample_ohlcv_df.select(
        pl.col("close").anta.ssl_channel("high", "low", period=10).alias("ssl")
    ).unnest("ssl")
    assert "ssl_up" in res_anta.columns

    res_ta = sample_ohlcv_df.select(
        pl.col("close").ta.ssl_channel("high", "low", period=10).alias("ssl")
    ).unnest("ssl")
    assert "ssl_down" in res_ta.columns

    # Invalid period
    with pytest.raises(ValueError, match="period must be a positive integer"):
        sample_ohlcv_df.select(ssl_channel("high", "low", "close", period=0))


def test_trend_intensity_index(sample_ohlcv_df: pl.DataFrame):
    # Direct function call
    result = sample_ohlcv_df.select(
        trend_intensity_index("close", period=10, signal_period=5).alias("tii")
    ).unnest("tii")

    assert "tii" in result.columns
    assert "signal" in result.columns

    vals = [x for x in result["tii"].to_list() if x is not None]
    sigs = [x for x in result["signal"].to_list() if x is not None]

    assert len(vals) > 0
    assert len(sigs) > 0

    for v in vals:
        assert 0.0 <= v <= 100.0

    # Namespace expression test (.anta and .ta)
    res_anta = sample_ohlcv_df.select(
        pl.col("close").anta.trend_intensity_index(period=10).alias("tii")
    ).unnest("tii")
    assert "tii" in res_anta.columns

    res_ta = sample_ohlcv_df.select(
        pl.col("close").ta.trend_intensity_index(period=10).alias("tii")
    ).unnest("tii")
    assert "signal" in res_ta.columns

    # Invalid period
    with pytest.raises(ValueError, match="period must be an integer greater than 1"):
        sample_ohlcv_df.select(trend_intensity_index("close", period=1))
    with pytest.raises(ValueError, match="signal_period must be a positive integer"):
        sample_ohlcv_df.select(trend_intensity_index("close", period=10, signal_period=0))
