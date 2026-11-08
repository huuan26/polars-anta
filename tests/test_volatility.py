"""Tests for volatility indicators."""

from __future__ import annotations

import math

import polars as pl
import pytest

from polars_anta.volatility import (
    atr,
    bollinger_bands,
    bollinger_delta,
    chaikin_volatility,
    chandelier_exit,
    donchian_channels,
    donchian_width,
    envelopes,
    garman_klass_volatility,
    historical_volatility,
    keltner_channels,
    keltner_width,
    natr,
    parkinson_volatility,
    rogers_satchell_volatility,
    starc_bands,
    stddev,
    true_range,
    ulcer_index,
)


def test_true_range(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(true_range("high", "low", "close").alias("tr"))
    tr_vals = result["tr"].to_list()

    assert len(tr_vals) == len(sample_ohlcv_df)
    for v in tr_vals[1:]:
        assert v > 0.0


def test_atr(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(atr("high", "low", "close", period=14).alias("atr"))
    atr_vals = [v for v in result["atr"].to_list() if v is not None]

    assert len(atr_vals) > 0
    for v in atr_vals:
        assert v > 0.0


def test_bollinger_bands(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        bollinger_bands("close", period=10, num_std=2.0).alias("bb")
    ).unnest("bb")

    assert set(result.columns) == {"upper", "middle", "lower", "bandwidth", "percent_b"}

    uppers = result["upper"].to_list()
    middles = result["middle"].to_list()
    lowers = result["lower"].to_list()

    for u, m, low_val in zip(uppers[9:], middles[9:], lowers[9:], strict=True):
        assert u > m > low_val


def test_keltner_channels(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        keltner_channels("high", "low", "close", ema_period=10, atr_period=5, multiplier=2.0).alias(
            "kc"
        )
    ).unnest("kc")

    assert set(result.columns) == {"upper", "middle", "lower"}

    uppers = result["upper"].to_list()
    middles = result["middle"].to_list()
    lowers = result["lower"].to_list()

    for u, m, low_val in zip(uppers[10:], middles[10:], lowers[10:], strict=True):
        assert u > m > low_val


def test_donchian_channels(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(donchian_channels("high", "low", period=10).alias("dc")).unnest(
        "dc"
    )

    assert set(result.columns) == {"upper", "middle", "lower"}
    uppers = result["upper"].to_list()
    middles = result["middle"].to_list()
    lowers = result["lower"].to_list()

    for u, m, low_val in zip(uppers[9:], middles[9:], lowers[9:], strict=True):
        assert u >= m >= low_val


def test_stddev(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(stddev("close", period=10).alias("std"))
    vals = [x for x in result["std"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_ulcer_index(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(ulcer_index("close", period=10).alias("ui"))
    vals = [x for x in result["ui"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_chandelier_exit(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        chandelier_exit("high", "low", "close", period=10).alias("ce")
    ).unnest("ce")
    assert "long_stop" in result.columns
    assert "short_stop" in result.columns


def test_natr(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(natr("high", "low", "close", period=10).alias("natr"))
    vals = [x for x in result["natr"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_historical_volatility(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(historical_volatility("close", period=10).alias("hv"))
    vals = [x for x in result["hv"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_parkinson_volatility(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(parkinson_volatility("high", "low", period=10).alias("pv"))
    vals = [x for x in result["pv"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_garman_klass_volatility(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        garman_klass_volatility("open", "high", "low", "close", period=10).alias("gk")
    )
    vals = [x for x in result["gk"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_chaikin_volatility(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        chaikin_volatility("high", "low", period=5, roc_period=5).alias("cv")
    )
    vals = [x for x in result["cv"].to_list() if x is not None]
    assert len(vals) > 0


def test_donchian_width(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(donchian_width("high", "low", period=10).alias("dw"))
    vals = [x for x in result["dw"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_envelopes(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        envelopes("close", period=10, percentage=0.025).alias("env")
    ).unnest("env")
    assert "upper" in result.columns
    assert "middle" in result.columns
    assert "lower" in result.columns
    assert result["upper"][-1] > result["middle"][-1] > result["lower"][-1]


def test_keltner_width(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        keltner_width("high", "low", "close", ema_period=10, atr_period=5, multiplier=2.0).alias(
            "kw"
        )
    )
    vals = [x for x in result["kw"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v > 0.0


def test_bollinger_delta(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(bollinger_delta("close", period=10, num_std=2.0).alias("delta"))
    vals = [x for x in result["delta"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_starc_bands(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        starc_bands("high", "low", "close", period=10, atr_period=5, multiplier=2.0).alias("starc")
    ).unnest("starc")
    assert "upper" in result.columns
    assert "middle" in result.columns
    assert "lower" in result.columns
    assert result["upper"][-1] > result["middle"][-1] > result["lower"][-1]


RS_ROWS = [
    (100.0, 102.0, 99.0, 101.0),
    (101.0, 103.5, 100.5, 102.5),
    (102.5, 104.0, 101.0, 103.0),
    (103.0, 105.5, 102.0, 104.5),
    (104.5, 106.0, 103.5, 105.0),
    (105.0, 107.5, 104.0, 106.0),
]


def _rs_term(op: float, hi: float, lo: float, cl: float) -> float:
    return math.log(hi / cl) * math.log(hi / op) + math.log(lo / cl) * math.log(lo / op)


def _rs_expected(period: int) -> float:
    terms = [_rs_term(*row) for row in RS_ROWS[-period:]]
    return math.sqrt(252.0 * sum(terms) / period) * 100.0


def _ohlc_frame() -> pl.DataFrame:
    op, hi, lo, cl = zip(*RS_ROWS, strict=True)
    return pl.DataFrame({"open": op, "high": hi, "low": lo, "close": cl})


def test_rogers_satchell_hand_computed():
    period = 5
    got = _ohlc_frame().select(
        rogers_satchell_volatility("open", "high", "low", "close", period=period).alias("rs")
    )["rs"][-1]

    assert got == pytest.approx(_rs_expected(period), rel=1e-9)


def test_rogers_satchell_constant_prices_is_zero():
    df = pl.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [100.0] * 8,
            "low": [100.0] * 8,
            "close": [100.0] * 8,
        }
    )
    vals = df.select(
        rogers_satchell_volatility("open", "high", "low", "close", period=5).alias("rs")
    )["rs"].to_list()

    assert len(vals) == 8
    assert all(v == 0.0 for v in vals)


def test_rogers_satchell_non_positive_prices_produce_no_nan():
    df = pl.DataFrame(
        {
            "open": [100.0, 0.0, 101.0, -5.0, 102.0, 103.0, 104.0],
            "high": [102.0, 0.0, 103.0, -5.0, 104.0, 105.0, 106.0],
            "low": [99.0, 0.0, 100.0, -5.0, 101.0, 102.0, 103.0],
            "close": [101.0, 0.0, 102.0, -5.0, 103.0, 104.0, 105.0],
        }
    )
    vals = df.select(
        rogers_satchell_volatility("open", "high", "low", "close", period=3).alias("rs")
    )["rs"].to_list()

    for v in vals:
        assert v is None or math.isfinite(v)


def test_rogers_satchell_warmup_rows_are_zero():
    """Warm-up rows emit 0.0, matching parkinson_volatility and garman_klass_volatility."""
    period = 3
    vals = (
        _ohlc_frame()
        .select(
            rogers_satchell_volatility("open", "high", "low", "close", period=period).alias("rs")
        )["rs"]
        .to_list()
    )

    assert vals[: period - 1] == [0.0] * (period - 1)
    assert all(v is not None and v > 0.0 for v in vals[period - 1 :])


def test_rogers_satchell_grouped_by_ticker():
    period = 5
    half = len(RS_ROWS)
    df = pl.DataFrame(
        {
            "ticker": ["A"] * half + ["B"] * half,
            "open": [r[0] for r in RS_ROWS] * 2,
            "high": [r[1] for r in RS_ROWS] * 2,
            "low": [r[2] for r in RS_ROWS] * 2,
            "close": [r[3] for r in RS_ROWS] * 2,
        }
    )
    out = df.with_columns(
        rogers_satchell_volatility("open", "high", "low", "close", period=period)
        .over("ticker")
        .alias("rs")
    )
    a = out.filter(pl.col("ticker") == "A")["rs"][-1]
    b = out.filter(pl.col("ticker") == "B")["rs"][-1]

    expected = _rs_expected(period)
    assert a == pytest.approx(expected, rel=1e-9)
    assert b == pytest.approx(expected, rel=1e-9)


def test_rogers_satchell_rejects_invalid_period():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            rogers_satchell_volatility("open", "high", "low", "close", period=bad)
