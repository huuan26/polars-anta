from __future__ import annotations

import polars as pl
import pytest

from polars_anta.alpha import (
    co_ratio,
    decay_linear,
    entropy,
    fractal_dimension,
    hl_ratio,
    hurst_exponent,
    relative_spread,
    returns_divergence,
    ts_argmax,
    ts_argmin,
    ts_rank,
    ts_zscore,
    twap,
    volume_weighted_momentum,
)


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
            "high": [10.5, 11.5, 12.8, 13.5, 14.8, 15.6, 16.9, 17.5, 18.8, 19.5, 20.8, 21.5],
            "low": [9.8, 10.7, 11.6, 12.8, 13.7, 14.8, 15.7, 16.8, 17.6, 18.7, 19.7, 20.8],
            "close": [10.2, 11.3, 12.1, 13.4, 14.2, 15.5, 16.2, 17.4, 18.2, 19.4, 20.2, 21.3],
            "volume": [100, 150, 120, 180, 200, 220, 190, 250, 230, 280, 300, 310],
        }
    )


def test_ts_rank_argmax_argmin(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(
        [
            ts_rank("close", period=5).alias("rank"),
            ts_argmax("close", period=5).alias("argmax"),
            ts_argmin("close", period=5).alias("argmin"),
        ]
    )
    assert len(res) == 12
    # In strictly increasing sequence, last element is highest -> rank is 1.0, argmax is 0 bars ago, argmin is 4 bars ago
    assert res["rank"][-1] == 1.0
    assert res["argmax"][-1] == 0.0
    assert res["argmin"][-1] == 4.0


def test_decay_linear_and_zscore(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(
        [
            decay_linear("close", period=5).alias("decay"),
            ts_zscore("close", period=5).alias("z"),
        ]
    )
    assert len(res) == 12
    assert res["decay"][-1] is not None
    assert res["z"][-1] is not None
    assert res["z"][-1] > 0.0


def test_ratios_and_divergence(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(
        [
            hl_ratio("high", "low").alias("hl"),
            co_ratio("close", "open").alias("co"),
            returns_divergence("close", "volume", period=5).alias("div"),
            volume_weighted_momentum("close", "volume", period=5).alias("vwm"),
            relative_spread("high", "low", "close").alias("spread"),
            twap("high", "low", "close", period=5).alias("twap"),
            entropy("close", period=6, bins=3).alias("entropy"),
        ]
    )
    assert len(res) == 12
    assert res["hl"][-1] > 0.0
    assert res["spread"][-1] > 0.0
    assert res["twap"][-1] > 0.0
    assert res["entropy"][-1] is not None


def test_fractal_dimension(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(fractal_dimension("high", "low", period=6).alias("fd"))
    vals = [x for x in res["fd"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert 1.0 <= v <= 2.0

    # Namespace expressions (.anta and .ta)
    res_anta = sample_ohlcv.select(
        pl.col("high").anta.fractal_dimension("low", period=6).alias("fd")
    )
    assert res_anta["fd"][-1] is not None

    res_ta = sample_ohlcv.select(pl.col("high").ta.fractal_dimension("low", period=6).alias("fd"))
    assert res_ta["fd"][-1] is not None

    # Invalid period
    with pytest.raises(ValueError, match="period must be an integer greater than 1"):
        sample_ohlcv.select(fractal_dimension("high", "low", period=1))


def test_hurst_exponent(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(hurst_exponent("close", period=6).alias("h"))
    vals = [x for x in res["h"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert 0.0 <= v <= 1.0

    # Namespace expressions (.anta and .ta)
    res_anta = sample_ohlcv.select(pl.col("close").anta.hurst_exponent(period=6).alias("h"))
    assert res_anta["h"][-1] is not None

    res_ta = sample_ohlcv.select(pl.col("close").ta.hurst_exponent(period=6).alias("h"))
    assert res_ta["h"][-1] is not None

    # Invalid period
    with pytest.raises(ValueError, match="period must be an integer greater than or equal to 4"):
        sample_ohlcv.select(hurst_exponent("close", period=3))

    # Flat line test (zero variance edge case)
    flat_df = pl.DataFrame({"close": [10.0] * 10})
    flat_res = flat_df.select(hurst_exponent("close", period=5).alias("h"))
    assert flat_res["h"][-1] == 0.5
