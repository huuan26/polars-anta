from __future__ import annotations

import polars as pl
import pytest

from polars_anta.alpha import (
    alpha_001,
    alpha_041,
    alpha_054,
    alpha_101,
    co_ratio,
    cs_rank,
    decay_linear,
    entropy,
    fractal_dimension,
    hl_ratio,
    hurst_exponent,
    neutralize,
    relative_spread,
    returns_divergence,
    signed_power,
    ts_argmax,
    ts_argmin,
    ts_correlation,
    ts_covariance,
    ts_delay,
    ts_delta,
    ts_max,
    ts_min,
    ts_product,
    ts_rank,
    ts_scale,
    ts_std,
    ts_sum,
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


def test_cs_rank():
    """Test cross-sectional ranking with and without group_by."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "ticker": ["AAPL", "MSFT", "GOOG"] * 2,
            "score": [10.0, 30.0, 20.0, 50.0, 40.0, 60.0],
        }
    )

    # Cross-sectional over date
    res = df.with_columns(cs_rank("score", over="date").alias("rank"))
    # In group 1: AAPL(10) -> 1/3, GOOG(20) -> 2/3, MSFT(30) -> 3/3 = 1.0
    group1 = res.filter(pl.col("date") == "2024-01-01")
    assert pytest.approx(group1.filter(pl.col("ticker") == "AAPL")["rank"][0]) == 1 / 3
    assert pytest.approx(group1.filter(pl.col("ticker") == "GOOG")["rank"][0]) == 2 / 3
    assert pytest.approx(group1.filter(pl.col("ticker") == "MSFT")["rank"][0]) == 1.0

    # Non-percentile integer ranks
    res_int = df.with_columns(cs_rank("score", over="date", percentile=False).alias("rank_int"))
    assert res_int["rank_int"].to_list() == [1.0, 3.0, 2.0, 2.0, 1.0, 3.0]

    # Without over (global rank)
    res_global = df.with_columns(cs_rank("score").alias("global_rank"))
    assert res_global["global_rank"].max() == 1.0

    # Without over and non-percentile
    res_glob_int = df.with_columns(cs_rank("score", percentile=False).alias("glob_int"))
    assert res_glob_int["glob_int"].max() == 6.0

    # Via .anta and .ta expression namespaces
    res_anta = df.with_columns(pl.col("score").anta.cs_rank(over="date").alias("r_anta"))
    res_ta = df.with_columns(pl.col("score").ta.cs_rank(over="date").alias("r_ta"))
    assert res_anta["r_anta"].equals(res_ta["r_ta"])


def test_neutralize():
    """Test industry / group neutralization."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 4,
            "sector": ["Tech", "Tech", "Fin", "Fin"],
            "subsector": ["A", "B", "A", "B"],
            "signal": [10.0, 20.0, 100.0, 200.0],
        }
    )
    # Demeaning
    res = df.with_columns(neutralize("signal", by="sector").alias("neut"))
    # Tech mean=15 -> [-5, 5]; Fin mean=150 -> [-50, 50]
    assert res["neut"].to_list() == [-5.0, 5.0, -50.0, 50.0]

    # Neutralization with over and sequence of by columns
    res_multi = df.with_columns(
        neutralize("signal", by=["sector", "subsector"], over="date").alias("neut_multi")
    )
    assert len(res_multi) == 4

    # Neutralization with sequence of over columns
    df_over_seq = df.with_columns(pl.lit("US").alias("country"))
    res_over_seq = df_over_seq.with_columns(
        neutralize("signal", by="sector", over=["date", "country"]).alias("neut_seq")
    )
    assert len(res_over_seq) == 4

    # Standardize (z-score within group)
    res_std = df.with_columns(neutralize("signal", by="sector", standardize=True).alias("neut_z"))
    assert pytest.approx(res_std["neut_z"][0]) == -0.70710678
    assert pytest.approx(res_std["neut_z"][1]) == 0.70710678

    # Zero variance group returns 0.0
    flat_df = pl.DataFrame({"sector": ["A", "A"], "signal": [5.0, 5.0]})
    res_flat = flat_df.with_columns(neutralize("signal", by="sector", standardize=True).alias("nz"))
    assert res_flat["nz"].to_list() == [0.0, 0.0]

    # Via .anta and .ta expression namespaces
    res_anta = df.with_columns(pl.col("signal").anta.neutralize(by="sector").alias("s_anta"))
    res_ta = df.with_columns(pl.col("signal").ta.neutralize(by="sector").alias("s_ta"))
    assert res_anta["s_anta"].equals(res_ta["s_ta"])


def test_ts_scale():
    """Test scaling such that sum of abs values equals a."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "val": [-2.0, 3.0, -5.0, 10.0, -10.0, 20.0],
        }
    )
    res = df.with_columns(ts_scale("val", a=1.0, over="date").alias("scaled"))
    # Sum of absolute values in each date group must equal 1.0
    d1 = res.filter(pl.col("date") == "2024-01-01")
    d2 = res.filter(pl.col("date") == "2024-01-02")
    assert pytest.approx(d1["scaled"].abs().sum()) == 1.0
    assert pytest.approx(d2["scaled"].abs().sum()) == 1.0

    # Zero sum edge case
    df_zero = pl.DataFrame({"val": [0.0, 0.0]})
    res_zero = df_zero.with_columns(ts_scale("val", a=1.0).alias("sc"))
    assert res_zero["sc"].to_list() == [0.0, 0.0]

    # Global scale without over
    res_glob = df.with_columns(ts_scale("val", a=2.0).alias("sc_glob"))
    assert pytest.approx(res_glob["sc_glob"].abs().sum()) == 2.0


def test_ts_delta_and_delay(sample_ohlcv: pl.DataFrame):
    """Test ts_delta and ts_delay."""
    res = sample_ohlcv.select(
        [
            ts_delta("close", period=1).alias("delta_1"),
            ts_delta("close", period=2).alias("delta_2"),
            ts_delay("close", period=1).alias("delay_1"),
            ts_delay("close", period=2).alias("delay_2"),
        ]
    )
    assert res["delay_1"][1] == sample_ohlcv["close"][0]
    assert res["delay_2"][2] == sample_ohlcv["close"][0]
    assert res["delta_1"][1] == pytest.approx(sample_ohlcv["close"][1] - sample_ohlcv["close"][0])
    assert res["delta_2"][2] == pytest.approx(sample_ohlcv["close"][2] - sample_ohlcv["close"][0])


def test_signed_power():
    """Test signed power transformation."""
    df = pl.DataFrame({"x": [-2.0, 3.0, 0.0]})
    res = df.select(
        [
            signed_power("x", exponent=2.0).alias("p2"),
            signed_power("x", exponent=0.5).alias("p05"),
        ]
    )
    # -2^2 = -4.0 (preserves sign)
    assert res["p2"].to_list() == [-4.0, 9.0, 0.0]
    assert res["p05"][0] < 0.0
    assert res["p05"][1] > 0.0


def test_ts_extrema_and_moments(sample_ohlcv: pl.DataFrame):
    """Test ts_min, ts_max, ts_sum, ts_std, ts_correlation, ts_covariance, ts_product."""
    res = sample_ohlcv.select(
        [
            ts_min("close", period=3).alias("min3"),
            ts_max("close", period=3).alias("max3"),
            ts_sum("close", period=3).alias("sum3"),
            ts_std("close", period=3).alias("std3"),
            ts_correlation("close", "volume", period=5).alias("corr5"),
            ts_covariance("close", "volume", period=5).alias("cov5"),
            ts_product("close", period=3).alias("prod3"),
        ]
    )
    assert res["min3"][2] == min(sample_ohlcv["close"][:3])
    assert res["max3"][2] == max(sample_ohlcv["close"][:3])
    assert pytest.approx(res["sum3"][2]) == sum(sample_ohlcv["close"][:3])
    assert res["std3"][2] > 0.0
    assert res["corr5"][-1] is not None
    assert res["cov5"][-1] is not None
    assert pytest.approx(res["prod3"][2]) == (
        sample_ohlcv["close"][0] * sample_ohlcv["close"][1] * sample_ohlcv["close"][2]
    )

    # Test ts_product and ts_rank with embedded nulls
    df_nulls = pl.DataFrame({"val": [1.0, None, 3.0, 4.0, 5.0]})
    res_nulls = df_nulls.select(
        [
            ts_product("val", period=3).alias("prod"),
            ts_rank("val", period=3).alias("rank"),
            ts_argmin("val", period=3).alias("argmin"),
        ]
    )
    assert res_nulls["prod"][1] is None
    assert res_nulls["prod"][2] is None
    assert res_nulls["prod"][4] == pytest.approx(3.0 * 4.0 * 5.0)


def test_worldquant_formulaic_alphas(sample_ohlcv: pl.DataFrame):
    """Test canonical WorldQuant formulaic alphas (Alpha 1, 41, 54, 101)."""
    res = sample_ohlcv.select(
        [
            alpha_001("close").alias("a001"),
            alpha_041("high", "low", "close", "volume").alias("a041"),
            alpha_054("open", "high", "low", "close").alias("a054"),
            alpha_101("open", "high", "low", "close").alias("a101"),
        ]
    )
    assert "a001" in res.columns
    assert "a041" in res.columns
    assert "a054" in res.columns
    assert "a101" in res.columns
    assert res["a101"][-1] is not None
    assert res["a041"][-1] is not None
    assert res["a054"][-1] is not None
