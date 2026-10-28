from __future__ import annotations

import polars as pl
import pytest

from polars_anta.alpha import (
    alpha_001,
    alpha_002,
    alpha_003,
    alpha_004,
    alpha_005,
    alpha_006,
    alpha_007,
    alpha_008,
    alpha_009,
    alpha_010,
    alpha_011,
    alpha_012,
    alpha_013,
    alpha_014,
    alpha_015,
    alpha_016,
    alpha_017,
    alpha_018,
    alpha_019,
    alpha_020,
    alpha_021,
    alpha_022,
    alpha_023,
    alpha_024,
    alpha_025,
    alpha_026,
    alpha_027,
    alpha_028,
    alpha_029,
    alpha_030,
    alpha_041,
    alpha_054,
    alpha_101,
    amihud_illiquidity,
    co_ratio,
    corwin_schultz_spread,
    cs_rank,
    cs_winsorize,
    cs_zscore,
    decay_linear,
    entropy,
    fractal_dimension,
    hl_ratio,
    hurst_exponent,
    kyle_lambda,
    neutralize,
    relative_spread,
    returns_divergence,
    roll_spread,
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


def test_cross_sectional_preprocessing():
    """Test cs_zscore and cs_winsorize across quantile, zscore, and mad methods."""
    df = pl.DataFrame(
        {
            "date": ["2026-01-01"] * 6 + ["2026-01-02"] * 6,
            "signal": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    res = df.with_columns(
        [
            cs_zscore("signal", over="date").alias("z"),
            cs_zscore("signal", over="date", clip_bound=1.5).alias("z_clipped"),
            cs_zscore("signal").alias("z_global"),
            cs_winsorize("signal", over="date", method="quantile", lower=0.1, upper=0.8).alias(
                "w_q"
            ),
            cs_winsorize("signal", over="date", method="zscore", threshold=1.5).alias("w_z"),
            cs_winsorize("signal", over="date", method="mad", threshold=2.0).alias("w_mad"),
            cs_winsorize("signal", method="mad", threshold=2.0).alias("w_mad_global"),
        ]
    )

    # Day 1 mean of z-score is 0
    day1 = res.filter(pl.col("date") == "2026-01-01")
    assert day1["z"].mean() == pytest.approx(0.0, abs=1e-9)
    assert day1["z_clipped"].max() <= 1.5
    assert day1["z_clipped"].min() >= -1.5

    # Day 2 has constant signal -> cs_zscore is 0.0
    day2 = res.filter(pl.col("date") == "2026-01-02")
    assert (day2["z"] == 0.0).all()

    # Winsorize clips the extreme outlier 100.0 down
    assert day1["w_q"][-1] < 100.0
    assert day1["w_z"][-1] < 100.0
    assert day1["w_mad"][-1] < 100.0

    # Namespace parity
    expr_df = df.select(
        [
            pl.col("signal").anta.cs_zscore(over="date", clip_bound=1.5).alias("z_clipped"),
            pl.col("signal")
            .ta.cs_winsorize(over="date", method="mad", threshold=2.0)
            .alias("w_mad"),
        ]
    )
    frame_df = df.anta.cs_zscore(
        col="signal", over="date", clip_bound=1.5, alias="z_clipped"
    ).ta.cs_winsorize(col="signal", over="date", method="mad", threshold=2.0, alias="w_mad")
    assert expr_df["z_clipped"].equals(frame_df["z_clipped"])
    assert expr_df["w_mad"].equals(frame_df["w_mad"])

    # Invalid arguments
    with pytest.raises(ValueError, match="clip_bound must be a positive"):
        cs_zscore("signal", clip_bound=0.0)
    with pytest.raises(ValueError, match="method must be one of"):
        cs_winsorize("signal", method="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lower and upper quantiles"):
        cs_winsorize("signal", method="quantile", lower=0.8, upper=0.2)
    with pytest.raises(ValueError, match="threshold must be a positive"):
        cs_winsorize("signal", method="zscore", threshold=-1.0)


def test_market_microstructure_indicators(sample_ohlcv: pl.DataFrame):
    """Test Amihud illiquidity, Roll spread, Corwin-Schultz spread, and Kyle's lambda."""
    res = sample_ohlcv.select(
        [
            amihud_illiquidity("close", "volume", period=5).alias("amihud"),
            roll_spread("close", period=5).alias("roll"),
            corwin_schultz_spread("high", "low", period=5).alias("cs_spread"),
            kyle_lambda("close", "volume", period=5).alias("kyle"),
        ]
    )

    for col in ("amihud", "roll", "cs_spread"):
        vals = [v for v in res[col].to_list() if v is not None]
        assert len(vals) > 0
        for v in vals:
            assert v >= 0.0

    kyle_vals = [v for v in res["kyle"].to_list() if v is not None]
    assert len(kyle_vals) > 0

    # Namespace tests on Expr and DataFrame
    ns_expr = sample_ohlcv.select(
        [
            pl.col("close").anta.amihud_illiquidity("volume", period=5).alias("amihud_5"),
            pl.col("close").ta.roll_spread(period=5).alias("roll_spread_5"),
            pl.col("high").anta.corwin_schultz_spread("low", period=5).alias("cs_spread_5"),
            pl.col("close").ta.kyle_lambda("volume", period=5).alias("kyle_lambda_5"),
        ]
    )
    ns_df = (
        sample_ohlcv.anta.amihud_illiquidity(period=5)
        .ta.roll_spread(period=5)
        .anta.corwin_schultz_spread(period=5)
        .ta.kyle_lambda(period=5)
    )
    for col in ("amihud_5", "roll_spread_5", "cs_spread_5", "kyle_lambda_5"):
        assert col in ns_df.columns
        assert ns_expr[col].equals(ns_df[col])

    # Validation tests
    with pytest.raises(ValueError, match="period must be a positive integer"):
        amihud_illiquidity("close", "volume", period=0)
    with pytest.raises(ValueError, match="period must be an integer greater than 1"):
        roll_spread("close", period=1)
    with pytest.raises(ValueError, match="period must be a positive integer"):
        corwin_schultz_spread("high", "low", period=0)
    with pytest.raises(ValueError, match="period must be an integer greater than 1"):
        kyle_lambda("close", "volume", period=1)


def test_worldquant_alphas_002_to_030():
    """Test WorldQuant formulaic alphas #2 through #30 across function, Expr, and Frame APIs."""
    n = 35
    df = pl.DataFrame(
        {
            "open": [100.0 + i * 0.5 + (i % 3) * 0.2 for i in range(n)],
            "high": [101.0 + i * 0.55 + (i % 4) * 0.3 for i in range(n)],
            "low": [99.0 + i * 0.45 - (i % 3) * 0.2 for i in range(n)],
            "close": [100.2 + i * 0.52 + ((-1) ** i) * 0.4 for i in range(n)],
            "volume": [1000.0 + (i * 37 % 500) + ((-1) ** i) * 120.0 for i in range(n)],
        }
    )

    # 1. Direct functional expressions
    func_df = df.select(
        [
            alpha_002("open", "close", "volume").alias("alpha_002"),
            alpha_003("open", "volume").alias("alpha_003"),
            alpha_004("low").alias("alpha_004"),
            alpha_005("open", "high", "low", "close", "volume").alias("alpha_005"),
            alpha_006("open", "volume").alias("alpha_006"),
            alpha_007("close", "volume", adv_period=5, rank_period=10).alias("alpha_007"),
            alpha_008("open", "close").alias("alpha_008"),
            alpha_009("close").alias("alpha_009"),
            alpha_010("close").alias("alpha_010"),
            alpha_011("high", "low", "close", "volume").alias("alpha_011"),
            alpha_012("close", "volume").alias("alpha_012"),
            alpha_013("close", "volume").alias("alpha_013"),
            alpha_014("open", "close", "volume").alias("alpha_014"),
            alpha_015("high", "volume").alias("alpha_015"),
            alpha_016("high", "volume").alias("alpha_016"),
            alpha_017("close", "volume", adv_period=5).alias("alpha_017"),
            alpha_018("open", "close").alias("alpha_018"),
            alpha_019("close", sum_period=10).alias("alpha_019"),
            alpha_020("open", "high", "low", "close").alias("alpha_020"),
            alpha_021("close", "volume", adv_period=5).alias("alpha_021"),
            alpha_022("high", "close", "volume").alias("alpha_022"),
            alpha_023("high", period=10).alias("alpha_023"),
            alpha_024("close", period=10).alias("alpha_024"),
            alpha_025("high", "low", "close", "volume", adv_period=5).alias("alpha_025"),
            alpha_026("high", "volume").alias("alpha_026"),
            alpha_027("high", "low", "close", "volume").alias("alpha_027"),
            alpha_028("high", "low", "close", "volume", adv_period=5).alias("alpha_028"),
            alpha_029("close").alias("alpha_029"),
            alpha_030("close", "volume").alias("alpha_030"),
        ]
    )

    for idx in range(2, 31):
        col = f"alpha_{idx:03d}"
        assert col in func_df.columns
        non_null = func_df[col].drop_nulls()
        assert len(non_null) > 0, f"{col} produced all nulls"

    # 2. Expression namespace (.anta / .ta)
    expr_df = df.select(
        [
            pl.col("close").anta.alpha_002("open", "volume").alias("alpha_002"),
            pl.col("open").ta.alpha_003("volume").alias("alpha_003"),
            pl.col("low").anta.alpha_004().alias("alpha_004"),
            pl.col("close").ta.alpha_005("open", "high", "low", "volume").alias("alpha_005"),
            pl.col("open").anta.alpha_006("volume").alias("alpha_006"),
            pl.col("close").ta.alpha_007("volume", adv_period=5, rank_period=10).alias("alpha_007"),
            pl.col("close").anta.alpha_008("open").alias("alpha_008"),
            pl.col("close").ta.alpha_009().alias("alpha_009"),
            pl.col("close").anta.alpha_010().alias("alpha_010"),
            pl.col("close").ta.alpha_011("high", "low", "volume").alias("alpha_011"),
            pl.col("close").anta.alpha_012("volume").alias("alpha_012"),
            pl.col("close").ta.alpha_013("volume").alias("alpha_013"),
            pl.col("close").anta.alpha_014("open", "volume").alias("alpha_014"),
            pl.col("high").ta.alpha_015("volume").alias("alpha_015"),
            pl.col("high").anta.alpha_016("volume").alias("alpha_016"),
            pl.col("close").ta.alpha_017("volume", adv_period=5).alias("alpha_017"),
            pl.col("close").anta.alpha_018("open").alias("alpha_018"),
            pl.col("close").ta.alpha_019(sum_period=10).alias("alpha_019"),
            pl.col("close").anta.alpha_020("open", "high", "low").alias("alpha_020"),
            pl.col("close").ta.alpha_021("volume", adv_period=5).alias("alpha_021"),
            pl.col("close").anta.alpha_022("high", "volume").alias("alpha_022"),
            pl.col("high").ta.alpha_023(period=10).alias("alpha_023"),
            pl.col("close").anta.alpha_024(period=10).alias("alpha_024"),
            pl.col("close").ta.alpha_025("high", "low", "volume", adv_period=5).alias("alpha_025"),
            pl.col("high").anta.alpha_026("volume").alias("alpha_026"),
            pl.col("close").ta.alpha_027("high", "low", "volume").alias("alpha_027"),
            pl.col("close")
            .anta.alpha_028("high", "low", "volume", adv_period=5)
            .alias("alpha_028"),
            pl.col("close").ta.alpha_029().alias("alpha_029"),
            pl.col("close").anta.alpha_030("volume").alias("alpha_030"),
        ]
    )

    # 3. DataFrame namespace (df.anta / df.ta)
    frame_df = (
        df.anta.alpha_002()
        .ta.alpha_003()
        .anta.alpha_004()
        .ta.alpha_005()
        .anta.alpha_006()
        .ta.alpha_007(adv_period=5, rank_period=10)
        .anta.alpha_008()
        .ta.alpha_009()
        .anta.alpha_010()
        .ta.alpha_011()
        .anta.alpha_012()
        .ta.alpha_013()
        .anta.alpha_014()
        .ta.alpha_015()
        .anta.alpha_016()
        .ta.alpha_017(adv_period=5)
        .anta.alpha_018()
        .ta.alpha_019(sum_period=10)
        .anta.alpha_020()
        .ta.alpha_021(adv_period=5)
        .anta.alpha_022()
        .ta.alpha_023(period=10)
        .anta.alpha_024(period=10)
        .ta.alpha_025(adv_period=5)
        .anta.alpha_026()
        .ta.alpha_027()
        .anta.alpha_028(adv_period=5)
        .ta.alpha_029()
        .anta.alpha_030()
    )

    for idx in range(2, 31):
        col = f"alpha_{idx:03d}"
        assert func_df[col].equals(expr_df[col])
        assert func_df[col].equals(frame_df[col])
