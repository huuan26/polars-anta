from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polars_anta.stats import (
    frac_diff,
    kurtosis,
    linear_regression_angle,
    linear_regression_forecast,
    linear_regression_intercept,
    linear_regression_r2,
    linear_regression_slope,
    linear_regression_stderr,
    ou_half_life,
    percentile_rank,
    rolling_alpha,
    rolling_beta,
    rolling_corr,
    rolling_cov,
    rolling_hedge_ratio,
    skewness,
    zscore,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "asset_ret": [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.01, -0.01, 0.02, 0.01],
            "bench_ret": [0.01, 0.01, -0.005, 0.02, 0.01, -0.01, 0.02, 0.005, -0.01, 0.01, 0.005],
        }
    )


def test_zscore(sample_data: pl.DataFrame):
    res = sample_data.select(zscore("close", period=5).alias("z")).get_column("z")
    assert len(res) == 11
    # For a strictly increasing series, the last element is greater than the rolling mean -> positive zscore
    assert res[-1] > 0.0


def test_percentile_rank(sample_data: pl.DataFrame):
    res = sample_data.select(percentile_rank("close", period=5).alias("pr")).get_column("pr")
    assert len(res) == 11
    # For strictly increasing sequence, current value is always highest in window -> 1.0
    assert res[-1] == 1.0


def test_skewness_kurtosis(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            skewness("close", period=5).alias("skew"),
            kurtosis("close", period=5).alias("kurt"),
        ]
    )
    assert len(res) == 11
    # Uniform linear progression has ~0 skewness
    assert abs(res["skew"][-1]) < 0.1
    assert res["kurt"][-1] is not None


def test_rolling_cov_corr(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            rolling_cov("asset_ret", "bench_ret", period=5).alias("cov"),
            rolling_corr("asset_ret", "bench_ret", period=5).alias("corr"),
        ]
    )
    assert len(res) == 11
    corr_val = res["corr"][-1]
    assert corr_val is not None
    assert -1.0 <= corr_val <= 1.0


def test_rolling_beta_alpha(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            rolling_beta("asset_ret", "bench_ret", period=5).alias("beta"),
            rolling_alpha("asset_ret", "bench_ret", rf=0.0, period=5).alias("alpha"),
        ]
    )
    assert len(res) == 11
    assert res["beta"][-1] is not None
    assert res["alpha"][-1] is not None


def test_linear_regression(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            linear_regression_slope("close", period=5).alias("slope"),
            linear_regression_intercept("close", period=5).alias("intercept"),
            linear_regression_angle("close", period=5).alias("angle"),
            linear_regression_forecast("close", period=5).alias("forecast"),
        ]
    )
    assert len(res) == 11
    # For series [10, 11, 12, ...], step is 1.0 so slope must be 1.0
    assert pytest.approx(res["slope"][-1], rel=1e-5) == 1.0
    # Current bar value is 20.0, forecast at current bar should be 20.0
    assert pytest.approx(res["forecast"][-1], rel=1e-5) == 20.0
    # Angle for slope 1.0 is 45 degrees
    assert pytest.approx(res["angle"][-1], rel=1e-5) == 45.0


def test_frac_diff(sample_data: pl.DataFrame) -> None:
    # d = 0.0 -> identity after warm-up window (w_0 = 1, w_k = 0 for k >= 1)
    res_d0 = sample_data.select(frac_diff("close", d=0.0, window=4).alias("fd0"))["fd0"]
    assert res_d0[:3].null_count() == 3
    assert res_d0[3:].to_list() == pytest.approx(sample_data["close"][3:].to_list())

    # d = 1.0 -> exact first difference (w_0 = 1, w_1 = -1, w_k = 0 for k >= 2)
    res_d1 = sample_data.select(frac_diff("close", d=1.0, window=4).alias("fd1"))["fd1"]
    assert res_d1[:3].null_count() == 3
    for val in res_d1[3:].to_list():
        assert val == pytest.approx(1.0)

    # d = 0.4 -> fractional differentiation with manual binomial weight verification
    # w_0 = 1.0, w_1 = -0.4, w_2 = -(-0.4) * (0.4 - 1) / 2 = -0.12, w_3 = -(-0.12) * (0.4 - 2) / 3 = -0.064
    res_d04 = sample_data.select(pl.col("close").anta.frac_diff(d=0.4, window=4).alias("fd04"))[
        "fd04"
    ]
    expected_idx3 = 1.0 * 13.0 - 0.4 * 12.0 - 0.12 * 11.0 - 0.064 * 10.0
    assert res_d04[3] == pytest.approx(expected_idx3, rel=1e-6)

    # DataFrame namespace (.anta / .ta) and multi-asset .over("ticker") support
    panel = pl.DataFrame(
        {
            "ticker": ["VN30F1M"] * 6 + ["VN30F2M"] * 6,
            "close": [1250.0, 1252.0, 1255.0, 1258.0, 1260.0, 1263.0]
            + [1260.0, 1261.0, 1262.0, 1263.0, 1264.0, 1265.0],
        }
    )
    out = panel.ta.frac_diff(d=0.4, window=3, group_by="ticker", alias="close_ffd")
    assert "close_ffd" in out.columns
    assert out.filter(pl.col("ticker") == "VN30F1M")["close_ffd"][:2].null_count() == 2
    assert out.filter(pl.col("ticker") == "VN30F2M")["close_ffd"][:2].null_count() == 2
    assert out.filter(pl.col("ticker") == "VN30F2M")["close_ffd"][2] is not None

    # Short series edge case (n < window)
    short_df = pl.DataFrame({"close": [100.0, 101.0]})
    short_res = short_df.select(frac_diff("close", d=0.4, window=5).alias("fd"))["fd"]
    assert short_res.null_count() == 2

    # Validation checks
    with pytest.raises(ValueError, match="d must be a non-negative"):
        frac_diff("close", d=-0.1, window=5)
    with pytest.raises(ValueError, match="window must be an integer greater than or equal to 2"):
        frac_diff("close", d=0.4, window=1)


def _linear_series(n: int = 30, slope: float = 2.0, intercept: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame({"close": [intercept + slope * i for i in range(n)]})


def _noisy_series(n: int = 40) -> pl.DataFrame:
    values = [100.0 + (i % 7) * 1.5 - (i % 3) * 2.0 + 0.1 * i for i in range(n)]
    return pl.DataFrame({"close": values})


def test_linear_regression_r2_perfect_linear_is_one():
    period = 10
    values = (
        _linear_series()
        .select(linear_regression_r2("close", period=period).alias("r2"))["r2"]
        .to_list()
    )

    assert values[: period - 1] == [0.0] * (period - 1)
    tail = values[period - 1 :]
    assert tail
    assert all(v == pytest.approx(1.0, abs=1e-9) for v in tail)


def test_linear_regression_r2_flat_series_is_zero():
    df = pl.DataFrame({"close": [50.0] * 20})
    vals = df.select(linear_regression_r2("close", period=5).alias("r2"))["r2"].to_list()

    assert all(v == 0.0 for v in vals)


def test_linear_regression_r2_stays_in_unit_interval():
    period = 10
    vals = _noisy_series().select(linear_regression_r2("close", period=period).alias("r2"))["r2"]
    tail = [v for v in vals.to_list() if v is not None]

    assert tail
    assert all(0.0 <= v <= 1.0 for v in tail)


def test_linear_regression_r2_rejects_invalid_period():
    for bad in (1, 0, -5):
        with pytest.raises(ValueError):
            linear_regression_r2("close", period=bad)


def test_linear_regression_stderr_perfect_linear_is_zero():
    period = 10
    vals = (
        _linear_series()
        .select(linear_regression_stderr("close", period=period).alias("se"))["se"]
        .to_list()
    )

    assert all(v == pytest.approx(0.0, abs=1e-9) for v in vals)


def test_linear_regression_stderr_flat_series_is_zero():
    df = pl.DataFrame({"close": [7.0] * 20})
    vals = df.select(linear_regression_stderr("close", period=6).alias("se"))["se"].to_list()

    assert all(v == 0.0 for v in vals)


def test_linear_regression_stderr_positive_for_noisy_series():
    period = 10
    vals = (
        _noisy_series()
        .select(linear_regression_stderr("close", period=period).alias("se"))["se"]
        .to_list()
    )
    tail = [v for v in vals if v is not None]

    assert tail
    assert all(v >= 0.0 for v in tail)
    assert max(tail) > 0.0


def test_linear_regression_stderr_rejects_invalid_period():
    for bad in (2, 1, 0):
        with pytest.raises(ValueError):
            linear_regression_stderr("close", period=bad)


def _numpy_skew(values, bias: bool) -> float:
    import math

    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    n = x.size
    dev = x - x.mean()
    m2 = float(np.mean(dev**2))
    m3 = float(np.mean(dev**3))
    g1 = m3 / m2**1.5
    if bias:
        return g1
    return math.sqrt(n * (n - 1.0)) / (n - 2.0) * g1


def _numpy_excess_kurtosis(values, bias: bool) -> float:
    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    n = x.size
    dev = x - x.mean()
    m2 = float(np.mean(dev**2))
    m4 = float(np.mean(dev**4))
    g2 = m4 / m2**2 - 3.0
    if bias:
        return g2
    return (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0)


BIAS_WINDOW = [1.0, 2.0, 3.0, 4.0, 100.0]


@pytest.mark.parametrize("bias", [True, False])
def test_skewness_matches_numpy(bias: bool):
    df = pl.DataFrame({"x": BIAS_WINDOW})
    got = df.select(skewness("x", period=5, bias=bias).alias("s"))["s"][-1]
    expected = _numpy_skew(BIAS_WINDOW, bias)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("bias", [True, False])
def test_kurtosis_matches_numpy(bias: bool):
    df = pl.DataFrame({"x": BIAS_WINDOW})
    got = df.select(kurtosis("x", period=5, bias=bias).alias("k"))["k"][-1]
    expected = _numpy_excess_kurtosis(BIAS_WINDOW, bias)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_skewness_default_is_adjusted_sample():
    df = pl.DataFrame({"x": BIAS_WINDOW})
    default = df.select(skewness("x", period=5).alias("s"))["s"][-1]
    sample = df.select(skewness("x", period=5, bias=False).alias("s"))["s"][-1]
    population = df.select(skewness("x", period=5, bias=True).alias("s"))["s"][-1]
    assert default == pytest.approx(sample)
    assert default != pytest.approx(population)


def test_kurtosis_default_is_adjusted_sample():
    df = pl.DataFrame({"x": BIAS_WINDOW})
    default = df.select(kurtosis("x", period=5).alias("k"))["k"][-1]
    sample = df.select(kurtosis("x", period=5, bias=False).alias("k"))["k"][-1]
    population = df.select(kurtosis("x", period=5, bias=True).alias("k"))["k"][-1]
    assert default == pytest.approx(sample)
    assert default != pytest.approx(population)


def test_bias_validation_period_too_small():
    df = pl.DataFrame({"x": BIAS_WINDOW})
    with pytest.raises(ValueError, match="period >= 3"):
        skewness("x", period=2)
    with pytest.raises(ValueError, match="period >= 4"):
        kurtosis("x", period=3)
    # The biased population estimators are still defined for two observations.
    df.select(skewness("x", period=2, bias=True).alias("s"))
    df.select(kurtosis("x", period=2, bias=True).alias("k"))


def test_namespace_forwards_bias():
    df = pl.DataFrame({"x": BIAS_WINDOW})
    expr = pl.col("x").anta.skewness(period=5, bias=True)
    got = df.select(expr.alias("s"))["s"][-1]
    expected = _numpy_skew(BIAS_WINDOW, True)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


def _ar1_series(n: int = 4000, phi: float = 0.9, seed: int = 7) -> np.ndarray:
    """Deterministic AR(1) path with a known mean-reversion half-life."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def test_ou_half_life_matches_full_sample_ols():
    """Rolling half-life must agree with the closed-form OLS estimate on the whole path."""
    x = _ar1_series(n=20000, phi=0.9, seed=7)
    df = pl.DataFrame({"spread": x})
    hl = df.select(ou_half_life(pl.col("spread"), period=250).alias("hl"))["hl"].drop_nulls()

    lag = x[:-1]
    delta = np.diff(x)
    beta, _ = np.polyfit(lag, delta, 1)
    expected = -math.log(2.0) / beta

    assert hl.len() > 1000
    assert abs(hl.mean() - expected) < 0.8
    # and it sits in the right neighbourhood of the textbook value
    assert abs(hl.mean() - 6.93) < 0.9


def test_ou_half_life_discrete_versus_continuous():
    """The exact AR(1) form is shorter than the continuous form for the same beta."""
    x = _ar1_series(n=20000, phi=0.9, seed=21)
    df = pl.DataFrame({"spread": x})
    cont = df.select(ou_half_life(pl.col("spread"), period=250).alias("h"))["h"].drop_nulls()
    disc = df.select(ou_half_life(pl.col("spread"), period=250, method="discrete").alias("h"))[
        "h"
    ].drop_nulls()

    lag = x[:-1]
    delta = np.diff(x)
    beta, _ = np.polyfit(lag, delta, 1)
    expected_disc = -math.log(2.0) / math.log(1.0 + beta)

    assert abs(disc.mean() - expected_disc) < 0.8
    assert disc.mean() < cont.mean()


def test_ou_half_life_null_on_exploding_series():
    """A diverging spread has beta > 0 in every window: no half-life, all nulls."""
    n = 3000
    x = np.empty(n)
    x[0] = 1.0
    rng = np.random.default_rng(5)
    eps = rng.normal(0.0, 0.01, n)
    for i in range(1, n):
        x[i] = 1.02 * x[i - 1] + eps[i]
    df = pl.DataFrame({"spread": x})
    hl = df.select(ou_half_life(pl.col("spread"), period=250).alias("hl"))["hl"]
    assert hl.is_null().all()


def test_ou_half_life_null_on_straight_trend():
    """A deterministic trend has zero first-difference variance: the slope is 0, so null."""
    trend = np.arange(3000, dtype=np.float64)
    df = pl.DataFrame({"spread": trend})
    hl = df.select(ou_half_life(pl.col("spread"), period=250).alias("hl"))["hl"]
    assert hl.is_null().all()


def test_ou_half_life_rejects_bad_method():
    """Unknown conversion methods fail loudly at expression-build time."""
    with pytest.raises(ValueError, match="method must be"):
        ou_half_life(pl.col("close"), period=20, method="bogus")


def test_rolling_hedge_ratio_recovers_slope():
    """A synthetic pair with beta = 2 must come back as ~2 on average."""
    rng = np.random.default_rng(11)
    b = np.cumsum(rng.normal(0.0, 1.0, 3000))
    a = 2.0 * b + rng.normal(0.0, 0.5, 3000)
    df = pl.DataFrame({"a": a, "b": b})
    beta = df.select(rolling_hedge_ratio(pl.col("a"), pl.col("b"), period=250).alias("beta"))[
        "beta"
    ].drop_nulls()
    assert beta.len() > 100
    assert abs(beta.mean() - 2.0) < 0.10


def test_ou_namespace_binding():
    """Both .anta and .ta expression namespaces match top-level functions."""
    import polars_anta as pa

    df = pl.DataFrame(
        {
            "spread": _ar1_series(n=500, seed=1),
            "a": 100.0 + _ar1_series(n=500, seed=2),
            "b": 100.0 + _ar1_series(n=500, seed=3),
        }
    )
    direct_hl = df.select(pa.ou_half_life(pl.col("spread"), period=60).alias("v"))["v"]
    assert direct_hl.equals(
        df.select(pl.col("spread").anta.ou_half_life(period=60).alias("v"))["v"]
    )
    assert direct_hl.equals(df.select(pl.col("spread").ta.ou_half_life(period=60).alias("v"))["v"])

    direct_hr = df.select(pa.rolling_hedge_ratio("a", "b", period=60).alias("v"))["v"]
    assert direct_hr.equals(
        df.select(pl.col("a").anta.rolling_hedge_ratio("b", period=60).alias("v"))["v"]
    )
    assert direct_hr.equals(
        df.select(pl.col("a").ta.rolling_hedge_ratio("b", period=60).alias("v"))["v"]
    )


def test_ou_and_hedge_ratio_validation_and_edge_cases():
    """Rejects period < 2, guards float cancellation on large-offset trends, flat prices, and oscillatory AR(1)."""
    with pytest.raises(ValueError, match="period"):
        ou_half_life("spread", period=1)
    with pytest.raises(ValueError, match="period"):
        rolling_hedge_ratio("a", "b", period=1)

    # Large-offset non-integer linear trend must not emit false half-lives from float cancellation
    offset_trend = 1e6 + 0.1 * np.arange(1500, dtype=np.float64)
    hl_trend = pl.DataFrame({"spread": offset_trend}).select(
        ou_half_life("spread", period=250).alias("hl")
    )["hl"]
    assert hl_trend.is_null().all()

    # Constant benchmark yields clean null (not NaN/inf)
    flat_df = pl.DataFrame({"a": np.linspace(10.0, 20.0, 100), "b": np.full(100, 50.0)})
    hr_flat = flat_df.select(rolling_hedge_ratio("a", "b", period=20).alias("hr"))["hr"]
    assert hr_flat.is_null().all()

    # Oscillatory alternating spread has beta = -2 <= -1 -> null in discrete mode
    osc = np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
    hl_osc = pl.DataFrame({"spread": osc}).select(
        ou_half_life("spread", period=20, method="discrete").alias("hl")
    )["hl"]
    assert hl_osc.is_null().all()

    # Large price scale accuracy (BTC-level offset with exact slope 2.0)
    rng = np.random.default_rng(1)
    dx = np.cumsum(rng.normal(0.0, 0.5, 250))
    btc_df = pl.DataFrame({"a": 90_000.0 + 2.0 * dx, "b": 90_000.0 + dx})
    hr_btc = btc_df.select(rolling_hedge_ratio("a", "b", period=60).alias("hr"))["hr"].drop_nulls()
    assert float((hr_btc - 2.0).abs().max()) < 1e-9


def test_rolling_hedge_ratio_log_prices_and_multi_asset_over():
    """Supports log_prices=True and isolates partitions cleanly under .over('pair')."""
    rng = np.random.default_rng(31)
    b1 = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 200)))
    a1 = (b1**1.5) * np.exp(rng.normal(0.0, 1e-4, 200))
    b2 = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 200)))
    a2 = (b2**0.5) * np.exp(rng.normal(0.0, 1e-4, 200))

    s1 = _ar1_series(n=200, phi=0.85, seed=101)
    s2 = _ar1_series(n=200, phi=0.95, seed=202)

    panel = pl.DataFrame(
        {
            "pair": ["P1"] * 200 + ["P2"] * 200,
            "a": np.concatenate([a1, a2]),
            "b": np.concatenate([b1, b2]),
            "spread": np.concatenate([s1, s2]),
        }
    )
    out = panel.with_columns(
        pl.col("a")
        .anta.rolling_hedge_ratio("b", period=60, log_prices=True)
        .over("pair")
        .alias("hr"),
        pl.col("spread").anta.ou_half_life(period=60).over("pair").alias("hl"),
    )

    p1_hr = out.filter(pl.col("pair") == "P1")["hr"].drop_nulls()
    p2_hr = out.filter(pl.col("pair") == "P2")["hr"].drop_nulls()
    assert abs(float(p1_hr.mean()) - 1.5) < 0.02
    assert abs(float(p2_hr.mean()) - 0.5) < 0.02

    # Multi-asset .over("pair") must match standalone single-pair evaluation bit-for-bit
    standalone_p2 = pl.DataFrame({"spread": s2}).select(
        ou_half_life("spread", period=60).alias("hl")
    )["hl"]
    assert out.filter(pl.col("pair") == "P2")["hl"].equals(standalone_p2)
