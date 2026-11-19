"""Mathematical invariant tests for HRP and Equal Risk Parity (ERC / IVP) allocation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_anta as pa
from polars_anta.covariance import ledoit_wolf_shrinkage, sample_covariance
from polars_anta.portfolio import equal_risk_parity, hierarchical_risk_parity


def _frame(x: np.ndarray, names: list[str]) -> pl.DataFrame:
    data = {n: x[:, i] for i, n in enumerate(names)}
    data["date"] = np.arange(x.shape[0])
    return pl.DataFrame(data)


def test_weights_sum_to_one_and_are_positive():
    """Both allocators return a normalized, strictly positive long-form frame."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, size=(300, 5))
    df = _frame(x, ["a", "b", "c", "d", "e"])

    for fn in (hierarchical_risk_parity, equal_risk_parity):
        w = fn(df, time_col="date")
        assert w.columns == ["asset", "weight"]
        assert w.height == 5
        assert abs(w["weight"].sum() - 1.0) < 1e-12
        assert (w["weight"] > 0).all()


def test_equal_risk_parity_exact_erc_risk_contributions():
    """Default ERC solves w_i (Sigma w)_i == (1/N) w^T Sigma w on correlated assets."""
    rng = np.random.default_rng(42)
    f_common = rng.normal(0, 0.015, size=(600, 1))
    x = np.hstack(
        [
            f_common + rng.normal(0, 0.005, size=(600, 1)),
            f_common + rng.normal(0, 0.005, size=(600, 1)),
            rng.normal(0, 0.010, size=(600, 1)),
            rng.normal(0, 0.030, size=(600, 1)),
        ]
    )
    df = _frame(x, ["c1", "c2", "ind_lo", "ind_hi"])

    for shrinkage in ("sample", "ledoit_wolf"):
        w = equal_risk_parity(df, time_col="date", method="erc", shrinkage=shrinkage)[
            "weight"
        ].to_numpy()
        cov = sample_covariance(x) if shrinkage == "sample" else ledoit_wolf_shrinkage(x)[0]
        rc = w * (cov @ w)
        rc_share = rc / rc.sum()
        np.testing.assert_allclose(rc_share, np.full(4, 0.25), atol=1e-10)

    # Correlated pair (c1, c2) receives less weight under ERC than under diagonal IVP
    w_erc = equal_risk_parity(df, time_col="date", method="erc", shrinkage="sample")[
        "weight"
    ].to_numpy()
    w_ivp = equal_risk_parity(df, time_col="date", method="ivp", shrinkage="sample")[
        "weight"
    ].to_numpy()
    assert w_erc[0] + w_erc[1] < w_ivp[0] + w_ivp[1]


def test_equal_risk_parity_ivp_is_inverse_volatility():
    """IVP mode must equal 1/sigma normalized on a sample covariance."""
    rng = np.random.default_rng(11)
    scales = np.array([1.0, 2.0, 4.0])
    x = rng.normal(0, 0.01, size=(500, 3)) * scales
    df = _frame(x, ["a", "b", "c"])
    w = equal_risk_parity(df, time_col="date", method="ivp", shrinkage="sample")
    vols = x.std(axis=0, ddof=0)
    expected = (1 / vols) / (1 / vols).sum()
    np.testing.assert_allclose(w["weight"].to_numpy(), expected, rtol=1e-10)


def test_two_asset_hrp_is_inverse_variance():
    """With two leaves recursive bisection reduces to inverse-variance weights."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, size=(400, 2)) * np.array([1.0, 3.0])
    df = _frame(x, ["lo", "hi"])
    w = hierarchical_risk_parity(df, time_col="date", shrinkage="sample")
    var = x.var(axis=0)
    expected = (1 / var) / (1 / var).sum()
    np.testing.assert_allclose(w["weight"].to_numpy(), expected, rtol=1e-10)


def test_hrp_balances_two_equal_clusters():
    """Two equally sized, internally correlated clusters share the risk budget evenly."""
    rng = np.random.default_rng(23)
    c1 = rng.normal(0, 0.01, size=(600, 1))
    c2 = rng.normal(0, 0.01, size=(600, 1))
    x = np.hstack(
        [
            c1 + rng.normal(0, 0.001, size=(600, 1)),
            c1 + rng.normal(0, 0.001, size=(600, 1)),
            c2 + rng.normal(0, 0.001, size=(600, 1)),
            c2 + rng.normal(0, 0.001, size=(600, 1)),
        ]
    )
    w = hierarchical_risk_parity(x)
    weights = w["weight"].to_numpy()
    assert abs(weights.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(weights, np.full(4, 0.25), atol=0.05)


def test_covariance_estimators_and_rank_deficient_stability():
    """Supports marchenko_pastur/exponential estimators and stays non-negative when K > T."""
    rng = np.random.default_rng(13)
    x = rng.normal(0, 0.01, size=(200, 6))
    for est in ("ledoit_wolf", "marchenko_pastur", "exponential", "sample"):
        w_hrp = hierarchical_risk_parity(x, shrinkage=est)
        w_erc = equal_risk_parity(x, shrinkage=est)
        assert abs(w_hrp["weight"].sum() - 1.0) < 1e-12
        assert abs(w_erc["weight"].sum() - 1.0) < 1e-12
        assert (w_hrp["weight"] > 0).all()
        assert (w_erc["weight"] > 0).all()

    # Rank-deficient K > T regime (sample covariance has zero/slightly-negative eigenvalues)
    x_rd = rng.normal(0, 0.01, size=(15, 25))
    w_hrp_rd = hierarchical_risk_parity(x_rd, shrinkage="sample")
    w_erc_rd = equal_risk_parity(x_rd, shrinkage="sample")
    assert abs(w_hrp_rd["weight"].sum() - 1.0) < 1e-12
    assert abs(w_erc_rd["weight"].sum() - 1.0) < 1e-12
    assert (w_hrp_rd["weight"] >= 0.0).all()
    assert (w_erc_rd["weight"] > 0.0).all()


def test_single_linkage_and_array_input():
    """Array input works and single linkage returns a valid allocation."""
    rng = np.random.default_rng(13)
    x = rng.normal(0, 0.01, size=(200, 4))
    w1 = hierarchical_risk_parity(x)
    w2 = hierarchical_risk_parity(x, method="single")
    assert w1.height == w2.height == 4
    assert abs(w2["weight"].sum() - 1.0) < 1e-12
    assert w1["asset"].to_list() == ["asset_0", "asset_1", "asset_2", "asset_3"]


def test_namespace_and_top_level_parity():
    """Both .anta and .ta namespaces on DataFrame/LazyFrame match top-level exports."""
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.01, size=(300, 4))
    df = _frame(x, ["w", "x", "y", "z"])
    lf = df.lazy()

    ref_hrp = pa.hierarchical_risk_parity(df, time_col="date")
    assert df.anta.hierarchical_risk_parity(time_col="date").equals(ref_hrp)
    assert df.ta.hierarchical_risk_parity(time_col="date").equals(ref_hrp)
    assert lf.anta.hierarchical_risk_parity(time_col="date").equals(ref_hrp)

    ref_erc = pa.equal_risk_parity(df, time_col="date")
    assert df.anta.equal_risk_parity(time_col="date").equals(ref_erc)
    assert df.ta.equal_risk_parity(time_col="date").equals(ref_erc)
    assert lf.ta.equal_risk_parity(time_col="date").equals(ref_erc)


def test_validation_errors():
    """Degenerate inputs are rejected with actionable messages."""
    rng = np.random.default_rng(19)
    x = rng.normal(0, 0.01, size=(50, 3))
    df = _frame(x, ["a", "b", "c"]).with_columns(pl.lit(0.0).alias("flat"))
    with pytest.raises(ValueError, match="constant return series"):
        hierarchical_risk_parity(df, time_col="date")
    with pytest.raises(ValueError, match="at least 10 complete observations"):
        hierarchical_risk_parity(x[:5])
    with pytest.raises(ValueError, match="method must be"):
        hierarchical_risk_parity(x, method="complete")
    with pytest.raises(ValueError, match="method must be"):
        equal_risk_parity(x, method="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shrinkage must be"):
        hierarchical_risk_parity(x, shrinkage="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two return columns"):
        hierarchical_risk_parity(x[:, :1])
