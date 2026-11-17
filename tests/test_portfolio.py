"""Mathematical invariant tests for HRP and inverse-volatility allocation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_anta.portfolio import equal_risk_parity, hierarchical_risk_parity


def _frame(x: np.ndarray, names: list[str]) -> pl.DataFrame:
    data = {n: x[:, i] for i, n in enumerate(names)}
    data["date"] = np.arange(x.shape[0])
    return pl.DataFrame(data)


def test_weights_sum_to_one_and_are_positive():
    """Both allocators return a normalised, strictly positive long-form frame."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, size=(300, 5))
    df = _frame(x, ["a", "b", "c", "d", "e"])

    for fn in (hierarchical_risk_parity, equal_risk_parity):
        w = fn(df, time_col="date")
        assert w.columns == ["asset", "weight"]
        assert w.height == 5
        assert abs(w["weight"].sum() - 1.0) < 1e-12
        assert (w["weight"] > 0).all()


def test_equal_risk_parity_is_inverse_volatility():
    """ERP weights must equal 1/sigma normalised, on a sample covariance."""
    rng = np.random.default_rng(11)
    scales = np.array([1.0, 2.0, 4.0])
    x = rng.normal(0, 0.01, size=(500, 3)) * scales
    df = _frame(x, ["a", "b", "c"])
    w = equal_risk_parity(df, time_col="date", shrinkage="sample")
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
    """Two equally sized, internally correlated clusters share the risk budget evenly.

    HRP bisects the quasi-diagonal order by position, so this only holds when the
    cluster sizes are comparable: with a 3-asset and a 1-asset cluster the first
    cut lands inside the larger cluster (documented behaviour of the algorithm,
    not a bug) and the budgets separate.
    """
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


def test_single_linkage_and_array_input():
    """Array input works and single linkage returns a valid allocation."""
    rng = np.random.default_rng(13)
    x = rng.normal(0, 0.01, size=(200, 4))
    w1 = hierarchical_risk_parity(x)
    w2 = hierarchical_risk_parity(x, method="single")
    assert w1.height == w2.height == 4
    assert abs(w2["weight"].sum() - 1.0) < 1e-12
    assert w1["asset"].to_list() == ["asset_0", "asset_1", "asset_2", "asset_3"]


def test_namespace_method():
    """The DataFrame namespace exposes the allocator through .anta."""
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.01, size=(300, 3))
    df = _frame(x, ["x", "y", "z"])
    out = df.anta.hierarchical_risk_parity(time_col="date")
    assert out.height == 3
    assert abs(out["weight"].sum() - 1.0) < 1e-12


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
    with pytest.raises(ValueError, match="at least two return columns"):
        hierarchical_risk_parity(x[:, :1])
