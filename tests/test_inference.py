"""Unit and statistical validation tests for polars_anta.inference."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_anta as pa
from polars_anta.evaluation import _newey_west_lrv, _nw_auto_lag
from polars_anta.inference import (
    _bootstrap_means,
    _to_1d_array,
    _to_matrix,
    adjust_pvalues,
    bootstrap_test,
    breakeven_cost,
    breakeven_cost_from_backtest,
    clark_west_test,
    cusum_mean_test,
    diebold_mariano_test,
    empirical_bayes_shrinkage,
    oos_r2,
    pbo_cscv,
    romano_wolf,
    spa_test,
    stationary_bootstrap_indices,
)


def test_ic_summary_nw_refactor_parity() -> None:
    """Verify exact numerical parity of ic_summary after extracting Newey-West helpers."""
    rng = np.random.default_rng(42)
    n_dates, n_assets = 80, 25
    dates = np.repeat(np.arange(n_dates), n_assets)
    assets = np.tile(np.arange(n_assets), n_dates)
    ret = rng.normal(0.0, 0.02, size=n_dates * n_assets)
    factor = 0.25 * ret + rng.normal(0.0, 0.02, size=n_dates * n_assets)
    df = pl.DataFrame({"date": dates, "asset": assets, "factor": factor, "fwd_ret": ret})

    expected = {
        None: 10.278339483747970,
        0: 10.937702275911013,
        5: 10.830998426443744,
    }
    for lags, exp_val in expected.items():
        res = pa.ic_summary(df, "factor", "fwd_ret", group_by="date", nw_lags=lags)
        assert math.isclose(res["nw_t_stat"], exp_val, rel_tol=1e-12)

    assert _nw_auto_lag(1) == 0
    u_2d = np.column_stack([ret[:50] - ret[:50].mean(), factor[:50] - factor[:50].mean()])
    lrv_2d = _newey_west_lrv(u_2d, 3)
    assert isinstance(lrv_2d, np.ndarray)
    assert lrv_2d.shape == (2,)
    assert math.isclose(float(lrv_2d[0]), float(_newey_west_lrv(u_2d[:, 0], 3)), rel_tol=1e-12)


def test_stationary_bootstrap_indices_and_means() -> None:
    """Test stationary bootstrap index generation, mean block length, and BLAS means parity."""
    idx1 = stationary_bootstrap_indices(n_obs=2000, n_boot=500, mean_block=10.0, seed=123)
    idx2 = stationary_bootstrap_indices(n_obs=2000, n_boot=500, mean_block=10.0, seed=123)
    assert idx1.shape == (500, 2000)
    assert idx1.dtype == np.int64
    assert int(idx1.min()) >= 0 and int(idx1.max()) < 2000
    np.testing.assert_array_equal(idx1, idx2)

    # Check empirical mean block length: contiguous step is (idx[:, t] - idx[:, t-1]) % T == 1
    is_cont = ((idx1[:, 1:] - idx1[:, :-1]) % 2000) == 1
    empirical_p = 1.0 - float(is_cont.mean())
    empirical_block = 1.0 / empirical_p
    assert 8.5 <= empirical_block <= 11.5

    # When mean_block=1, draws are i.i.d. uniform => adjacent step rate ~ 1/T
    idx_iid = stationary_bootstrap_indices(n_obs=1000, n_boot=300, mean_block=1.0, seed=7)
    adj_rate = float((((idx_iid[:, 1:] - idx_iid[:, :-1]) % 1000) == 1).mean())
    assert adj_rate < 0.01

    # _bootstrap_means exact parity with X[idx].mean(axis=1) and chunking parity
    rng = np.random.default_rng(99)
    x_mat = rng.normal(size=(60, 5))
    idx_small = stationary_bootstrap_indices(60, n_boot=40, mean_block=5.0, seed=99)
    direct = x_mat[idx_small].mean(axis=1)
    blas_full = _bootstrap_means(x_mat, idx_small)
    blas_chunked = _bootstrap_means(x_mat, idx_small, chunk_bytes=8 * 60 * 7)
    np.testing.assert_allclose(blas_full, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(blas_chunked, direct, rtol=1e-12, atol=1e-12)


def test_bootstrap_test_mean_and_sharpe() -> None:
    """Test stationary bootstrap inference for mean and Sharpe ratio."""
    rng = np.random.default_rng(42)
    null_series = rng.normal(0.0, 1.0, size=1000)
    res_null = bootstrap_test(null_series, statistic="mean", n_boot=499, seed=42)
    assert res_null["p_value"] > 0.05
    assert res_null["ci_lower"] <= res_null["estimate"] <= res_null["ci_upper"]

    alt_series = rng.normal(0.20, 1.0, size=1000)
    res_alt = bootstrap_test(alt_series, statistic="mean", n_boot=499, seed=42)
    assert res_alt["p_value"] < 0.01

    res_gt = bootstrap_test(
        alt_series, statistic="mean", alternative="greater", n_boot=499, seed=42
    )
    res_lt = bootstrap_test(alt_series, statistic="mean", alternative="less", n_boot=499, seed=42)
    assert res_gt["p_value"] < 0.01
    assert res_lt["p_value"] > 0.90

    # Sharpe ratio estimate parity with pa.sharpe_ratio(..., annualization=252)
    s_pl = pl.Series("ret", alt_series)
    res_sr = bootstrap_test(s_pl, statistic="sharpe", n_boot=199, annualization=252.0, seed=42)
    df_sr = pl.DataFrame({"ret": alt_series}).select(
        pa.sharpe_ratio("ret", annualization=252.0).alias("sr")
    )
    assert math.isclose(res_sr["estimate"], float(df_sr["sr"][0]), rel_tol=1e-12)


def test_adjust_pvalues_canonical_and_nan() -> None:
    """Test Bonferroni, Holm, BH, and BY p-value adjustments against analytical values."""
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(
        adjust_pvalues(p, method="bonferroni"), [0.04, 0.16, 0.12, 0.02], rtol=1e-12
    )
    np.testing.assert_allclose(
        adjust_pvalues(p, method="holm"), [0.03, 0.06, 0.06, 0.02], rtol=1e-12
    )
    bh = adjust_pvalues(p, method="bh")
    np.testing.assert_allclose(bh, [0.02, 0.04, 0.04, 0.02], rtol=1e-12)

    harmonic_4 = 1.0 + 0.5 + 1.0 / 3.0 + 0.25
    np.testing.assert_allclose(
        adjust_pvalues(p, method="by"), np.minimum(1.0, bh * harmonic_4), rtol=1e-12
    )

    p_with_nan = pl.Series("p", [0.01, None, 0.04, float("nan"), 0.03, 0.005])
    adj_nan = adjust_pvalues(p_with_nan, method="bh")
    assert np.isnan(adj_nan[1]) and np.isnan(adj_nan[3])
    np.testing.assert_allclose(adj_nan[[0, 2, 4, 5]], bh, rtol=1e-12)
    assert np.all(np.isnan(adjust_pvalues([float("nan"), float("nan")], method="bh")))


def test_inference_input_validation_part1() -> None:
    """Test input validation errors on bootstrap and p-value adjustment helpers."""
    with pytest.raises(ValueError, match="n_obs"):
        stationary_bootstrap_indices(1)
    with pytest.raises(ValueError, match="n_boot"):
        stationary_bootstrap_indices(20, n_boot=0)
    with pytest.raises(ValueError, match="mean_block"):
        stationary_bootstrap_indices(20, mean_block=0.0)
    with pytest.raises(ValueError, match="statistic"):
        bootstrap_test(np.ones(20), statistic="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alternative"):
        bootstrap_test(np.ones(20), alternative="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_test(np.ones(20), ci_level=1.0)
    with pytest.raises(ValueError, match="annualization"):
        bootstrap_test(np.ones(20), annualization=0.0)
    with pytest.raises(ValueError, match="at least 10"):
        bootstrap_test(np.ones(5))
    with pytest.raises(ValueError, match="NaN"):
        _to_1d_array(np.array([1.0] * 10 + [float("nan")]))
    with pytest.raises(ValueError, match="null"):
        _to_1d_array(pl.Series("x", [1.0] * 10 + [None]))
    with pytest.raises(ValueError, match="1-dimensional"):
        _to_1d_array(np.ones((10, 2)))
    with pytest.raises(ValueError, match="method"):
        adjust_pvalues([0.1, 0.2], method="sidak")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        adjust_pvalues([0.1, 1.2])
    with pytest.raises(ValueError, match="1-dimensional"):
        adjust_pvalues(np.ones((2, 2)))
    with pytest.raises(ValueError, match="No columns"):
        _to_matrix(pl.DataFrame({"a": range(15)}), exclude=["a"])
    with pytest.raises(ValueError, match="null values"):
        _to_matrix(pl.DataFrame({"a": [1.0] * 14 + [None]}))


def test_romano_wolf_stepdown_properties() -> None:
    """Test Romano-Wolf stepdown power, monotonicity, p_rw >= p_raw, constant column, and K=1."""
    rng = np.random.default_rng(77)
    t_obs = 1000
    strong_col = rng.normal(0.15, 1.0, size=(t_obs, 1))
    null_cols = rng.normal(0.0, 1.0, size=(t_obs, 49))
    mat = np.hstack([strong_col, null_cols])
    df = pl.DataFrame({f"s_{i}": mat[:, i] for i in range(50)})

    rw = romano_wolf(df, n_boot=499, seed=77)
    assert rw.columns == ["name", "mean", "t_stat", "p_raw", "p_rw"]
    assert rw["name"].to_list() == [f"s_{i}" for i in range(50)]
    assert float(rw.row(0, named=True)["p_rw"]) < 0.01

    p_raw = rw["p_raw"].to_numpy()
    p_rw = rw["p_rw"].to_numpy()
    assert np.all(p_rw + 1e-12 >= p_raw)

    # Monotonicity along |t_stat| descending order
    t_abs = np.abs(rw["t_stat"].to_numpy())
    order = np.argsort(-t_abs)
    p_rw_sorted = p_rw[order]
    assert np.all(np.diff(p_rw_sorted) >= -1e-12)

    # Constant column => t_stat = 0, p_raw = 1, p_rw = 1; K=1 works
    df_const = pl.DataFrame({"const": np.zeros(100), "sig": rng.normal(0.2, 1.0, size=100)})
    rw_const = romano_wolf(df_const, n_boot=199, alternative="greater", seed=77)
    const_row = rw_const.filter(pl.col("name") == "const").row(0, named=True)
    assert const_row["t_stat"] == 0.0
    assert const_row["p_raw"] == 1.0
    assert const_row["p_rw"] == 1.0

    rw_k1 = romano_wolf(strong_col[:, 0], n_boot=199, seed=77)
    assert len(rw_k1) == 1
    assert math.isclose(float(rw_k1["p_raw"][0]), float(rw_k1["p_rw"][0]), rel_tol=1e-12)


def test_spa_test_properties() -> None:
    """Test Hansen SPA / White Reality Check size, power, ordering, and column benchmark."""
    rng = np.random.default_rng(42)
    null_mat = rng.normal(0.0, 1.0, size=(1000, 50))
    res_null = spa_test(null_mat, n_boot=499, seed=42)
    assert res_null["p_consistent"] > 0.10
    assert res_null["p_lower"] <= res_null["p_consistent"] <= res_null["p_upper"]

    alt_mat = np.hstack([null_mat, rng.normal(0.15, 1.0, size=(1000, 1))])
    res_alt = spa_test(alt_mat, n_boot=499, seed=42)
    assert res_alt["p_consistent"] < 0.05
    assert res_alt["p_lower"] <= res_alt["p_consistent"] <= res_alt["p_upper"]
    assert res_alt["best_name"] == "col_50"

    # Benchmark as column name in DataFrame
    df = pl.DataFrame(
        {
            "bench": null_mat[:, 0],
            "m1": null_mat[:, 1],
            "m2": null_mat[:, 0] + 0.15 + rng.normal(0.0, 0.5, size=1000),
        }
    )
    res_df = spa_test(df.lazy(), benchmark="bench", n_boot=199, seed=42)
    assert res_df["n_models"] == 2.0
    assert res_df["best_name"] == "m2"
    assert res_df["p_consistent"] < 0.05


def test_pbo_cscv_properties() -> None:
    """Test CSCV PBO under pure noise, dominant strategy, subsampling, and odd n_splits."""
    rng = np.random.default_rng(123)
    noise_mat = rng.normal(0.0, 1.0, size=(1605, 20))  # 1605 trims 5 leading rows to 1600
    res_noise = pbo_cscv(noise_mat, n_splits=16)
    assert res_noise["n_combinations"] == 12870
    assert res_noise["n_strategies"] == 20
    assert 0.30 <= float(res_noise["pbo"]) <= 0.70

    dom_mat = np.hstack([noise_mat, rng.normal(0.15, 1.0, size=(1605, 1))])
    res_dom = pbo_cscv(dom_mat, n_splits=16)
    assert float(res_dom["pbo"]) < 0.10

    res_sub = pbo_cscv(noise_mat, n_splits=16, max_combinations=250, seed=42)
    assert res_sub["n_combinations"] == 250
    assert isinstance(res_sub["logits"], np.ndarray) and res_sub["logits"].shape == (250,)

    with pytest.raises(ValueError, match="even integer"):
        pbo_cscv(noise_mat, n_splits=15)
    with pytest.raises(ValueError, match="at least 2 strategies"):
        pbo_cscv(noise_mat[:, :1], n_splits=8)
    with pytest.raises(ValueError, match="max_combinations"):
        pbo_cscv(noise_mat, n_splits=8, max_combinations=0)


def test_empirical_bayes_shrinkage() -> None:
    """Test EB shrinkage under tau2=0, small SE, grand_mean prior, and grouped priors."""
    # Case 1: tau2 = 0 (SE larger than dispersion around prior) => full shrinkage to prior
    eb_zero = empirical_bayes_shrinkage(
        estimates=[0.01, -0.01, 0.005],
        std_errors=[0.10, 0.10, 0.10],
        names=["f1", "f2", "f3"],
        prior="zero",
    )
    assert np.allclose(eb_zero["tau2"].to_numpy(), 0.0)
    assert np.allclose(eb_zero["shrinkage"].to_numpy(), 1.0)
    assert np.allclose(eb_zero["posterior_mean"].to_numpy(), 0.0)
    assert np.allclose(eb_zero["posterior_sd"].to_numpy(), 0.0)
    assert np.allclose(eb_zero["prob_positive"].to_numpy(), 0.5)

    # Case 2: Tiny SE => posterior_mean ~= estimate
    est = np.array([0.05, -0.04, 0.08, 0.02])
    se = np.array([1e-4, 1e-4, 1e-4, 1e-4])
    eb_prec = empirical_bayes_shrinkage(est, se, prior="grand_mean")
    np.testing.assert_allclose(eb_prec["posterior_mean"].to_numpy(), est, rtol=1e-4)
    probs = eb_prec["prob_positive"].to_numpy()
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert probs[0] > 0.99 and probs[1] < 0.01

    # Case 3: Grouped shrinkage
    eb_grp = empirical_bayes_shrinkage(
        estimates=[0.10, 0.12, -0.10, 0.001, -0.001, 0.002],
        std_errors=[0.01, 0.01, 0.01, 0.05, 0.05, 0.05],
        groups=["strong", "strong", "strong", "weak", "weak", "weak"],
        prior="zero",
    )
    strong_rows = eb_grp.filter(pl.col("group") == "strong")
    weak_rows = eb_grp.filter(pl.col("group") == "weak")
    assert float(strong_rows["tau2"][0]) > 0.0
    assert float(weak_rows["tau2"][0]) == 0.0
    assert np.allclose(weak_rows["posterior_mean"].to_numpy(), 0.0)


def test_oos_r2_clark_west_and_diebold_mariano() -> None:
    """Test Campbell-Thompson OOS R2, Clark-West nested test, and Diebold-Mariano symmetry."""
    rng = np.random.default_rng(55)
    y = rng.normal(0.01, 0.10, size=200)

    # Perfect forecast => R2_oos == 1.0
    r2_perf = oos_r2(y, y, min_train=20)
    assert math.isclose(r2_perf["r2_oos"], 1.0, rel_tol=1e-12)
    assert r2_perf["n_obs"] == 180.0

    # Forecast equal to expanding mean => R2_oos == 0.0
    exp_mean = np.zeros_like(y)
    exp_mean[1:] = np.cumsum(y[:-1]) / np.arange(1, y.size, dtype=np.float64)
    r2_bench = oos_r2(y, exp_mean, min_train=20)
    assert math.isclose(r2_bench["r2_oos"], 0.0, abs_tol=1e-12)

    # Clark-West: true predictive signal vs zero benchmark => p < 0.05
    signal = rng.normal(0.0, 1.0, size=300)
    y_cw = 0.25 * signal + rng.normal(0.0, 1.0, size=300)
    pred_r = np.zeros(300)
    pred_u = 0.25 * signal
    cw = clark_west_test(y_cw, pred_r, pred_u)
    assert cw["statistic"] > 2.0
    assert cw["p_value"] < 0.05

    # Diebold-Mariano: symmetry when swapping pred_1 and pred_2
    dm_12 = diebold_mariano_test(y_cw, pred_u, pred_r, loss="squared", horizon=1)
    dm_21 = diebold_mariano_test(y_cw, pred_r, pred_u, loss="squared", horizon=1)
    assert math.isclose(dm_12["statistic"], -dm_21["statistic"], rel_tol=1e-12)
    assert math.isclose(dm_12["p_value"], dm_21["p_value"], rel_tol=1e-12)
    assert dm_12["statistic"] < 0.0 and dm_12["p_value"] < 0.05

    dm_abs = diebold_mariano_test(y_cw, pred_u, pred_r, loss="absolute", horizon=3)
    assert dm_abs["statistic"] < 0.0


def test_cusum_mean_test_and_breakeven_cost() -> None:
    """Test OLS-CUSUM structural break detection and breakeven transaction cost."""
    rng = np.random.default_rng(88)
    stable = rng.normal(0.05, 0.10, size=400)
    cusum_stable = cusum_mean_test(stable)
    assert cusum_stable["p_value"] > 0.05

    shifted = np.concatenate([rng.normal(0.10, 0.05, size=200), rng.normal(-0.10, 0.05, size=200)])
    cusum_shift = cusum_mean_test(shifted)
    assert cusum_shift["p_value"] < 0.01
    assert 160.0 <= cusum_shift["break_index"] <= 240.0
    assert cusum_shift["mean_before"] > 0.05 and cusum_shift["mean_after"] < -0.05

    # Breakeven cost hand-calculable + BacktestResult integration
    gross = [0.01, 0.02, -0.005, 0.015]  # sum = 0.04
    to = [1.0, 0.0, 1.0, 0.0]  # sum = 2.0
    assert math.isclose(breakeven_cost(gross, to), 0.02, rel_tol=1e-12)
    assert math.isinf(breakeven_cost([0.01, 0.01], [0.0, 0.0]))
    assert breakeven_cost([-0.01, 0.0], [0.0, 0.0]) == 0.0

    df_bt = pl.DataFrame(
        {
            "close": [100.0, 102.0, 105.0, 103.0, 106.0, 108.0],
            "signal": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        }
    )
    bt_res = df_bt.anta.backtest(signal="signal", fee_rate=0.0)
    c_star = breakeven_cost_from_backtest(bt_res)
    expected_c = float(bt_res.data["gross_return"].sum()) / float(bt_res.data["turnover"].sum())
    assert math.isclose(c_star, expected_c, rel_tol=1e-12)
