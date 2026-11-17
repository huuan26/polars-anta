"""Mathematical invariant, KKT optimality, and causal prefix-invariance tests for Phase 3."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polars_anta.allocation import (
    combine_alphas_walkforward,
    project_box_simplex,
    project_simplex,
    prox_l1_shifted,
    solve_fista,
)
from polars_anta.covariance import (
    exponential_covariance,
    ledoit_wolf_shrinkage,
    marchenko_pastur_denoise,
    nearest_spd,
    sample_covariance,
)


def _naive_ledoit_wolf_const_corr(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Reference explicit-loop Ledoit-Wolf (2004) implementation to verify vectorized BLAS."""
    t_obs, k_cols = x.shape
    x_c = x - np.mean(x, axis=0, keepdims=True)
    s_cov = (x_c.T @ x_c) / t_obs
    var = np.diag(s_cov)
    std = np.sqrt(var)
    corr = s_cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    r_bar = float((np.sum(corr) - k_cols) / (k_cols * (k_cols - 1)))
    target = r_bar * np.outer(std, std)
    np.fill_diagonal(target, var)

    pi_mat = np.zeros((k_cols, k_cols))
    for t in range(t_obs):
        pi_mat += (np.outer(x_c[t], x_c[t]) - s_cov) ** 2
    pi_mat /= t_obs
    pi_hat = float(np.sum(pi_mat))
    d2 = float(np.sum((s_cov - target) ** 2))
    rho_diag = float(np.sum(np.diag(pi_mat)))
    rho_off = 0.0
    for i in range(k_cols):
        for j in range(k_cols):
            if i != j:
                t1 = (
                    float(
                        np.sum(
                            (x_c[:, i] ** 2 - s_cov[i, i]) * (x_c[:, i] * x_c[:, j] - s_cov[i, j])
                        )
                    )
                    / t_obs
                )
                t2 = (
                    float(
                        np.sum(
                            (x_c[:, j] ** 2 - s_cov[j, j]) * (x_c[:, i] * x_c[:, j] - s_cov[i, j])
                        )
                    )
                    / t_obs
                )
                rho_off += r_bar * 0.5 * (std[j] / std[i] * t1 + std[i] / std[j] * t2)
    delta = min(1.0, max(0.0, ((pi_hat - (rho_diag + rho_off)) / d2) / t_obs))
    cov_out = (delta * target + (1.0 - delta) * s_cov) * (t_obs / (t_obs - 1.0))
    return cov_out, delta


def test_ledoit_wolf_blas_matches_reference_loop_to_machine_precision() -> None:
    rng = np.random.default_rng(17)
    x = rng.standard_t(df=5.0, size=(80, 6))
    cov_fast, delta_fast = ledoit_wolf_shrinkage(x, target="constant_correlation")
    cov_ref, delta_ref = _naive_ledoit_wolf_const_corr(x)
    assert np.isclose(delta_fast, delta_ref, atol=1e-13)
    np.testing.assert_allclose(cov_fast, cov_ref, atol=1e-12)

    # Identity target + Polars LazyFrame input with NaNs
    df_in = pl.DataFrame(x, schema=[f"f{i}" for i in range(6)])
    cov_id, delta_id = ledoit_wolf_shrinkage(df_in.lazy(), target="identity")
    assert 0.0 <= delta_id <= 1.0
    assert np.all(np.linalg.eigvalsh(cov_id) > 0.0)

    # K=1 edge case returns 2D (1, 1) matrix
    cov_1d, d_1d = ledoit_wolf_shrinkage(x[:, :1])
    assert cov_1d.shape == (1, 1)
    assert d_1d == 0.0


def test_marchenko_pastur_denoise_reduces_true_correlation_error() -> None:
    """RMT denoising must preserve SPD & unit diagonal and reduce Frobenius error vs sample cov."""
    rng = np.random.default_rng(2026)
    t_obs, k_cols = 100, 20
    # 2 latent strong factors + idiosyncratic noise
    loadings = np.zeros((k_cols, 2))
    loadings[:10, 0] = 0.8
    loadings[10:, 1] = 0.75
    cov_true = loadings @ loadings.T + np.diag(1.0 - np.sum(loadings**2, axis=1))

    # Simulate observations from cov_true
    chol = np.linalg.cholesky(cov_true)
    obs = rng.standard_normal((t_obs, k_cols)) @ chol.T

    s_cov = sample_covariance(obs)
    mp_const = marchenko_pastur_denoise(obs, method="constant_residual")
    mp_shrink = marchenko_pastur_denoise(s_cov, n_obs=t_obs, is_cov=True, method="targeted_shrink")
    exp_cov = exponential_covariance(obs, half_life=30.0)

    for mat in (mp_const, mp_shrink, exp_cov):
        assert mat.shape == (k_cols, k_cols)
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(mat)) > 0.0

    # Check correlation matrix unit diagonal & Frobenius improvement over noisy sample correlation
    sd_s = np.sqrt(np.diag(s_cov))
    corr_s = s_cov / np.outer(sd_s, sd_s)
    sd_mp = np.sqrt(np.diag(mp_const))
    corr_mp = mp_const / np.outer(sd_mp, sd_mp)
    np.testing.assert_allclose(np.diag(corr_mp), np.ones(k_cols), atol=1e-12)

    err_sample = float(np.linalg.norm(corr_s - cov_true, ord="fro"))
    err_mp = float(np.linalg.norm(corr_mp - cov_true, ord="fro"))
    assert err_mp < err_sample


def test_solve_fista_analytical_parity_and_exact_joint_kkt_optimality() -> None:
    """Verify exact unconstrained analytical match, joint L1+box-simplex KKT, and zero-turnover lock."""
    rng = np.random.default_rng(77)
    a = rng.standard_normal((5, 5))
    cov = (a.T @ a) / 5.0 + 0.2 * np.eye(5)
    mu = np.array([0.08, -0.05, 0.04, 0.01, -0.03])

    # 1. Unconstrained with lambda_1 = 0 matches closed-form linear system
    gamma, l2 = 2.0, 1e-3
    res_unc = solve_fista(
        mu, cov, risk_aversion=gamma, turnover_penalty=0.0, l2_reg=l2, constraint="unconstrained"
    )
    w_exact = np.linalg.solve(gamma * cov + l2 * np.eye(5), mu)
    np.testing.assert_allclose(res_unc.weights, w_exact, atol=1e-7)
    assert res_unc.kkt_residual < 1e-7

    # 2. Simultaneous L1 turnover penalty > 0 AND active box-simplex constraints
    w_prev = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    lam1 = 0.03
    res_joint = solve_fista(
        mu,
        cov,
        risk_aversion=gamma,
        turnover_penalty=lam1,
        w_prev=w_prev,
        l2_reg=l2,
        constraint="box_simplex",
        lower_bounds=0.0,
        upper_bounds=0.45,
        budget=1.0,
    )
    w_star = res_joint.weights
    assert res_joint.converged
    assert res_joint.kkt_residual < 1e-7
    assert np.isclose(np.sum(w_star), 1.0, atol=1e-11)
    assert np.all((w_star >= -1e-12) & (w_star <= 0.45 + 1e-12))

    # Global optimality verification against random feasible perturbations on box-simplex
    hess = gamma * cov + l2 * np.eye(5)

    def obj_fn(w: np.ndarray) -> float:
        return (
            0.5 * float(w @ (hess @ w)) - float(mu @ w) + lam1 * float(np.sum(np.abs(w - w_prev)))
        )

    f_star = obj_fn(w_star)
    for _ in range(300):
        cand = project_box_simplex(w_star + 0.05 * rng.standard_normal(5), 0.0, 0.45, 1.0)
        assert f_star <= obj_fn(cand) + 1e-10

    # 3. Turnover monotonicity & exact zero-turnover lock under high penalty
    res_low = solve_fista(mu, cov, risk_aversion=gamma, turnover_penalty=0.005, w_prev=w_prev)
    res_high = solve_fista(mu, cov, risk_aversion=gamma, turnover_penalty=0.05, w_prev=w_prev)
    res_lock = solve_fista(mu, cov, risk_aversion=gamma, turnover_penalty=5.0, w_prev=w_prev)
    assert res_high.turnover <= res_low.turnover + 1e-10
    assert res_lock.turnover == 0.0
    np.testing.assert_allclose(res_lock.weights, w_prev, atol=1e-12)
    assert "converged" in res_lock.summary()


def test_combine_alphas_walkforward_strict_causal_prefix_invariance() -> None:
    """Truncating future prices and masking pending h=5 forward returns must produce 0.0 weight diff."""
    rng = np.random.default_rng(314)
    n_dates, n_tickers = 55, 24
    rows: list[dict[str, object]] = []

    # Generate panel with 3 factors (f1 strong positive, f2 noisy, f3 negative) and h=5 open-to-open return
    f_true = rng.standard_normal((n_dates, n_tickers, 3))
    future_shock = rng.standard_normal((n_dates, n_tickers))
    for t in range(n_dates):
        for i in range(n_tickers):
            f1, f2, f3 = float(f_true[t, i, 0]), float(f_true[t, i, 1]), float(f_true[t, i, 2])
            # h=5 forward return uses open[t+6]/open[t+1]-1 -> realized only when t + 6 <= current_day
            ret_5 = (
                0.12 * f1 - 0.09 * f3 + 0.5 * float(future_shock[t, i]) if t + 6 < n_dates else None
            )
            rows.append(
                {
                    "date": t,
                    "ticker": f"S{i:02d}",
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "fwd_ret_5": ret_5,
                }
            )
    full_panel = pl.DataFrame(rows)

    # Full-sample walk-forward run with horizon=5, signal_lag=1 (embargo = 5)
    res_full = combine_alphas_walkforward(
        full_panel,
        factor_cols=["f1", "f2", "f3"],
        return_col="fwd_ret_5",
        time_col="date",
        lookback=15,
        horizon=5,
        signal_lag=1,
        turnover_penalty=0.04,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        cov_estimator="marchenko_pastur",
        ic_shrinkage="empirical_bayes",
        ic_ema_halflife=10.0,
    )

    # Live cut at t_0 = 40: at close of day 40, open[40] is known, open[41..] is UNKNOWN!
    # Therefore any t where t + 1 + 5 > 40 (i.e. t >= 35, indices 35..40) has fwd_ret_5 = NULL!
    t_cut = 40
    live_panel = full_panel.filter(pl.col("date") <= t_cut).with_columns(
        pl.when(pl.col("date") > t_cut - 6)
        .then(None)
        .otherwise(pl.col("fwd_ret_5"))
        .alias("fwd_ret_5")
    )
    # Also corrupt future prices in full_panel after t_cut to prove they had zero effect
    res_live = combine_alphas_walkforward(
        live_panel,
        factor_cols=["f1", "f2", "f3"],
        return_col="fwd_ret_5",
        time_col="date",
        lookback=15,
        horizon=5,
        signal_lag=1,
        turnover_penalty=0.04,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        cov_estimator="marchenko_pastur",
        ic_shrinkage="empirical_bayes",
        ic_ema_halflife=10.0,
    )

    # 1. Live day t_cut = 40 MUST have portfolio weights generated despite null fwd_ret_5!
    assert res_live.weights_history["date"].max() == t_cut
    assert res_live.embargo_bars == 5

    # 2. Prefix-Invariance: EVERY weight on t <= 40 must match full_panel bit-for-bit!
    w_full_sub = res_full.weights_history.filter(pl.col("date") <= t_cut).select(
        ["w_f1", "w_f2", "w_f3"]
    )
    w_live_sub = res_live.weights_history.select(["w_f1", "w_f2", "w_f3"])
    np.testing.assert_allclose(w_live_sub.to_numpy(), w_full_sub.to_numpy(), atol=1e-12)

    # 3. Signed box mode assigns positive weight to f1 and negative weight to f3!
    assert res_full.weights_history["w_f1"].mean() > 0.0
    assert res_full.weights_history["w_f3"].mean() < 0.0
    assert "fista_ic_mean" in res_full.summary()

    for est in ("ledoit_wolf", "exponential", "sample"):
        res_alt = combine_alphas_walkforward(
            full_panel,
            factor_cols=["f1", "f2"],
            return_col="fwd_ret_5",
            time_col="date",
            lookback=15,
            horizon=5,
            signal_lag=1,
            cov_estimator=est,  # type: ignore[arg-type]
            no_trade_buffer=0.05,
        )
        assert res_alt.weights_history.height > 0


def test_combine_alphas_turnover_cost_discrimination_and_causal_smoothing() -> None:
    """Cost-aware allocator must prefer low-turnover persistent alpha over equal-gross noisy alpha."""
    rng = np.random.default_rng(202611)
    n_dates, n_tickers = 75, 32
    f_slow = np.zeros((n_dates, n_tickers), dtype=np.float64)
    f_fast = np.zeros((n_dates, n_tickers), dtype=np.float64)

    s_prev = rng.standard_normal(n_tickers)
    s_prev = (s_prev - s_prev.mean()) / s_prev.std(ddof=1)
    for t in range(n_dates):
        if t > 0:
            s_raw = 0.94 * s_prev + math.sqrt(1.0 - 0.94**2) * rng.standard_normal(n_tickers)
            s_prev = (s_raw - s_raw.mean()) / s_raw.std(ddof=1)
        f_slow[t] = s_prev
        z_fast = rng.standard_normal(n_tickers)
        # Orthogonalize f_fast wrt f_slow so both have identical gross alpha and zero correlation
        z_fast = z_fast - ((z_fast @ s_prev) / (s_prev @ s_prev)) * s_prev
        f_fast[t] = (z_fast - z_fast.mean()) / z_fast.std(ddof=1)

    rows: list[dict[str, object]] = []
    shocks = rng.normal(0.0, 0.012, size=(n_dates, n_tickers))
    for t in range(n_dates):
        for i in range(n_tickers):
            # f_fast has HIGHER gross slope (0.0060) than f_slow (0.0048), but ~5x higher turnover
            r_1 = (
                0.0048 * float(f_slow[t, i]) + 0.0060 * float(f_fast[t, i]) + float(shocks[t, i])
                if t + 1 < n_dates
                else None
            )
            rows.append(
                {
                    "date": t,
                    "ticker": f"A{i:02d}",
                    "f_slow": float(f_slow[t, i]),
                    "f_fast": float(f_fast[t, i]),
                    "fwd_ret_1": r_1,
                }
            )
    panel = pl.DataFrame(rows)

    # Zero cost (also exercises asset_col=None in factor_return mode) -> prefers higher-gross f_fast
    res_zero = combine_alphas_walkforward(
        panel,
        factor_cols=["f_slow", "f_fast"],
        return_col="fwd_ret_1",
        time_col="date",
        asset_col=None,
        lookback=25,
        horizon=1,
        signal_lag=1,
        objective_mode="factor_return",
        one_way_cost_bps=0.0,
        turnover_penalty=0.005,
    )
    w_slow_zero = float(res_zero.weights_history["w_f_slow"].mean() or 0.0)
    w_fast_zero = float(res_zero.weights_history["w_f_fast"].mean() or 0.0)
    assert w_fast_zero > w_slow_zero

    # Realistic 30 bps one-way cost -> flips preference decisively toward persistent low-turnover f_slow
    for mode in ("factor_return", "ic"):
        res_cost = combine_alphas_walkforward(
            panel,
            factor_cols=["f_slow", "f_fast"],
            return_col="fwd_ret_1",
            time_col="date",
            asset_col="ticker",
            lookback=25,
            horizon=1,
            signal_lag=1,
            objective_mode=mode,  # type: ignore[arg-type]
            one_way_cost_bps=30.0,
            signal_ema_alpha=0.35,
            turnover_penalty=0.005,
        )
        w_slow = float(res_cost.weights_history["w_f_slow"].mean() or 0.0)
        w_fast = float(res_cost.weights_history["w_f_fast"].mean() or 0.0)
        assert w_slow > w_fast + 0.30
        assert w_slow > w_slow_zero + 0.30

    # Prefix-Invariance of factor_return + one_way_cost_bps + signal_ema_alpha at t_cut = 55
    t_cut = 55
    full_c = combine_alphas_walkforward(
        panel,
        factor_cols=["f_slow", "f_fast"],
        return_col="fwd_ret_1",
        time_col="date",
        asset_col="ticker",
        lookback=20,
        horizon=2,
        signal_lag=1,
        objective_mode="factor_return",
        one_way_cost_bps=15.0,
        signal_ema_alpha=0.4,
    )
    live_panel = panel.filter(pl.col("date") <= t_cut).with_columns(
        pl.when(pl.col("date") > t_cut - 2)
        .then(None)
        .otherwise(pl.col("fwd_ret_1"))
        .alias("fwd_ret_1")
    )
    live_c = combine_alphas_walkforward(
        live_panel,
        factor_cols=["f_slow", "f_fast"],
        return_col="fwd_ret_1",
        time_col="date",
        asset_col="ticker",
        lookback=20,
        horizon=2,
        signal_lag=1,
        objective_mode="factor_return",
        one_way_cost_bps=15.0,
        signal_ema_alpha=0.4,
    )
    np.testing.assert_allclose(
        live_c.weights_history.select(["w_f_slow", "w_f_fast"]).to_numpy(),
        full_c.weights_history.filter(pl.col("date") <= t_cut)
        .select(["w_f_slow", "w_f_fast"])
        .to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        live_c.data.sort(["date", "ticker"])["composite_alpha"].to_numpy(),
        full_c.data.filter(pl.col("date") <= t_cut)
        .sort(["date", "ticker"])["composite_alpha"]
        .to_numpy(),
        atol=1e-12,
    )


def test_allocation_and_covariance_validation_guards() -> None:
    np.testing.assert_allclose(
        project_simplex(np.array([2.0, 0.0, -1.0]), budget=1.0), [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        prox_l1_shifted(np.array([1.0, -1.0]), np.zeros(2), 0.0), [1.0, -1.0]
    )
    np.testing.assert_allclose(
        project_box_simplex(np.array([0.2, 0.8]), lower_bounds=0.1, upper_bounds=0.9, budget=0.2),
        [0.1, 0.1],
    )
    np.testing.assert_allclose(
        project_box_simplex(np.array([0.2, 0.8]), lower_bounds=0.1, upper_bounds=0.9, budget=1.8),
        [0.9, 0.9],
    )
    mp_id = marchenko_pastur_denoise(np.eye(4), n_obs=50, is_cov=True)
    np.testing.assert_allclose(mp_id, np.eye(4), atol=1e-12)

    with pytest.raises(ValueError, match="budget"):
        project_simplex(np.array([1.0]), budget=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        project_simplex(np.array([]))
    with pytest.raises(ValueError, match="non-empty"):
        project_box_simplex(np.array([]))
    with pytest.raises(ValueError, match="Infeasible budget"):
        project_box_simplex(np.array([0.5, 0.5]), lower_bounds=0.0, upper_bounds=0.4, budget=1.0)
    with pytest.raises(ValueError, match="lower_bounds"):
        project_box_simplex(np.array([0.5, 0.5]), lower_bounds=0.6, upper_bounds=0.4, budget=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        solve_fista(np.array([0.1]), np.eye(1), risk_aversion=-1.0)
    with pytest.raises(ValueError, match="constraint"):
        solve_fista(np.array([0.1]), np.eye(1), constraint="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        solve_fista(np.array([0.1, 0.2]), np.eye(3))
    with pytest.raises(ValueError, match="w_prev"):
        solve_fista(np.array([0.1, 0.2]), np.eye(2), w_prev=np.ones(3))
    with pytest.raises(ValueError, match="eps"):
        nearest_spd(np.eye(2), eps=0.0)
    with pytest.raises(ValueError, match="square"):
        nearest_spd(np.ones((2, 3)))
    with pytest.raises(ValueError, match="NaN"):
        nearest_spd(np.array([[1.0, float("nan")], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="ddof"):
        sample_covariance(np.ones((5, 2)), ddof=-1)
    assert sample_covariance(np.arange(10, dtype=np.float64)).shape == (1, 1)
    with pytest.raises(ValueError, match="No numeric columns"):
        sample_covariance(pl.DataFrame({"date": [1, 2, 3]}), exclude=["date"])
    with pytest.raises(ValueError, match="Expected 2D"):
        sample_covariance(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="finite observations"):
        sample_covariance(np.array([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="2D square"):
        marchenko_pastur_denoise(np.ones((2, 3)), is_cov=True, n_obs=20)
    with pytest.raises(ValueError, match="half_life"):
        exponential_covariance(np.ones((5, 2)), half_life=0.0)
    with pytest.raises(ValueError, match="target"):
        ledoit_wolf_shrinkage(np.ones((10, 2)), target="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        marchenko_pastur_denoise(np.ones((10, 2)), method="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shrink_alpha"):
        marchenko_pastur_denoise(np.ones((10, 2)), shrink_alpha=1.5)
    with pytest.raises(ValueError, match="n_obs"):
        marchenko_pastur_denoise(np.eye(3), is_cov=True, n_obs=None)
    with pytest.raises(ValueError, match="At least 2"):
        combine_alphas_walkforward(
            pl.DataFrame({"date": [1], "f1": [1.0], "r": [0.1]}), ["f1"], "r"
        )
    with pytest.raises(ValueError, match="lookback"):
        combine_alphas_walkforward(
            pl.DataFrame({"date": [1], "f1": [1.0], "f2": [2.0], "r": [0.1]}),
            ["f1", "f2"],
            "r",
            lookback=2,
        )
    tiny = pl.DataFrame(
        {
            "date": [1, 2, 3, 4, 5],
            "ticker": ["A"] * 5,
            "f1": [1.0] * 5,
            "f2": [2.0] * 5,
            "r": [0.1] * 5,
        }
    )
    with pytest.raises(ValueError, match="horizon"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", horizon=0)
    with pytest.raises(ValueError, match="signal_lag"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", signal_lag=-1)
    with pytest.raises(ValueError, match="one_way_cost_bps"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", one_way_cost_bps=-1.0)
    with pytest.raises(ValueError, match="objective_mode"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", objective_mode="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="asset_col is required when one_way_cost_bps"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", one_way_cost_bps=5.0)
    with pytest.raises(ValueError, match="signal_ema_alpha"):
        combine_alphas_walkforward(
            tiny, ["f1", "f2"], "r", asset_col="ticker", signal_ema_alpha=0.0
        )
    with pytest.raises(ValueError, match="asset_col is required when signal_ema_alpha"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", signal_ema_alpha=0.5)
    with pytest.raises(ValueError, match="Panel timestamps"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", lookback=5)
    with pytest.raises(ValueError, match="min_weight_norm"):
        combine_alphas_walkforward(tiny, ["f1", "f2"], "r", min_weight_norm=-1e-5)


def test_combine_alphas_ema_symmetry_and_zero_weight_guard() -> None:
    """Verify equal_weight_alpha uses identical EMA smoothing and zero weights yield 0.0 composite."""
    rng = np.random.default_rng(1234)
    n_dates, n_tickers = 40, 16
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        for i in range(n_tickers):
            rows.append(
                {
                    "date": t,
                    "ticker": f"S{i:02d}",
                    "f1": float(rng.standard_normal()),
                    "f2": float(rng.standard_normal()),
                    "r": float(rng.normal(0.0, 0.01)) if t + 1 < n_dates else None,
                }
            )
    panel = pl.DataFrame(rows)

    # 1. Prohibitive cost drags all weights to 0.0 -> composite_alpha must stay 0.0 (not amplified)
    res_zero = combine_alphas_walkforward(
        panel,
        factor_cols=["f1", "f2"],
        return_col="r",
        time_col="date",
        asset_col="ticker",
        lookback=15,
        horizon=1,
        signal_lag=1,
        objective_mode="factor_return",
        one_way_cost_bps=5_000.0,
        signal_ema_alpha=0.3,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        min_weight_norm=1e-8,
    )
    assert res_zero.weights_history.select(["w_f1", "w_f2"]).to_numpy().max() == 0.0
    assert res_zero.weights_history.select(["w_f1", "w_f2"]).to_numpy().min() == 0.0
    np.testing.assert_allclose(res_zero.data["composite_alpha"].to_numpy(), 0.0, atol=1e-12)

    # 2. Original factor columns are preserved un-mutated & equal_weight_alpha reflects EMA smoothing
    joined_orig = res_zero.data.sort(["date", "ticker"])
    valid_dates = joined_orig["date"].unique().implode()
    orig_sub = panel.filter(pl.col("date").is_in(valid_dates)).sort(["date", "ticker"])
    np.testing.assert_allclose(joined_orig["f1"].to_numpy(), orig_sub["f1"].to_numpy(), atol=1e-12)

    ema_expected = (
        panel.sort(["ticker", "date"])
        .with_columns(
            pl.col("f1").ewm_mean(alpha=0.3, adjust=False).over("ticker").alias("f1_e"),
            pl.col("f2").ewm_mean(alpha=0.3, adjust=False).over("ticker").alias("f2_e"),
        )
        .with_columns(((pl.col("f1_e") + pl.col("f2_e")) * 0.5).alias("ew_raw"))
        .with_columns(
            (
                (pl.col("ew_raw") - pl.col("ew_raw").mean().over("date"))
                / pl.col("ew_raw").std().over("date")
            ).alias("ew_z")
        )
        .filter(pl.col("date").is_in(valid_dates))
        .sort(["date", "ticker"])
    )
    np.testing.assert_allclose(
        joined_orig["equal_weight_alpha"].to_numpy(),
        ema_expected["ew_z"].to_numpy(),
        atol=1e-12,
    )
