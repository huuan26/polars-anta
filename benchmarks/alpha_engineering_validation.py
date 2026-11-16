"""Mathematical invariant, RMT denoising, and Long-Short walk-forward ablation benchmark.

Generates ``docs/assets/alpha_engineering_validation.md`` from deterministic execution:
1. Exact numerical invariants: FWL simultaneous sector+risk null-space error, batched Lowdin
   orthogonality & permutation equivariance, joint FISTA KKT stationarity residual, and
   strictly causal h=5 Prefix-Invariance weight difference.
2. Monte Carlo covariance conditioning (R=100, T=80, K=24): Sample vs. Ledoit-Wolf vs.
   Marchenko-Pastur RMT Frobenius error and condition number against true factor covariance.
3. End-to-end 4-Stage Dollar-Neutral Long-Short Walk-Forward Ablation (N=150 assets, 10 sectors,
   T=750 trading days, K=8 crowded/noisy signals, 15 bps one-way execution cost).

Run from the repository root:
    python benchmarks/alpha_engineering_validation.py
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import polars as pl

import polars_anta as pa

SEED = 202605


def run_table1_invariants() -> dict[str, float]:
    """Verify machine-precision algebraic, KKT, and causal prefix-invariance bounds."""
    rng = np.random.default_rng(SEED)
    n_dates, n_assets = 50, 80
    sectors = ["Tech", "Fin", "Health", "Ind", "Cons"]

    rows: list[dict[str, object]] = []
    f_raw = rng.standard_normal((n_dates, n_assets, 4))
    for t in range(n_dates):
        beta = rng.normal(1.0, 0.35, size=n_assets)
        size = rng.normal(9.5, 1.5, size=n_assets)
        sec_i = rng.integers(0, len(sectors), size=n_assets)
        sec_shock = np.array([0.7, -0.5, 0.3, -0.4, 0.2])[sec_i]
        ret_5 = (
            0.10 * f_raw[t, :, 0] - 0.07 * f_raw[t, :, 2] + 0.4 * rng.standard_normal(n_assets)
            if t + 6 < n_dates
            else np.full(n_assets, np.nan)
        )
        for i in range(n_assets):
            r_val = None if np.isnan(ret_5[i]) else float(ret_5[i])
            rows.append(
                {
                    "date": t,
                    "ticker": f"EQ{i:03d}",
                    "sector": sectors[int(sec_i[i])],
                    "beta": float(beta[i]),
                    "log_mcap": float(size[i]),
                    "f1": float(f_raw[t, i, 0] + 1.4 * sec_shock[i] + 0.9 * beta[i]),
                    "f2": float(0.8 * f_raw[t, i, 0] + 0.4 * f_raw[t, i, 1] - 0.8 * size[i]),
                    "f3": float(f_raw[t, i, 2] - 1.1 * sec_shock[i] + 0.6 * beta[i]),
                    "f4": float(f_raw[t, i, 3] + 0.5 * size[i]),
                    "fwd_ret_5": r_val,
                }
            )
    panel = pl.DataFrame(rows)
    f_cols = ["f1", "f2", "f3", "f4"]

    # 1. FWL sector + continuous risk null-space max inner product
    neut = pa.neutralize_panel(
        panel, f_cols, time_col="date", risk_cols=["beta", "log_mcap"], group_col="sector"
    )
    neut_cols = [f"{c}_neut" for c in f_cols]
    max_fwl_err = 0.0
    for d in (0, 15, 35):
        sub = neut.filter(pl.col("date") == d)
        f_mat = sub.select(neut_cols).to_numpy()
        for s_name in sectors:
            d_vec = (sub["sector"] == s_name).to_numpy().astype(np.float64)
            max_fwl_err = max(max_fwl_err, float(np.max(np.abs(d_vec @ f_mat))))
        x_mat = sub.select(["beta", "log_mcap"]).to_numpy()
        max_fwl_err = max(max_fwl_err, float(np.max(np.abs(x_mat.T @ f_mat))))

    # 2. Batched Lowdin orthogonality & permutation equivariance
    ortho_a = pa.orthogonalize_panel(neut, neut_cols, time_col="date", method="lowdin")
    perm_cols = [neut_cols[2], neut_cols[0], neut_cols[3], neut_cols[1]]
    ortho_p = pa.orthogonalize_panel(neut, perm_cols, time_col="date", method="lowdin")

    sub_a = (
        ortho_a.filter(pl.col("date") == 10).select([f"{c}_ortho" for c in neut_cols]).to_numpy()
    )
    sub_p = (
        ortho_p.filter(pl.col("date") == 10).select([f"{c}_ortho" for c in perm_cols]).to_numpy()
    )
    gram_err = float(np.linalg.norm((sub_a.T @ sub_a) / (n_assets - 1) - np.eye(4), ord="fro"))
    perm_err = float(np.max(np.abs(sub_p - sub_a[:, [2, 0, 3, 1]])))

    # 3. Joint FISTA KKT stationarity residual under active L1 + box-simplex constraints
    a_mat = rng.standard_normal((6, 6))
    cov_kkt = (a_mat.T @ a_mat) / 6.0 + 0.15 * np.eye(6)
    mu_kkt = np.array([0.07, -0.02, 0.05, 0.01, 0.04, -0.03])
    w_prev = np.full(6, 1.0 / 6.0)
    opt_res = pa.solve_fista(
        mu_kkt,
        cov_kkt,
        risk_aversion=2.0,
        turnover_penalty=0.025,
        w_prev=w_prev,
        constraint="box_simplex",
        lower_bounds=0.0,
        upper_bounds=0.40,
        budget=1.0,
        tol=1e-11,
    )

    # 4. Causal Prefix-Invariance on h=5 overlapping forward returns
    wf_full = pa.combine_alphas_walkforward(
        ortho_a,
        factor_cols=[f"{c}_ortho" for c in neut_cols],
        return_col="fwd_ret_5",
        time_col="date",
        lookback=15,
        horizon=5,
        signal_lag=1,
        turnover_penalty=0.03,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    t_cut = 35
    live_cut = ortho_a.filter(pl.col("date") <= t_cut).with_columns(
        pl.when(pl.col("date") > t_cut - 6)
        .then(None)
        .otherwise(pl.col("fwd_ret_5"))
        .alias("fwd_ret_5")
    )
    wf_live = pa.combine_alphas_walkforward(
        live_cut,
        factor_cols=[f"{c}_ortho" for c in neut_cols],
        return_col="fwd_ret_5",
        time_col="date",
        lookback=15,
        horizon=5,
        signal_lag=1,
        turnover_penalty=0.03,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    w_cols = [f"w_{c}_ortho" for c in neut_cols]
    w_diff = float(
        np.max(
            np.abs(
                wf_full.weights_history.filter(pl.col("date") <= t_cut).select(w_cols).to_numpy()
                - wf_live.weights_history.select(w_cols).to_numpy()
            )
        )
    )

    return {
        "max_fwl_err": max_fwl_err,
        "lowdin_gram_err": gram_err,
        "lowdin_perm_err": perm_err,
        "fista_kkt_res": opt_res.kkt_residual,
        "prefix_leakage_diff": w_diff,
    }


def run_table2_rmt_monte_carlo() -> list[dict[str, float | str]]:
    """Monte Carlo comparison of correlation/covariance estimators (T=80, K=24, R=100)."""
    rng = np.random.default_rng(SEED + 100)
    t_obs, k_cols, n_reps = 80, 24, 100

    # True 3-block factor correlation structure
    loadings = np.zeros((k_cols, 3))
    loadings[:8, 0] = 0.75
    loadings[8:16, 1] = 0.70
    loadings[16:, 2] = 0.65
    c_true = loadings @ loadings.T + np.diag(1.0 - np.sum(loadings**2, axis=1))
    chol = np.linalg.cholesky(c_true)

    errs: dict[str, list[float]] = {"Sample": [], "Ledoit-Wolf": [], "Marchenko-Pastur (RMT)": []}
    conds: dict[str, list[float]] = {"Sample": [], "Ledoit-Wolf": [], "Marchenko-Pastur (RMT)": []}

    def _to_corr_and_cond(cov: np.ndarray) -> tuple[np.ndarray, float]:
        sd = np.sqrt(np.diag(cov))
        c = cov / np.outer(sd, sd)
        np.fill_diagonal(c, 1.0)
        eigs = np.linalg.eigvalsh(c)
        return c, float(np.max(eigs) / max(float(np.min(eigs)), 1e-12))

    for _ in range(n_reps):
        x = rng.standard_normal((t_obs, k_cols)) @ chol.T
        s_cov = pa.sample_covariance(x)
        lw_cov, _ = pa.ledoit_wolf_shrinkage(x, target="constant_correlation")
        mp_cov = pa.marchenko_pastur_denoise(x, method="constant_residual")

        for label, mat in [
            ("Sample", s_cov),
            ("Ledoit-Wolf", lw_cov),
            ("Marchenko-Pastur (RMT)", mp_cov),
        ]:
            c_hat, kappa = _to_corr_and_cond(mat)
            errs[label].append(float(np.linalg.norm(c_hat - c_true, ord="fro")))
            conds[label].append(kappa)

    return [
        {
            "estimator": name,
            "mean_fro_err": float(np.mean(errs[name])),
            "sd_fro_err": float(np.std(errs[name], ddof=1)),
            "mean_cond_num": float(np.mean(conds[name])),
        }
        for name in ("Sample", "Ledoit-Wolf", "Marchenko-Pastur (RMT)")
    ]


def run_table3_long_short_ablation() -> list[dict[str, float | str]]:
    """Dollar-neutral Long-Short multi-factor ablation (N=150, T=600, K=6, 10 bps one-way cost)."""
    rng = np.random.default_rng(SEED + 999)
    n_dates, n_assets = 600, 150
    n_sectors = 10
    cost_bps = 10.0

    sec_ids = np.arange(n_assets) % n_sectors
    sec_names = [f"S{s:02d}" for s in sec_ids]
    asset_betas = rng.uniform(0.6, 1.5, size=n_assets)

    a_slow = np.zeros((n_dates, n_assets))
    a_med = np.zeros((n_dates, n_assets))
    for t in range(1, n_dates):
        a_slow[t] = 0.94 * a_slow[t - 1] + np.sqrt(1.0 - 0.94**2) * rng.standard_normal(n_assets)
        a_med[t] = 0.80 * a_med[t - 1] + np.sqrt(1.0 - 0.80**2) * rng.standard_normal(n_assets)
    a_fast_noise = rng.standard_normal((n_dates, n_assets))

    mkt_ret = rng.normal(0.0003, 0.011, size=n_dates)
    sec_ret = rng.normal(0.0, 0.008, size=(n_dates, n_sectors))
    idio_ret = rng.normal(0.0, 0.015, size=(n_dates, n_assets))

    # Realistic cross-sectional alpha SNR (composite IC ~ 0.03-0.04, Gross Sharpe ~ 3-5 before costs)
    rets = np.zeros((n_dates, n_assets))
    for t in range(1, n_dates):
        rets[t] = (
            asset_betas * mkt_ret[t]
            + sec_ret[t, sec_ids]
            + 0.00085 * a_slow[t - 1]
            + 0.00060 * a_med[t - 1]
            + idio_ret[t]
        )
    closes = 100.0 * np.exp(np.cumsum(rets, axis=0))
    vols = 1_000_000.0 * np.exp(rng.normal(0.0, 0.3, size=(n_dates, n_assets)))

    sec_bias = rng.standard_normal((n_dates, n_sectors))
    f1 = (
        0.75 * a_slow
        + 1.6 * sec_bias[:, sec_ids]
        + 1.1 * asset_betas[None, :]
        + 0.55 * rng.standard_normal((n_dates, n_assets))
    )
    f2 = f1 + 0.20 * rng.standard_normal((n_dates, n_assets))
    f3 = f1 - 0.18 * rng.standard_normal((n_dates, n_assets))
    f4 = 0.70 * a_med - 1.4 * sec_bias[:, sec_ids] + 0.55 * rng.standard_normal((n_dates, n_assets))
    f5 = a_fast_noise + 1.1 * sec_bias[:, sec_ids]
    f6 = rng.standard_normal((n_dates, n_assets)) - 0.8 * sec_bias[:, sec_ids]

    panel = pl.DataFrame(
        {
            "date": np.repeat(np.arange(n_dates), n_assets),
            "ticker": np.tile([f"US{i:03d}" for i in range(n_assets)], n_dates),
            "sector": np.tile(sec_names, n_dates),
            "beta": np.tile(asset_betas, n_dates),
            "close": closes.ravel(),
            "volume": vols.ravel(),
            "s1": f1.ravel(),
            "s2": f2.ravel(),
            "s3": f3.ravel(),
            "s4": f4.ravel(),
            "s5": f5.ravel(),
            "s6": f6.ravel(),
        }
    )
    panel = pa.forward_returns(
        panel, close_col="close", periods=(1,), group_by="ticker", prefix="fwd_"
    )
    raw_cols = ["s1", "s2", "s3", "s4", "s5", "s6"]

    # Stage 0: Naive Equal-Weight of raw z-scored signals (evaluated on OOS window t >= 80)
    oos_start = 80
    s0_df = panel.with_columns(
        [pa.cs_zscore(c, over="date").alias(f"{c}_z") for c in raw_cols]
    ).with_columns(pl.mean_horizontal([pl.col(f"{c}_z") for c in raw_cols]).alias("raw_ew"))

    # Stage 1: Gaussianize + FWL Sector & Beta Neutralization + Batched Lowdin Orthogonalization
    g_df = panel.with_columns([pa.cs_gaussianize(c, over="date").alias(f"{c}_g") for c in raw_cols])
    n_df = pa.neutralize_panel(
        g_df, [f"{c}_g" for c in raw_cols], time_col="date", risk_cols=["beta"], group_col="sector"
    )
    neut_cols = [f"{c}_g_neut" for c in raw_cols]
    o_df = pa.orthogonalize_panel(n_df, neut_cols, time_col="date", method="lowdin")
    ortho_cols = [f"{c}_ortho" for c in neut_cols]
    o_df = o_df.with_columns(pl.mean_horizontal([pl.col(c) for c in ortho_cols]).alias("ortho_ew"))

    # Stage 2: Walk-Forward FISTA without turnover penalty & noisy rolling window (lookback=30)
    wf_unpen = pa.combine_alphas_walkforward(
        o_df,
        factor_cols=ortho_cols,
        return_col="fwd_1",
        time_col="date",
        lookback=30,
        horizon=1,
        signal_lag=1,
        risk_aversion=1.0,
        turnover_penalty=0.0,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        cov_estimator="sample",
        composite_col="wf_unpen",
    )

    # Stage 3: Full Pipeline -> Factor-Return RMT + Cost Drag + FISTA L1 Penalty + Causal Smoothing
    wf_full = pa.combine_alphas_walkforward(
        o_df,
        factor_cols=ortho_cols,
        return_col="fwd_1",
        time_col="date",
        asset_col="ticker",
        lookback=80,
        horizon=1,
        signal_lag=1,
        objective_mode="factor_return",
        one_way_cost_bps=cost_bps,
        signal_ema_alpha=0.35,
        risk_aversion=2.5,
        turnover_penalty=0.006,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        cov_estimator="marchenko_pastur",
        ic_shrinkage="empirical_bayes",
        no_trade_buffer=0.02,
        composite_col="wf_cost_aware",
    )

    results: list[dict[str, float | str]] = []
    configs = [
        ("0. Raw Signals (Equal-Weight)", s0_df.filter(pl.col("date") >= oos_start), "raw_ew"),
        (
            "1. FWL + Lowdin Ortho (Equal-Weight)",
            o_df.filter(pl.col("date") >= oos_start),
            "ortho_ew",
        ),
        (
            "2. Unpenalized Walk-Forward (Sample Cov)",
            wf_unpen.data.filter(pl.col("date") >= oos_start),
            "wf_unpen",
        ),
        (
            "3. Full Pipeline (RMT + Cost-Aware FISTA + EMA)",
            wf_full.data.filter(pl.col("date") >= oos_start),
            "wf_cost_aware",
        ),
    ]
    for label, frame, col in configs:
        diag = pa.analyze_alpha(
            frame,
            factor_col=col,
            return_col=None,
            close_col="close",
            volume_col="volume",
            asset_col="ticker",
            time_col="date",
            horizon=1,
            signal_lag=1,
            weight_scheme="dollar_neutral",
            one_way_cost_bps=cost_bps,
        )
        s = diag.summary()
        results.append(
            {
                "stage": label,
                "ic_mean": s["ic_mean"],
                "icir": s["icir"],
                "turnover": s["mean_turnover"],
                "gross_margin_bps": s["gross_margin_bps"],
                "breakeven_bps": s["breakeven_cost_bps"],
                "gross_sharpe": s["gross_sharpe"],
                "net_sharpe": s["net_sharpe"],
            }
        )
    return results


def main() -> None:
    t0 = time.perf_counter()
    inv = run_table1_invariants()
    rmt = run_table2_rmt_monte_carlo()
    abl = run_table3_long_short_ablation()
    elapsed = time.perf_counter() - t0

    lines = [
        "# Cross-Sectional Alpha Engineering - Mathematical & Ablation Validation",
        "",
        "Generated automatically by `benchmarks/alpha_engineering_validation.py`.",
        "",
        "## Environment",
        "",
        f"- **Python**: `{platform.python_version()}` (`{platform.system()} {platform.release()}`)",
        f"- **Polars**: `{pl.__version__}` | **NumPy**: `{np.__version__}` | **polars-anta**: `{pa.__version__}`",
        f"- **Random Seed**: `{SEED}` | **Total Benchmark Runtime**: `{elapsed:.2f} s`",
        "",
        "## Table 1 - Exact Algebraic, KKT & Causal Prefix-Invariance Checks",
        "",
        "| Mathematical Invariant | Observed Error / Residual | Acceptance Bound |",
        "| :--- | :---: | :---: |",
        f"| FWL Simultaneous Sector + Continuous Risk Exposure $\\max |\\mathbf{{X}}_t^T \\mathbf{{F}}_{{\\text{{neut}},t}}|$ | `{inv['max_fwl_err']:.2e}` | `< 1.00e-10` |",
        f"| Batched Lowdin Orthogonality $\\|\\widetilde{{\\mathbf{{F}}}}_t^T \\widetilde{{\\mathbf{{F}}}}_t / (N_t - 1) - \\mathbf{{I}}_K\\|_F$ | `{inv['lowdin_gram_err']:.2e}` | `< 1.00e-10` |",
        f"| Lowdin Permutation Equivariance $\\max |\\widetilde{{\\mathbf{{F}}}}_{{\\pi}} - \\widetilde{{\\mathbf{{F}}}}\\mathbf{{P}}|$ | `{inv['lowdin_perm_err']:.2e}` | `< 1.00e-11` |",
        f"| Joint Box-Simplex + $L_1$ FISTA KKT Stationarity Residual | `{inv['fista_kkt_res']:.2e}` | `< 1.00e-07` |",
        f"| Causal Embargo Prefix-Invariance Weight Leakage ($h=5, d=1$) | `{inv['prefix_leakage_diff']:.2e}` | `== 0.00e+00` |",
        "",
        "## Table 2 - Covariance Conditioning Monte Carlo ($T = 80, K = 24, R = 100$)",
        "",
        "| Estimator | Mean Frobenius Error vs. $\\mathbf{C}_{\\text{true}}$ | SD(Error) | Mean Condition Number $\\kappa_2(\\widehat{\\mathbf{C}})$ |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for r in rmt:
        lines.append(
            f"| {r['estimator']} | `{r['mean_fro_err']:.3f}` | `{r['sd_fro_err']:.3f}` | `{r['mean_cond_num']:.2f}` |"
        )

    lines.extend(
        [
            "",
            "## Table 3 - Controlled Synthetic Ablation: Planted Factor Structure ($N = 150, T = 600$, `10 bps` Cost)",
            "",
            "> **Note**: Controlled synthetic simulation isolating each algebraic and optimization mechanism under known planted factor SNR and sector/beta contamination. For real-market out-of-sample backtests across Global Crypto, US Equities, and Vietnam HOSE, see `docs/assets/real_market_alpha_engineering.md`.",
            "",
            "| Pipeline Stage | Rank IC | ICIR | Daily L1 Turnover | Gross Margin (`bps`) | Gross Sharpe | Net Sharpe (`10 bps`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in abl:
        lines.append(
            f"| {row['stage']} | `{row['ic_mean']:+.4f}` | `{row['icir']:+.3f}` | "
            f"`{row['turnover']:.3f}` | `{row['gross_margin_bps']:+.2f}` | "
            f"`{row['gross_sharpe']:+.2f}` | `{row['net_sharpe']:+.2f}` |"
        )
    lines.append("")

    out_path = (
        Path(__file__).resolve().parents[1] / "docs" / "assets" / "alpha_engineering_validation.md"
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
