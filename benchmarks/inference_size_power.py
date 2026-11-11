"""Monte Carlo size, FDR, power, PBO, and throughput validation for polars_anta.inference.

Reproduces three econometric validation tables plus a panel throughput benchmark:
1. Empirical Family-Wise Error Rate (FWER) under the global null (K=50, T=1000, reps=200,
   n_boot=499) for i.i.d. N(0, 1) and AR(1) (phi=0.3) return processes.
2. Empirical False Discovery Rate (FDR) and average statistical power with 5 true strategies
   (annualized Sharpe = 1.0, i.e. mu = 1/sqrt(252)) and 45 null strategies.
3. CSCV Probability of Backtest Overfitting (PBO) under a pure-noise strategy matrix vs.
   a matrix containing 1 dominant strategy.
4. End-to-end throughput of ``falsify_factors`` on a synthetic panel of 150 factors x
   2,500 periods x 100 assets (n_boot=1000).

Run from the repository root:
    python benchmarks/inference_size_power.py
"""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path

import numpy as np
import polars as pl

from polars_anta.evaluation import _newey_west_lrv, _nw_auto_lag
from polars_anta.inference import (
    adjust_pvalues,
    falsify_factors,
    pbo_cscv,
    romano_wolf,
)

SEED = 42
K_STRATEGIES = 50
T_OBS = 1000
N_REPS = 200
N_BOOT = 499
AR1_PHI = 0.30
ALPHA_FWER = 0.05
FDR_Q = 0.10


def _generate_ar1_matrix(
    rng: np.random.Generator,
    n_obs: int,
    k_cols: int,
    phi: float,
) -> np.ndarray:
    """Generate stationary AR(1) matrix with unit unconditional variance."""
    eps = rng.standard_normal((n_obs, k_cols))
    if abs(phi) < 1e-12:
        return eps
    scale = math.sqrt(1.0 - phi * phi)
    out = np.empty((n_obs, k_cols), dtype=np.float64)
    out[0] = eps[0]
    for t in range(1, n_obs):
        out[t] = phi * out[t - 1] + scale * eps[t]
    return out


def _ols_two_sided_pvalues(x_mat: np.ndarray) -> np.ndarray:
    """Two-sided asymptotic normal p-values using naive i.i.d. standard errors."""
    n_obs = x_mat.shape[0]
    means = x_mat.mean(axis=0)
    stds = np.std(x_mat, axis=0, ddof=1)
    t_stats = np.sqrt(float(n_obs)) * np.abs(means) / np.maximum(stds, 1e-15)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return np.array([math.erfc(float(t) * inv_sqrt2) for t in t_stats], dtype=np.float64)


def _hac_two_sided_pvalues(x_mat: np.ndarray) -> np.ndarray:
    """Two-sided asymptotic normal p-values using Newey-West HAC standard errors."""
    n_obs, k_cols = x_mat.shape
    p_vals = np.empty(k_cols, dtype=np.float64)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    nw_lag = _nw_auto_lag(n_obs)
    for k in range(k_cols):
        col = x_mat[:, k]
        mu = float(np.mean(col))
        lrv = _newey_west_lrv(col - mu, nw_lag)
        se = math.sqrt(lrv / n_obs) if lrv > 0.0 else 0.0
        t_abs = abs(mu / se) if se > 1e-15 else 0.0
        p_vals[k] = math.erfc(t_abs * inv_sqrt2)
    return p_vals


def run_table1_size(seed: int = SEED) -> dict[str, dict[str, float]]:
    """Run Monte Carlo global-null FWER experiment under i.i.d. and AR(1)."""
    results: dict[str, dict[str, float]] = {}

    for dgp_name, phi in [("iid_N(0,1)", 0.0), ("AR(1)_phi=0.3", AR1_PHI)]:
        rng = np.random.default_rng(seed if phi == 0.0 else seed + 10_000)
        counts = {
            "Unadjusted (OLS t)": 0,
            "Bonferroni (OLS t)": 0,
            "Holm (OLS t)": 0,
            "Bonferroni (HAC t)": 0,
            "Holm (HAC t)": 0,
            "Romano-Wolf (iid, mean_block=1)": 0,
            "Romano-Wolf (Stationary Bootstrap)": 0,
        }

        for rep in range(N_REPS):
            x_mat = _generate_ar1_matrix(rng, T_OBS, K_STRATEGIES, phi=phi)
            p_ols = _ols_two_sided_pvalues(x_mat)
            p_hac = _hac_two_sided_pvalues(x_mat)

            if np.any(p_ols < ALPHA_FWER):
                counts["Unadjusted (OLS t)"] += 1
            if np.any(adjust_pvalues(p_ols, method="bonferroni") < ALPHA_FWER):
                counts["Bonferroni (OLS t)"] += 1
            if np.any(adjust_pvalues(p_ols, method="holm") < ALPHA_FWER):
                counts["Holm (OLS t)"] += 1
            if np.any(adjust_pvalues(p_hac, method="bonferroni") < ALPHA_FWER):
                counts["Bonferroni (HAC t)"] += 1
            if np.any(adjust_pvalues(p_hac, method="holm") < ALPHA_FWER):
                counts["Holm (HAC t)"] += 1

            rw_iid = romano_wolf(
                x_mat,
                n_boot=N_BOOT,
                mean_block=1.0,
                alternative="two-sided",
                seed=seed + rep * 2,
            )
            if np.any(rw_iid["p_rw"].to_numpy() < ALPHA_FWER):
                counts["Romano-Wolf (iid, mean_block=1)"] += 1

            rw_sb = romano_wolf(
                x_mat,
                n_boot=N_BOOT,
                mean_block=None,
                alternative="two-sided",
                seed=seed + rep * 2 + 1,
            )
            if np.any(rw_sb["p_rw"].to_numpy() < ALPHA_FWER):
                counts["Romano-Wolf (Stationary Bootstrap)"] += 1

        results[dgp_name] = {k: v / float(N_REPS) for k, v in counts.items()}

    return results


def run_table2_fdr_power(seed: int = SEED + 50_000) -> dict[str, float]:
    """Run Monte Carlo FDR & Power experiment (5 true Sharpe~1.0 + 45 null strategies)."""
    rng = np.random.default_rng(seed)
    n_true = 5
    mu_true = 1.0 / math.sqrt(252.0)  # Annualized Sharpe = 1.0 on unit-variance daily returns

    fdr_bh_list: list[float] = []
    power_bh_list: list[float] = []
    power_rw_list: list[float] = []
    power_bonf_list: list[float] = []
    fwer_rw_null_list: list[float] = []
    fwer_bonf_null_list: list[float] = []

    for rep in range(N_REPS):
        # Equicorrelated factor zoo (rho = 0.50 across candidate strategies) + signal in first 5
        common = rng.standard_normal((T_OBS, 1))
        idio = rng.standard_normal((T_OBS, K_STRATEGIES))
        rho = 0.50
        x_mat = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * idio
        x_mat[:, :n_true] += mu_true

        rw_df = romano_wolf(
            x_mat,
            n_boot=N_BOOT,
            mean_block=None,
            alternative="two-sided",
            seed=seed + rep,
        )
        p_hac = _hac_two_sided_pvalues(x_mat)
        p_rw = rw_df["p_rw"].to_numpy()
        p_bh = adjust_pvalues(p_hac, method="bh")
        p_bonf = adjust_pvalues(p_hac, method="bonferroni")

        # BH at q = 0.10
        rej_bh = p_bh < FDR_Q
        n_rej_bh = int(np.sum(rej_bh))
        false_rej_bh = int(np.sum(rej_bh[n_true:]))
        fdr_bh_list.append(false_rej_bh / n_rej_bh if n_rej_bh > 0 else 0.0)
        power_bh_list.append(float(np.mean(rej_bh[:n_true])))

        # Romano-Wolf & Bonferroni at alpha = 0.05
        rej_rw = p_rw < ALPHA_FWER
        rej_bonf = p_bonf < ALPHA_FWER
        power_rw_list.append(float(np.mean(rej_rw[:n_true])))
        power_bonf_list.append(float(np.mean(rej_bonf[:n_true])))
        fwer_rw_null_list.append(1.0 if np.any(rej_rw[n_true:]) else 0.0)
        fwer_bonf_null_list.append(1.0 if np.any(rej_bonf[n_true:]) else 0.0)

    return {
        "bh_empirical_fdr_q10": float(np.mean(fdr_bh_list)),
        "bh_power_q10": float(np.mean(power_bh_list)),
        "rw_power_alpha05": float(np.mean(power_rw_list)),
        "bonf_power_alpha05": float(np.mean(power_bonf_list)),
        "rw_null_fwer_alpha05": float(np.mean(fwer_rw_null_list)),
        "bonf_null_fwer_alpha05": float(np.mean(fwer_bonf_null_list)),
    }


def run_table3_pbo(seed: int = SEED + 90_000) -> dict[str, float]:
    """Run Monte Carlo CSCV PBO experiment under pure noise vs. 1 dominant strategy."""
    rng = np.random.default_rng(seed)
    n_pbo_reps = 50
    n_strat = 20
    t_pbo = 1600
    n_slices = 16

    pbo_noise: list[float] = []
    pbo_dominant: list[float] = []
    deg_noise: list[float] = []
    deg_dominant: list[float] = []

    for _ in range(n_pbo_reps):
        mat_noise = rng.standard_normal((t_pbo, n_strat))
        res_noise = pbo_cscv(mat_noise, n_splits=n_slices)
        pbo_noise.append(float(res_noise["pbo"]))  # type: ignore[arg-type]
        deg_noise.append(float(res_noise["prob_oos_loss"]))  # type: ignore[arg-type]

        mat_dom = rng.standard_normal((t_pbo, n_strat))
        mat_dom[:, 0] += 0.12  # Dominant strategy with high genuine Sharpe
        res_dom = pbo_cscv(mat_dom, n_splits=n_slices)
        pbo_dominant.append(float(res_dom["pbo"]))  # type: ignore[arg-type]
        deg_dominant.append(float(res_dom["prob_oos_loss"]))  # type: ignore[arg-type]

    return {
        "noise_mean_pbo": float(np.mean(pbo_noise)),
        "noise_sd_pbo": float(np.std(pbo_noise, ddof=1)),
        "noise_prob_loss_oos": float(np.mean(deg_noise)),
        "dominant_mean_pbo": float(np.mean(pbo_dominant)),
        "dominant_sd_pbo": float(np.std(pbo_dominant, ddof=1)),
        "dominant_prob_loss_oos": float(np.mean(deg_dominant)),
    }


def run_throughput_benchmark(seed: int = SEED + 120_000) -> dict[str, float]:
    """Benchmark falsify_factors on 150 factors x 2,500 periods x 100 assets (n_boot=1000)."""
    rng = np.random.default_rng(seed)
    n_periods = 2500
    n_assets = 100
    n_factors = 150
    n_rows = n_periods * n_assets

    dates = np.repeat(np.arange(n_periods, dtype=np.int32), n_assets)
    fwd_ret = rng.standard_normal(n_rows) * 0.02
    cols: dict[str, np.ndarray] = {"date": dates, "fwd_ret": fwd_ret}
    factor_names: list[str] = []
    for k in range(n_factors):
        name = f"f_{k:03d}"
        factor_names.append(name)
        if k < 5:
            cols[name] = 0.25 * fwd_ret + rng.standard_normal(n_rows) * 0.02
        else:
            cols[name] = rng.standard_normal(n_rows)

    df = pl.DataFrame(cols)
    t0 = time.perf_counter()
    res = falsify_factors(
        df,
        factor_cols=factor_names,
        return_col="fwd_ret",
        group_by="date",
        n_boot=1000,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return {
        "n_factors": float(n_factors),
        "n_periods": float(n_periods),
        "n_assets": float(n_assets),
        "n_rows": float(n_rows),
        "n_boot": 1000.0,
        "elapsed_sec": float(elapsed),
        "n_robust": float(res.summary["n_robust"]),
    }


def format_markdown(
    t1: dict[str, dict[str, float]],
    t2: dict[str, float],
    t3: dict[str, float],
    tp: dict[str, float],
    total_elapsed: float,
) -> str:
    """Format Monte Carlo validation results as a Markdown document."""
    iid = t1["iid_N(0,1)"]
    ar1 = t1["AR(1)_phi=0.3"]

    lines = [
        "# Statistical Inference Engine — Monte Carlo Size, Power & PBO Validation",
        "",
        "Generated automatically by `benchmarks/inference_size_power.py`.",
        "",
        "## Simulation Parameters & Environment",
        "",
        f"- **Python**: `{platform.python_version()}` (`{platform.system()} {platform.release()}`)",
        f"- **Polars**: `{pl.__version__}` | **NumPy**: `{np.__version__}`",
        f"- **Hypotheses ($K$)**: `{K_STRATEGIES}` | **Sample Length ($T$)**: `{T_OBS}`",
        f"- **Monte Carlo Replications**: `{N_REPS}` | **Bootstrap Replicates ($B$)**: `{N_BOOT}`",
        f"- **Nominal FWER ($\\alpha$)**: `{ALPHA_FWER:.2f}` | **Target FDR ($q$)**: `{FDR_Q:.2f}`",
        f"- **Random Seed**: `{SEED}` | **Total Benchmark Runtime**: `{total_elapsed:.2f} s`",
        "",
        "## Table 1 — Empirical Family-Wise Error Rate (Global Null, $\\alpha = 0.05$)",
        "",
        "Fraction of Monte Carlo replications ($R = 200$) with at least one false rejection across",
        "$K = 50$ pure-noise strategies ($T = 1000$). Under $\\text{AR}(1)$ serial dependence",
        "($\\phi = 0.30$), the long-run variance is $\\frac{1+\\phi}{1-\\phi} \\approx 1.86\\times$ the",
        "naive sample variance, causing i.i.d. procedures (`OLS t` and `mean_block=1` bootstrap) to",
        "severely over-reject while HAC and the Politis–Romano Stationary Bootstrap preserve size",
        "near the nominal $5\\%$ level.",
        "",
        "| Multiple Testing Procedure | Empirical FWER — i.i.d. $\\mathcal{N}(0,1)$ | Empirical FWER — $\\text{AR}(1)$ ($\\phi = 0.30$) | Nominal Target |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for method_name in iid:
        lines.append(
            f"| {method_name} | {iid[method_name] * 100:.1f}% | {ar1[method_name] * 100:.1f}% | 5.0% |"
        )

    lines.extend(
        [
            "",
            "## Table 2 — Empirical FDR & Statistical Power (5 True + 45 Null Strategies)",
            "",
            "Equicorrelated factor zoo ($\\rho = 0.50$, $K = 50$, $T = 1000$) containing $5$ true",
            "strategies with annualized Sharpe $= 1.0$ ($\\mu = 1/\\sqrt{252} \\approx 0.063$) and $45$",
            "zero-mean strategies.",
            "",
            "| Metric | Value | Reference Target |",
            "| :--- | :---: | :---: |",
            f"| Benjamini–Hochberg ($q = 0.10$) Empirical FDR | {t2['bh_empirical_fdr_q10'] * 100:.1f}% | $\\le 10.0\\%$ ($\\frac{{m_0}}{{m}} q = 9.0\\%$) |",
            f"| Benjamini–Hochberg ($q = 0.10$) Average Power | {t2['bh_power_q10'] * 100:.1f}% | — |",
            f"| Romano–Wolf Stepdown ($\\alpha = 0.05$) Average Power | {t2['rw_power_alpha05'] * 100:.1f}% | $\\ge$ Bonferroni |",
            f"| Bonferroni ($\\alpha = 0.05$) Average Power | {t2['bonf_power_alpha05'] * 100:.1f}% | — |",
            f"| Romano–Wolf Null-Subset FWER ($\\alpha = 0.05$) | {t2['rw_null_fwer_alpha05'] * 100:.1f}% | $\\le 5.0\\%$ |",
            f"| Bonferroni Null-Subset FWER ($\\alpha = 0.05$) | {t2['bonf_null_fwer_alpha05'] * 100:.1f}% | $\\le 5.0\\%$ |",
            "",
            "## Table 3 — CSCV Probability of Backtest Overfitting (PBO)",
            "",
            "Exact Combinatorially Symmetric Cross-Validation ($S = 16$, $\\binom{16}{8} = 12,870$",
            "partitions) over $N = 20$ candidate strategies ($T = 1600$, 50 replications).",
            "",
            "| Strategy Universe | Mean PBO | SD(PBO) | Mean Prob. OOS Loss | Theoretical Expectation |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| Pure Noise ($N = 20$ i.i.d. $\\mathcal{{N}}(0,1)$) | {t3['noise_mean_pbo']:.3f} | {t3['noise_sd_pbo']:.3f} | {t3['noise_prob_loss_oos']:.3f} | $\\approx 0.500$ |",
            f"| 1 Dominant Strategy ($\\mu_1 = 0.12$) + 19 Noise | {t3['dominant_mean_pbo']:.3f} | {t3['dominant_sd_pbo']:.3f} | {t3['dominant_prob_loss_oos']:.3f} | $< 0.050$ |",
            "",
            "## Table 4 — End-to-End Panel Falsification Throughput",
            "",
            "| Panel Dimensions | Rows | Bootstrap Replicates ($B$) | Wall-Clock Time | Surviving Robust Factors |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| `{int(tp['n_factors'])}` factors $\\times$ `{int(tp['n_periods'])}` periods $\\times$ `{int(tp['n_assets'])}` assets | `{int(tp['n_rows']):,}` | `{int(tp['n_boot'])}` | `{tp['elapsed_sec']:.2f} s` | `{int(tp['n_robust'])} / 5` planted |",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    t_start = time.perf_counter()
    print("Running Table 1: Global-null size (i.i.d. vs AR(1))...")
    t1 = run_table1_size()
    print("Running Table 2: Empirical FDR and Power...")
    t2 = run_table2_fdr_power()
    print("Running Table 3: CSCV PBO validation...")
    t3 = run_table3_pbo()
    print("Running Table 4: Panel falsify_factors throughput (150 x 2,500 x 100)...")
    tp = run_throughput_benchmark()
    total_elapsed = time.perf_counter() - t_start

    md = format_markdown(t1, t2, t3, tp, total_elapsed)
    out_path = (
        Path(__file__).resolve().parent.parent / "docs" / "assets" / "inference_size_power.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved validation report to {out_path}")


if __name__ == "__main__":
    main()
