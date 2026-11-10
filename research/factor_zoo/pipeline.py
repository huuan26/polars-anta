"""Confirmatory discovery and locked-holdout evaluation pipeline for the Factor Zoo study."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

import polars_anta as pa
from polars_anta.evaluation import _newey_west_lrv
from polars_anta.inference import (
    adjust_pvalues,
    bootstrap_test,
    breakeven_cost,
    cusum_mean_test,
    falsify_factors,
    ic_matrix,
    romano_wolf,
)
from research.factor_zoo.config import DEFAULT_CONFIG, REPO_ROOT, RESULTS_DIR, StudyConfig
from research.factor_zoo.data import load_panel
from research.factor_zoo.registry import (
    EXCLUDED_SIGNALS,
    SignalSpec,
    build_registry,
    compute_signals_panel,
    family_counts,
    registry_sha256,
)


def _compute_participation_ratio(ic_df: pl.DataFrame, factor_cols: list[str]) -> dict[str, float]:
    """Compute eigenvalue participation ratio N_eff = (sum lambda)^2 / sum(lambda^2)."""
    mat = ic_df.select(factor_cols).to_numpy()
    # Standardize columns and compute correlation matrix
    stds = np.nanstd(mat, axis=0, ddof=1)
    valid = stds > 1e-12
    mat_valid = mat[:, valid]
    corr = np.corrcoef(mat_valid, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.maximum(eigvals, 0.0)
    sum_l = float(np.sum(eigvals))
    sum_l2 = float(np.sum(eigvals * eigvals))
    n_eff = (sum_l * sum_l) / sum_l2 if sum_l2 > 0.0 else 0.0
    top5_share = float(np.sum(np.sort(eigvals)[::-1][:5]) / sum_l) if sum_l > 0.0 else 0.0
    return {
        "n_factors": float(len(factor_cols)),
        "n_eff": float(n_eff),
        "top1_eigenvalue": float(np.max(eigvals)),
        "top5_variance_share": top5_share,
    }


def _evaluate_long_only_cost_stage(
    panel_df: pl.DataFrame,
    candidates: list[tuple[str, float, str, float, float]],
    config: StudyConfig,
    ret_col: str = "fwd_ret_5",
) -> pl.DataFrame:
    """Evaluate 5-day rebalanced long-only top-quintile portfolios vs equal-weight universe.

    :param candidates: List of ``(factor_name, disc_ic_mean, verdict, p_rw, q_bh)``.
    """
    if not candidates:
        return pl.DataFrame(
            schema={
                "factor": pl.String,
                "verdict_discovery": pl.String,
                "p_rw_discovery": pl.Float64,
                "q_bh_discovery": pl.Float64,
                "ic_direction": pl.Float64,
                "n_rebalance_periods": pl.Int64,
                "gross_excess_mean_5d": pl.Float64,
                "l1_turnover_mean_5d": pl.Float64,
                "one_way_turnover_mean_5d": pl.Float64,
                "net_excess_mean_5d": pl.Float64,
                "net_excess_ann_pct": pl.Float64,
                "net_ir_ann": pl.Float64,
                "p_boot_greater": pl.Float64,
                "q_cost_bh": pl.Float64,
                "breakeven_bps_l1": pl.Float64,
                "breakeven_bps_roundtrip": pl.Float64,
                "survives_cost_raw_p05": pl.Boolean,
                "survives_rw_and_cost_raw_p05": pl.Boolean,
                "survives_cost": pl.Boolean,
                "survives_rw_and_cost": pl.Boolean,
            }
        )

    dates = panel_df["date"].unique().sort()
    rebal_dates = dates.gather_every(config.rebalance_every)
    rebal_df = panel_df.filter(pl.col("date").is_in(rebal_dates)).sort(["date", "ticker"])

    # Pivot ticker weights/returns on rebalance dates
    all_tickers = sorted(rebal_df["ticker"].unique().to_list())
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
    rebal_date_list = rebal_df["date"].unique().sort().to_list()
    date_to_idx = {d: i for i, d in enumerate(rebal_date_list)}
    n_T = len(rebal_date_list)
    n_N = len(all_tickers)

    date_idx_arr = np.array([date_to_idx[d] for d in rebal_df["date"].to_list()], dtype=np.int64)
    tick_idx_arr = np.array(
        [ticker_to_idx[t] for t in rebal_df["ticker"].to_list()], dtype=np.int64
    )
    ret_arr = rebal_df[ret_col].to_numpy()

    ret_mat = np.full((n_T, n_N), np.nan, dtype=np.float64)
    ret_mat[date_idx_arr, tick_idx_arr] = ret_arr
    univ_ret = np.nanmean(ret_mat, axis=1)

    l1_fee_rate = config.round_trip_cost_rate  # 0.0030 per L1 unit (0.0060 per 1-way replacement)
    ann_scale = 252.0 / float(config.rebalance_every)

    rows: list[dict[str, object]] = []
    for idx_k, (f_name, ic_mean, verdict, p_rw, q_bh) in enumerate(candidates):
        direction = 1.0 if ic_mean >= 0.0 else -1.0
        f_vals = rebal_df[f_name].to_numpy()
        f_mat = np.full((n_T, n_N), np.nan, dtype=np.float64)
        f_mat[date_idx_arr, tick_idx_arr] = f_vals * direction

        gross_excess = np.zeros(n_T, dtype=np.float64)
        l1_turnover = np.zeros(n_T, dtype=np.float64)
        prev_w = np.zeros(n_N, dtype=np.float64)
        valid_steps = np.zeros(n_T, dtype=bool)

        for t in range(n_T):
            row_f = f_mat[t]
            row_r = ret_mat[t]
            mask = np.isfinite(row_f) & np.isfinite(row_r)
            n_valid = int(np.sum(mask))
            if n_valid < config.n_quantiles or not np.isfinite(univ_ret[t]):
                continue
            cutoff = np.quantile(row_f[mask], 1.0 - 1.0 / float(config.n_quantiles))
            sel = mask & (row_f >= cutoff)
            n_sel = int(np.sum(sel))
            if n_sel == 0:
                continue
            w = np.zeros(n_N, dtype=np.float64)
            w[sel] = 1.0 / float(n_sel)
            l1_turn = float(np.sum(np.abs(w - prev_w))) if np.any(prev_w > 0) else 1.0
            q5_ret = float(np.sum(w[sel] * row_r[sel]))
            gross_excess[t] = q5_ret - float(univ_ret[t])
            l1_turnover[t] = l1_turn
            prev_w = w
            valid_steps[t] = True

        g_ex = gross_excess[valid_steps]
        turn_l1 = l1_turnover[valid_steps]
        net_ex = g_ex - l1_fee_rate * turn_l1

        boot = bootstrap_test(
            net_ex,
            statistic="mean",
            alternative="greater",
            n_boot=config.n_boot,
            seed=config.seed + 500 + idx_k,
        )
        be_rate = float(breakeven_cost(g_ex, turn_l1))
        be_bps_l1 = be_rate * 10_000.0
        net_mean = float(np.mean(net_ex))
        net_std = float(np.std(net_ex, ddof=1)) if net_ex.size > 1 else 0.0
        net_ir_ann = (net_mean / net_std) * math.sqrt(ann_scale) if net_std > 1e-12 else 0.0

        rows.append(
            {
                "factor": f_name,
                "verdict_discovery": verdict,
                "p_rw_discovery": float(p_rw),
                "q_bh_discovery": float(q_bh),
                "ic_direction": direction,
                "n_rebalance_periods": int(net_ex.size),
                "gross_excess_mean_5d": float(np.mean(g_ex)),
                "l1_turnover_mean_5d": float(np.mean(turn_l1)),
                "one_way_turnover_mean_5d": float(0.5 * np.mean(turn_l1)),
                "net_excess_mean_5d": net_mean,
                "net_excess_ann_pct": float(net_mean * ann_scale * 100.0),
                "net_ir_ann": float(net_ir_ann),
                "p_boot_greater": float(boot["p_value"]),
                "breakeven_bps_l1": float(be_bps_l1),
                "breakeven_bps_roundtrip": float(be_bps_l1 * 2.0),
            }
        )

    cost_df = pl.DataFrame(rows)
    p_vec = cost_df["p_boot_greater"].to_numpy()
    q_cost = adjust_pvalues(p_vec, method="bh")
    cost_df = cost_df.with_columns(
        pl.Series("q_cost_bh", q_cost, dtype=pl.Float64),
    ).with_columns(
        (
            pl.col("verdict_discovery").is_in(["robust", "fdr_only"])
            & (pl.col("net_excess_mean_5d") > 0.0)
            & (pl.col("p_boot_greater") <= config.cost_alpha)
            & (pl.col("breakeven_bps_l1") > l1_fee_rate * 10_000.0)
        ).alias("survives_cost_raw_p05"),
        (
            (pl.col("verdict_discovery") == "robust")
            & (pl.col("net_excess_mean_5d") > 0.0)
            & (pl.col("p_boot_greater") <= config.cost_alpha)
            & (pl.col("breakeven_bps_l1") > l1_fee_rate * 10_000.0)
        ).alias("survives_rw_and_cost_raw_p05"),
        (
            pl.col("verdict_discovery").is_in(["robust", "fdr_only"])
            & (pl.col("net_excess_mean_5d") > 0.0)
            & (pl.col("p_boot_greater") <= config.cost_alpha)
            & (pl.col("q_cost_bh") <= config.fdr_q)
            & (pl.col("breakeven_bps_l1") > l1_fee_rate * 10_000.0)
        ).alias("survives_cost"),
        (
            (pl.col("verdict_discovery") == "robust")
            & (pl.col("net_excess_mean_5d") > 0.0)
            & (pl.col("p_boot_greater") <= config.cost_alpha)
            & (pl.col("q_cost_bh") <= config.fdr_q)
            & (pl.col("breakeven_bps_l1") > l1_fee_rate * 10_000.0)
        ).alias("survives_rw_and_cost"),
    )
    return cost_df


def _run_robustness_checks(
    disc_df: pl.DataFrame,
    factor_cols: list[str],
    top_factors: list[str],
    config: StudyConfig,
) -> dict[str, object]:
    """Run sub-period, liquidity-tier, and non-limit-day robustness checks on discovery."""
    mid_date = pl.lit(config.discovery_start).dt.offset_by("4y")
    first_half = disc_df.filter(pl.col("date") < mid_date)
    second_half = disc_df.filter(pl.col("date") >= mid_date)

    ic_h1 = ic_matrix(first_half, factor_cols, "fwd_ret_5", group_by="date")
    ic_h2 = ic_matrix(second_half, factor_cols, "fwd_ret_5", group_by="date")

    means_h1 = np.array([float(ic_h1[c].drop_nulls().mean() or 0.0) for c in factor_cols])
    means_h2 = np.array([float(ic_h2[c].drop_nulls().mean() or 0.0) for c in factor_cols])
    split_corr = float(np.corrcoef(means_h1, means_h2)[0, 1])

    # Top-50 vs Rank 51-150 liquidity tiers
    ranked_liq = disc_df.with_columns(
        pl.col("adv_60").rank(method="ordinal", descending=True).over("date").alias("_r_liq")
    )
    top50_df = ranked_liq.filter(pl.col("_r_liq") <= 50)
    mid100_df = ranked_liq.filter(pl.col("_r_liq") > 50)

    ic_top50 = ic_matrix(top50_df, top_factors, "fwd_ret_5", group_by="date")
    ic_mid100 = ic_matrix(mid100_df, top_factors, "fwd_ret_5", group_by="date")

    # Exclude stocks near daily price limit (|close/open - 1| >= 0.065)
    no_limit_df = disc_df.filter((pl.col("close") / pl.col("open") - 1.0).abs() < 0.065)
    ic_nolimit = ic_matrix(no_limit_df, top_factors, "fwd_ret_5", group_by="date")

    factor_breakdown: list[dict[str, object]] = []
    for f_name in top_factors:
        idx = factor_cols.index(f_name)
        factor_breakdown.append(
            {
                "factor": f_name,
                "ic_2014_2017": float(means_h1[idx]),
                "ic_2018_2021": float(means_h2[idx]),
                "ic_top50_liquid": float(ic_top50[f_name].drop_nulls().mean() or 0.0),
                "ic_mid51_150_liquid": float(ic_mid100[f_name].drop_nulls().mean() or 0.0),
                "ic_excluding_limit_days": float(ic_nolimit[f_name].drop_nulls().mean() or 0.0),
            }
        )

    return {
        "split_half_ic_correlation_all_150": split_corr,
        "top_factors_breakdown": factor_breakdown,
    }


def run_discovery(
    config: StudyConfig = DEFAULT_CONFIG,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, object]:
    """Execute the complete pre-registered Discovery pipeline (2014-01-01 to 2021-12-31)."""
    t0 = time.perf_counter()
    panel_df, data_summary = load_panel(config=config)
    specs: tuple[SignalSpec, ...] = build_registry()
    sig_panel = compute_signals_panel(panel_df, specs=specs)

    disc_df = sig_panel.filter(
        (pl.col("date") >= config.discovery_start) & (pl.col("date") <= config.discovery_end)
    )
    factor_cols = [s.name for s in specs]
    spec_map = {s.name: s for s in specs}

    # Confirmatory h=5 falsification funnel
    res_h5 = falsify_factors(
        disc_df,
        factor_cols=factor_cols,
        return_col=f"fwd_ret_{config.primary_horizon}",
        group_by="date",
        horizon=config.primary_horizon,
        method=config.ic_method,  # type: ignore[arg-type]
        n_boot=config.n_boot,
        fdr_q=config.fdr_q,
        fwer_alpha=config.fwer_alpha,
        prior=config.eb_prior,  # type: ignore[arg-type]
        seed=config.seed,
    )

    table = res_h5.table.with_columns(
        pl.col("factor").replace_strict({k: v.family for k, v in spec_map.items()}).alias("family"),
        pl.col("factor")
        .replace_strict({k: v.transform for k, v in spec_map.items()})
        .alias("transform"),
    )

    # Participation ratio N_eff on discovery IC matrix
    eff_info = _compute_participation_ratio(res_h5.ic_series, factor_cols)

    # Identify statistical survivors per PREREGISTRATION §4 (verdict in {"robust", "fdr_only"})
    stat_survivors = table.filter(pl.col("verdict").is_in(["robust", "fdr_only"])).sort("p_rw")
    candidates = [
        (
            str(r["factor"]),
            float(r["ic_mean"]),
            str(r["verdict"]),
            float(r["p_rw"]),
            float(r["q_bh"]),
        )
        for r in stat_survivors.iter_rows(named=True)
    ]

    cost_df = _evaluate_long_only_cost_stage(
        disc_df,
        candidates=candidates,
        config=config,
        ret_col=f"fwd_ret_{config.primary_horizon}",
    )

    # Robustness checks on top 10 factors by |nw_t|
    top10_factors = (
        table.with_columns(pl.col("nw_t").abs().alias("_abs_t"))
        .sort("_abs_t", descending=True)
        .head(10)["factor"]
        .to_list()
    )
    robustness_info = _run_robustness_checks(disc_df, factor_cols, top10_factors, config)

    # Exploratory horizons h=1 and h=20
    exploratory_summary: dict[str, dict[str, object]] = {}
    for h_exp in config.exploratory_horizons:
        res_exp = falsify_factors(
            disc_df,
            factor_cols=factor_cols,
            return_col=f"fwd_ret_{h_exp}",
            group_by="date",
            horizon=h_exp,
            method=config.ic_method,  # type: ignore[arg-type]
            n_boot=config.n_boot,
            fdr_q=config.fdr_q,
            fwer_alpha=config.fwer_alpha,
            prior=config.eb_prior,  # type: ignore[arg-type]
            seed=config.seed + h_exp,
        )
        t_exp = res_exp.table
        top5_exp = (
            t_exp.sort("p_rw")
            .head(5)
            .select(["factor", "ic_mean", "nw_t", "p_raw", "q_bh", "p_rw", "verdict"])
            .to_dicts()
        )
        exploratory_summary[f"h_{h_exp}"] = {
            "horizon": h_exp,
            "n_tested": int(res_exp.summary["n_tested"]),
            "n_raw_p05": int((t_exp["p_raw"] <= 0.05).sum()),
            "n_nw_t196": int((t_exp["nw_t"].abs() >= 1.96).sum()),
            "n_bh_q10": int((t_exp["q_bh"] <= config.fdr_q).sum()),
            "n_rw_p05": int((t_exp["p_rw"] <= config.fwer_alpha).sum()),
            "n_robust": int(res_exp.summary["n_robust"]),
            "n_fdr_only": int(res_exp.summary["n_fdr_only"]),
            "n_decaying": int(res_exp.summary["n_decaying"]),
            "n_noise": int(res_exp.summary["n_noise"]),
            "top5_factors": top5_exp,
        }

    # Family breakdown in discovery
    fam_breakdown: list[dict[str, object]] = []
    for fam_name, fam_cnt in family_counts(specs).items():
        sub = table.filter(pl.col("family") == fam_name)
        fam_breakdown.append(
            {
                "family": fam_name,
                "n_tested": fam_cnt,
                "n_raw_p05": int((sub["p_raw"] <= 0.05).sum()),
                "n_bh_q10": int((sub["q_bh"] <= config.fdr_q).sum()),
                "n_rw_p05": int((sub["p_rw"] <= config.fwer_alpha).sum()),
                "n_robust": int((sub["verdict"] == "robust").sum()),
                "n_fdr_only": int((sub["verdict"] == "fdr_only").sum()),
                "n_decaying": int((sub["verdict"] == "decaying").sum()),
                "median_abs_ic": float(sub["ic_mean"].abs().median() or 0.0),
                "median_abs_posterior_ic": float(sub["posterior_ic"].abs().median() or 0.0),
            }
        )

    n_tested = len(factor_cols)
    n_raw_p05 = int((table["p_raw"] <= 0.05).sum())
    n_nw_t196 = int((table["nw_t"].abs() >= 1.96).sum())
    n_bh_q10 = int((table["q_bh"] <= config.fdr_q).sum())
    n_rw_p05 = int((table["p_rw"] <= config.fwer_alpha).sum())
    n_robust = int(res_h5.summary["n_robust"])
    n_fdr_only = int(res_h5.summary["n_fdr_only"])
    n_decaying = int(res_h5.summary["n_decaying"])
    n_noise = int(res_h5.summary["n_noise"])
    n_positive_net_excess = (
        int((cost_df["net_excess_mean_5d"] > 0.0).sum()) if cost_df.height > 0 else 0
    )
    n_cost_survivors_raw_p05 = (
        int(cost_df["survives_cost_raw_p05"].sum()) if cost_df.height > 0 else 0
    )
    n_cost_survivors_rw_raw_p05 = (
        int(cost_df["survives_rw_and_cost_raw_p05"].sum()) if cost_df.height > 0 else 0
    )
    n_cost_survivors_fdr = int(cost_df["survives_cost"].sum()) if cost_df.height > 0 else 0
    n_cost_survivors_rw = int(cost_df["survives_rw_and_cost"].sum()) if cost_df.height > 0 else 0

    elapsed = time.perf_counter() - t0
    disc_dir = results_dir / "discovery"
    disc_dir.mkdir(parents=True, exist_ok=True)

    table.sort("p_rw").write_csv(disc_dir / "factor_table.csv")
    cost_df.write_csv(disc_dir / "cost_stage.csv")

    summary_payload: dict[str, object] = {
        "stage": "discovery",
        "registry_sha256": registry_sha256(specs),
        "seed": config.seed,
        "n_boot": config.n_boot,
        "primary_horizon": config.primary_horizon,
        "discovery_start": str(config.discovery_start),
        "discovery_end": str(config.discovery_end),
        "environment": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "numpy": np.__version__,
            "polars_anta": pa.__version__,
            "elapsed_sec": round(elapsed, 2),
        },
        "data_summary": data_summary,
        "effective_independent_tests": eff_info,
        "funnel": {
            "n_tested": n_tested,
            "expected_false_positives_at_5pct": round(0.05 * n_tested, 1),
            "n_raw_p05": n_raw_p05,
            "n_nw_t196": n_nw_t196,
            "n_bh_q10": n_bh_q10,
            "n_rw_p05": n_rw_p05,
            "n_verdict_robust": n_robust,
            "n_verdict_fdr_only": n_fdr_only,
            "n_verdict_decaying": n_decaying,
            "n_verdict_noise": n_noise,
            "n_cost_candidates_evaluated": int(cost_df.height),
            "n_positive_net_excess": n_positive_net_excess,
            "n_survive_costs_raw_p05": n_cost_survivors_raw_p05,
            "n_survive_costs_rw_raw_p05": n_cost_survivors_rw_raw_p05,
            "n_survive_costs_fdr": n_cost_survivors_fdr,
            "n_survive_costs_rw": n_cost_survivors_rw,
        },
        "family_breakdown": fam_breakdown,
        "robustness": robustness_info,
        "exploratory_horizons": exploratory_summary,
    }
    (disc_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    return summary_payload


def verify_preregistration_guard(repo_root: Path = REPO_ROOT) -> str:
    """Enforce that PREREGISTRATION.md is committed and frozen files have no dirty edits."""
    log_proc = subprocess.run(
        ["git", "log", "-n", "1", "--format=%H", "--", "research/factor_zoo/PREREGISTRATION.md"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_hash = log_proc.stdout.strip()
    if not commit_hash:
        raise RuntimeError(
            "Holdout guard failed: research/factor_zoo/PREREGISTRATION.md has not been committed to git!"
        )

    status_proc = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "research/factor_zoo/PREREGISTRATION.md",
            "research/factor_zoo/config.py",
            "research/factor_zoo/registry.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = status_proc.stdout.strip()
    if dirty:
        raise RuntimeError(
            f"Holdout guard failed: frozen pre-registration files have uncommitted changes:\n{dirty}"
        )
    return commit_hash


def run_holdout(
    unlock_holdout: bool = False,
    config: StudyConfig = DEFAULT_CONFIG,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, object]:
    """Execute the locked Holdout evaluation (2022-01-01 to end of sample)."""
    if not unlock_holdout:
        raise RuntimeError(
            "Holdout evaluation is locked. Pass --unlock-holdout after pre-registration is committed."
        )
    prereg_commit = verify_preregistration_guard()

    disc_summary_path = results_dir / "discovery" / "summary.json"
    disc_table_path = results_dir / "discovery" / "factor_table.csv"
    disc_cost_path = results_dir / "discovery" / "cost_stage.csv"
    if not (disc_summary_path.exists() and disc_table_path.exists() and disc_cost_path.exists()):
        raise RuntimeError("Discovery results not found. Run --stage discovery first.")

    t0 = time.perf_counter()
    disc_summary = json.loads(disc_summary_path.read_text(encoding="utf-8"))
    disc_table = pl.read_csv(disc_table_path)
    disc_cost = pl.read_csv(disc_cost_path)

    panel_df, _ = load_panel(config=config)
    specs = build_registry()
    sig_panel = compute_signals_panel(panel_df, specs=specs)

    full_study_df = sig_panel.filter(pl.col("date") >= config.discovery_start)
    hold_df = sig_panel.filter(pl.col("date") >= config.holdout_start)
    factor_cols = [s.name for s in specs]

    # Compute holdout and full-sample IC matrices
    ic_hold = ic_matrix(
        hold_df,
        factor_cols=factor_cols,
        return_col=f"fwd_ret_{config.primary_horizon}",
        group_by="date",
        method=config.ic_method,  # type: ignore[arg-type]
    )
    ic_full = ic_matrix(
        full_study_df,
        factor_cols=factor_cols,
        return_col=f"fwd_ret_{config.primary_horizon}",
        group_by="date",
        method=config.ic_method,  # type: ignore[arg-type]
    )

    # Evaluate discovery survivors (verdict in {"robust", "fdr_only"}) in holdout
    survivor_df = disc_table.filter(pl.col("verdict").is_in(["robust", "fdr_only"])).sort("p_rw")
    survivor_names = survivor_df["factor"].to_list()

    if survivor_names:
        rw_hold_survivors = romano_wolf(
            ic_hold.select(survivor_names),
            n_boot=config.n_boot,
            mean_block=max(float(config.primary_horizon), float(ic_hold.height) ** (1.0 / 3.0)),
            alternative="two-sided",
            seed=config.seed + 9000,
        )
        rw_hold_map = {
            str(r["name"]): (float(r["p_raw"]), float(r["p_rw"]))
            for r in rw_hold_survivors.iter_rows(named=True)
        }
        q_bh_hold_arr = adjust_pvalues(rw_hold_survivors["p_raw"].to_numpy(), method="bh")
        q_bh_hold_map = {name: float(q_bh_hold_arr[i]) for i, name in enumerate(survivor_names)}
    else:
        rw_hold_map = {}
        q_bh_hold_map = {}

    # Evaluate holdout long-only cost stage on discovery survivors
    candidates_hold = [
        (
            str(r["factor"]),
            float(r["ic_mean"]),
            str(r["verdict"]),
            float(r["p_rw"]),
            float(r["q_bh"]),
        )
        for r in survivor_df.iter_rows(named=True)
    ]
    hold_cost_df = _evaluate_long_only_cost_stage(
        hold_df,
        candidates=candidates_hold,
        config=config,
        ret_col=f"fwd_ret_{config.primary_horizon}",
    )
    hold_cost_map = {str(r["factor"]): r for r in hold_cost_df.iter_rows(named=True)}
    disc_cost_map = {str(r["factor"]): r for r in disc_cost.iter_rows(named=True)}

    holdout_rows: list[dict[str, object]] = []
    for r in survivor_df.iter_rows(named=True):
        f_name = str(r["factor"])
        ic_disc = float(r["ic_mean"])
        post_ic_disc = float(r["posterior_ic"])
        s_hold = ic_hold[f_name].drop_nulls().to_numpy()
        s_full = ic_full[f_name].drop_nulls().to_numpy()

        ic_hold_mean = float(np.mean(s_hold)) if s_hold.size > 0 else 0.0
        ic_hold_std = float(np.std(s_hold, ddof=1)) if s_hold.size > 1 else 0.0
        lrv_hold = _newey_west_lrv(s_hold - ic_hold_mean, max(1, config.primary_horizon - 1))
        se_hold = math.sqrt(lrv_hold / s_hold.size) if lrv_hold > 0.0 and s_hold.size > 0 else 0.0
        t_nw_hold = ic_hold_mean / se_hold if se_hold > 1e-15 else 0.0

        cusum_full = cusum_mean_test(s_full, nw_lags=max(1, config.primary_horizon - 1))
        p_raw_h, p_rw_h = rw_hold_map.get(f_name, (1.0, 1.0))
        q_bh_h = q_bh_hold_map.get(f_name, 1.0)
        same_sign = (ic_disc * ic_hold_mean) > 0.0
        retention_ratio = ic_hold_mean / ic_disc if abs(ic_disc) > 1e-12 else 0.0

        dc = disc_cost_map.get(f_name, {})
        hc = hold_cost_map.get(f_name, {})
        surv_disc_cost_raw = bool(dc.get("survives_cost_raw_p05", False))
        surv_disc_cost = bool(dc.get("survives_cost", False))
        surv_disc_rw_cost = bool(dc.get("survives_rw_and_cost", False))
        hold_net_ex = float(hc.get("net_excess_mean_5d", 0.0))
        hold_net_ann = float(hc.get("net_excess_ann_pct", 0.0))
        hold_p_cost = float(hc.get("p_boot_greater", 1.0))

        confirmed_raw_p05 = (
            surv_disc_cost_raw
            and same_sign
            and (p_raw_h <= config.fwer_alpha)
            and (hold_net_ex > 0.0)
        )
        confirmed_rw = (
            surv_disc_rw_cost
            and same_sign
            and (p_rw_h <= config.fwer_alpha)
            and (hold_net_ex > 0.0)
        )
        confirmed_fdr = (
            surv_disc_cost and same_sign and (q_bh_h <= config.fdr_q) and (hold_net_ex > 0.0)
        )

        holdout_rows.append(
            {
                "factor": f_name,
                "family": str(r["family"]),
                "verdict_discovery": str(r["verdict"]),
                "ic_discovery": ic_disc,
                "posterior_ic_discovery": post_ic_disc,
                "nw_t_discovery": float(r["nw_t"]),
                "p_rw_discovery": float(r["p_rw"]),
                "q_bh_discovery": float(r["q_bh"]),
                "survives_discovery_cost_raw_p05": surv_disc_cost_raw,
                "survives_discovery_cost": surv_disc_cost,
                "survives_discovery_rw_and_cost": surv_disc_rw_cost,
                "breakeven_bps_l1_discovery": float(dc.get("breakeven_bps_l1", 0.0)),
                "breakeven_bps_roundtrip_discovery": float(dc.get("breakeven_bps_roundtrip", 0.0)),
                "net_excess_ann_pct_discovery": float(dc.get("net_excess_ann_pct", 0.0)),
                "p_cost_discovery": float(dc.get("p_boot_greater", 1.0)),
                "ic_holdout": ic_hold_mean,
                "ic_std_holdout": ic_hold_std,
                "t_nw_holdout": t_nw_hold,
                "p_raw_holdout": p_raw_h,
                "q_bh_holdout": q_bh_h,
                "p_rw_holdout": p_rw_h,
                "same_sign": same_sign,
                "retention_ratio": retention_ratio,
                "holdout_net_excess_5d": hold_net_ex,
                "holdout_net_excess_ann_pct": hold_net_ann,
                "holdout_p_cost": hold_p_cost,
                "full_sample_cusum_stat": float(cusum_full["statistic"]),
                "full_sample_cusum_p": float(cusum_full["p_value"]),
                "full_sample_break_index": int(cusum_full["break_index"]),
                "confirmed_holdout_raw_p05": confirmed_raw_p05,
                "confirmed_holdout_rw": confirmed_rw,
                "confirmed_holdout_fdr": confirmed_fdr,
            }
        )

    hold_table = pl.DataFrame(holdout_rows) if holdout_rows else pl.DataFrame()
    n_confirmed_raw_p05 = (
        int(hold_table["confirmed_holdout_raw_p05"].sum()) if hold_table.height > 0 else 0
    )
    n_confirmed_rw = int(hold_table["confirmed_holdout_rw"].sum()) if hold_table.height > 0 else 0
    n_confirmed_fdr = int(hold_table["confirmed_holdout_fdr"].sum()) if hold_table.height > 0 else 0
    n_holdout_rw_stat_only = (
        int(
            (
                (hold_table["verdict_discovery"] == "robust")
                & hold_table["same_sign"]
                & (hold_table["p_rw_holdout"] <= config.fwer_alpha)
            ).sum()
        )
        if hold_table.height > 0
        else 0
    )
    n_holdout_bh_stat_only = (
        int(
            (
                hold_table["verdict_discovery"].is_in(["robust", "fdr_only"])
                & hold_table["same_sign"]
                & (hold_table["q_bh_holdout"] <= config.fdr_q)
            ).sum()
        )
        if hold_table.height > 0
        else 0
    )
    n_holdout_same_sign_robust = (
        int(((hold_table["verdict_discovery"] == "robust") & hold_table["same_sign"]).sum())
        if hold_table.height > 0
        else 0
    )
    median_retention_robust = (
        float(
            hold_table.filter(pl.col("verdict_discovery") == "robust")["retention_ratio"].median()
            or 0.0
        )
        if hold_table.height > 0
        else 0.0
    )

    # Compute cross-sectional correlation of IC across all 150 signals between discovery and holdout
    all_ic_disc = np.array(
        [float(disc_table.filter(pl.col("factor") == c)["ic_mean"][0]) for c in factor_cols]
    )
    all_ic_hold = np.array([float(ic_hold[c].drop_nulls().mean() or 0.0) for c in factor_cols])
    oos_ic_corr_all150 = float(np.corrcoef(all_ic_disc, all_ic_hold)[0, 1])

    elapsed = time.perf_counter() - t0
    hold_dir = results_dir / "holdout"
    hold_dir.mkdir(parents=True, exist_ok=True)
    if hold_table.height > 0:
        hold_table.write_csv(hold_dir / "holdout_table.csv")

    # Save lightweight CSV of top 5 discovery survivors' daily IC series for Figure 5
    top5_surv = (
        disc_table.filter(pl.col("verdict") == "robust")
        .with_columns(pl.col("nw_t").abs().alias("_abs_t"))
        .sort("_abs_t", descending=True)
        .head(5)["factor"]
        .to_list()
    )
    ic_full.select(["date", *top5_surv]).write_csv(hold_dir / "top5_ic_series.csv")

    # Save full 150-signal discovery vs holdout IC comparison for plotting
    pl.DataFrame(
        {
            "factor": factor_cols,
            "family": [s.family for s in specs],
            "ic_discovery": all_ic_disc,
            "ic_holdout": all_ic_hold,
        }
    ).write_csv(hold_dir / "all150_discovery_vs_holdout.csv")

    hold_summary: dict[str, object] = {
        "stage": "holdout",
        "preregistration_commit": prereg_commit,
        "registry_sha256": registry_sha256(specs),
        "holdout_start": str(config.holdout_start),
        "holdout_end": str(disc_summary["data_summary"]["date_max_study"]),
        "holdout_days": int(ic_hold.height),
        "elapsed_sec": round(elapsed, 2),
        "oos_ic_correlation_all_150": oos_ic_corr_all150,
        "median_retention_ratio_robust": median_retention_robust,
        "funnel_final": {
            **disc_summary["funnel"],
            "n_holdout_same_sign_robust": n_holdout_same_sign_robust,
            "n_holdout_bh_stat_only": n_holdout_bh_stat_only,
            "n_holdout_rw_stat_only": n_holdout_rw_stat_only,
            "n_confirmed_holdout_raw_p05": n_confirmed_raw_p05,
            "n_confirmed_holdout_rw": n_confirmed_rw,
            "n_confirmed_holdout_fdr": n_confirmed_fdr,
        },
        "survivors_holdout": holdout_rows,
        "excluded_signals": [
            {"name": name, "family": fam, "reason": reason}
            for name, fam, reason in EXCLUDED_SIGNALS
        ],
    }
    (hold_dir / "summary.json").write_text(
        json.dumps(hold_summary, indent=2),
        encoding="utf-8",
    )
    return hold_summary
