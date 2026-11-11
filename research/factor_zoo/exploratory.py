"""Exploratory post-holdout analyses for the Vietnamese Equity Factor Zoo study (Plan E §5 E-C5).

Conducted after the 2022-2026 holdout was unlocked; explicitly labeled non-confirmatory.
Includes:
(a) Incremental cross-sectional IC after orthogonalizing each signal against 5-day reversal (rev_5).
(b) Q1-Q5 quintile active excess return decomposition (long-only Q5 vs. Q1 short leg asymmetry).
(c) Breakeven transaction-cost survival curve across 0..60 bps round-trip costs.
(d) 146-unique-signal discovery funnel sensitivity check (dropping the 4 exact duplicate pairs).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

import polars_anta as pa
from polars_anta.inference import adjust_pvalues, bootstrap_test, falsify_factors
from research.factor_zoo.config import DEFAULT_CONFIG, RESULTS_DIR, StudyConfig
from research.factor_zoo.data import load_panel
from research.factor_zoo.pipeline import _compute_participation_ratio
from research.factor_zoo.registry import SignalSpec, build_registry, compute_signals_panel

EXACT_DUPLICATE_SECOND_ENTRIES: tuple[str, ...] = (
    "pivot_fibonacci",
    "natr_14",
    "decay_linear_10",
    "ts_zscore_20",
)

RT_COST_GRID_BPS: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60)


def _run_unique_146_sensitivity(
    disc_df: pl.DataFrame,
    specs: tuple[SignalSpec, ...],
    config: StudyConfig,
    seed_base: int,
) -> dict[str, object]:
    """Run Discovery falsify_factors on the 146 unique signals (dropping 4 exact duplicates)."""
    unique_cols = [s.name for s in specs if s.name not in EXACT_DUPLICATE_SECOND_ENTRIES]
    res_146 = falsify_factors(
        disc_df,
        factor_cols=unique_cols,
        return_col="fwd_ret_5",
        group_by="date",
        horizon=5,
        method="spearman",
        fdr_q=config.fdr_q,
        fwer_alpha=config.fwer_alpha,
        prior=config.eb_prior,  # type: ignore[arg-type]
        n_boot=config.n_boot,
        seed=seed_base,
    )
    tbl_146 = res_146.table
    neff_146 = _compute_participation_ratio(res_146.ic_series, unique_cols)

    verdicts = tbl_146["verdict"].to_list()
    return {
        "n_tested": len(unique_cols),
        "dropped_exact_duplicates": list(EXACT_DUPLICATE_SECOND_ENTRIES),
        "n_raw_p05": int((tbl_146["p_raw"] <= 0.05).sum()),
        "n_nw_t196": int((tbl_146["nw_t"].abs() >= 1.96).sum()),
        "n_bh_q10": int((tbl_146["q_bh"] <= config.fdr_q).sum()),
        "n_rw_p05": int((tbl_146["p_rw"] <= config.fwer_alpha).sum()),
        "n_verdict_robust": int(sum(1 for v in verdicts if v == "robust")),
        "n_verdict_fdr_only": int(sum(1 for v in verdicts if v == "fdr_only")),
        "n_verdict_decaying": int(sum(1 for v in verdicts if v == "decaying")),
        "n_verdict_noise": int(sum(1 for v in verdicts if v == "noise")),
        "effective_independent_tests": neff_146,
    }


def _run_incremental_vs_reversal(
    panel_df: pl.DataFrame,
    sig_panel: pl.DataFrame,
    specs: tuple[SignalSpec, ...],
    config: StudyConfig,
    seed_base: int,
    out_dir: Path,
) -> dict[str, object]:
    """Orthogonalize all 150 signals against 5-day close-to-close reversal (rev_5_rank) per date."""
    # Compute 5-day reversal on full cleaned ticker series before filtering in_universe
    rev_df = (
        panel_df.sort(["ticker", "date"])
        .with_columns(
            (-(pl.col("close") / pl.col("close").shift(5).over("ticker") - 1.0)).alias("rev_5")
        )
        .filter(pl.col("in_universe"))
        .with_columns(pa.cs_rank(pl.col("rev_5"), over="date").alias("rev_5_rank"))
        .select(["date", "ticker", "rev_5", "rev_5_rank"])
    )

    joined = sig_panel.join(rev_df, on=["date", "ticker"], how="inner").filter(
        pl.col("rev_5_rank").is_not_null() & pl.col("rev_5_rank").is_finite()
    )

    factor_cols = [s.name for s in specs]
    spec_map = {s.name: s for s in specs}

    # Cross-sectionally orthogonalize every signal against rev_5_rank per date
    orth_exprs = [
        pa.orthogonalize(
            pl.col(f),
            factors=["rev_5_rank"],
            over="date",
            standardize=True,
        ).alias(f)
        for f in factor_cols
    ]
    orth_panel = joined.with_columns(orth_exprs)

    disc_orth = orth_panel.filter(
        (pl.col("date") >= config.discovery_start) & (pl.col("date") <= config.discovery_end)
    )
    hold_orth = orth_panel.filter(pl.col("date") >= config.holdout_start)

    res_disc = falsify_factors(
        disc_orth,
        factor_cols=factor_cols,
        return_col="fwd_ret_5",
        group_by="date",
        horizon=5,
        method="spearman",
        fdr_q=config.fdr_q,
        fwer_alpha=config.fwer_alpha,
        prior=config.eb_prior,  # type: ignore[arg-type]
        n_boot=config.n_boot,
        seed=seed_base + 100,
    )
    res_hold = falsify_factors(
        hold_orth,
        factor_cols=factor_cols,
        return_col="fwd_ret_5",
        group_by="date",
        horizon=5,
        method="spearman",
        fdr_q=config.fdr_q,
        fwer_alpha=config.fwer_alpha,
        prior=config.eb_prior,  # type: ignore[arg-type]
        n_boot=config.n_boot,
        seed=seed_base + 200,
    )

    neff_orth = _compute_participation_ratio(res_disc.ic_series, factor_cols)

    # Also evaluate standalone rev_5_rank IC in Discovery and Holdout
    rev_disc_res_obj = falsify_factors(
        joined.filter(
            (pl.col("date") >= config.discovery_start) & (pl.col("date") <= config.discovery_end)
        ),
        factor_cols=["rev_5_rank"],
        return_col="fwd_ret_5",
        group_by="date",
        horizon=5,
        method="spearman",
        fdr_q=config.fdr_q,
        fwer_alpha=config.fwer_alpha,
        prior=config.eb_prior,  # type: ignore[arg-type]
        n_boot=config.n_boot,
        seed=seed_base + 50,
    )
    rev_hold_res_obj = falsify_factors(
        joined.filter(pl.col("date") >= config.holdout_start),
        factor_cols=["rev_5_rank"],
        return_col="fwd_ret_5",
        group_by="date",
        horizon=5,
        method="spearman",
        fdr_q=config.fdr_q,
        fwer_alpha=config.fwer_alpha,
        prior=config.eb_prior,  # type: ignore[arg-type]
        n_boot=config.n_boot,
        seed=seed_base + 60,
    )
    rev_disc_res = rev_disc_res_obj.table.row(0, named=True)
    rev_hold_res = rev_hold_res_obj.table.row(0, named=True)

    # Merge baseline discovery/holdout vs orthogonalized discovery/holdout
    base_disc = pl.read_csv(RESULTS_DIR / "discovery" / "factor_table.csv").select(
        [
            pl.col("factor"),
            pl.col("ic_mean").alias("raw_disc_ic"),
            pl.col("nw_t").alias("raw_disc_t_nw"),
            pl.col("p_rw").alias("raw_disc_p_rw"),
            pl.col("verdict").alias("raw_disc_verdict"),
        ]
    )
    hold_map = {r["factor"]: r for r in res_hold.table.iter_rows(named=True)}

    rows: list[dict[str, object]] = []
    for r in res_disc.table.iter_rows(named=True):
        fname = r["factor"]
        h_row = hold_map[fname]
        rows.append(
            {
                "factor": fname,
                "family": spec_map[fname].family,
                "orth_disc_ic": float(r["ic_mean"]),
                "orth_disc_posterior_ic": float(r["posterior_ic"]),
                "orth_disc_t_nw": float(r["nw_t"]),
                "orth_disc_p_boot": float(r["p_raw"]),
                "orth_disc_q_bh": float(r["q_bh"]),
                "orth_disc_p_rw": float(r["p_rw"]),
                "orth_disc_p_cusum": float(r["p_cusum"]),
                "orth_disc_verdict": str(r["verdict"]),
                "orth_hold_ic": float(h_row["ic_mean"]),
                "orth_hold_t_nw": float(h_row["nw_t"]),
                "orth_hold_q_bh": float(h_row["q_bh"]),
                "orth_hold_p_rw": float(h_row["p_rw"]),
            }
        )

    orth_df = pl.DataFrame(rows).join(base_disc, on="factor", how="left")
    orth_df = orth_df.with_columns(
        (
            (pl.col("orth_disc_verdict") == "robust")
            & (pl.col("orth_disc_ic") * pl.col("orth_hold_ic") > 0.0)
        ).alias("orth_hold_same_sign_robust"),
        (
            (pl.col("orth_disc_verdict") == "robust")
            & (pl.col("orth_disc_ic") * pl.col("orth_hold_ic") > 0.0)
            & (pl.col("orth_hold_q_bh") <= config.fdr_q)
        ).alias("orth_hold_bh_replicated"),
        (
            (pl.col("orth_disc_verdict") == "robust")
            & (pl.col("orth_disc_ic") * pl.col("orth_hold_ic") > 0.0)
            & (pl.col("orth_hold_p_rw") <= config.fwer_alpha)
        ).alias("orth_hold_rw_replicated"),
    )
    orth_df.write_csv(out_dir / "incremental_vs_reversal.csv")

    verdicts = orth_df["orth_disc_verdict"].to_list()
    top_orth = (
        orth_df.with_columns(pl.col("orth_disc_t_nw").abs().alias("_abs_t"))
        .sort("_abs_t", descending=True)
        .drop("_abs_t")
        .head(10)
        .to_dicts()
    )

    summary = {
        "rev_5_benchmark": {
            "discovery_mean_ic": float(rev_disc_res["ic_mean"]),
            "discovery_t_nw": float(rev_disc_res["nw_t"]),
            "holdout_mean_ic": float(rev_hold_res["ic_mean"]),
            "holdout_t_nw": float(rev_hold_res["nw_t"]),
            "discovery_n_days": int(rev_disc_res_obj.ic_series.height),
            "holdout_n_days": int(rev_hold_res_obj.ic_series.height),
        },
        "before_orthogonalization": {
            "n_tested": 150,
            "n_raw_p05": 91,
            "n_nw_t196": 95,
            "n_bh_q10": 94,
            "n_rw_p05": 71,
            "n_verdict_robust": 64,
            "n_verdict_fdr_only": 16,
            "n_verdict_decaying": 14,
            "n_verdict_noise": 56,
            "n_holdout_rw_replicated": 38,
            "n_holdout_bh_replicated": 49,
            "n_eff": 6.0041,
            "top1_eigenvalue": 55.2072,
            "top5_variance_share": 0.6660,
        },
        "after_orthogonalization": {
            "n_tested": int(orth_df.height),
            "n_raw_p05": int((orth_df["orth_disc_p_boot"] <= 0.05).sum()),
            "n_nw_t196": int((orth_df["orth_disc_t_nw"].abs() >= 1.96).sum()),
            "n_bh_q10": int((orth_df["orth_disc_q_bh"] <= config.fdr_q).sum()),
            "n_rw_p05": int((orth_df["orth_disc_p_rw"] <= config.fwer_alpha).sum()),
            "n_verdict_robust": int(sum(1 for v in verdicts if v == "robust")),
            "n_verdict_fdr_only": int(sum(1 for v in verdicts if v == "fdr_only")),
            "n_verdict_decaying": int(sum(1 for v in verdicts if v == "decaying")),
            "n_verdict_noise": int(sum(1 for v in verdicts if v == "noise")),
            "n_holdout_same_sign_robust": int(orth_df["orth_hold_same_sign_robust"].sum()),
            "n_holdout_bh_replicated": int(orth_df["orth_hold_bh_replicated"].sum()),
            "n_holdout_rw_replicated": int(orth_df["orth_hold_rw_replicated"].sum()),
            "n_eff": float(neff_orth["n_eff"]),
            "top1_eigenvalue": float(neff_orth["top1_eigenvalue"]),
            "top5_variance_share": float(neff_orth["top5_variance_share"]),
        },
        "top_10_orthogonalized_signals": top_orth,
    }
    (out_dir / "incremental_vs_reversal.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _compute_quintile_excess_for_sample(
    sample_df: pl.DataFrame,
    target_signals: list[tuple[str, str, float]],
    config: StudyConfig,
    sample_label: str,
) -> list[dict[str, object]]:
    """Compute non-overlapping 5-day excess returns for Q1..Q5 vs equal-weight universe."""
    dates = sample_df["date"].unique().sort()
    rebal_dates = dates.gather_every(config.rebalance_every)
    rebal_df = sample_df.filter(pl.col("date").is_in(rebal_dates)).sort(["date", "ticker"])

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
    ret_arr = rebal_df["fwd_ret_5"].to_numpy()

    ret_mat = np.full((n_T, n_N), np.nan, dtype=np.float64)
    ret_mat[date_idx_arr, tick_idx_arr] = ret_arr
    univ_ret = np.nanmean(ret_mat, axis=1)
    ann_scale = 252.0 / float(config.rebalance_every)

    out_rows: list[dict[str, object]] = []
    for f_name, family, disc_ic in target_signals:
        direction = 1.0 if disc_ic >= 0.0 else -1.0
        f_vals = rebal_df[f_name].to_numpy()
        f_mat = np.full((n_T, n_N), np.nan, dtype=np.float64)
        f_mat[date_idx_arr, tick_idx_arr] = f_vals * direction

        q_excess = np.full((n_T, 5), np.nan, dtype=np.float64)
        for t in range(n_T):
            row_f = f_mat[t]
            row_r = ret_mat[t]
            mask = np.isfinite(row_f) & np.isfinite(row_r)
            if int(np.sum(mask)) < 5 or not np.isfinite(univ_ret[t]):
                continue
            f_valid = row_f[mask]
            r_valid = row_r[mask]
            edges = np.quantile(f_valid, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            for q_idx in range(5):
                lo = edges[q_idx]
                hi = edges[q_idx + 1]
                if q_idx < 4:
                    q_sel = (f_valid >= lo) & (f_valid < hi)
                else:
                    q_sel = f_valid >= lo
                if np.any(q_sel):
                    q_excess[t, q_idx] = float(np.mean(r_valid[q_sel]) - univ_ret[t])

        q_means_5d = [float(np.nanmean(q_excess[:, q])) for q in range(5)]
        q_means_ann_pct = [m * ann_scale * 100.0 for m in q_means_5d]
        q5_minus_q1_5d = q_means_5d[4] - q_means_5d[0]
        q5_minus_q1_ann_pct = q5_minus_q1_5d * ann_scale * 100.0
        short_leg_share = (
            float(-q_means_5d[0] / q5_minus_q1_5d) if abs(q5_minus_q1_5d) > 1e-12 else 0.0
        )

        out_rows.append(
            {
                "sample": sample_label,
                "factor": f_name,
                "family": family,
                "ic_direction": direction,
                "q1_excess_5d": q_means_5d[0],
                "q2_excess_5d": q_means_5d[1],
                "q3_excess_5d": q_means_5d[2],
                "q4_excess_5d": q_means_5d[3],
                "q5_excess_5d": q_means_5d[4],
                "q5_minus_q1_5d": q5_minus_q1_5d,
                "q1_excess_ann_pct": q_means_ann_pct[0],
                "q2_excess_ann_pct": q_means_ann_pct[1],
                "q3_excess_ann_pct": q_means_ann_pct[2],
                "q4_excess_ann_pct": q_means_ann_pct[3],
                "q5_excess_ann_pct": q_means_ann_pct[4],
                "q5_minus_q1_ann_pct": q5_minus_q1_ann_pct,
                "q1_short_leg_share_of_spread": short_leg_share,
                "q1_short_leg_share_pct": short_leg_share * 100.0,
            }
        )
    return out_rows


def _run_quintile_decomposition(
    disc_df: pl.DataFrame,
    hold_df: pl.DataFrame,
    config: StudyConfig,
    out_dir: Path,
) -> dict[str, object]:
    """Run Q1-Q5 active return decomposition for top-10 |t_NW| signals + trix_15, tsi."""
    disc_table = pl.read_csv(RESULTS_DIR / "discovery" / "factor_table.csv")
    top10 = (
        disc_table.with_columns(pl.col("nw_t").abs().alias("_abs_t"))
        .sort("_abs_t", descending=True)
        .drop("_abs_t")
        .head(10)
    )
    extra = disc_table.filter(pl.col("factor").is_in(["trix_15", "tsi", "alpha_018"]))
    combined = pl.concat([top10, extra]).unique(subset=["factor"], maintain_order=True)
    target_signals = [
        (str(r["factor"]), str(r["family"]), float(r["ic_mean"]))
        for r in combined.iter_rows(named=True)
    ]

    disc_rows = _compute_quintile_excess_for_sample(disc_df, target_signals, config, "discovery")
    hold_rows = _compute_quintile_excess_for_sample(hold_df, target_signals, config, "holdout")
    all_rows = disc_rows + hold_rows

    q_df = pl.DataFrame(all_rows)
    q_df.write_csv(out_dir / "quintile_spread.csv")

    top10_names = set(top10["factor"].to_list())
    disc_top10 = [r for r in disc_rows if r["factor"] in top10_names]
    hold_top10 = [r for r in hold_rows if r["factor"] in top10_names]

    summary = {
        "n_signals_analyzed": len(target_signals),
        "discovery_top10_mean_q1_ann_pct": float(
            np.mean([r["q1_excess_ann_pct"] for r in disc_top10])
        ),
        "discovery_top10_mean_q5_ann_pct": float(
            np.mean([r["q5_excess_ann_pct"] for r in disc_top10])
        ),
        "discovery_top10_mean_q5_minus_q1_ann_pct": float(
            np.mean([r["q5_minus_q1_ann_pct"] for r in disc_top10])
        ),
        "discovery_top10_mean_short_leg_share": float(
            np.mean([r["q1_short_leg_share_of_spread"] for r in disc_top10])
        ),
        "holdout_top10_mean_q1_ann_pct": float(
            np.mean([r["q1_excess_ann_pct"] for r in hold_top10])
        ),
        "holdout_top10_mean_q5_ann_pct": float(
            np.mean([r["q5_excess_ann_pct"] for r in hold_top10])
        ),
        "holdout_top10_mean_q5_minus_q1_ann_pct": float(
            np.mean([r["q5_minus_q1_ann_pct"] for r in hold_top10])
        ),
        "holdout_top10_mean_short_leg_share": float(
            np.mean([r["q1_short_leg_share_of_spread"] for r in hold_top10])
        ),
        "rows": all_rows,
    }
    (out_dir / "quintile_spread.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _extract_q5_gross_and_turnover_series(
    sample_df: pl.DataFrame,
    candidates: list[tuple[str, float, str]],
    config: StudyConfig,
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Precompute 5-day Q5 gross excess and L1 turnover series for all 80 candidates."""
    dates = sample_df["date"].unique().sort()
    rebal_dates = dates.gather_every(config.rebalance_every)
    rebal_df = sample_df.filter(pl.col("date").is_in(rebal_dates)).sort(["date", "ticker"])

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
    ret_arr = rebal_df["fwd_ret_5"].to_numpy()

    ret_mat = np.full((n_T, n_N), np.nan, dtype=np.float64)
    ret_mat[date_idx_arr, tick_idx_arr] = ret_arr
    univ_ret = np.nanmean(ret_mat, axis=1)

    series_list: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for f_name, ic_mean, verdict in candidates:
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

        series_list.append((f_name, verdict, gross_excess[valid_steps], l1_turnover[valid_steps]))
    return series_list


def _run_breakeven_curve(
    disc_df: pl.DataFrame,
    hold_df: pl.DataFrame,
    config: StudyConfig,
    seed_base: int,
    out_dir: Path,
) -> dict[str, object]:
    """Evaluate how many of the 80 candidates survive across 0..60 bps round-trip costs."""
    disc_table = pl.read_csv(RESULTS_DIR / "discovery" / "factor_table.csv")
    cand_df = disc_table.filter(pl.col("verdict").is_in(["robust", "fdr_only"]))
    candidates = [
        (str(r["factor"]), float(r["ic_mean"]), str(r["verdict"]))
        for r in cand_df.iter_rows(named=True)
    ]

    disc_series = _extract_q5_gross_and_turnover_series(disc_df, candidates, config)
    hold_series = _extract_q5_gross_and_turnover_series(hold_df, candidates, config)

    curve_rows: list[dict[str, object]] = []
    for sample_label, s_list, sample_offset in (
        ("discovery", disc_series, 0),
        ("holdout", hold_series, 1000),
    ):
        for cost_idx, rt_bps in enumerate(RT_COST_GRID_BPS):
            l1_rate = (float(rt_bps) / 2.0) / 10_000.0
            p_vals: list[float] = []
            net_means: list[float] = []
            verdicts: list[str] = []

            for sig_idx, (_fname, verdict, g_ex, turn_l1) in enumerate(s_list):
                net_ex = g_ex - l1_rate * turn_l1
                net_means.append(float(np.mean(net_ex)))
                verdicts.append(verdict)
                boot = bootstrap_test(
                    net_ex,
                    statistic="mean",
                    alternative="greater",
                    n_boot=config.n_boot,
                    seed=seed_base + 300 + sample_offset + cost_idx * 100 + sig_idx,
                )
                p_vals.append(float(boot["p_value"]))

            p_arr = np.array(p_vals, dtype=np.float64)
            m_arr = np.array(net_means, dtype=np.float64)
            q_arr = adjust_pvalues(p_arr, method="bh")

            n_pos = int(np.sum(m_arr > 0.0))
            n_raw = int(np.sum((m_arr > 0.0) & (p_arr <= config.cost_alpha)))
            n_bh = int(
                np.sum((m_arr > 0.0) & (p_arr <= config.cost_alpha) & (q_arr <= config.fdr_q))
            )
            curve_rows.append(
                {
                    "sample": sample_label,
                    "round_trip_cost_bps": int(rt_bps),
                    "l1_cost_bps": float(rt_bps) / 2.0,
                    "n_candidates": len(s_list),
                    "n_positive_net": n_pos,
                    "n_raw_p05": n_raw,
                    "n_bh_q10": n_bh,
                }
            )

    curve_df = pl.DataFrame(curve_rows)
    curve_df.write_csv(out_dir / "breakeven_curve.csv")

    summary = {
        "n_candidates": len(candidates),
        "round_trip_cost_grid_bps": list(RT_COST_GRID_BPS),
        "curve": curve_rows,
    }
    (out_dir / "breakeven_curve.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_exploratory(
    config: StudyConfig = DEFAULT_CONFIG,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, object]:
    """Execute all exploratory post-holdout analyses (E-04 + E-05 a, b, c)."""
    t0 = time.perf_counter()
    out_dir = results_dir / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = config.seed + 7000
    panel_df, _ = load_panel(config=config)
    specs = build_registry()
    sig_panel = compute_signals_panel(panel_df, specs=specs)

    disc_df = sig_panel.filter(
        (pl.col("date") >= config.discovery_start) & (pl.col("date") <= config.discovery_end)
    )
    hold_df = sig_panel.filter(pl.col("date") >= config.holdout_start)

    unique_146 = _run_unique_146_sensitivity(disc_df, specs, config, seed_base)
    (out_dir / "unique_146_sensitivity.json").write_text(
        json.dumps(unique_146, indent=2), encoding="utf-8"
    )

    inc_rev = _run_incremental_vs_reversal(panel_df, sig_panel, specs, config, seed_base, out_dir)
    quintile = _run_quintile_decomposition(disc_df, hold_df, config, out_dir)
    breakeven = _run_breakeven_curve(disc_df, hold_df, config, seed_base, out_dir)

    elapsed = round(time.perf_counter() - t0, 2)
    master_summary = {
        "stage": "exploratory_post_holdout",
        "seed_base": seed_base,
        "n_boot": config.n_boot,
        "elapsed_sec": elapsed,
        "unique_146_sensitivity": unique_146,
        "incremental_vs_reversal": {
            "rev_5_benchmark": inc_rev["rev_5_benchmark"],
            "before_orthogonalization": inc_rev["before_orthogonalization"],
            "after_orthogonalization": inc_rev["after_orthogonalization"],
        },
        "quintile_decomposition_summary": {k: v for k, v in quintile.items() if k != "rows"},
        "breakeven_curve": breakeven["curve"],
    }
    (out_dir / "summary.json").write_text(json.dumps(master_summary, indent=2), encoding="utf-8")
    return master_summary


if __name__ == "__main__":
    res = run_exploratory()
    print(json.dumps(res, indent=2))
