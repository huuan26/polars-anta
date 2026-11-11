"""Generate publication figures and Markdown tables from Factor Zoo JSON/CSV results."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from research.factor_zoo.config import DEFAULT_CONFIG, REPO_ROOT, RESULTS_DIR
from research.factor_zoo.registry import EXCLUDED_SIGNALS


def _summarize_cost_subset(df: pl.DataFrame) -> dict[str, float | int]:
    """Compute descriptive statistics for a subset of cost_stage.csv."""
    l1 = df["l1_turnover_mean_5d"]
    ow = df["one_way_turnover_mean_5d"]
    be_rt = df["breakeven_bps_roundtrip"]
    drag_rate = DEFAULT_CONFIG.round_trip_cost_rate * l1 * (252.0 / DEFAULT_CONFIG.primary_horizon)
    pos_be = df.filter(pl.col("breakeven_bps_roundtrip") > 0.0)["breakeven_bps_roundtrip"]
    return {
        "n_signals": int(df.height),
        "l1_turnover_min": float(l1.min()),
        "l1_turnover_max": float(l1.max()),
        "one_way_turnover_min": float(ow.min()),
        "one_way_turnover_max": float(ow.max()),
        "one_way_turnover_pct_min": float(ow.min()) * 100.0,
        "one_way_turnover_pct_max": float(ow.max()) * 100.0,
        "cost_drag_ann_min": float(drag_rate.min()),
        "cost_drag_ann_max": float(drag_rate.max()),
        "cost_drag_ann_pct_min": float(drag_rate.min()) * 100.0,
        "cost_drag_ann_pct_max": float(drag_rate.max()) * 100.0,
        "breakeven_rt_bps_min": float(be_rt.min()),
        "breakeven_rt_bps_max": float(be_rt.max()),
        "breakeven_rt_bps_pos_min": float(pos_be.min()) if pos_be.len() > 0 else 0.0,
        "breakeven_rt_bps_pos_max": float(pos_be.max()) if pos_be.len() > 0 else 0.0,
    }


def render_all(results_dir: Path = RESULTS_DIR) -> dict[str, str]:
    """Render all 5 Factor Zoo figures and copy them to docs/assets/research/."""
    disc_dir = results_dir / "discovery"
    hold_dir = results_dir / "holdout"
    fig_dir = results_dir / "figures"
    docs_fig_dir = REPO_ROOT / "docs" / "assets" / "research"
    fig_dir.mkdir(parents=True, exist_ok=True)
    docs_fig_dir.mkdir(parents=True, exist_ok=True)

    disc_summary = json.loads((disc_dir / "summary.json").read_text(encoding="utf-8"))
    hold_summary = json.loads((hold_dir / "summary.json").read_text(encoding="utf-8"))
    disc_table = pl.read_csv(disc_dir / "factor_table.csv")
    cost_table = pl.read_csv(disc_dir / "cost_stage.csv")
    all150_cmp = pl.read_csv(hold_dir / "all150_discovery_vs_holdout.csv")
    ic_full = pl.read_csv(hold_dir / "top5_ic_series.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
        }
    )

    funnel = hold_summary["funnel_final"]

    # ------------------------------------------------------------------
    # Figure 1: Two-Track Falsification Funnel (Statistical vs Economic)
    # ------------------------------------------------------------------
    stat_stages = [
        "1. Registered Signals\n(146 Unique / 150 Total)",
        "2. Benjamini–Hochberg\nFDR q ≤ 0.10 (Discovery)",
        "3. Romano–Wolf Stepdown\nFWER p ≤ 0.05 (Discovery)",
        "4. Non-Decaying `robust`\n(CUSUM Stability Screen)",
        "5. Replicate in Holdout\n(2022–2026, RW p ≤ 0.05)",
    ]
    stat_counts = [
        int(funnel["n_tested"]),
        int(funnel["n_bh_q10"]),
        int(funnel["n_rw_p05"]),
        int(funnel["n_verdict_robust"]),
        int(funnel["n_holdout_rw_stat_only"]),
    ]
    stat_colors = ["#475569", "#3b82f6", "#6366f1", "#0ea5e9", "#10b981"]

    econ_stages = [
        "1. Non-Decaying Candidates\n(64 robust + 16 fdr_only)",
        "2. Positive Net Excess\n(Long-Only Q5, 60 bps RT)",
        "3. Unadjusted Bootstrap\np ≤ 0.05 (trix_15, tsi)",
        "4. Survive BH FDR q ≤ 0.10\nAcross 80 Cost Candidates",
        "5. Confirmed in Holdout\n(2022–2026, Post-Cost)",
    ]
    econ_counts = [
        int(funnel["n_cost_candidates_evaluated"]),
        int(funnel["n_positive_net_excess"]),
        int(funnel["n_survive_costs_raw_p05"]),
        int(funnel["n_survive_costs_fdr"]),
        int(funnel["n_confirmed_holdout_rw"]),
    ]
    econ_colors = ["#475569", "#f59e0b", "#fb7185", "#ef4444", "#dc2626"]

    fig, (ax_stat, ax_econ) = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=160)
    y_stat = np.arange(len(stat_stages))[::-1]
    bars_stat = ax_stat.barh(
        y_stat, stat_counts, color=stat_colors, height=0.60, edgecolor="#0f172a", linewidth=0.7
    )
    exp_fp = float(funnel["expected_false_positives_at_5pct"])
    ax_stat.axvline(
        exp_fp,
        color="#ef4444",
        linestyle="--",
        linewidth=1.4,
        label=(f"i.i.d. Noise Benchmark: {exp_fp:.1f} (Raw p ≤ 0.05: {funnel['n_raw_p05']})"),
    )
    for bar, val in zip(bars_stat, stat_counts, strict=True):
        pct = 100.0 * val / float(funnel["n_tested"])
        ax_stat.text(
            bar.get_width() + 2.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.0,
            fontweight="bold",
            color="#0f172a",
        )
    ax_stat.set_yticks(y_stat)
    ax_stat.set_yticklabels(stat_stages, fontsize=8.8)
    ax_stat.set_xlim(0, max(stat_counts) * 1.24)
    ax_stat.set_xlabel("Signals Passing Stage", fontsize=9.5)
    ax_stat.set_title(
        "Track A — Statistical Predictability (Rank IC, h = 5d)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_stat.grid(axis="x", linestyle=":")
    ax_stat.legend(loc="lower right", frameon=True, fontsize=8.0)

    y_econ = np.arange(len(econ_stages))[::-1]
    bars_econ = ax_econ.barh(
        y_econ, econ_counts, color=econ_colors, height=0.60, edgecolor="#0f172a", linewidth=0.7
    )
    for bar, val in zip(bars_econ, econ_counts, strict=True):
        pct_cand = 100.0 * val / float(funnel["n_cost_candidates_evaluated"])
        ax_econ.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val} ({pct_cand:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.0,
            fontweight="bold",
            color="#0f172a",
        )
    ax_econ.set_yticks(y_econ)
    ax_econ.set_yticklabels(econ_stages, fontsize=8.8)
    ax_econ.set_xlim(0, max(econ_counts) * 1.28)
    ax_econ.set_xlabel("Signals Passing Stage", fontsize=9.5)
    ax_econ.set_title(
        "Track B — Economic Implementability (Long-Only Q5, HOSE Costs)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_econ.grid(axis="x", linestyle=":")

    fig.suptitle(
        "Two-Track Factor Falsification Funnel on Vietnamese Equities (HOSE, 2014–2026)",
        fontsize=12.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "01_falsification_funnel.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Newey-West t-Statistic Distribution vs N(0, 1)
    # ------------------------------------------------------------------
    t_vals = disc_table["nw_t"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=160)
    ax.hist(
        t_vals,
        bins=24,
        density=True,
        color="#3b82f6",
        alpha=0.65,
        edgecolor="#1e293b",
        label="Empirical Newey–West $t$-stats (150 Signals)",
    )
    xs = np.linspace(
        min(-4.5, float(np.min(t_vals)) - 0.5), max(4.5, float(np.max(t_vals)) + 0.5), 400
    )
    norm_pdf = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * xs * xs)
    ax.plot(
        xs,
        norm_pdf,
        color="#0f172a",
        linewidth=2.0,
        label="Standard Normal $\\mathcal{N}(0, 1)$ Null",
    )
    ax.axvline(
        1.96,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.3,
        label="Naive $|t| = 1.96$ ($p = 0.05$)",
    )
    ax.axvline(-1.96, color="#f59e0b", linestyle="--", linewidth=1.3)
    # Approximate RW threshold from smallest |nw_t| with p_rw <= 0.05
    rw_sig = disc_table.filter(pl.col("p_rw") <= 0.05)
    if rw_sig.height > 0:
        rw_thresh = float(rw_sig["nw_t"].abs().min())
        ax.axvline(
            rw_thresh,
            color="#ef4444",
            linestyle="-.",
            linewidth=1.5,
            label=f"Romano–Wolf FWER 5% Threshold ($|t| \\approx {rw_thresh:.2f}$)",
        )
        ax.axvline(-rw_thresh, color="#ef4444", linestyle="-.", linewidth=1.5)
    ax.set_xlabel("Newey–West HAC $t$-Statistic (Discovery 2014–2021, $h = 5$)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "Distribution of Cross-Sectional Rank IC $t$-Statistics Across 150 Signals",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_nw_tstat_distribution.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Empirical-Bayes Shrinkage (Raw IC vs Posterior IC)
    # ------------------------------------------------------------------
    raw_ic = disc_table["ic_mean"].to_numpy()
    post_ic = disc_table["posterior_ic"].to_numpy()
    verdicts = disc_table["verdict"].to_list()
    v_colors = {
        "robust": "#10b981",
        "fdr_only": "#3b82f6",
        "decaying": "#f59e0b",
        "noise": "#94a3b8",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    lim = max(float(np.max(np.abs(raw_ic))) * 1.12, 0.03)
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color="#64748b",
        linestyle="--",
        linewidth=1.0,
        label="45° No-Shrinkage Line",
    )
    ax.axhline(0.0, color="#cbd5e1", linewidth=0.8)
    ax.axvline(0.0, color="#cbd5e1", linewidth=0.8)

    for v_label in ("noise", "decaying", "fdr_only", "robust"):
        mask = np.array([v == v_label for v in verdicts], dtype=bool)
        if np.any(mask):
            ax.scatter(
                raw_ic[mask],
                post_ic[mask],
                color=v_colors[v_label],
                label=f"{v_label} ({int(np.sum(mask))})",
                s=36 if v_label != "noise" else 22,
                alpha=0.88 if v_label != "noise" else 0.55,
                edgecolors="#0f172a" if v_label != "noise" else "none",
                linewidths=0.5,
            )

    for i in range(len(raw_ic)):
        if verdicts[i] != "noise":
            ax.annotate(
                "",
                xy=(raw_ic[i], post_ic[i]),
                xytext=(raw_ic[i], raw_ic[i]),
                arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.7, "alpha": 0.6},
            )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Sample Discovery Rank IC ($\\overline{\\mathrm{IC}}_k$)", fontsize=10)
    ax.set_ylabel(
        "Empirical-Bayes Posterior Rank IC ($\\widehat{\\mathrm{IC}}^{\\mathrm{EB}}_k$)",
        fontsize=10,
    )
    ax.set_title(
        "Efron–Morris Empirical-Bayes Shrinkage of Factor Rank ICs",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_empirical_bayes_shrinkage.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Discovery vs Locked Holdout IC (All 150 Signals)
    # ------------------------------------------------------------------
    rw_map = {str(r["factor"]): float(r["p_rw"]) <= 0.05 for r in disc_table.iter_rows(named=True)}
    ic_d_all = all150_cmp["ic_discovery"].to_numpy()
    ic_h_all = all150_cmp["ic_holdout"].to_numpy()
    f_names_all = all150_cmp["factor"].to_list()
    is_rw = np.array([rw_map.get(f, False) for f in f_names_all], dtype=bool)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
    ax.axvline(0.0, color="#94a3b8", linewidth=0.9)
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color="#64748b",
        linestyle=":",
        linewidth=1.0,
        label="1:1 Perfect Persistence",
    )
    ax.scatter(
        ic_d_all[~is_rw],
        ic_h_all[~is_rw],
        color="#94a3b8",
        alpha=0.55,
        s=24,
        label=f"Rejected / Non-RW Signals ({int(np.sum(~is_rw))})",
    )
    if np.any(is_rw):
        ax.scatter(
            ic_d_all[is_rw],
            ic_h_all[is_rw],
            color="#10b981",
            s=48,
            edgecolors="#0f172a",
            linewidths=0.7,
            label=f"Discovery Romano–Wolf Survivors ({int(np.sum(is_rw))})",
        )
    oos_corr = float(hold_summary["oos_ic_correlation_all_150"])
    ax.set_xlabel("Discovery Rank IC (2014–2021)", fontsize=10)
    ax.set_ylabel("Locked Holdout Rank IC (2022–2026)", fontsize=10)
    ax.set_title(
        f"In-Sample Discovery vs. Out-of-Sample Holdout Rank IC (Corr = {oos_corr:.2f})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_discovery_vs_holdout_ic.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: Cumulative Rank IC & CUSUM Breakpoints of Top Survivors
    # ------------------------------------------------------------------
    top_surv = [c for c in ic_full.columns if c != "date"]

    dates_full = ic_full["date"].to_list()
    holdout_split_idx = 0
    for idx_d, d_val in enumerate(dates_full):
        if str(d_val) >= str(hold_summary["holdout_start"]):
            holdout_split_idx = idx_d
            break

    hold_surv_map = {str(r["factor"]): r for r in hold_summary.get("survivors_holdout", [])}

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    palette = ["#10b981", "#3b82f6", "#6366f1", "#f59e0b", "#ec4899"]
    x_idx = np.arange(len(dates_full))
    for i, f_name in enumerate(top_surv):
        series = ic_full[f_name].fill_null(0.0).fill_nan(0.0).to_numpy()
        direction = 1.0 if float(np.mean(series[:holdout_split_idx])) >= 0.0 else -1.0
        cum_ic = np.cumsum(series * direction)
        color = palette[i % len(palette)]
        ax.plot(x_idx, cum_ic, color=color, linewidth=1.7, label=f"{f_name} (sign-oriented)")
        if f_name in hold_surv_map:
            b_idx = int(hold_surv_map[f_name]["full_sample_break_index"])
            if 0 <= b_idx < len(cum_ic):
                ax.scatter([b_idx], [cum_ic[b_idx]], color=color, marker="X", s=55, zorder=5)

    ax.axvline(
        holdout_split_idx,
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        label="Holdout Lock Boundary (2022-01-01)",
    )
    tick_step = max(1, len(dates_full) // 8)
    tick_positions = list(range(0, len(dates_full), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(dates_full[p])[:7] for p in tick_positions], rotation=25, fontsize=8.5)
    ax.set_ylabel("Cumulative Sign-Oriented Rank IC", fontsize=10)
    ax.set_title(
        "Cumulative Rank IC Trajectories & OLS-CUSUM Structural Breakpoints (2014–2026)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_cumulative_ic_cusum.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figures 6-8: Exploratory Post-Holdout Analyses (if present)
    # ------------------------------------------------------------------
    exp_dir = results_dir / "exploratory"
    if exp_dir.exists():
        inc_csv = exp_dir / "incremental_vs_reversal.csv"
        inc_json = exp_dir / "incremental_vs_reversal.json"
        if inc_csv.exists() and inc_json.exists():
            inc_df = pl.read_csv(inc_csv)
            inc_meta = json.loads(inc_json.read_text(encoding="utf-8"))
            fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=160)

            ax0 = axes[0]
            raw_t = inc_df["raw_disc_t_nw"].to_numpy()
            orth_t = inc_df["orth_disc_t_nw"].to_numpy()
            is_rep = inc_df["orth_hold_rw_replicated"].to_numpy()
            ax0.scatter(
                raw_t[~is_rep],
                orth_t[~is_rep],
                color="#64748b",
                alpha=0.55,
                s=26,
                label="Not RW-replicated in Holdout",
            )
            ax0.scatter(
                raw_t[is_rep],
                orth_t[is_rep],
                color="#10b981",
                alpha=0.85,
                s=34,
                label=f"Holdout RW p ≤ 0.05 ({int(np.sum(is_rep))} signals)",
            )
            lim = max(float(np.max(np.abs(raw_t))), float(np.max(np.abs(orth_t)))) * 1.1
            ax0.plot([-lim, lim], [-lim, lim], color="#94a3b8", linestyle="--", linewidth=1.0)
            ax0.axhline(0.0, color="#475569", linewidth=0.8)
            ax0.axvline(0.0, color="#475569", linewidth=0.8)
            ax0.set_xlim(-lim, lim)
            ax0.set_ylim(-lim, lim)
            ax0.set_xlabel("Raw Discovery Newey–West t-Stat", fontsize=9.5)
            ax0.set_ylabel("Orthogonalized Discovery NW t-Stat (vs. rev_5)", fontsize=9.5)
            ax0.set_title(
                "A. Signal NW t-Stats Before vs. After Orthogonalizing rev_5",
                fontsize=10.2,
                fontweight="bold",
            )
            ax0.grid(True, linestyle=":")
            ax0.legend(loc="upper left", fontsize=8.2)

            ax1 = axes[1]
            b_meta = inc_meta["before_orthogonalization"]
            a_meta = inc_meta["after_orthogonalization"]
            cats = [
                "Discovery\nBH q ≤ 0.10",
                "Discovery\nRW p ≤ 0.05",
                "Discovery\nrobust",
                "Holdout RW\nReplicated",
            ]
            b_vals = [
                b_meta["n_bh_q10"],
                b_meta["n_rw_p05"],
                b_meta["n_verdict_robust"],
                b_meta["n_holdout_rw_replicated"],
            ]
            a_vals = [
                a_meta["n_bh_q10"],
                a_meta["n_rw_p05"],
                a_meta["n_verdict_robust"],
                a_meta["n_holdout_rw_replicated"],
            ]
            x_pos = np.arange(len(cats))
            width = 0.36
            bars1 = ax1.bar(
                x_pos - width / 2,
                b_vals,
                width,
                color="#3b82f6",
                label=f"Raw Signals (N_eff = {b_meta['n_eff']:.2f})",
            )
            bars2 = ax1.bar(
                x_pos + width / 2,
                a_vals,
                width,
                color="#10b981",
                label=f"Orthogonalized vs. rev_5 (N_eff = {a_meta['n_eff']:.2f})",
            )
            for b in list(bars1) + list(bars2):
                h_val = int(b.get_height())
                ax1.text(
                    b.get_x() + b.get_width() / 2,
                    h_val + 2,
                    f"{h_val}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(cats, fontsize=8.5)
            ax1.set_ylim(0, 130)
            ax1.set_ylabel("Number of Signals", fontsize=9.5)
            ax1.set_title(
                "B. Statistical Survival Before vs. After Removing 5d Reversal",
                fontsize=10.2,
                fontweight="bold",
            )
            ax1.grid(True, axis="y", linestyle=":")
            ax1.legend(loc="upper right", fontsize=8.2)

            fig.suptitle(
                "Exploratory Analysis (a): Incremental Predictability Beyond 5-Day Return Reversal",
                fontsize=11.5,
                fontweight="bold",
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(fig_dir / "06_incremental_vs_reversal.png", bbox_inches="tight")
            plt.close(fig)

        q_json = exp_dir / "quintile_spread.json"
        if q_json.exists():
            q_meta = json.loads(q_json.read_text(encoding="utf-8"))
            q_rows = q_meta["rows"]
            top10_disc = [r for r in q_rows if r["sample"] == "discovery"][:10]
            top10_hold = [r for r in q_rows if r["sample"] == "holdout"][:10]
            d_means = [
                float(np.mean([r[f"q{i}_excess_ann_pct"] for r in top10_disc])) for i in range(1, 6)
            ]
            h_means = [
                float(np.mean([r[f"q{i}_excess_ann_pct"] for r in top10_hold])) for i in range(1, 6)
            ]
            a018_disc = next(
                r for r in q_rows if r["sample"] == "discovery" and r["factor"] == "alpha_018"
            )
            a018_vals = [float(a018_disc[f"q{i}_excess_ann_pct"]) for i in range(1, 6)]

            fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=160)
            q_labels = ["Q1 (Short Leg)", "Q2", "Q3", "Q4", "Q5 (Long Leg)"]
            x_q = np.arange(5)
            w_q = 0.36

            ax0 = axes[0]
            b_d = ax0.bar(
                x_q - w_q / 2, d_means, w_q, color="#3b82f6", label="Discovery (2014–2021)"
            )
            b_h = ax0.bar(x_q + w_q / 2, h_means, w_q, color="#10b981", label="Holdout (2022–2026)")
            ax0.axhline(0.0, color="#475569", linewidth=0.9)
            for b in list(b_d) + list(b_h):
                v = float(b.get_height())
                ax0.text(
                    b.get_x() + b.get_width() / 2,
                    v + (0.4 if v >= 0 else -1.1),
                    f"{v:+.1f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.8,
                )
            ax0.set_xticks(x_q)
            ax0.set_xticklabels(q_labels, fontsize=8.5)
            ax0.set_ylim(-15.5, 8.5)
            ax0.set_ylabel("Mean Annualized Active Return vs. EW Universe (%)", fontsize=9)
            ax0.set_title(
                "A. Mean Q1–Q5 Active Returns (Top-10 |t_NW| Discovery Signals)",
                fontsize=10.2,
                fontweight="bold",
            )
            ax0.grid(True, axis="y", linestyle=":")
            ax0.legend(loc="lower right", fontsize=8.5)

            ax1 = axes[1]
            colors_a18 = ["#ef4444" if v < 0 else "#6366f1" for v in a018_vals]
            b_a18 = ax1.bar(x_q, a018_vals, width=0.52, color=colors_a18)
            ax1.axhline(0.0, color="#475569", linewidth=0.9)
            for b in b_a18:
                v = float(b.get_height())
                ax1.text(
                    b.get_x() + b.get_width() / 2,
                    v + (0.5 if v >= 0 else -1.3),
                    f"{v:+.1f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8.2,
                    fontweight="bold",
                )
            ax1.set_xticks(x_q)
            ax1.set_xticklabels(q_labels, fontsize=8.5)
            ax1.set_ylim(-19.5, 13.0)
            ax1.set_ylabel("Annualized Active Return vs. EW Universe (%)", fontsize=9)
            ax1.set_title(
                "B. Non-Monotonic Tail Profile of alpha_018 (Discovery IC = +0.0327)",
                fontsize=10.2,
                fontweight="bold",
            )
            ax1.grid(True, axis="y", linestyle=":")

            fig.suptitle(
                "Exploratory Analysis (b): Q1–Q5 Active Return Asymmetry & Short-Leg Dominance",
                fontsize=11.5,
                fontweight="bold",
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(fig_dir / "07_quintile_spread.png", bbox_inches="tight")
            plt.close(fig)

        be_json = exp_dir / "breakeven_curve.json"
        if be_json.exists():
            be_meta = json.loads(be_json.read_text(encoding="utf-8"))
            curve = be_meta["curve"]
            d_curve = [r for r in curve if r["sample"] == "discovery"]
            h_curve = [r for r in curve if r["sample"] == "holdout"]
            grid = [int(r["round_trip_cost_bps"]) for r in d_curve]

            fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160, sharey=True)
            for ax_i, (s_title, s_rows) in enumerate(
                [
                    ("A. Discovery (2014–2021, 80 Candidates)", d_curve),
                    ("B. Locked Holdout (2022–2026, 80 Candidates)", h_curve),
                ]
            ):
                ax_c = axes[ax_i]
                ax_c.plot(
                    grid,
                    [r["n_positive_net"] for r in s_rows],
                    marker="o",
                    linewidth=2.0,
                    color="#3b82f6",
                    label="Positive Net Excess Return (> 0)",
                )
                ax_c.plot(
                    grid,
                    [r["n_raw_p05"] for r in s_rows],
                    marker="s",
                    linewidth=2.0,
                    color="#f59e0b",
                    label="Unadjusted Bootstrap p ≤ 0.05",
                )
                ax_c.plot(
                    grid,
                    [r["n_bh_q10"] for r in s_rows],
                    marker="^",
                    linewidth=2.2,
                    color="#10b981",
                    label="Benjamini–Hochberg FDR q ≤ 0.10",
                )
                ax_c.axvline(
                    60,
                    color="#ef4444",
                    linestyle="--",
                    linewidth=1.3,
                    label="Baseline Retail HOSE Cost (60 bps RT)",
                )
                ax_c.set_xticks(grid)
                ax_c.set_xlabel("Round-Trip Transaction Cost Assumption (bps)", fontsize=9.5)
                if ax_i == 0:
                    ax_c.set_ylabel("Number of Surviving Signals (out of 80)", fontsize=9.5)
                ax_c.set_ylim(-2, 68)
                ax_c.set_title(s_title, fontsize=10.2, fontweight="bold")
                ax_c.grid(True, linestyle=":")
                ax_c.legend(loc="upper right", fontsize=8.2)

            fig.suptitle(
                "Exploratory Analysis (c): Long-Only Q5 Signal Survival Across Transaction Cost Assumptions",
                fontsize=11.5,
                fontweight="bold",
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(fig_dir / "08_breakeven_curve.png", bbox_inches="tight")
            plt.close(fig)

    # Copy all figures to docs/assets/research/
    for png_file in fig_dir.glob("*.png"):
        shutil.copy2(png_file, docs_fig_dir / png_file.name)

    # Generate Markdown tables fragment from JSON/CSV results
    n_test = float(funnel["n_tested"])
    n_cand = float(funnel["n_cost_candidates_evaluated"])
    md_lines = [
        "# Generated Factor Zoo Tables (Auto-rendered from JSON/CSV)",
        "",
        "## Funnel Table (Two-Track Falsification)",
        "",
        "### Track A — Statistical Predictability Funnel (Cross-Sectional Rank IC, $h = 5$ Days)",
        "",
        "| Stage | Signals | Share of Registry | Notes |",
        "| :--- | ---: | ---: | :--- |",
        f"| **A1. Tested in Registry** | **{funnel['n_tested']}** | `{100.0 * funnel['n_tested'] / n_test:.1f}%` | 146 unique signals + 4 exact duplicate pairs ($N_{{\\text{{eff}}}} = {disc_summary['effective_independent_tests']['n_eff']:.2f}$) |",
        f"| **A2. Benjamini–Hochberg FDR $q \\le 0.10$** | **{funnel['n_bh_q10']}** | `{100.0 * funnel['n_bh_q10'] / n_test:.1f}%` | Raw Stationary Bootstrap $p \\le 0.05$: **{funnel['n_raw_p05']}** (NW $|t| \\ge 1.96$: **{funnel['n_nw_t196']}**; $\\approx {funnel['expected_false_positives_at_5pct']:.1f}$ under i.i.d. noise) |",
        f"| **A3. Romano–Wolf Stepdown FWER $p_{{\\text{{RW}}}} \\le 0.05$** | **{funnel['n_rw_p05']}** | `{100.0 * funnel['n_rw_p05'] / n_test:.1f}%` | Joint studentized $\\max$-$t$ bootstrap across all {funnel['n_tested']} IC series |",
        f"| **A4. Structurally Stable (`robust` Verdict)** | **{funnel['n_verdict_robust']}** | `{100.0 * funnel['n_verdict_robust'] / n_test:.1f}%` | RW $p \\le 0.05$ and non-decaying (+ **{funnel['n_verdict_fdr_only']}** `fdr_only`, **{funnel['n_verdict_decaying']}** `decaying`, **{funnel['n_verdict_noise']}** `noise`) |",
        f"| **A5. Replicate in Locked Holdout (`{hold_summary['holdout_start']}`–`{hold_summary['holdout_end']}`)** | **{funnel['n_holdout_rw_stat_only']}** | `{100.0 * funnel['n_holdout_rw_stat_only'] / n_test:.1f}%` | **{funnel['n_holdout_rw_stat_only']}** of **{funnel['n_verdict_robust']}** (`{100.0 * funnel['n_holdout_rw_stat_only'] / funnel['n_verdict_robust']:.1f}%`) pass Holdout RW $p \\le 0.05$ (**{funnel['n_holdout_bh_stat_only']}** or `{100.0 * funnel['n_holdout_bh_stat_only'] / funnel['n_verdict_robust']:.1f}%` pass BH $q \\le 0.10$; **{funnel['n_holdout_same_sign_robust']}** or `{100.0 * funnel['n_holdout_same_sign_robust'] / funnel['n_verdict_robust']:.1f}%` retain sign) |",
        "",
        "### Track B — Economic Implementability Funnel (Long-Only Q5 vs. Equal-Weight Universe, HOSE Costs)",
        "",
        "| Stage | Signals | Share of Registry | Share of 80 Candidates |",
        "| :--- | ---: | ---: | ---: |",
        f"| **B1. Non-Decaying Discovery Survivors Evaluated** | **{funnel['n_cost_candidates_evaluated']}** | `{100.0 * funnel['n_cost_candidates_evaluated'] / n_test:.1f}%` | `100.0%` (**{funnel['n_verdict_robust']}** `robust` + **{funnel['n_verdict_fdr_only']}** `fdr_only`) |",
        f"| **B2. Positive Net Excess Return After HOSE Costs (`60 bps` RT)** | **{funnel['n_positive_net_excess']}** | `{100.0 * funnel['n_positive_net_excess'] / n_test:.1f}%` | `{100.0 * funnel['n_positive_net_excess'] / n_cand:.1f}%` |",
        f"| **B3. Unadjusted One-Sided Bootstrap $p \\le 0.05$ on Net Excess** | **{funnel['n_survive_costs_raw_p05']}** | `{100.0 * funnel['n_survive_costs_raw_p05'] / n_test:.1f}%` | `{100.0 * funnel['n_survive_costs_raw_p05'] / n_cand:.1f}%` (`trix_15`, `tsi`) |",
        f"| **B4. Survive Benjamini–Hochberg FDR $q \\le 0.10$ Across Cost Stage** | **{funnel['n_survive_costs_fdr']}** | `{100.0 * funnel['n_survive_costs_fdr'] / n_test:.1f}%` | `{100.0 * funnel['n_survive_costs_fdr'] / n_cand:.1f}%` |",
        f"| **B5. Confirmed Post-Cost Survivors in Locked Holdout** | **{funnel['n_confirmed_holdout_rw']}** | `{100.0 * funnel['n_confirmed_holdout_rw'] / n_test:.1f}%` | `{100.0 * funnel['n_confirmed_holdout_rw'] / n_cand:.1f}%` |",
        "",
        "## Family Breakdown (Discovery 2014–2021, h = 5)",
        "",
        "| Family | Tested | Raw p ≤ 0.05 | BH q ≤ 0.10 | RW p ≤ 0.05 | Robust | FDR-Only | Decaying | Median |IC| | Median |Posterior IC| |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fb in disc_summary["family_breakdown"]:
        md_lines.append(
            f"| `{fb['family']}` | {fb['n_tested']} | {fb['n_raw_p05']} | {fb['n_bh_q10']} | "
            f"{fb['n_rw_p05']} | {fb['n_robust']} | {fb['n_fdr_only']} | {fb['n_decaying']} | "
            f"{fb['median_abs_ic']:.4f} | {fb['median_abs_posterior_ic']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Top Positive Net Excess Return Signals in Cost Stage (Discovery 2014–2021)",
            "",
            "| Factor | Verdict | Gross Excess (5d) | L1 Turnover (5d) | Net Excess (% Ann.) | Net IR (Ann.) | Raw Bootstrap p | BH q | Breakeven Cost (Round-Trip bps) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    pos_cost = cost_table.filter(pl.col("net_excess_mean_5d") > 0.0).sort(
        "net_excess_ann_pct", descending=True
    )
    neg_cost = cost_table.filter(pl.col("net_excess_mean_5d") <= 0.0)
    top6_names = (
        disc_table.with_columns(pl.col("nw_t").abs().alias("_abs_t"))
        .sort("_abs_t", descending=True)
        .head(6)["factor"]
        .to_list()
    )
    top6_cost = cost_table.filter(pl.col("factor").is_in(top6_names))
    alpha018_row = cost_table.filter(pl.col("factor") == "alpha_018").row(0, named=True)

    cost_stage_summary = {
        "top6_abs_t_signals": top6_names,
        "top6_abs_t": _summarize_cost_subset(top6_cost),
        "negative_net": _summarize_cost_subset(neg_cost),
        "positive_net": _summarize_cost_subset(pos_cost),
        "alpha_018_gross_excess_5d": float(alpha018_row["gross_excess_mean_5d"]),
        "alpha_018_breakeven_rt_bps": float(alpha018_row["breakeven_bps_roundtrip"]),
    }
    (disc_dir / "cost_stage_summary.json").write_text(
        json.dumps(cost_stage_summary, indent=2), encoding="utf-8"
    )
    t6 = cost_stage_summary["top6_abs_t"]
    neg_s = cost_stage_summary["negative_net"]
    pos_s = cost_stage_summary["positive_net"]
    md_lines.extend(
        [
            "",
            "### Cost-Stage Descriptive Ranges (Auto-derived from `cost_stage.csv`)",
            "",
            f"- **Top-6 $|t_{{\\text{{NW}}}}|$ fast reversal signals** (`{', '.join(top6_names)}`): "
            f"5-day L1 turnover `{t6['l1_turnover_min']:.2f}–{t6['l1_turnover_max']:.2f}` "
            f"(`{t6['one_way_turnover_pct_min']:.1f}%–{t6['one_way_turnover_pct_max']:.1f}%` one-way replacement), "
            f"annualized cost drag `{t6['cost_drag_ann_pct_min']:.1f}%–{t6['cost_drag_ann_pct_max']:.1f}%/yr`, "
            f"breakeven round-trip cost `{t6['breakeven_rt_bps_min']:.1f}–{t6['breakeven_rt_bps_max']:.1f} bps` "
            f"(`{t6['breakeven_rt_bps_pos_min']:.1f}–{t6['breakeven_rt_bps_pos_max']:.1f} bps` across the 5 positive-gross-excess signals; "
            f"`alpha_018` has negative long-only Q5 gross excess `{cost_stage_summary['alpha_018_gross_excess_5d']:.4f}` and breakeven `{cost_stage_summary['alpha_018_breakeven_rt_bps']:.1f} bps`).",
            f"- **All {neg_s['n_signals']} negative-net-excess candidates**: "
            f"5-day L1 turnover `{neg_s['l1_turnover_min']:.2f}–{neg_s['l1_turnover_max']:.2f}` "
            f"(`{neg_s['one_way_turnover_pct_min']:.1f}%–{neg_s['one_way_turnover_pct_max']:.1f}%` one-way), "
            f"annualized cost drag `{neg_s['cost_drag_ann_pct_min']:.1f}%–{neg_s['cost_drag_ann_pct_max']:.1f}%/yr`, "
            f"breakeven round-trip cost `{neg_s['breakeven_rt_bps_min']:.1f}–{neg_s['breakeven_rt_bps_max']:.1f} bps`.",
            f"- **All {pos_s['n_signals']} positive-net-excess candidates**: "
            f"5-day L1 turnover `{pos_s['l1_turnover_min']:.2f}–{pos_s['l1_turnover_max']:.2f}` "
            f"(`{pos_s['one_way_turnover_pct_min']:.1f}%–{pos_s['one_way_turnover_pct_max']:.1f}%` one-way), "
            f"annualized cost drag `{pos_s['cost_drag_ann_pct_min']:.1f}%–{pos_s['cost_drag_ann_pct_max']:.1f}%/yr`, "
            f"breakeven round-trip cost `{pos_s['breakeven_rt_bps_min']:.1f}–{pos_s['breakeven_rt_bps_max']:.1f} bps`.",
            "",
        ]
    )
    for r in pos_cost.iter_rows(named=True):
        md_lines.append(
            f"| `{r['factor']}` | `{r['verdict_discovery']}` | {r['gross_excess_mean_5d']:.4f} | "
            f"{r['l1_turnover_mean_5d']:.2f} | +{r['net_excess_ann_pct']:.2f}% | "
            f"{r['net_ir_ann']:.2f} | {r['p_boot_greater']:.4f} | {r['q_cost_bh']:.2f} | "
            f"{r['breakeven_bps_roundtrip']:.1f} |"
        )

    hold_table = pl.read_csv(hold_dir / "holdout_table.csv")
    md_lines.extend(
        [
            "",
            "## Top 12 Discovery Survivors & Locked Holdout Replication (2022–2026, h = 5)",
            "",
            "| Factor | Family | Disc IC | Posterior IC | Disc NW t | Disc RW p | Breakeven (RT bps) | Holdout IC | Holdout NW t | Holdout RW p | Retention Ratio | CUSUM p |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    top_hold = (
        hold_table.with_columns(pl.col("nw_t_discovery").abs().alias("_abs_t"))
        .sort("_abs_t", descending=True)
        .drop("_abs_t")
        .head(10)
    )
    extra_cost = hold_table.filter(pl.col("factor").is_in(["trix_15", "tsi"]))
    combined_hold = pl.concat([top_hold, extra_cost]).unique(subset=["factor"], maintain_order=True)
    for r in combined_hold.iter_rows(named=True):
        md_lines.append(
            f"| `{r['factor']}` | `{r['family']}` | {r['ic_discovery']:.4f} | "
            f"{r['posterior_ic_discovery']:.4f} | {r['nw_t_discovery']:.2f} | "
            f"{r['p_rw_discovery']:.4f} | {r['breakeven_bps_roundtrip_discovery']:.1f} | "
            f"{r['ic_holdout']:.4f} | {r['t_nw_holdout']:.2f} | {r['p_rw_holdout']:.4f} | "
            f"{r['retention_ratio']:.2f} | {r['full_sample_cusum_p']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            f"## Appendix A. Excluded Library Exports (`{len(EXCLUDED_SIGNALS)}` Export Groups)",
            "",
            "| Export / Function Group | Family / Module | Exclusion Reason |",
            "| :--- | :--- | :--- |",
        ]
    )
    for export_name, fam, reason in EXCLUDED_SIGNALS:
        md_lines.append(f"| `{export_name}` | `{fam}` | {reason} |")

    timeline_path = results_dir / "audit" / "timeline.txt"
    if timeline_path.exists():
        md_lines.extend(
            [
                "",
                "## Execution Timeline (Auto-derived from `results/audit/timeline.txt`)",
                "",
                "| Commit / Event | Timestamp (`+07:00`) | Description |",
                "| :--- | :--- | :--- |",
            ]
        )
        for raw_line in timeline_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split("|", 2)
            if len(parts) == 3:
                evt, ts, desc = parts
                md_lines.append(f"| `{evt}` | `{ts}` | {desc} |")

    if exp_dir.exists():
        u146_path = exp_dir / "unique_146_sensitivity.json"
        inc_path = exp_dir / "incremental_vs_reversal.json"
        q_path = exp_dir / "quintile_spread.json"
        be_path = exp_dir / "breakeven_curve.json"

        if u146_path.exists() and inc_path.exists():
            u146 = json.loads(u146_path.read_text(encoding="utf-8"))
            inc = json.loads(inc_path.read_text(encoding="utf-8"))
            b_o = inc["before_orthogonalization"]
            a_o = inc["after_orthogonalization"]
            md_lines.extend(
                [
                    "",
                    "## Exploratory Sensitivity & Incremental Predictability vs. 5-Day Reversal",
                    "",
                    "| Specification | Tested | Raw $p \\le 0.05$ | NW $|t| \\ge 1.96$ | BH $q \\le 0.10$ | RW $p \\le 0.05$ | `robust` | `fdr_only` | `decaying` | `noise` | $N_{\\text{eff}}$ | $\\lambda_1$ | Top-5 Share |",
                    "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    f"| **Baseline Frozen Registry (150 Signals)** | {b_o['n_tested']} | {b_o['n_raw_p05']} | {b_o['n_nw_t196']} | {b_o['n_bh_q10']} | {b_o['n_rw_p05']} | {b_o['n_verdict_robust']} | {b_o['n_verdict_fdr_only']} | {b_o['n_verdict_decaying']} | {b_o['n_verdict_noise']} | {b_o['n_eff']:.2f} | {b_o['top1_eigenvalue']:.2f} | {100.0 * b_o['top5_variance_share']:.1f}% |",
                    f"| **Unique Signals Only (146 Signals, Drop 4 Exact Duplicates)** | {u146['n_tested']} | {u146['n_raw_p05']} | {u146['n_nw_t196']} | {u146['n_bh_q10']} | {u146['n_rw_p05']} | {u146['n_verdict_robust']} | {u146['n_verdict_fdr_only']} | {u146['n_verdict_decaying']} | {u146['n_verdict_noise']} | {u146['effective_independent_tests']['n_eff']:.2f} | {u146['effective_independent_tests']['top1_eigenvalue']:.2f} | {100.0 * u146['effective_independent_tests']['top5_variance_share']:.1f}% |",
                    f"| **Orthogonalized Against `rev_5_rank` (150 Signals)** | {a_o['n_tested']} | {a_o['n_raw_p05']} | {a_o['n_nw_t196']} | {a_o['n_bh_q10']} | {a_o['n_rw_p05']} | {a_o['n_verdict_robust']} | {a_o['n_verdict_fdr_only']} | {a_o['n_verdict_decaying']} | {a_o['n_verdict_noise']} | {a_o['n_eff']:.2f} | {a_o['top1_eigenvalue']:.2f} | {100.0 * a_o['top5_variance_share']:.1f}% |",
                ]
            )

        if q_path.exists():
            q_data = json.loads(q_path.read_text(encoding="utf-8"))
            md_lines.extend(
                [
                    "",
                    "## Exploratory Q1–Q5 Annualized Active Return Decomposition (Discovery 2014–2021)",
                    "",
                    "| Factor | Family | Sign | Q1 (% Ann.) | Q2 (% Ann.) | Q3 (% Ann.) | Q4 (% Ann.) | Q5 (% Ann.) | Q5 − Q1 (% Ann.) | Q1 Share of Spread |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for r in q_data["rows"]:
                if r["sample"] == "discovery":
                    md_lines.append(
                        f"| `{r['factor']}` | `{r['family']}` | {r['ic_direction']:+.0f} | "
                        f"{r['q1_excess_ann_pct']:+.2f}% | {r['q2_excess_ann_pct']:+.2f}% | "
                        f"{r['q3_excess_ann_pct']:+.2f}% | {r['q4_excess_ann_pct']:+.2f}% | "
                        f"{r['q5_excess_ann_pct']:+.2f}% | {r['q5_minus_q1_ann_pct']:+.2f}% | "
                        f"{100.0 * r['q1_short_leg_share_of_spread']:.1f}% |"
                    )

        if be_path.exists():
            be_data = json.loads(be_path.read_text(encoding="utf-8"))
            d_map = {
                int(r["round_trip_cost_bps"]): r
                for r in be_data["curve"]
                if r["sample"] == "discovery"
            }
            h_map = {
                int(r["round_trip_cost_bps"]): r
                for r in be_data["curve"]
                if r["sample"] == "holdout"
            }
            md_lines.extend(
                [
                    "",
                    "## Exploratory Breakeven Transaction-Cost Survival Curve (80 Cost-Stage Candidates)",
                    "",
                    "| Round-Trip Cost (bps) | L1 Cost (bps) | Disc Net > 0 | Disc Raw $p \\le 0.05$ | Disc BH $q \\le 0.10$ | Holdout Net > 0 | Holdout Raw $p \\le 0.05$ | Holdout BH $q \\le 0.10$ |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for bps in sorted(d_map):
                dr = d_map[bps]
                hr = h_map[bps]
                md_lines.append(
                    f"| `{bps}` | `{dr['l1_cost_bps']:.1f}` | {dr['n_positive_net']} | "
                    f"{dr['n_raw_p05']} | {dr['n_bh_q10']} | {hr['n_positive_net']} | "
                    f"{hr['n_raw_p05']} | {hr['n_bh_q10']} |"
                )

    tables_path = results_dir / "tables.md"
    tables_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "figures_dir": str(fig_dir),
        "docs_figures_dir": str(docs_fig_dir),
        "tables_md": str(tables_path),
    }


if __name__ == "__main__":
    out = render_all()
    print(json.dumps(out, indent=2))
