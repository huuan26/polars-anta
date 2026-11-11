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
    # Figure 1: Falsification Funnel
    # ------------------------------------------------------------------
    stages = [
        "1. Pre-Registered\nRegistry",
        "2. Raw Bootstrap\np < 0.05",
        "3. Benjamini–Hochberg\nFDR q ≤ 0.10",
        "4. Romano–Wolf\nFWER p ≤ 0.05",
        "5. Survive HOSE\nCosts & Tax (RW)",
        "6. Confirmed in\nHoldout (2022–2026)",
    ]
    counts = [
        int(funnel["n_tested"]),
        int(funnel["n_raw_p05"]),
        int(funnel["n_bh_q10"]),
        int(funnel["n_rw_p05"]),
        int(funnel["n_survive_costs_rw"]),
        int(funnel["n_confirmed_holdout_rw"]),
    ]
    colors = ["#475569", "#f59e0b", "#3b82f6", "#6366f1", "#0ea5e9", "#10b981"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    y_pos = np.arange(len(stages))[::-1]
    bars = ax.barh(y_pos, counts, color=colors, height=0.62, edgecolor="#0f172a", linewidth=0.7)
    exp_fp = float(funnel["expected_false_positives_at_5pct"])
    ax.axvline(
        exp_fp,
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Expected False Positives if All Noise (0.05 × {funnel['n_tested']} = {exp_fp:.1f})",
    )
    for bar, val in zip(bars, counts, strict=True):
        pct = 100.0 * val / float(funnel["n_tested"])
        ax.text(
            bar.get_width() + 2.2,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val}  ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=9.5)
    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_xlabel("Number of Signals Surviving Stage", fontsize=10)
    ax.set_title(
        "The Factor Zoo Falsification Funnel on Vietnamese Equities (HOSE, h = 5 Days)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", linestyle=":")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
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

    # Copy all 5 figures to docs/assets/research/
    for png_file in fig_dir.glob("*.png"):
        shutil.copy2(png_file, docs_fig_dir / png_file.name)

    # Generate Markdown tables fragment from JSON/CSV results
    md_lines = [
        "# Generated Factor Zoo Tables (Auto-rendered from JSON/CSV)",
        "",
        "## Funnel Table",
        "",
        "| Stage | Signals |",
        "| :--- | ---: |",
        f"| Tested (pre-registered registry) | {funnel['n_tested']} |",
        f"| Raw p < 0.05 (Stationary Bootstrap / Newey–West) | {funnel['n_raw_p05']} *(≈ {funnel['expected_false_positives_at_5pct']} expected by chance)* |",
        f"| Benjamini–Hochberg q ≤ 0.10 | {funnel['n_bh_q10']} |",
        f"| Romano–Wolf FWER ≤ 0.05 | {funnel['n_rw_p05']} ({funnel['n_verdict_robust']} non-decaying `robust`) |",
        f"| Survive costs (long-only, VN fees & tax, BH q ≤ 0.10) | {funnel['n_survive_costs_rw']} ({funnel['n_survive_costs_raw_p05']} at raw p ≤ 0.05) |",
        f"| Confirmed in locked holdout ({hold_summary['holdout_start']}–{hold_summary['holdout_end']}) | {funnel['n_confirmed_holdout_rw']} ({funnel['n_holdout_rw_stat_only']} statistical IC survivors) |",
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
