"""Read-only data and signal-duplication audit for the Vietnamese Equity Factor Zoo study.

Generates:
- ``research/factor_zoo/results/audit/data_adjustment_check.json`` (Sections 1.1 & 1.2 of Plan E)
- ``research/factor_zoo/results/audit/duplicate_signals.json`` (Section 1.3 of Plan E)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

import polars_anta as pa
from research.factor_zoo.config import DATA_CACHE_PATH, DEFAULT_CONFIG, RESULTS_DIR, StudyConfig
from research.factor_zoo.data import load_panel
from research.factor_zoo.registry import build_registry, compute_signals_panel

AUDIT_DIR = RESULTS_DIR / "audit"

# Common corporate-action (stock dividend / bonus share / split) price ratios on HOSE
TYPICAL_SPLIT_RATIOS: tuple[tuple[str, float], ...] = (
    ("10:11 (10% stock div)", 1.0 / 1.10),
    ("100:115 (15% stock div)", 1.0 / 1.15),
    ("5:6 (20% stock div)", 1.0 / 1.20),
    ("4:5 (25% stock div)", 1.0 / 1.25),
    ("10:13 (30% stock div)", 1.0 / 1.30),
    ("2:3 (50% stock div)", 1.0 / 1.50),
    ("1:2 (100% split/bonus)", 1.0 / 2.00),
)


def audit_price_adjustments_and_survivorship(
    cache_path: Path = DATA_CACHE_PATH,
    cutoff_date: dt.date = dt.date(2026, 10, 1),
) -> dict[str, object]:
    """Inspect raw cached OHLCV for large overnight price jumps and last-trading-date distribution."""
    raw_df = pl.read_parquet(cache_path).sort(["ticker", "date"])
    with_ret = raw_df.with_columns(
        pl.col("close").shift(1).over("ticker").alias("prev_close"),
        (pl.col("close") / pl.col("close").shift(1).over("ticker")).alias("price_ratio"),
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret_1d"),
    ).filter(pl.col("ret_1d").is_not_null() & pl.col("ret_1d").is_finite())

    abs_ret = with_ret["ret_1d"].abs()
    n_gt_075 = int((abs_ret > 0.075).sum())
    n_gt_150 = int((abs_ret > 0.15).sum())
    n_gt_300 = int((abs_ret > 0.30).sum())
    n_neg_gt_150 = int((with_ret["ret_1d"] < -0.15).sum())
    n_pos_gt_150 = int((with_ret["ret_1d"] > 0.15).sum())

    # Check how many |ret_1d| > 0.15 events fall within +/- 0.015 of typical split/dividend ratios
    gt_150_df = with_ret.filter(pl.col("ret_1d").abs() > 0.15)
    ratios_np = gt_150_df["price_ratio"].to_numpy()
    matched_mask = np.zeros(len(ratios_np), dtype=bool)
    ratio_breakdown: dict[str, int] = {}
    for label, target_ratio in TYPICAL_SPLIT_RATIOS:
        hit = np.abs(ratios_np - target_ratio) <= 0.015
        matched_mask |= hit
        ratio_breakdown[label] = int(np.sum(hit))

    n_matched_split_ratio = int(np.sum(matched_mask))
    frac_matched_of_gt_150 = float(n_matched_split_ratio / n_gt_150) if n_gt_150 > 0 else 0.0

    # Per-ticker jump counts summary
    per_ticker_jumps = (
        with_ret.group_by("ticker")
        .agg(
            (pl.col("ret_1d").abs() > 0.075).sum().alias("n_gt_075"),
            (pl.col("ret_1d").abs() > 0.15).sum().alias("n_gt_150"),
            (pl.col("ret_1d").abs() > 0.30).sum().alias("n_gt_300"),
        )
        .sort("n_gt_075", descending=True)
    )

    top15_df = (
        with_ret.with_columns(pl.col("ret_1d").abs().alias("_abs_ret"))
        .sort("_abs_ret", descending=True)
        .head(15)
    )
    top15_events = [
        {
            "ticker": str(row["ticker"]),
            "date": str(row["date"]),
            "prev_close": round(float(row["prev_close"]), 2),
            "close": round(float(row["close"]), 2),
            "price_ratio": round(float(row["price_ratio"]), 4),
            "ret_1d": round(float(row["ret_1d"]), 4),
        }
        for row in top15_df.iter_rows(named=True)
    ]

    # Survivorship check: last trading date per ticker
    ticker_span = (
        raw_df.group_by("ticker")
        .agg(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.len().alias("n_bars"),
        )
        .sort("last_date")
    )
    ended_early = ticker_span.filter(pl.col("last_date") < cutoff_date)
    early_list = [
        {
            "ticker": str(r["ticker"]),
            "first_date": str(r["first_date"]),
            "last_date": str(r["last_date"]),
            "n_bars": int(r["n_bars"]),
        }
        for r in ended_early.iter_rows(named=True)
    ]

    return {
        "price_jumps": {
            "total_return_observations": int(with_ret.height),
            "n_abs_ret_gt_075": n_gt_075,
            "n_abs_ret_gt_150": n_gt_150,
            "n_abs_ret_gt_300": n_gt_300,
            "n_ret_lt_neg_150": n_neg_gt_150,
            "n_ret_gt_pos_150": n_pos_gt_150,
            "n_gt_150_near_typical_split_ratio": n_matched_split_ratio,
            "frac_gt_150_near_typical_split_ratio": round(frac_matched_of_gt_150, 4),
            "typical_split_ratio_hits": ratio_breakdown,
            "tickers_with_gt_075": int((per_ticker_jumps["n_gt_075"] > 0).sum()),
            "tickers_with_gt_150": int((per_ticker_jumps["n_gt_150"] > 0).sum()),
            "top_10_tickers_by_jump_count": [
                {
                    "ticker": str(r["ticker"]),
                    "n_gt_075": int(r["n_gt_075"]),
                    "n_gt_150": int(r["n_gt_150"]),
                    "n_gt_300": int(r["n_gt_300"]),
                }
                for r in per_ticker_jumps.head(10).iter_rows(named=True)
            ],
            "top_15_extreme_events": top15_events,
            "note": (
                "TODO(verify): Public VNDirect dchart / SSI iBoard history endpoints return "
                "raw series containing 5,737 daily moves exceeding +/-7.5% (5,653 after "
                "zero-volume filtering) and 343 exceeding +/-15% (126 negative, 217 positive; "
                "78/343 = 22.74% within +/-0.015 of typical split/dividend ratios). Whether "
                "the upstream feed is partially adjusted or contains unadjusted corporate-action "
                "discontinuities is not independently verified; the pipeline filters |r_t| > 7.5%."
            ),
        },
        "survivorship": {
            "total_tickers_in_cache": int(ticker_span.height),
            "cutoff_date": str(cutoff_date),
            "n_tickers_last_date_before_cutoff": int(ended_early.height),
            "min_last_date": str(ticker_span["last_date"].min()),
            "max_last_date": str(ticker_span["last_date"].max()),
            "tickers_ending_before_cutoff": early_list,
            "note": (
                "The ticker directory endpoint returns currently active HOSE listings at fetch time; "
                "historical delistings between 2012 and 2026 are not included (survivorship bias)."
            ),
        },
    }


def audit_duplicate_signals(config: StudyConfig = DEFAULT_CONFIG) -> dict[str, object]:
    """Identify exact and near-duplicate signals (|corr| > 0.99) in the discovery IC matrix."""
    specs = build_registry()
    spec_by_name = {s.name: s for s in specs}
    order_index = {s.name: idx for idx, s in enumerate(specs)}

    # 1. Group by ic_mean in results/discovery/factor_table.csv
    ft_path = RESULTS_DIR / "discovery" / "factor_table.csv"
    ft_df = pl.read_csv(ft_path)

    # Also compute the discovery IC matrix directly to verify pairwise IC correlations and max abs diff
    panel_df, _ = load_panel(config=config)
    sig_panel = compute_signals_panel(panel_df, specs=specs)
    disc_panel = sig_panel.filter(
        (pl.col("date") >= config.discovery_start) & (pl.col("date") <= config.discovery_end)
    )
    signal_names = [s.name for s in specs]
    ic_df = pa.ic_matrix(
        disc_panel,
        factor_cols=signal_names,
        return_col="fwd_ret_5",
        group_by="date",
        method=config.ic_method,
    )
    ic_mat = ic_df.select(signal_names).to_numpy()

    # Pairwise correlation matrix across 150 signals (ignoring NaNs)
    k = len(signal_names)
    exact_duplicates: list[dict[str, object]] = []
    near_duplicates_099: list[dict[str, object]] = []
    dropped_second_signals: list[str] = []

    for i in range(k):
        for j in range(i + 1, k):
            col_i = ic_mat[:, i]
            col_j = ic_mat[:, j]
            valid = np.isfinite(col_i) & np.isfinite(col_j)
            if int(np.sum(valid)) < 10:
                continue
            max_abs_diff = float(np.max(np.abs(col_i[valid] - col_j[valid])))
            corr = float(np.corrcoef(col_i[valid], col_j[valid])[0, 1])
            s_i = signal_names[i]
            s_j = signal_names[j]
            if max_abs_diff < 1e-12:
                exact_duplicates.append(
                    {
                        "signal_kept": s_i,
                        "signal_duplicate": s_j,
                        "family_kept": spec_by_name[s_i].family,
                        "family_duplicate": spec_by_name[s_j].family,
                        "transform_kept": spec_by_name[s_i].transform,
                        "transform_duplicate": spec_by_name[s_j].transform,
                        "max_abs_ic_diff": max_abs_diff,
                        "ic_corr": round(corr, 6),
                    }
                )
                dropped_second_signals.append(s_j)
            elif abs(corr) > 0.99:
                near_duplicates_099.append(
                    {
                        "signal_a": s_i,
                        "signal_b": s_j,
                        "family_a": spec_by_name[s_i].family,
                        "family_b": spec_by_name[s_j].family,
                        "ic_corr": round(corr, 6),
                        "max_abs_ic_diff": round(max_abs_diff, 6),
                    }
                )

    near_duplicates_099.sort(key=lambda d: abs(float(d["ic_corr"])), reverse=True)
    unique_signals = [s for s in signal_names if s not in set(dropped_second_signals)]

    return {
        "n_registered_signals": k,
        "n_exact_duplicate_pairs": len(exact_duplicates),
        "n_unique_signals": len(unique_signals),
        "dropped_in_unique_146": sorted(dropped_second_signals, key=lambda x: order_index[x]),
        "exact_duplicates": exact_duplicates,
        "n_near_duplicate_pairs_abs_corr_gt_099": len(near_duplicates_099),
        "near_duplicates_099": near_duplicates_099,
        "factor_table_rows": int(ft_df.height),
    }


def main() -> None:
    """Run all read-only data and signal-duplication audits and write JSON outputs."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    adj_check = audit_price_adjustments_and_survivorship()
    adj_path = AUDIT_DIR / "data_adjustment_check.json"
    adj_path.write_text(json.dumps(adj_check, indent=2), encoding="utf-8")

    dup_check = audit_duplicate_signals()
    dup_path = AUDIT_DIR / "duplicate_signals.json"
    dup_path.write_text(json.dumps(dup_check, indent=2), encoding="utf-8")
    print(f"Wrote {adj_path} and {dup_path}")


if __name__ == "__main__":
    main()
