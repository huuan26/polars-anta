"""Cross-sectional alpha diagnostics, margin attribution, sub-universe, and marginal alpha tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from .conditioning import collinearity_summary, neutralize_panel
from .evaluation import (
    ic_decay,
    ic_summary,
    max_drawdown,
    quantile_analysis,
    sharpe_ratio,
)

__all__ = [
    "AlphaDiagnosticReport",
    "analyze_alpha",
    "marginal_alpha_report",
]


@dataclass(frozen=True)
class AlphaDiagnosticReport:
    """Comprehensive cross-sectional alpha tearsheet container."""

    factor_name: str
    ic_stats: dict[str, float]
    implementability: dict[str, float]
    quantiles: pl.DataFrame
    sub_universe: pl.DataFrame
    period_breakdown: pl.DataFrame

    def summary(self) -> dict[str, float]:
        """Return flat scalar metrics for automated alpha screening."""
        return {
            "ic_mean": self.ic_stats["ic_mean"],
            "icir": self.ic_stats["icir"],
            "nw_t_stat": self.ic_stats["nw_t_stat"],
            "half_life": self.ic_stats["half_life"],
            "gross_sharpe": self.implementability["gross_sharpe"],
            "net_sharpe": self.implementability["net_sharpe"],
            "mean_turnover": self.implementability["mean_turnover"],
            "gross_margin_bps": self.implementability["gross_margin_bps"],
            "breakeven_cost_bps": self.implementability["breakeven_cost_bps"],
            "max_drawdown": self.implementability["max_drawdown"],
            "long_leg_share": self.implementability["long_leg_share"],
            "short_leg_share": self.implementability["short_leg_share"],
        }

    def report(self) -> str:
        """Render a concise plain-text diagnostic summary."""
        s = self.summary()
        lines = [
            f"=== Alpha Diagnostic Report: {self.factor_name} ===",
            (
                f"IC Mean: {s['ic_mean']:+.4f} | ICIR: {s['icir']:+.3f} | "
                f"NW t-stat: {s['nw_t_stat']:+.2f} | Half-Life: {s['half_life']:.2f}"
            ),
            (
                f"Gross Sharpe: {s['gross_sharpe']:+.2f} | Net Sharpe: {s['net_sharpe']:+.2f} | "
                f"Max DD: {s['max_drawdown']:.2%}"
            ),
            (
                f"Turnover: {s['mean_turnover']:.3f} | Gross Margin: {s['gross_margin_bps']:+.2f} bps | "
                f"Breakeven Cost: {s['breakeven_cost_bps']:+.2f} bps"
            ),
            (
                f"Spread Attribution -> Long Leg: {s['long_leg_share']:.1%} | "
                f"Short Leg: {s['short_leg_share']:.1%}"
            ),
        ]
        return "\n".join(lines)


def _weights_expr(
    sig_col: str,
    time_col: str,
    scheme: Literal["dollar_neutral", "long_only", "quantile_spread"],
    quantiles: int,
) -> pl.Expr:
    s = pl.col(sig_col)
    if scheme == "dollar_neutral":
        dm = s - s.mean().over(time_col)
        denom = dm.abs().sum().over(time_col)
        return pl.when(denom > 1e-15).then(dm / denom).otherwise(0.0)
    if scheme == "long_only":
        pos = pl.when(s > 0.0).then(s).otherwise(0.0)
        denom = pos.sum().over(time_col)
        return pl.when(denom > 1e-15).then(pos / denom).otherwise(0.0)

    # quantile_spread: +0.5 equal-weight top quantile, -0.5 equal-weight bottom quantile
    cnt = s.count().over(time_col)
    pct = (s.rank(method="ordinal").over(time_col) - 1.0) / cnt.cast(pl.Float64)
    q = (pct * float(quantiles)).floor().clip(0.0, float(quantiles - 1)).cast(pl.Int32) + 1
    is_top = (s.is_not_null()) & (q == quantiles)
    is_bot = (s.is_not_null()) & (q == 1)
    n_top = is_top.cast(pl.Float64).sum().over(time_col)
    n_bot = is_bot.cast(pl.Float64).sum().over(time_col)
    w_long = pl.when(is_top & (n_top > 0.0)).then(0.5 / n_top).otherwise(0.0)
    w_short = pl.when(is_bot & (n_bot > 0.0)).then(-0.5 / n_bot).otherwise(0.0)
    return w_long + w_short


def _extract_period_key(df: pl.DataFrame, time_col: str) -> pl.Expr:
    dtype = df.schema[time_col]
    if dtype in (pl.Date, pl.Datetime):
        return pl.col(time_col).dt.year().cast(pl.String)
    if dtype == pl.String:
        return pl.col(time_col).str.slice(0, 4)
    return pl.lit("P") + ((pl.int_range(0, pl.len()) * 4 // pl.len().clip(lower_bound=1)) + 1).cast(
        pl.String
    )


def analyze_alpha(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str | None = None,
    close_col: str = "close",
    volume_col: str | None = "volume",
    asset_col: str = "ticker",
    time_col: str = "date",
    horizon: int = 1,
    signal_lag: int = 1,
    quantiles: int = 5,
    weight_scheme: Literal["dollar_neutral", "long_only", "quantile_spread"] = "dollar_neutral",
    one_way_cost_bps: float = 10.0,
    adv_window: int = 20,
    sub_universe_bins: int = 3,
    annualization: float = 252.0,
    ic_method: Literal["spearman", "pearson"] = "spearman",
) -> AlphaDiagnosticReport:
    """Run causal cross-sectional alpha diagnostics (IC, Margin bps, Sub-Universe, Yearly)."""
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if signal_lag < 0:
        raise ValueError("signal_lag must be non-negative")
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    if one_way_cost_bps < 0.0:
        raise ValueError("one_way_cost_bps must be non-negative")
    if weight_scheme not in ("dollar_neutral", "long_only", "quantile_spread"):
        raise ValueError(
            "weight_scheme must be 'dollar_neutral', 'long_only', or 'quantile_spread'"
        )

    eager = (df.collect() if isinstance(df, pl.LazyFrame) else df).sort([asset_col, time_col])
    has_close = close_col in eager.columns
    if return_col is None and not has_close:
        raise ValueError("Either return_col or close_col must be present in df")

    work = eager.with_columns(pl.col(factor_col).cast(pl.Float64).fill_nan(None).alias("_anta_f"))

    if return_col is None:
        # Forward return aligned after signal_lag bars: close[t + lag + h - 1] / close[t + lag - 1] - 1
        lag_base = max(0, int(signal_lag) - 1)
        p_start = pl.col(close_col).cast(pl.Float64).shift(-lag_base).over(asset_col)
        p_end = pl.col(close_col).cast(pl.Float64).shift(-(lag_base + int(horizon))).over(asset_col)
        work = work.with_columns(((p_end / p_start) - 1.0).fill_nan(None).alias("_anta_fwd"))
        fwd_col = "_anta_fwd"
    else:
        work = work.with_columns(
            pl.col(return_col).cast(pl.Float64).fill_nan(None).alias("_anta_fwd")
        )
        fwd_col = "_anta_fwd"

    # 1. Predictive IC + Half-Life
    nw_lags = max(1, int(horizon) - 1)
    ic_stats = ic_summary(
        work,
        factor_col="_anta_f",
        return_col=fwd_col,
        group_by=time_col,
        method=ic_method,
        nw_lags=nw_lags,
    )
    if has_close:
        dec_df = ic_decay(
            work,
            factor_col="_anta_f",
            close_col=close_col,
            horizons=(1, 2, 5, 10),
            asset_col=asset_col,
            group_by=time_col,
            method=ic_method,
        )
        ic_stats["half_life"] = float(dec_df["half_life"][0])
    else:
        ic_stats["half_life"] = float("nan")

    # 2. Quantile Monotonicity & Daily Equal-Weight Unbalanced-Safe Long/Short Attribution
    q_df = quantile_analysis(
        work,
        factor_col="_anta_f",
        return_col=fwd_col,
        quantiles=quantiles,
        group_by=time_col,
        annualization=annualization / float(horizon),
    )
    univ_valid = work.filter(pl.col("_anta_f").is_not_null() & pl.col(fwd_col).is_not_null())
    long_share, short_share = 0.5, 0.5
    if univ_valid.height > 0 and q_df.height >= 2:
        pct_q = (pl.col("_anta_f").rank(method="ordinal").over(time_col) - 1.0) / pl.len().over(
            time_col
        )
        q_col = (pct_q * float(quantiles)).floor().clip(0.0, float(quantiles - 1)).cast(
            pl.Int32
        ) + 1
        daily_legs = (
            univ_valid.with_columns(q_col.alias("_q_leg"))
            .group_by(time_col)
            .agg(
                [
                    pl.col(fwd_col).mean().alias("_r_univ"),
                    pl.col(fwd_col).filter(pl.col("_q_leg") == 1).mean().alias("_r_q1"),
                    pl.col(fwd_col).filter(pl.col("_q_leg") == quantiles).mean().alias("_r_qk"),
                ]
            )
            .filter(pl.col("_r_q1").is_not_null() & pl.col("_r_qk").is_not_null())
        )
        if daily_legs.height > 0:
            r_u = float(daily_legs["_r_univ"].mean() or 0.0)
            r_q1 = float(daily_legs["_r_q1"].mean() or 0.0)
            r_qk = float(daily_legs["_r_qk"].mean() or 0.0)
            if ic_stats["ic_mean"] >= 0.0:
                long_act, short_act = r_qk - r_u, r_u - r_q1
            else:
                long_act, short_act = r_q1 - r_u, r_u - r_qk
            tot_spread = long_act + short_act
            if abs(tot_spread) > 1e-15:
                long_share = long_act / tot_spread
                short_share = short_act / tot_spread

    # 3. Causal Portfolio Simulation, Turnover, Margin (bps), and Breakeven Cost
    work = work.with_columns(
        _weights_expr("_anta_f", time_col, weight_scheme, quantiles).alias("_anta_w_raw")
    )
    if has_close:
        r1_expr = (
            pl.col(close_col)
            .cast(pl.Float64)
            .pct_change()
            .over(asset_col)
            .fill_null(0.0)
            .fill_nan(0.0)
        )
        w_book = (
            pl.col("_anta_w_raw")
            .rolling_mean(window_size=int(horizon), min_samples=1)
            .shift(int(signal_lag), fill_value=0.0)
            .over(asset_col)
        )
        ann_factor = float(annualization)
        sim = work.with_columns([w_book.alias("_w"), r1_expr.alias("_r_bar")])
    else:
        # Pre-aligned forward return panel without raw prices
        sim = work.filter(pl.col(fwd_col).is_not_null()).with_columns(
            [pl.col("_anta_w_raw").fill_null(0.0).alias("_w"), pl.col(fwd_col).alias("_r_bar")]
        )
        ann_factor = float(annualization) / float(horizon)

    sim = sim.with_columns(
        (pl.col("_w") - pl.col("_w").shift(1, fill_value=0.0).over(asset_col)).abs().alias("_turn"),
        (pl.col("_w") * pl.col("_r_bar")).alias("_gross"),
    )
    cost_rate = float(one_way_cost_bps) * 1e-4
    ts_df = (
        sim.group_by(time_col)
        .agg(
            [
                pl.col("_turn").sum().alias("turnover"),
                pl.col("_gross").sum().alias("gross_return"),
            ]
        )
        .sort(time_col)
        .with_columns((pl.col("gross_return") - pl.col("turnover") * cost_rate).alias("net_return"))
    )
    # Trim leading zero-position warm-up rows before first trade
    active_ts = ts_df.filter(pl.col("turnover").cum_sum() > 0.0)
    if active_ts.height == 0:
        active_ts = ts_df

    g_arr = active_ts["gross_return"].to_numpy()
    n_arr = active_ts["net_return"].to_numpy()
    t_arr = active_ts["turnover"].to_numpy()
    tot_turn = float(np.sum(t_arr))
    gross_margin_bps = (1e4 * float(np.sum(g_arr)) / tot_turn) if tot_turn > 1e-15 else 0.0
    net_margin_bps = gross_margin_bps - float(one_way_cost_bps)

    def _arr_sharpe(r: np.ndarray, ann: float) -> float:
        if len(r) < 2:
            return 0.0
        sd = float(np.std(r, ddof=1))
        return float(np.mean(r) / sd * math.sqrt(ann)) if sd > 1e-15 else 0.0

    mdd_val = float(
        active_ts.select(max_drawdown(pl.col("net_return"), is_returns=True)).item() or 0.0
    )
    implementability = {
        "gross_sharpe": _arr_sharpe(g_arr, ann_factor),
        "net_sharpe": _arr_sharpe(n_arr, ann_factor),
        "gross_return_ann": float(np.mean(g_arr)) * ann_factor if len(g_arr) else 0.0,
        "net_return_ann": float(np.mean(n_arr)) * ann_factor if len(n_arr) else 0.0,
        "mean_turnover": float(np.mean(t_arr)) if len(t_arr) else 0.0,
        "gross_margin_bps": gross_margin_bps,
        "net_margin_bps": net_margin_bps,
        "breakeven_cost_bps": gross_margin_bps,
        "max_drawdown": mdd_val,
        "long_leg_share": float(long_share),
        "short_leg_share": float(short_share),
    }

    # 4. Strictly Causal Sub-Universe Breakdown by Trailing ADV
    sub_rows: list[dict[str, Any]] = []
    if (
        has_close
        and volume_col is not None
        and volume_col in work.columns
        and sub_universe_bins >= 2
    ):
        adv_expr = (
            (pl.col(close_col).cast(pl.Float64) * pl.col(volume_col).cast(pl.Float64))
            .rolling_median(window_size=max(1, int(adv_window)), min_samples=1)
            .over(asset_col)
        )
        with_adv = work.with_columns(adv_expr.alias("_anta_adv")).filter(
            pl.col("_anta_adv").is_not_null()
            & pl.col("_anta_f").is_not_null()
            & pl.col(fwd_col).is_not_null()
        )
        if with_adv.height > 0:
            pct_adv = (
                pl.col("_anta_adv").rank(method="ordinal").over(time_col) - 1.0
            ) / pl.len().over(time_col)
            b_expr = (pct_adv * float(sub_universe_bins)).floor().clip(
                0.0, float(sub_universe_bins - 1)
            ).cast(pl.Int32) + 1
            binned = with_adv.sort([asset_col, time_col]).with_columns(b_expr.alias("_adv_bin"))
            labels = {1: "Illiquid/Small", sub_universe_bins: "Liquid/Large"}
            for b_id in range(1, int(sub_universe_bins) + 1):
                sub_b = binned.filter(pl.col("_adv_bin") == b_id)
                if sub_b.height < 4:
                    continue
                b_ic = ic_summary(
                    sub_b,
                    factor_col="_anta_f",
                    return_col=fwd_col,
                    group_by=time_col,
                    method=ic_method,
                    nw_lags=nw_lags,
                )
                w_b = sub_b.with_columns(
                    _weights_expr("_anta_f", time_col, weight_scheme, quantiles).alias("_wb")
                )
                # Map tier weights onto full sorted panel to measure exact tier turnover per asset
                tier_panel = (
                    binned.join(
                        w_b.select([asset_col, time_col, "_wb"]),
                        on=[asset_col, time_col],
                        how="left",
                    )
                    .sort([asset_col, time_col])
                    .with_columns(pl.col("_wb").fill_null(0.0).alias("_wb_full"))
                    .with_columns(
                        (
                            pl.col("_wb_full")
                            - pl.col("_wb_full").shift(int(horizon), fill_value=0.0).over(asset_col)
                        )
                        .abs()
                        .alias("_tb")
                    )
                )
                ts_b = (
                    tier_panel.group_by(time_col)
                    .agg(
                        [
                            (pl.col("_wb_full") * pl.col(fwd_col)).sum().alias("r"),
                            pl.col("_tb").sum().alias("t"),
                        ]
                    )
                    .sort(time_col)
                )
                ret_b = ts_b["r"].to_numpy()
                turn_b = ts_b["t"].to_numpy()
                sum_tb = float(np.sum(turn_b))
                margin_b = (1e4 * float(np.sum(ret_b)) / sum_tb) if sum_tb > 1e-15 else 0.0
                sub_rows.append(
                    {
                        "bucket": b_id,
                        "label": labels.get(b_id, f"Tier-{b_id}"),
                        "ic_mean": b_ic["ic_mean"],
                        "nw_t_stat": b_ic["nw_t_stat"],
                        "gross_sharpe": _arr_sharpe(
                            ret_b, ann_factor / float(horizon) if has_close else ann_factor
                        ),
                        "mean_turnover": float(np.mean(turn_b)) / float(horizon)
                        if len(turn_b)
                        else 0.0,
                        "gross_margin_bps": margin_b,
                        "n_obs": int(sub_b.height),
                    }
                )
    sub_df = (
        pl.DataFrame(sub_rows)
        if sub_rows
        else pl.DataFrame(
            schema={
                "bucket": pl.Int32,
                "label": pl.String,
                "ic_mean": pl.Float64,
                "nw_t_stat": pl.Float64,
                "gross_sharpe": pl.Float64,
                "mean_turnover": pl.Float64,
                "gross_margin_bps": pl.Float64,
                "n_obs": pl.Int64,
            }
        )
    )

    # 5. Period / Calendar-Year Stability Breakdown
    p_expr = _extract_period_key(active_ts, time_col).alias("period")
    ts_p = active_ts.with_columns(p_expr)
    period_df = ts_p.group_by("period", maintain_order=True).agg(
        [
            pl.len().cast(pl.Int64).alias("n_periods"),
            sharpe_ratio(pl.col("gross_return"), annualization=ann_factor).alias("gross_sharpe"),
            sharpe_ratio(pl.col("net_return"), annualization=ann_factor).alias("net_sharpe"),
            pl.col("turnover").mean().alias("mean_turnover"),
            max_drawdown(pl.col("net_return"), is_returns=True).alias("max_drawdown"),
        ]
    )

    return AlphaDiagnosticReport(
        factor_name=factor_col,
        ic_stats=ic_stats,
        implementability=implementability,
        quantiles=q_df,
        sub_universe=sub_df,
        period_breakdown=period_df,
    )


def marginal_alpha_report(
    df: pl.DataFrame | pl.LazyFrame,
    candidate_col: str,
    benchmark_cols: Sequence[str],
    return_col: str,
    time_col: str = "date",
    horizon: int = 1,
    ic_method: Literal["spearman", "pearson"] = "spearman",
) -> dict[str, float]:
    """Measure incremental dimension ($\\Delta N_{\\text{eff}}$), spanned $R^2$, and FWL Residual IC."""
    benches = list(benchmark_cols)
    if not benches:
        raise ValueError("benchmark_cols must contain at least one column")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")

    all_cols = [*benches, candidate_col]
    eager = df.collect() if isinstance(df, pl.LazyFrame) else df
    valid = pl.all_horizontal(
        [pl.col(c).cast(pl.Float64).fill_nan(None).is_not_null() for c in [*all_cols, return_col]]
    )
    clean = eager.filter(valid)
    if clean.height < 4:
        raise ValueError("At least 4 complete observations are required for marginal_alpha_report")

    # Cross-sectionally standardize candidate and benchmarks to unit variance per period
    z_exprs = [
        (
            (pl.col(c).cast(pl.Float64) - pl.col(c).cast(pl.Float64).mean().over(time_col))
            / pl.col(c).cast(pl.Float64).std().over(time_col)
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias(f"_z_{c}")
        for c in all_cols
    ]
    z_df = clean.with_columns(z_exprs)
    z_benches = [f"_z_{c}" for c in benches]
    z_cand = f"_z_{candidate_col}"

    n_eff_before = float(collinearity_summary(z_df, z_benches, time_col=None)["effective_rank"][0])
    n_eff_after = float(
        collinearity_summary(z_df, [*z_benches, z_cand], time_col=None)["effective_rank"][0]
    )

    mat_b = z_df.select(z_benches).to_numpy()
    vec_c = z_df[z_cand].to_numpy()
    sd_c = float(np.std(vec_c, ddof=1))
    corrs: list[float] = []
    for j in range(len(z_benches)):
        sd_b = float(np.std(mat_b[:, j], ddof=1))
        if sd_c > 1e-15 and sd_b > 1e-15:
            c_val = float(np.corrcoef(vec_c, mat_b[:, j])[0, 1])
            corrs.append(abs(c_val) if math.isfinite(c_val) else 0.0)
        else:
            corrs.append(0.0)
    max_abs_corr = max(corrs) if corrs else 0.0

    # FWL orthogonal projection of standardized candidate against benchmarks within each time_col
    neut_df = neutralize_panel(
        z_df,
        factor_cols=[z_cand],
        time_col=time_col,
        risk_cols=z_benches,
        standardize=False,
        suffix="_res",
    )
    res_col = f"{z_cand}_res"
    res_vec = neut_df[res_col].fill_null(0.0).to_numpy()
    var_cand = float(np.var(vec_c, ddof=1))
    var_res = float(np.var(res_vec, ddof=1))
    spanned_r2 = float(np.clip(1.0 - (var_res / var_cand), 0.0, 1.0)) if var_cand > 1e-15 else 1.0

    nw_lags = max(1, int(horizon) - 1)
    raw_ic = ic_summary(
        z_df,
        factor_col=z_cand,
        return_col=return_col,
        group_by=time_col,
        method=ic_method,
        nw_lags=nw_lags,
    )
    if var_res <= 1e-12:
        res_ic_mean, res_nw_t = 0.0, 0.0
    else:
        res_ic = ic_summary(
            neut_df,
            factor_col=res_col,
            return_col=return_col,
            group_by=time_col,
            method=ic_method,
            nw_lags=nw_lags,
        )
        res_ic_mean = float(res_ic["ic_mean"])
        res_nw_t = float(res_ic["nw_t_stat"])

    return {
        "raw_ic_mean": float(raw_ic["ic_mean"]),
        "raw_nw_t_stat": float(raw_ic["nw_t_stat"]),
        "residual_ic_mean": res_ic_mean,
        "residual_nw_t_stat": res_nw_t,
        "max_abs_corr": max_abs_corr,
        "spanned_r2": spanned_r2,
        "n_eff_before": n_eff_before,
        "n_eff_after": n_eff_after,
        "delta_n_eff": n_eff_after - n_eff_before,
    }
