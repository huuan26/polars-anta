"""Unit, accounting identity, and causal sub-universe tests for polars_anta.diagnostics."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_anta.diagnostics import analyze_alpha, marginal_alpha_report


def _build_panel(seed: int = 42, n_days: int = 80, n_tickers: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates: list[date] = []
    cur = date(2024, 6, 3)
    while len(dates) < n_days:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)

    frames: list[pl.DataFrame] = []
    for i in range(n_tickers):
        # First 10 tickers are illiquid small-caps; last 20 are liquid large-caps
        is_small = i < 10
        base_vol = 20_000.0 if is_small else 2_500_000.0
        sig = rng.standard_normal(n_days)
        # Planted signal works strongly in small-caps and moderately in large-caps
        alpha_beta = 0.008 if is_small else 0.003
        noise = rng.normal(0.0, 0.015, size=n_days)
        rets = np.zeros(n_days, dtype=np.float64)
        rets[1:] = alpha_beta * sig[:-1] + noise[1:]
        close = 50.0 * np.exp(np.cumsum(rets))
        vol = base_vol * np.exp(rng.normal(0.0, 0.2, size=n_days))
        fwd_1 = np.empty(n_days, dtype=np.float64)
        fwd_1[:-1] = close[1:] / close[:-1] - 1.0
        fwd_1[-1] = np.nan

        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ticker": [f"TK{i:02d}"] * n_days,
                    "close": close,
                    "volume": vol,
                    "alpha_pos": sig,
                    "alpha_neg": -sig,
                    "fwd_ret_1": fwd_1,
                }
            )
        )
    return pl.concat(frames).sort(["date", "ticker"])


def test_analyze_alpha_breakeven_margin_accounting_identity_and_schemes() -> None:
    """At cost == breakeven_cost_bps, net return and net margin must equal 0.0 to 1e-12."""
    panel = _build_panel(seed=42)

    rep = analyze_alpha(
        panel,
        factor_col="alpha_pos",
        return_col=None,
        close_col="close",
        volume_col="volume",
        asset_col="ticker",
        time_col="date",
        horizon=1,
        signal_lag=1,
        weight_scheme="dollar_neutral",
        one_way_cost_bps=10.0,
    )
    s = rep.summary()
    assert s["ic_mean"] > 0.05
    assert s["gross_sharpe"] > s["net_sharpe"]
    assert np.isclose(s["long_leg_share"] + s["short_leg_share"], 1.0, atol=1e-12)
    assert "Alpha Diagnostic Report: alpha_pos" in rep.report()
    assert rep.period_breakdown.height >= 1
    assert set(rep.period_breakdown["period"].to_list()).issubset({"2024", "2025"})

    # Exact Breakeven Cost Identity (net_return_ann == 0.0 and net_margin_bps == 0.0 at breakeven)
    be_bps = rep.implementability["breakeven_cost_bps"]
    assert be_bps > 0.0
    rep_be = analyze_alpha(
        panel,
        factor_col="alpha_pos",
        return_col=None,
        close_col="close",
        volume_col="volume",
        asset_col="ticker",
        time_col="date",
        horizon=1,
        signal_lag=1,
        weight_scheme="dollar_neutral",
        one_way_cost_bps=be_bps,
    )
    assert np.isclose(rep_be.implementability["net_return_ann"], 0.0, atol=1e-11)
    assert np.isclose(rep_be.implementability["net_margin_bps"], 0.0, atol=1e-11)
    assert np.isclose(rep_be.implementability["net_sharpe"], 0.0, atol=1e-11)

    # Sub-universe liquidity breakdown detects stronger IC in illiquid tier (bucket 1) than tier 3
    assert rep.sub_universe.height == 3
    ic_small = float(rep.sub_universe.filter(pl.col("bucket") == 1)["ic_mean"][0])
    ic_large = float(rep.sub_universe.filter(pl.col("bucket") == 3)["ic_mean"][0])
    assert ic_small > ic_large > 0.0

    # Negative-IC factor orientation + quantile_spread & long_only weight schemes
    for scheme in ("long_only", "quantile_spread"):
        rep_neg = analyze_alpha(
            panel.lazy(),
            factor_col="alpha_neg",
            return_col="fwd_ret_1",
            weight_scheme=scheme,  # type: ignore[arg-type]
            horizon=2,
        )
        assert rep_neg.ic_stats["ic_mean"] < -0.05
        assert np.isclose(
            rep_neg.implementability["long_leg_share"]
            + rep_neg.implementability["short_leg_share"],
            1.0,
            atol=1e-12,
        )


def test_analyze_alpha_causal_adv_sub_universe_invariance_and_factor_only_mode() -> None:
    """Future volume/price shocks after t_cut must have zero effect on past ADV sub-universe."""
    panel = _build_panel(seed=101, n_days=50, n_tickers=24)
    all_dates = panel.select("date").unique().sort("date")["date"].to_list()
    t_cut = all_dates[34]

    # Mutate volume and close wildly on dates > t_cut
    corrupted = panel.with_columns(
        pl.when(pl.col("date") > t_cut)
        .then(pl.col("volume") * 1e7)
        .otherwise(pl.col("volume"))
        .alias("volume"),
        pl.when(pl.col("date") > t_cut)
        .then(pl.col("close") * 50.0)
        .otherwise(pl.col("close"))
        .alias("close"),
    )

    rep_clean = analyze_alpha(
        panel.filter(pl.col("date") <= t_cut),
        factor_col="alpha_pos",
        return_col="fwd_ret_1",
    )
    rep_corrupt_prefix = analyze_alpha(
        corrupted.filter(pl.col("date") <= t_cut),
        factor_col="alpha_pos",
        return_col="fwd_ret_1",
    )
    np.testing.assert_allclose(
        rep_clean.sub_universe["ic_mean"].to_numpy(),
        rep_corrupt_prefix.sub_universe["ic_mean"].to_numpy(),
        atol=1e-13,
    )

    # Factor-only mode (no close/volume columns, integer time_col)
    slim = panel.with_columns(pl.col("date").rank("dense").cast(pl.Int32).alias("step")).select(
        ["step", "ticker", "alpha_pos", "fwd_ret_1"]
    )
    rep_slim = analyze_alpha(
        slim,
        factor_col="alpha_pos",
        return_col="fwd_ret_1",
        time_col="step",
        asset_col="ticker",
        volume_col=None,
    )
    assert rep_slim.sub_universe.height == 0
    assert rep_slim.period_breakdown.height >= 1
    assert math.isnan(rep_slim.ic_stats["half_life"])


def test_analyze_alpha_unbalanced_spread_attribution_and_sub_universe_margins() -> None:
    """Daily equal-weight spread attribution must remain unbiased on expanding universes."""
    rng = np.random.default_rng(707)
    rows: list[dict[str, object]] = []
    for d in range(40):
        # Early bull regime has 10 stocks; later bear regime has 50 stocks
        n_stk = 10 if d < 15 else 50
        mkt_ret = 0.045 if d < 15 else -0.015
        sig = np.linspace(-1.0, 1.0, n_stk)
        ret = mkt_ret + 0.01 * sig + 0.002 * rng.standard_normal(n_stk)
        for i in range(n_stk):
            rows.append(
                {
                    "date": f"2025-{(d // 20) + 1:02d}-{(d % 20) + 1:02d}",
                    "ticker": f"U{i:02d}",
                    "close": 100.0 * (1.0 + float(ret[i])),
                    "volume": 100_000.0 * (i + 1),
                    "sig": float(sig[i]),
                    "fwd_ret": float(ret[i]),
                }
            )
    unbal = pl.DataFrame(rows)
    rep = analyze_alpha(
        unbal,
        factor_col="sig",
        return_col="fwd_ret",
        close_col="close",
        volume_col="volume",
        quantiles=5,
    )
    assert np.isclose(
        rep.implementability["long_leg_share"] + rep.implementability["short_leg_share"],
        1.0,
        atol=1e-12,
    )
    # Symmetric linear signal around daily market return must attribute ~50% long / ~50% short
    assert abs(rep.implementability["long_leg_share"] - 0.5) < 0.05
    assert "mean_turnover" in rep.sub_universe.columns
    assert "gross_margin_bps" in rep.sub_universe.columns
    assert np.all(rep.sub_universe["mean_turnover"].to_numpy() > 0.0)


def test_marginal_alpha_report_redundancy_vs_orthogonal_edge() -> None:
    """Redundant linear combination must yield spanned_r2=1 & residual_ic=0; pure alpha > 0."""
    rng = np.random.default_rng(2026)
    n_days, n_tickers = 60, 40
    rows: list[dict[str, object]] = []
    for d in range(n_days):
        # Generate cross-sectionally standardized benchmarks so linear combo stays exact after zscore
        b1 = rng.standard_normal(n_tickers)
        b1 = (b1 - b1.mean()) / b1.std(ddof=1)
        b2 = rng.standard_normal(n_tickers)
        b2 = (b2 - b2.mean()) / b2.std(ddof=1)
        pure_new = rng.standard_normal(n_tickers)
        # Orthogonalize pure_new wrt b1, b2 so it is 100% independent
        pure_new = (
            pure_new - ((pure_new @ b1) / (b1 @ b1)) * b1 - ((pure_new @ b2) / (b2 @ b2)) * b2
        )
        pure_new = (pure_new - pure_new.mean()) / pure_new.std(ddof=1)

        ret = 0.06 * b1 + 0.09 * pure_new + 0.4 * rng.standard_normal(n_tickers)
        cand_redundant = 2.5 * b1 - 0.8 * b2
        cand_novel = 0.5 * b1 + 0.866 * pure_new

        for i in range(n_tickers):
            rows.append(
                {
                    "date": f"2026-{(d // 20) + 1:02d}-{(d % 20) + 1:02d}",
                    "ticker": f"A{i:02d}",
                    "b1": float(b1[i]),
                    "b2": float(b2[i]),
                    "cand_red": float(cand_redundant[i]),
                    "cand_new": float(cand_novel[i]),
                    "fwd_ret": float(ret[i]),
                }
            )
    df = pl.DataFrame(rows)

    rep_red = marginal_alpha_report(
        df,
        candidate_col="cand_red",
        benchmark_cols=["b1", "b2"],
        return_col="fwd_ret",
        time_col="date",
        ic_method="pearson",
    )
    assert np.isclose(rep_red["spanned_r2"], 1.0, atol=1e-10)
    assert rep_red["delta_n_eff"] <= 0.05
    assert rep_red["residual_ic_mean"] == 0.0
    assert rep_red["residual_nw_t_stat"] == 0.0

    rep_new = marginal_alpha_report(
        df,
        candidate_col="cand_new",
        benchmark_cols=["b1", "b2"],
        return_col="fwd_ret",
        time_col="date",
        ic_method="pearson",
    )
    assert rep_new["spanned_r2"] < 0.35
    assert rep_new["delta_n_eff"] > 0.5
    assert rep_new["residual_ic_mean"] > 0.08
    assert rep_new["residual_nw_t_stat"] > 3.0


def test_diagnostics_validation_guards() -> None:
    df = pl.DataFrame(
        {
            "date": ["2026-01-01"] * 5,
            "ticker": ["A", "B", "C", "D", "E"],
            "f": [1, 2, 3, 4, 5],
            "r": [0.1] * 5,
        }
    )
    with pytest.raises(ValueError, match="horizon"):
        analyze_alpha(df, "f", return_col="r", horizon=0)
    with pytest.raises(ValueError, match="signal_lag"):
        analyze_alpha(df, "f", return_col="r", signal_lag=-1)
    with pytest.raises(ValueError, match="quantiles"):
        analyze_alpha(df, "f", return_col="r", quantiles=1)
    with pytest.raises(ValueError, match="one_way_cost_bps"):
        analyze_alpha(df, "f", return_col="r", one_way_cost_bps=-5.0)
    with pytest.raises(ValueError, match="weight_scheme"):
        analyze_alpha(df, "f", return_col="r", weight_scheme="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="return_col or close_col"):
        analyze_alpha(df.drop("r"), "f")
    with pytest.raises(ValueError, match="benchmark_cols"):
        marginal_alpha_report(df, "f", [], "r")
    with pytest.raises(ValueError, match="horizon"):
        marginal_alpha_report(df, "f", ["f"], "r", horizon=0)
    with pytest.raises(ValueError, match="At least 4"):
        marginal_alpha_report(df.head(2), "f", ["f"], "r")
