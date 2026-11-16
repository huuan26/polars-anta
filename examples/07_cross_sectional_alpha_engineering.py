"""End-to-end Cross-Sectional Alpha Engineering & Cost-Aware Dollar-Neutral Long-Short Workflow.

Demonstrates the 4-stage `polars-anta` pipeline designed for institutional equity / crypto
cross-sectional research and WorldQuant BRAIN-style alpha engineering:
  Stage 1: Signal Conditioning (cs_gaussianize, FWL Sector+Beta Neutralization, Batched Lowdin SVD)
  Stage 2: Alphalens-Grade Diagnostics (IC, Half-Life, Turnover, Margin bps, Causal ADV Sub-Universe)
           & Marginal Alpha Redundancy Check (Delta N_eff, Spanned R^2, Residual IC)
  Stage 3: Random Matrix Theory (Marchenko-Pastur) & Ledoit-Wolf Covariance Conditioning
  Stage 4: Strictly Causal Walk-Forward FISTA Allocation with L1 Turnover Control

Run offline (deterministic multi-sector global equity simulation):
    python examples/07_cross_sectional_alpha_engineering.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polars_anta as pa


def generate_global_long_short_panel(
    n_dates: int = 320,
    n_assets: int = 120,
    seed: int = 42,
) -> pl.DataFrame:
    """Simulate a 120-stock, 8-sector dollar-neutral universe with realistic crowding and costs."""
    rng = np.random.default_rng(seed)
    sectors = [
        "Tech",
        "Financials",
        "Healthcare",
        "Industrials",
        "Energy",
        "Consumer",
        "Materials",
        "Utilities",
    ]
    sec_idx = np.arange(n_assets) % len(sectors)
    sec_labels = [sectors[i] for i in sec_idx]
    betas = rng.uniform(0.65, 1.45, size=n_assets)

    # Persistent cross-sectional value/quality & medium-horizon momentum states
    state_slow = np.zeros((n_dates, n_assets))
    state_med = np.zeros((n_dates, n_assets))
    for t in range(1, n_dates):
        state_slow[t] = 0.93 * state_slow[t - 1] + np.sqrt(1.0 - 0.93**2) * rng.standard_normal(
            n_assets
        )
        state_med[t] = 0.78 * state_med[t - 1] + np.sqrt(1.0 - 0.78**2) * rng.standard_normal(
            n_assets
        )

    mkt = rng.normal(0.0003, 0.011, size=n_dates)
    sec_shocks = rng.normal(0.0, 0.008, size=(n_dates, len(sectors)))
    rets = np.zeros((n_dates, n_assets))
    for t in range(1, n_dates):
        rets[t] = (
            betas * mkt[t]
            + sec_shocks[t, sec_idx]
            + 0.0009 * state_slow[t - 1]
            + 0.0006 * state_med[t - 1]
            + rng.normal(0.0, 0.015, size=n_assets)
        )

    closes = 100.0 * np.exp(np.cumsum(rets, axis=0))
    opens = closes * np.exp(rng.normal(0.0, 0.002, size=(n_dates, n_assets)))
    highs = np.maximum(opens, closes) * (
        1.0 + np.abs(rng.normal(0.004, 0.002, size=(n_dates, n_assets)))
    )
    lows = np.minimum(opens, closes) * (
        1.0 - np.abs(rng.normal(0.004, 0.002, size=(n_dates, n_assets)))
    )
    # Liquidity tiers (small, mid, large ADV)
    adv_tier = np.where(
        np.arange(n_assets) < 40,
        150_000.0,
        np.where(np.arange(n_assets) < 80, 800_000.0, 4_000_000.0),
    )
    volumes = adv_tier[None, :] * np.exp(rng.normal(0.0, 0.25, size=(n_dates, n_assets)))

    dates = [f"2025-{(d // 21) + 1:02d}-{(d % 21) + 1:02d}" for d in range(n_dates)]
    sec_contam = rng.standard_normal((n_dates, len(sectors)))
    raw_a1 = (
        0.75 * state_slow
        + 1.5 * sec_contam[:, sec_idx]
        + 1.0 * betas[None, :]
        + 0.5 * rng.standard_normal((n_dates, n_assets))
    )
    raw_a1_clone = raw_a1 + 0.15 * rng.standard_normal((n_dates, n_assets))
    raw_a2 = (
        0.70 * state_med
        - 1.3 * sec_contam[:, sec_idx]
        + 0.5 * rng.standard_normal((n_dates, n_assets))
    )
    raw_noise = rng.standard_normal((n_dates, n_assets)) + 0.9 * sec_contam[:, sec_idx]

    return pl.DataFrame(
        {
            "date": np.repeat(dates, n_assets),
            "ticker": np.tile([f"GLB{i:03d}" for i in range(n_assets)], n_dates),
            "sector": np.tile(sec_labels, n_dates),
            "beta": np.tile(betas, n_dates),
            "open": opens.ravel(),
            "high": highs.ravel(),
            "low": lows.ravel(),
            "close": closes.ravel(),
            "volume": volumes.ravel(),
            "alpha_value": raw_a1.ravel(),
            "alpha_value_clone": raw_a1_clone.ravel(),
            "alpha_momentum": raw_a2.ravel(),
            "alpha_fast_noise": raw_noise.ravel(),
        }
    )


def main() -> None:
    print(f"Running polars-anta v{pa.__version__} Dollar-Neutral Long-Short Pipeline\n")
    panel = generate_global_long_short_panel()
    panel = panel.anta.forward_returns(
        close_col="close", periods=(1,), group_by="ticker", prefix="fwd_"
    )
    raw_cols = ["alpha_value", "alpha_value_clone", "alpha_momentum", "alpha_fast_noise"]

    print("1. Raw Collinearity Summary (Before Conditioning):")
    print(panel.anta.collinearity_summary(raw_cols, time_col="date"))

    # Stage 1: Rank-Gaussianize -> FWL Sector + Beta Neutralization -> Batched Lowdin SVD
    conditioned = panel.with_columns(
        [pl.col(c).anta.cs_gaussianize(over="date").alias(f"{c}_g") for c in raw_cols]
    ).anta.neutralize_panel(
        [f"{c}_g" for c in raw_cols],
        time_col="date",
        risk_cols=["beta"],
        group_col="sector",
    )
    neut_cols = [f"{c}_g_neut" for c in raw_cols]

    # Stage 2: Marginal Redundancy Check (detecting crowded clone vs genuinely novel momentum)
    marg_clone = conditioned.anta.marginal_alpha_report(
        candidate_col="alpha_value_clone_g_neut",
        benchmark_cols=["alpha_value_g_neut"],
        return_col="fwd_1",
        time_col="date",
    )
    marg_novel = conditioned.anta.marginal_alpha_report(
        candidate_col="alpha_momentum_g_neut",
        benchmark_cols=["alpha_value_g_neut"],
        return_col="fwd_1",
        time_col="date",
    )
    print("\n2. Marginal Alpha Screening (Crowded Clone vs. Independent Signal):")
    print(
        f"   Clone  -> Spanned R^2: {marg_clone['spanned_r2']:.3f} | Delta N_eff: {marg_clone['delta_n_eff']:+.3f} | Residual NW t: {marg_clone['residual_nw_t_stat']:+.2f}"
    )
    print(
        f"   Novel  -> Spanned R^2: {marg_novel['spanned_r2']:.3f} | Delta N_eff: {marg_novel['delta_n_eff']:+.3f} | Residual NW t: {marg_novel['residual_nw_t_stat']:+.2f}"
    )

    # Stage 3 & 4: Batched Lowdin Orthogonalization + RMT/EB/FISTA Walk-Forward Allocation
    ortho_df = conditioned.anta.orthogonalize_panel(neut_cols, time_col="date", method="lowdin")
    ortho_cols = [f"{c}_ortho" for c in neut_cols]

    wf = ortho_df.anta.combine_alphas_walkforward(
        factor_cols=ortho_cols,
        return_col="fwd_1",
        time_col="date",
        lookback=60,
        horizon=1,
        signal_lag=1,
        risk_aversion=2.5,
        turnover_penalty=0.006,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        cov_estimator="marchenko_pastur",
        ic_shrinkage="empirical_bayes",
        composite_col="composite_raw",
    )
    final_df = wf.data.sort(["ticker", "date"]).with_columns(
        pl.col("composite_raw")
        .ewm_mean(alpha=0.35, adjust=False)
        .over("ticker")
        .alias("composite_alpha")
    )

    diag = final_df.anta.analyze_alpha(
        factor_col="composite_alpha",
        close_col="close",
        volume_col="volume",
        asset_col="ticker",
        time_col="date",
        horizon=1,
        signal_lag=1,
        weight_scheme="dollar_neutral",
        one_way_cost_bps=10.0,
    )
    print("\n3. Out-of-Sample Dollar-Neutral Long-Short Tearsheet (10 bps Cost):")
    print(diag.report())
    print("\n4. Causal Liquidity Sub-Universe Breakdown:")
    print(diag.sub_universe)


if __name__ == "__main__":
    main()
