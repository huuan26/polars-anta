"""polars-anta Backtest & Alpha Evaluation Showcase Demo.

Demonstrates:
1. Easy Entry/Exit Strategy Backtesting (EMA Crossover, RSI Mean-Reversion, SuperTrend)
2. Side-by-side Multi-Strategy Comparison & Formatted Tearsheet Reporting
3. Institutional Multi-Asset Alpha Pipeline (Rank IC, 5-Quantile Spread & Dollar-Neutral Backtest)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_anta as pa


def generate_single_asset_ohlcv(n_bars: int = 1_500, seed: int = 42) -> pl.DataFrame:
    """Generate realistic synthetic daily OHLCV data with regime shifts (trend + mean-reversion)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars, dtype=np.float64)

    # Combine secular drift + cyclical waves + stochastic noise
    drift = 0.0004
    cycle = 0.004 * np.sin(2.0 * np.pi * t / 90.0)
    noise = rng.normal(0.0, 0.014, size=n_bars)
    ret = drift + cycle + noise

    close = 100.0 * np.cumprod(1.0 + ret)
    open_ = np.roll(close, 1) * (1.0 + rng.normal(0.0, 0.002, size=n_bars))
    open_[0] = 100.0
    spread = np.abs(rng.normal(0.008, 0.004, size=n_bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=14.0, sigma=0.5, size=n_bars)

    start_dt = datetime(2020, 1, 1)
    dates = [start_dt + timedelta(days=i) for i in range(n_bars)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def generate_multi_asset_panel(
    n_tickers: int = 50, n_days: int = 500, seed: int = 7
) -> pl.DataFrame:
    """Generate a 50-ticker cross-sectional panel with embedded alpha signal predictability."""
    rng = np.random.default_rng(seed)
    start_dt = datetime(2024, 1, 1)
    dates = [start_dt + timedelta(days=i) for i in range(n_days)]

    frames: list[pl.DataFrame] = []
    for idx in range(n_tickers):
        sym = f"TICK_{idx + 1:02d}"
        latent_alpha = rng.normal(0.0, 1.0, size=n_days)
        # Forward return partially driven by lagged latent alpha + market + idiosyncratic noise
        fwd_noise = rng.normal(0.0, 0.015, size=n_days)
        ret = 0.0003 + 0.0035 * np.roll(latent_alpha, 1) + fwd_noise
        ret[0] = 0.0

        close = (50.0 + idx) * np.cumprod(1.0 + ret)
        open_ = close * (1.0 - 0.002 * latent_alpha)
        high = np.maximum(open_, close) * 1.006
        low = np.minimum(open_, close) * 0.994
        volume = rng.lognormal(mean=13.5, sigma=0.4, size=n_days) * (
            1.0 + 0.25 * np.abs(latent_alpha)
        )

        frames.append(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": [sym] * n_days,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "alpha_signal": latent_alpha,
                }
            )
        )
    return pl.concat(frames)


def main() -> None:
    pl.Config.set_ascii_tables(True)
    pl.Config.set_tbl_rows(25)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_fmt_str_lengths(40)

    print("=" * 88)
    print(" POLARS-ANTA BACKTEST & ALPHA RESEARCH SHOWCASE")
    print("=" * 88)

    # =========================================================================
    # PART 1: Single-Asset Entry/Exit Strategy Backtesting & Comparison
    # =========================================================================
    df = generate_single_asset_ohlcv(n_bars=1_500)
    t0 = time.perf_counter()

    # Step 1: Compute technical indicators in one vectorized Polars pass
    df_ta = (
        df.anta.ema(period=12, alias="ema_12")
        .anta.ema(period=26, alias="ema_26")
        .anta.rsi(period=14, alias="rsi_14")
        .anta.supertrend(period=10, multiplier=2.5, unnest=True)
    )

    # Strategy A: EMA(12/26) Crossover (using .crosses_above / .crosses_below + entries/exits)
    bt_ema = df_ta.anta.backtest(
        entries=pl.col("ema_12").anta.crosses_above("ema_26"),
        exits=pl.col("ema_12").anta.crosses_below("ema_26"),
        fee_rate=0.0005,
        slippage_rate=0.0002,
        initial_capital=100_000.0,
    )

    # Strategy B: RSI(14) Mean-Reversion (Buy oversold < 30, Exit overbought > 65)
    bt_rsi = df_ta.anta.backtest(
        entries=pl.col("rsi_14").anta.crosses_below(30),
        exits=pl.col("rsi_14").anta.crosses_above(65),
        fee_rate=0.0005,
        slippage_rate=0.0002,
        initial_capital=100_000.0,
    )

    # Strategy C: SuperTrend Long/Short Regime Follower
    bt_supertrend = df_ta.anta.backtest(
        signal=pl.col("direction"),
        fee_rate=0.0005,
        slippage_rate=0.0002,
        initial_capital=100_000.0,
    )
    elapsed_single = (time.perf_counter() - t0) * 1000.0

    print(
        f"\n[1] Formatted Tearsheet - EMA(12/26) Crossover (Computed 3 strategies in {elapsed_single:.2f} ms):"
    )
    print(bt_ema.report())

    print("\n[2] Multi-Strategy Side-by-Side Comparison Table:")
    comp_df = pa.compare_strategies(
        {
            "EMA(12/26) Crossover": bt_ema,
            "RSI(14) Mean Reversion": bt_rsi,
            "SuperTrend(10, 2.5) L/S": bt_supertrend,
        }
    ).select(
        [
            "strategy",
            (pl.col("total_return") * 100).round(2).alias("total_ret_%"),
            (pl.col("benchmark_return") * 100).round(2).alias("bench_ret_%"),
            (pl.col("excess_return") * 100).round(2).alias("excess_ret_%"),
            pl.col("sharpe_ratio").round(2),
            pl.col("sortino_ratio").round(2),
            (pl.col("max_drawdown") * 100).round(2).alias("max_dd_%"),
            (pl.col("win_rate") * 100).round(1).alias("win_rate_%"),
            pl.col("n_trades").cast(pl.Int64),
        ]
    )
    print(comp_df)

    # =========================================================================
    # PART 2: Multi-Asset Cross-Sectional Alpha Evaluation & Portfolio Backtest
    # =========================================================================
    panel_df = generate_multi_asset_panel(n_tickers=50, n_days=500)
    t1 = time.perf_counter()

    # Compute WorldQuant Alpha #101 + Forward Returns across 50 tickers (25,000 rows)
    panel_df = panel_df.anta.alpha_101().anta.forward_returns(periods=(1, 5), group_by="symbol")

    # 1. Rank IC Summary
    ic_stats = panel_df.anta.ic_summary(
        factor_col="alpha_signal",
        return_col="fwd_ret_1",
        group_by="timestamp",
        method="spearman",
    )

    # 2. 5-Quantile Monotonicity Analysis
    q_table = panel_df.anta.quantile_analysis(
        factor_col="alpha_signal",
        return_col="fwd_ret_1",
        quantiles=5,
        group_by="timestamp",
    )

    # 3. Multi-Asset Portfolio Backtest (Dollar-Neutral vs Long-Only)
    bt_dn = panel_df.anta.backtest(
        signal="alpha_signal",
        group_by="symbol",
        time_col="timestamp",
        weight_scheme="dollar_neutral",
        fee_rate=0.0003,
        slippage_rate=0.0002,
        signal_lag=1,
        initial_capital=1_000_000.0,
    )
    bt_lo = panel_df.anta.backtest(
        signal="alpha_signal",
        group_by="symbol",
        time_col="timestamp",
        weight_scheme="long_only",
        fee_rate=0.0003,
        slippage_rate=0.0002,
        signal_lag=1,
        initial_capital=1_000_000.0,
    )
    elapsed_panel = (time.perf_counter() - t1) * 1000.0

    print(
        f"\n[3] 50-Ticker Cross-Sectional Factor IC Summary (25,000 rows in {elapsed_panel:.2f} ms):"
    )
    for k, v in ic_stats.items():
        print(f"    {k:<15}: {v:.4f}")

    print("\n[4] 5-Quantile Return Monotonicity Table:")
    print(q_table)

    print("\n[5] Multi-Asset Portfolio Backtest (Dollar-Neutral L/S vs Long-Only):")
    port_comp = pa.compare_strategies(
        {
            "50-Ticker Dollar-Neutral L/S": bt_dn,
            "50-Ticker Factor Long-Only": bt_lo,
        }
    ).select(
        [
            "strategy",
            pl.col("final_equity").round(2),
            (pl.col("total_return") * 100).round(2).alias("total_ret_%"),
            pl.col("sharpe_ratio").round(2),
            pl.col("sortino_ratio").round(2),
            (pl.col("max_drawdown") * 100).round(2).alias("max_dd_%"),
            (pl.col("win_rate") * 100).round(1).alias("win_rate_%"),
        ]
    )
    print(port_comp)


if __name__ == "__main__":
    main()
