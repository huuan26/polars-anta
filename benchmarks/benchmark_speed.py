"""High-throughput performance benchmark: polars-anta vs Pandas / NumPy.

Measures computation time across single-asset 1,000,000 bars and
multi-asset 500 tickers x 2,000 bars (1,000,000 total rows).
Generates high-resolution visualization `benchmarks/benchmark_speed.png`.
"""

from __future__ import annotations

import gc
import shutil
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import polars_anta  # noqa: F401


def _measure(fn: Callable[[], object], repeats: int = 5, warmup: int = 1) -> float:
    """Return median wall-clock execution time in seconds with GC isolated per trial."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        finally:
            if gc_was_enabled:
                gc.enable()
    return float(statistics.median(samples))


def generate_single_asset_data(n_rows: int = 1_000_000) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Generates synthetic OHLCV data with random walk for 1,000,000 bars."""
    print(f"Generating {n_rows:,} rows of synthetic single-asset OHLCV data...")
    np.random.seed(42)

    returns = np.random.normal(0.0001, 0.01, n_rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.005, n_rows)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.005, n_rows)))
    open_ = (high + low) / 2.0
    volume = np.random.randint(100, 10000, n_rows).astype(np.float64)

    df_dict = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }

    df_pl = pl.DataFrame(df_dict)
    df_pd = pd.DataFrame(df_dict)
    return df_pl, df_pd


def generate_multi_asset_data(
    n_tickers: int = 500, n_bars: int = 2_000
) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Generates 500 tickers x 2,000 bars (1,000,000 rows) for .over('ticker') benchmarking."""
    total_rows = n_tickers * n_bars
    print(
        f"Generating multi-asset data: {n_tickers} tickers x {n_bars} bars = {total_rows:,} rows..."
    )
    np.random.seed(42)

    tickers = [f"TICKER_{i:03d}" for i in range(n_tickers)]
    ticker_col = np.repeat(tickers, n_bars)

    # Random walk per ticker
    close_list = []
    for _ in range(n_tickers):
        ret = np.random.normal(0.0001, 0.015, n_bars)
        price = 50.0 * np.exp(np.cumsum(ret))
        close_list.append(price)

    close_arr = np.concatenate(close_list)
    high_arr = close_arr * (1.0 + np.abs(np.random.normal(0, 0.005, total_rows)))
    low_arr = close_arr * (1.0 - np.abs(np.random.normal(0, 0.005, total_rows)))
    vol_arr = np.random.randint(100, 5000, total_rows).astype(np.float64)

    df_dict = {
        "ticker": ticker_col,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
        "volume": vol_arr,
    }

    df_pl = pl.DataFrame(df_dict)
    df_pd = pd.DataFrame(df_dict)
    return df_pl, df_pd


def benchmark_single_asset(df_pl: pl.DataFrame, df_pd: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Runs single-asset benchmarks on 1,000,000 bars."""
    results: dict[str, dict[str, float]] = {}

    # 1. SMA (20)
    print("Benchmarking SMA(20)...")
    t_pd = _measure(lambda: df_pd["close"].rolling(20).mean())
    t_pl = _measure(lambda: df_pl.select(pl.col("close").anta.sma(20)))
    results["SMA (20)"] = {"pandas": t_pd, "polars_anta": t_pl}

    # 2. EMA (20)
    print("Benchmarking EMA(20)...")
    t_pd = _measure(lambda: df_pd["close"].ewm(span=20, adjust=False).mean())
    t_pl = _measure(lambda: df_pl.select(pl.col("close").anta.ema(20)))
    results["EMA (20)"] = {"pandas": t_pd, "polars_anta": t_pl}

    # 3. Bollinger Bands (20, 2) - 5 fields: upper, middle, lower, bandwidth, percent_b
    print("Benchmarking Bollinger Bands(20, 2)...")

    def _bb_pd() -> pd.DataFrame:
        c = df_pd["close"]
        middle = c.rolling(20).mean()
        std = c.rolling(20).std()
        upper = middle + 2.0 * std
        lower = middle - 2.0 * std
        spread = upper - lower
        bandwidth = np.where(middle > 0.0, spread / middle, 0.0)
        percent_b = np.where(spread > 0.0, (c - lower) / spread, 0.5)
        return pd.DataFrame(
            {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "bandwidth": bandwidth,
                "percent_b": percent_b,
            }
        )

    t_pd = _measure(_bb_pd)
    t_pl = _measure(lambda: df_pl.select(pl.col("close").anta.bollinger_bands(20, 2.0)))
    results["Bollinger (20,2)"] = {"pandas": t_pd, "polars_anta": t_pl}

    # 4. Z-Score (20)
    print("Benchmarking Z-Score(20)...")

    def _zscore_pd() -> pd.Series:
        m = df_pd["close"].rolling(20).mean()
        s = df_pd["close"].rolling(20).std()
        return (df_pd["close"] - m) / s

    t_pd = _measure(_zscore_pd)
    t_pl = _measure(lambda: df_pl.select(pl.col("close").anta.zscore(20)))
    results["Z-Score (20)"] = {"pandas": t_pd, "polars_anta": t_pl}

    # 5. Linear Regression Slope (14)
    print("Benchmarking Linear Regression Slope(14)...")
    w = np.arange(1, 15, dtype=np.float64)
    w_sum = float(w.sum())

    def _linreg_pd() -> pd.Series:
        wma_pd = df_pd["close"].rolling(14).apply(lambda x: np.dot(x, w) / w_sum, raw=True)
        sma_pd = df_pd["close"].rolling(14).mean()
        return (wma_pd - sma_pd) * (6.0 / 13.0)

    t_pd = _measure(_linreg_pd, repeats=3, warmup=1)
    t_pl = _measure(lambda: df_pl.select(pl.col("close").anta.linear_regression_slope(14)))
    results["Linear Reg (14)"] = {"pandas": t_pd, "polars_anta": t_pl}

    return results


def benchmark_multi_asset(df_pl: pl.DataFrame, df_pd: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Runs multi-asset parallel benchmark: 500 tickers x 2,000 bars."""
    results: dict[str, dict[str, float]] = {}

    print("Benchmarking Multi-Asset (500 tickers): SMA(20)...")
    t_pd = _measure(lambda: df_pd.groupby("ticker", group_keys=False)["close"].rolling(20).mean())
    t_pl = _measure(lambda: df_pl.select(pl.col("close").anta.sma(20).over("ticker")))
    results["Multi-Asset SMA"] = {"pandas": t_pd, "polars_anta": t_pl}

    print("Benchmarking Multi-Asset (500 tickers): RSI(14)...")

    def _calc_rsi_pd(s: pd.Series) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    t_pd = _measure(
        lambda: df_pd.groupby("ticker", group_keys=False)["close"].apply(_calc_rsi_pd),
        repeats=3,
        warmup=1,
    )
    t_pl = _measure(lambda: df_pl.select(pl.col("close").anta.rsi(14).over("ticker")))
    results["Multi-Asset RSI"] = {"pandas": t_pd, "polars_anta": t_pl}

    return results


def plot_benchmark_chart(all_results: dict[str, dict[str, float]], output_path: Path):
    """Renders a modern, publication-grade benchmark chart."""
    labels = list(all_results.keys())
    pd_times = [all_results[k]["pandas"] for k in labels]
    pl_times = [all_results[k]["polars_anta"] for k in labels]
    speedups = [pd_t / pl_t for pd_t, pl_t in zip(pd_times, pl_times, strict=False)]

    # Modern plot styling
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    y_pos = np.arange(len(labels))
    bar_height = 0.38

    ax.barh(
        y_pos + bar_height / 2,
        pd_times,
        height=bar_height,
        label="Pandas (Single-threaded)",
        color="#d9534f",
        alpha=0.9,
    )
    ax.barh(
        y_pos - bar_height / 2,
        pl_times,
        height=bar_height,
        label="polars-anta (Multi-threaded Rust)",
        color="#0275d8",
        alpha=0.95,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel(
        "Execution Time in Seconds (Lower is Better - Log Scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_xscale("log")
    ax.set_title(
        "polars-anta Performance Benchmark: 1,000,000 Rows / 500 Tickers Universe",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="lower right")

    # Annotate speedup factors
    max_time = max(max(pl_times), max(pd_times))
    min_time = min(min(pl_times), min(pd_times))
    ax.set_xlim(min_time * 0.65, max_time * 3.8)
    for i, speedup in enumerate(speedups):
        val_x = max(pl_times[i], pd_times[i]) * 1.22
        badge_text = f"{speedup:.1f}x FASTER" if speedup >= 1.05 else f"{speedup:.1f}x"
        ax.text(
            val_x,
            y_pos[i] - bar_height / 2,
            badge_text,
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color="#0275d8",
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor="#e8f4fd", edgecolor="#b6dcfb", alpha=0.9
            ),
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    docs_chart_path = output_path.parent.parent / "docs" / "assets" / "benchmark_speed.png"
    if docs_chart_path.parent.exists():
        shutil.copy2(output_path, docs_chart_path)
    print(f"\n[OK] Benchmark chart successfully saved to: {output_path}")


def main():
    print("=" * 70)
    print("polars-anta High-Performance Benchmarking Suite")
    print("=" * 70)

    # 1. Single Asset Benchmark
    df_pl_single, df_pd_single = generate_single_asset_data(1_000_000)
    single_results = benchmark_single_asset(df_pl_single, df_pd_single)

    # 2. Multi-Asset Benchmark (500 tickers x 2,000 bars)
    df_pl_multi, df_pd_multi = generate_multi_asset_data(500, 2_000)
    multi_results = benchmark_multi_asset(df_pl_multi, df_pd_multi)

    all_results = {**single_results, **multi_results}

    # Print summary table
    print("\n" + "=" * 70)
    print(
        f"{'Indicator / Operation':<25} | {'Pandas (s)':<12} | {'polars-anta (s)':<14} | {'Speedup':<10}"
    )
    print("-" * 70)
    for k, v in all_results.items():
        pd_t = v["pandas"]
        pl_t = v["polars_anta"]
        ratio = pd_t / pl_t
        print(f"{k:<25} | {pd_t:<12.4f} | {pl_t:<14.4f} | {ratio:<10.1f}x")
    print("=" * 70)

    # Generate Chart
    chart_path = Path(__file__).resolve().parent / "benchmark_speed.png"
    plot_benchmark_chart(all_results, chart_path)


if __name__ == "__main__":
    main()
