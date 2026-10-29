<div align="center">
  <img src="assets/banner.png" alt="polars-anta banner" width="100%" />
</div>

<div align="center">

### Quantitative technical analysis, cross-sectional alpha research and vectorized backtesting in Polars

**Transparent mathematical expressions. Institutional factor analytics and risk modeling. An intuitive dual `.anta` / `.ta` interface. Polars clarity with Rust-accelerated performance.**

</div>

---

## What is `polars-anta`?

**`polars-anta`** (**A**lpha & **N**umerical **T**echnical **A**nalysis) is an open-source quantitative finance library for **technical analysis, market microstructure estimation, cross-sectional factor engineering, and vectorized backtesting** in Python and Polars.

Designed for quantitative researchers, algorithmic traders, data scientists, and students, `polars-anta` implements technical indicators, statistical regressions, formulaic alpha primitives, and portfolio evaluation tools directly as Polars Expressions (`pl.Expr`)—combining readable, inspectable mathematical formulas with Polars' multi-threaded Rust execution engine.

> **A Simple Core Idea:**
> <p align="center"><strong>Quantitative Pipeline = OHLCV Data + Native Polars Expressions (<code>.anta</code> / <code>.ta</code>) + Cross-Sectional Alpha + Vectorized Backtest</strong></p>

---

## Why `polars-anta`?

- **End-to-end quantitative coverage** — move seamlessly from raw OHLCV validation and technical indicators to cross-sectional factor neutralization, IC/quantile evaluation, and vectorized backtesting.
- **Transparent implementations** — inspect clean, closed-form algebraic Polars expressions rather than opaque C/Cython binaries.
- **Rust-accelerated performance** — up to **112.4x faster** than Pandas on 1,000,000 rows via multi-threaded Rust query execution and zero-copy stride tricks.
- **Zero C compilation** — installs in seconds via `pip install polars-anta` across Windows, macOS, and Linux with no external C headers or MSVC/GCC toolchains.
- **Learn by example** — explore ready-to-run Python scripts, 1-click Google Colab notebooks, and modular API reference guides.

---

## Documentation and Learning Resources

| Resource | Description |
| :--- | :--- |
| [**API Reference & Catalog**](indicators/index.md) | Complete module-by-module function signatures, struct fields, and mathematical formulas |
| [**Getting Started Guide**](getting-started.md) | Installation, expression syntax, `pl.Struct` unnesting, and fluent DataFrame chaining |
| [**Architecture & Vectorization**](architecture.md) | How closed-form OLS identities, Wilder recurrences, and zero-copy stride windows work |
| [**Multi-Asset Universe Guide**](guides/multi-asset.md) | Scaling indicators and cross-sectional factors across hundreds of symbols with `.over("ticker")` |
| [**Resampling & Data Validation**](guides/resampling-and-dataframe-api.md) | Multi-timeframe OHLCV aggregation (`resample_ohlcv`) and integrity checks (`validate_ohlcv`) |
| [**Factor Evaluation & Backtesting**](indicators/evaluation-and-backtest.md) | Rank IC, ICIR, quantile monotonicity, strategy tearsheets, and `df.anta.backtest()` |
| [**Performance Benchmarks**](benchmarks.md) | 1,000,000-row latency comparisons against Pandas and reproduction instructions |
| [**Interactive Colab Notebooks**](https://github.com/huuan26/polars-anta/tree/master/examples) | Runnable Jupyter / Google Colab tutorials covering screening, momentum, and backtesting |

---

## 3-Step Quick Start

### 1. Install

```bash
pip install polars-anta
```

### 2. Compute Indicators & Alpha Factors

```python
import polars as pl
import polars_anta as pa  # Registers .anta and .ta namespaces

df = pl.read_parquet("market_data.parquet")

features = df.with_columns(
    pl.col("close").anta.ema(20).over("symbol").alias("ema_20"),
    pl.col("close").anta.rsi(14).over("symbol").alias("rsi_14"),
    pl.col("close").anta.linear_regression_slope(14).over("symbol").alias("ols_slope"),
    pl.col("close").anta.bollinger_bands(20, 2.0).over("symbol").alias("bb"),
).unnest("bb")
```

### 3. Run a 1-Line Vectorized Backtest

```python
bt = (
    df.anta.ema(12, alias="ema_12")
    .anta.ema(26, alias="ema_26")
    .anta.backtest(
        entries=pl.col("ema_12").anta.crosses_above("ema_26"),
        exits=pl.col("ema_12").anta.crosses_below("ema_26"),
        fee_rate=0.0005,
        initial_capital=100_000.0,
    )
)
print(bt.report())
```
