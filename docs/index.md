# polars-anta

<p align="center">
  <strong>ANTA (Alpha & Numerical Technical Analysis): Ultra-fast, zero-C-dependency technical analysis and factor library powered natively by Polars Expressions.</strong>
</p>

---

## What is `polars-anta`?

`polars-anta` is an ultra-fast, modern financial technical analysis and alpha factor library designed from the ground up for quantitative researchers, algorithmic traders, and financial engineers.

Unlike traditional libraries (`TA-Lib`, `pandas-ta`) that require external C compilers, introduce cumbersome binary dependencies, or execute through slow Python `for` loops, `polars-anta` implements **all 102 indicators natively as Polars Expressions (`pl.Expr`)**.

Every calculation is executed directly inside Polars' multi-threaded Rust query engine with zero-copy Apache Arrow memory layouts.

---

## Key Advantages

=== "Multi-Threaded Rust Engine"
    Calculations bypass the Python GIL and run across all CPU cores automatically. Compute indicators on 1,000,000 rows in milliseconds.

=== "Dual Expression Namespaces"
    Both `pl.col("close").anta.<indicator>()` and `pl.col("close").ta.<indicator>()` are supported natively with zero overhead.

=== "Zero C Dependencies"
    No MSVC, no GCC, no headers. `pip install polars-anta` installs instantly on Linux, macOS, and Windows.

=== "Native Multi-Asset Windowing"
    Calculate indicators across entire universes (500+ tickers simultaneously) in parallel via `.over("ticker")` without slow `.groupby().apply()`.

=== "Full LazyFrame & Streaming"
    Integrates directly with Polars' `LazyFrame` query optimizer, enabling predicate pushdown and streaming out-of-core evaluation for datasets exceeding RAM.

---

## Performance at a Glance

| Indicator / Operation | Scale | Pandas Execution | `polars-anta` Execution | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression Slope** | 1,000,000 rows | 1.8018 s | **0.0154 s** | **116.8x Faster** |
| **Multi-Asset RSI** | 500 tickers x 2k bars | 0.4560 s | **0.0610 s** | **7.5x Faster** |
| **Multi-Asset SMA** | 500 tickers x 2k bars | 0.1229 s | **0.0176 s** | **7.0x Faster** |
| **Z-Score Normalization** | 1,000,000 rows | 0.0378 s | **0.0182 s** | **2.1x Faster** |

---

## 3-Line Quickstart

```python
import polars as pl
import polars_anta  # Registers both .anta and .ta namespaces

df = pl.read_parquet("market_data.parquet")

# Compute indicators cleanly within expressions
features = df.with_columns(
    pl.col("close").anta.sma(20).alias("sma_20"),
    pl.col("close").anta.rsi(14).alias("rsi_14"),
    pl.col("close").anta.bollinger_bands(20, 2.0).alias("bb"),
)
```
