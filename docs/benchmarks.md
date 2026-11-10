# Benchmarks & Performance Analysis

To evaluate real-world performance, `polars-anta` provides an automated benchmarking suite (`benchmarks/benchmark_speed.py`) comparing execution speeds against Pandas / NumPy baselines across 1,000,000 rows.

---

## Benchmark Results

| Operation / Indicator | Dataset Scale | Pandas (s) | `polars-anta` (s) | Measured Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression Slope** | 1,000,000 bars | 1.8018 s | **0.0154 s** | **116.8x Faster** |
| **Multi-Asset RSI** | 500 tickers x 2k bars | 0.4560 s | **0.0610 s** | **7.5x Faster** |
| **Multi-Asset SMA** | 500 tickers x 2k bars | 0.1229 s | **0.0176 s** | **7.0x Faster** |
| **Z-Score Normalization** | 1,000,000 bars | 0.0378 s | **0.0182 s** | **2.1x Faster** |
| **Exponential Moving Average** | 1,000,000 bars | 0.0093 s | **0.0058 s** | **1.6x Faster** |
| **Simple Moving Average** | 1,000,000 bars | 0.0187 s | **0.0097 s** | **1.9x Faster** |

> *Baselines are straightforward pandas implementations (`rolling().mean()`, `groupby().apply()`, and `rolling().apply()` for the WMA term of OLS). A fully vectorized pandas/NumPy OLS would narrow the 112.4x gap considerably; speedups vary by hardware. Reproduce with `python benchmarks/benchmark_speed.py`.*

---

## Why is `polars-anta` so much faster?

### 1. Vectorized Linear Regression Decomposition
Traditional packages compute rolling linear regression by taking 1,000,000 windows and repeatedly solving $y = \beta_0 + \beta_1 t$ with NumPy dot products.

`polars-anta` decomposes the OLS closed-form equation into Linearly Weighted Moving Average (WMA) and Simple Moving Average (SMA):

$$
\beta_1 = \frac{6}{N - 1} (\text{WMA} - \text{SMA})
$$

Because WMA and SMA execute in SIMD-vectorized Rust kernels, the rolling regression evaluates in **0.015 seconds** instead of 1.80 seconds—over **116x faster**!

### 2. Multi-Threaded Partitioning (`.over()`)
When processing multiple tickers, Polars automatically partitions data and distributes computation across all physical and logical CPU cores in Rust. Pandas `.groupby().apply()`, by contrast, iterates sequentially in a single Python thread.

---

## How to Reproduce

Run the benchmark script directly on your own machine:

```bash
pip install -e .[benchmark]
python benchmarks/benchmark_speed.py
```
