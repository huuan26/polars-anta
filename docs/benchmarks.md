# Benchmarks & Performance Analysis

To evaluate real-world performance, `polars-anta` provides an automated benchmarking suite (`benchmarks/benchmark_speed.py`) comparing execution speeds against Pandas / NumPy baselines across 1,000,000 rows.

---

## Benchmark Results

| Operation / Indicator | Dataset Scale | Pandas (s) | `polars-anta` (s) | Measured Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression Slope (OLS)** | 1,000,000 bars | 2.3529 s | **0.0209 s** | **112.4x Faster** |
| **Multi-Asset RSI (14)** | 500 tickers x 2k bars | 0.4398 s | **0.0545 s** | **8.1x Faster** |
| **Multi-Asset SMA (20)** | 500 tickers x 2k bars | 0.1345 s | **0.0168 s** | **8.0x Faster** |
| **Z-Score Normalization** | 1,000,000 bars | 0.0450 s | **0.0267 s** | **1.7x Faster** |
| **Exponential Moving Average** | 1,000,000 bars | 0.0100 s | **0.0069 s** | **1.5x Faster** |
| **Simple Moving Average** | 1,000,000 bars | 0.0170 s | **0.0153 s** | **1.1x Faster** |

> *Baselines are straightforward pandas implementations (`rolling().mean()`, `groupby().apply()`, and `rolling().apply()` for the WMA term of OLS). A fully vectorized pandas/NumPy OLS would narrow the 112.4x gap considerably; speedups vary by hardware. Reproduce with `python benchmarks/benchmark_speed.py`.*

---

## Why is `polars-anta` so much faster?

### 1. Vectorized Linear Regression Decomposition
Traditional packages compute rolling linear regression by taking 1,000,000 windows and repeatedly solving $y = \beta_0 + \beta_1 t$ with NumPy dot products.

`polars-anta` decomposes the OLS closed-form equation into Linearly Weighted Moving Average (WMA) and Simple Moving Average (SMA):

$$
\beta_1 = \frac{6}{N - 1} (\text{WMA} - \text{SMA})
$$

Because WMA and SMA execute in SIMD-vectorized Rust kernels, the rolling regression evaluates in **0.0209 seconds** instead of 2.35 seconds - over **112x faster**!

### 2. Multi-Threaded Partitioning (`.over()`)
When processing multiple tickers, Polars automatically partitions data and distributes computation across all physical and logical CPU cores in Rust. Pandas `.groupby().apply()`, by contrast, iterates sequentially in a single Python thread.

---

## How to Reproduce

Run the speed or cross-sectional alpha benchmarks directly on your own machine:

```bash
pip install -e .[benchmark]
python benchmarks/benchmark_speed.py
python benchmarks/alpha_engineering_validation.py
python benchmarks/real_market_alpha_engineering.py
```
