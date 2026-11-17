# Benchmarks & Performance Analysis

To evaluate real-world performance, `polars-anta` provides an automated benchmarking suite (`benchmarks/benchmark_speed.py`) comparing execution speeds against Pandas / NumPy baselines across 1,000,000 rows.

---

## Benchmark Results

| Operation / Indicator | Dataset Scale | Pandas (s) | `polars-anta` (s) | Measured Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression Slope (OLS)** | 1,000,000 bars | 1.8652 s | **0.0301 s** | **62.0x Faster** |
| **Multi-Asset RSI (14)** | 500 tickers x 2k bars | 0.7096 s | **0.0589 s** | **12.0x Faster** |
| **Multi-Asset SMA (20)** | 500 tickers x 2k bars | 0.2014 s | **0.0174 s** | **11.6x Faster** |
| **Simple Moving Average (20)** | 1,000,000 bars | 0.0599 s | **0.0227 s** | **2.6x Faster** |
| **Z-Score Normalization (20)** | 1,000,000 bars | 0.0873 s | **0.0427 s** | **2.0x Faster** |
| **Exponential Moving Average (20)** | 1,000,000 bars | 0.0353 s | **0.0205 s** | **1.7x Faster** |
| **Bollinger Bands (20, 2)** | 1,000,000 bars | 0.1464 s | **0.0964 s** | **1.5x Faster** |

> *Baselines are straightforward pandas implementations (`rolling().mean()`, `groupby().apply()`, and `rolling().apply()` for the WMA term of OLS), measured as the median of repeated GC-isolated runs after warmup. A fully vectorized pandas/NumPy OLS would narrow the 62.0x gap considerably; speedups vary by hardware. Reproduce with `python benchmarks/benchmark_speed.py`.*

---

## Why is `polars-anta` so much faster?

### 1. Vectorized Linear Regression Decomposition
Traditional packages compute rolling linear regression by taking 1,000,000 windows and repeatedly solving $y = \beta_0 + \beta_1 t$ with NumPy dot products.

`polars-anta` decomposes the OLS closed-form equation into Linearly Weighted Moving Average (WMA) and Simple Moving Average (SMA):

$$
\beta_1 = \frac{6}{N - 1} (\text{WMA} - \text{SMA})
$$

Because WMA and SMA execute in SIMD-vectorized Rust kernels, the rolling regression evaluates in **0.0301 seconds** instead of 1.87 seconds - **62x faster**!

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
