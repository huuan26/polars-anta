<div align="center">

<img src="assets/banner.png" alt="polars-anta banner" width="100%" />

### Quantitative technical analysis, cross-sectional alpha research and vectorized backtesting in Polars

**Transparent mathematical expressions. Institutional factor analytics and risk modeling. An intuitive dual `.anta` / `.ta` interface. Polars clarity with Rust-accelerated performance.**

An open-source quantitative finance library for technical analysis, market microstructure estimation, cross-sectional factor engineering, predictive signal evaluation, and loop-free vectorized backtesting. Built for quants, algorithmic traders, researchers, and students who want both execution speed and mathematical transparency.

[![CI](https://github.com/huuan26/polars-anta/actions/workflows/ci.yml/badge.svg)](https://github.com/huuan26/polars-anta/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/polars-anta.svg?label=PyPI&color=0077b6)](https://pypi.org/project/polars-anta/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776ab)](https://pypi.org/project/polars-anta/)
[![Polars Version](https://img.shields.io/badge/polars-%3E%3D1.0.0-orange)](https://pola.rs)
[![Engine](https://img.shields.io/badge/engine-native%20Polars%20expressions-10b981)](#api-reference--indicator-catalog)
[![Docs](https://github.com/huuan26/polars-anta/actions/workflows/docs.yml/badge.svg)](https://huuan26.github.io/polars-anta/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huuan26/polars-anta/blob/master/examples/quickstart_colab.ipynb)
[![License](https://img.shields.io/badge/license-Apache%202.0-555.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/huuan26/polars-anta?style=flat&color=e0a526)](https://github.com/huuan26/polars-anta/stargazers)

[**Documentation**](https://huuan26.github.io/polars-anta/) • [**Quick start**](#quick-start) • [**API Reference**](#api-reference--indicator-catalog) • [**Learning Resources**](#documentation-and-learning-resources) • [**Examples**](https://github.com/huuan26/polars-anta/tree/master/examples) • [**Report an issue**](https://github.com/huuan26/polars-anta/issues)

</div>

---

## What is `polars-anta`?

**`polars-anta`** (**A**lpha & **N**umerical **T**echnical **A**nalysis) is an open-source Python library for **high-speed technical analysis, cross-sectional alpha factor engineering, signal evaluation, and vectorized strategy backtesting** powered natively by [Polars](https://pola.rs/).

It unifies trend overlays, momentum oscillators, volatility channels, volume dynamics, closed-form OLS regressions, formulaic alpha primitives, market microstructure estimators, and portfolio risk analytics into a single composable framework—with an emphasis on code that can be inspected, understood, and scaled across multi-asset universes without C compiler dependencies.

### Why `polars-anta`?

- **End-to-end quantitative coverage** — move seamlessly from raw OHLCV validation and technical indicators to cross-sectional factor neutralization, IC/quantile evaluation, and vectorized backtesting in one unified package.
- **Transparent implementations** — inspect clean, closed-form algebraic Polars expressions (`pl.Expr`) written in readable Python rather than opaque C/Cython binaries.
- **Rust-accelerated performance** — computations push directly into Polars' multi-threaded Rust query engine and SIMD kernels, achieving up to **112.4x speedups** over Pandas without the Python GIL.
- **Zero C compilation** — installs cleanly via `pip install polars-anta` on Windows, macOS, and Linux with no external C libraries or MSVC/GCC toolchains.
- **Expression-first & composable** — invoke via `.anta` or `.ta` on any Polars `Expr`, `DataFrame`, or `LazyFrame`, and scale across thousands of symbols in parallel with `.over("ticker")`.
- **Learn by example** — explore runnable scripts, 1-click Google Colab notebooks, and modular API documentation.

> **A Simple Core Idea:**
> <p align="center"><strong>Quantitative Pipeline = OHLCV Data + Native Polars Expressions (<code>.anta</code> / <code>.ta</code>) + Cross-Sectional Alpha + Vectorized Backtest</strong></p>

| Capability | TA-Lib | pandas-ta | `polars-anta` |
| :--- | :---: | :---: | :---: |
| **Dependencies** | External C library & MSVC / GCC | Pure Python / NumPy | **Zero C compilation** (pure Polars & NumPy) |
| **Execution Engine** | Single-threaded C loops | Single-threaded Pandas / GIL | **Multi-threaded Rust** query engine + SIMD |
| **Multi-Asset Scaling** | Manual sequential loops | Slow `.groupby().apply()` | **Native `.over("symbol")`** parallelization |
| **Dual Namespace API** | Unsupported | Single `.ta` accessor | **Both `.anta` & `.ta`** on `Expr`, `DataFrame`, `LazyFrame` |
| **Cross-Sectional Factor Ops** | Unsupported | Unsupported | **Built-in `cs_winsorize`, `cs_zscore`, `cs_rank`, `neutralize`** |
| **Market Microstructure** | Unsupported | Unsupported | **Amihud Illiquidity, Kyle's $\lambda$, Roll & Corwin-Schultz Spreads** |
| **Factor Eval & Backtesting** | Unsupported | Unsupported | **Rank IC, ICIR, Quantiles, `df.anta.backtest()` & Tearsheets** |
| **Lazy Evaluation** | Unsupported | Unsupported | **Full `pl.LazyFrame`** query optimization & streaming |

---

## Quick start

### 1. Install

`polars-anta` is available on PyPI with zero C compiler requirements:

```bash
python -m pip install polars-anta
```

To upgrade to the latest release:

```bash
python -m pip install --upgrade polars-anta
```

Or using [`uv`](https://github.com/astral-sh/uv):

```bash
uv add polars-anta
```

### 2. Verify your installation

Importing `polars_anta` automatically registers both `.anta` and `.ta` namespaces on Polars `Expr`, `DataFrame`, and `LazyFrame` objects:

```python
import polars as pl
import polars_anta as pa

df = pl.DataFrame({"close": [100.0, 102.0, 101.5, 104.0, 106.5, 105.0, 108.0]})
out = df.with_columns(
    pl.col("close").anta.ema(period=3).round(2).alias("ema_3"),
    pl.col("close").anta.rsi(period=3).round(2).alias("rsi_3"),
)
print(out.tail(3))
# shape: (3, 3)
# ┌───────┬────────┬───────┐
# │ close ┆ ema_3  ┆ rsi_3 │
# │ ---   ┆ ---    ┆ ---   │
# │ f64   ┆ f64    ┆ f64   │
# ╞═══════╪════════╪═══════╡
# │ 106.5 ┆ 104.59 ┆ 88.14 │
# │ 105.0 ┆ 104.80 ┆ 62.61 │
# │ 108.0 ┆ 106.40 ┆ 80.52 │
# └───────┴────────┴───────┘
```

### 3. Explore quantitative workflows

Start with our ready-to-run [example scripts](examples/) or [1-click Google Colab notebooks](#interactive-jupyter--colab-notebooks), or explore the four core workflows below:

#### A. Fluent DataFrame & LazyFrame Chaining (`.anta` or `.ta`)

```python
# Validate OHLCV integrity, resample timeframes, and chain indicators on DataFrames or LazyFrames:
df_features = (
    df.anta.validate_ohlcv()
    .anta.resample("15m", group_by="symbol")
    .anta.heikin_ashi(unnest=True)
    .anta.sma(20)
    .anta.rsi(14)
    .anta.macd(unnest=True)
    .anta.bollinger_bands(period=20, unnest=True)
)

# Or generate a comprehensive quantitative feature suite in one call:
df_all = df.anta.add_all(group_by="symbol")
```

#### B. Multi-Asset Parallel Execution (`.over("ticker")`)

```python
# Compute indicators across thousands of tickers simultaneously in Polars' Rust engine:
universe_features = universe.with_columns(
    pl.col("close").anta.sma(20).over("ticker").alias("sma_20"),
    pl.col("close").anta.rsi(14).over("ticker").alias("rsi_14"),
    pl.col("close").anta.linear_regression_slope(14).over("ticker").alias("ols_slope"),
    pl.col("close").anta.amihud_illiquidity("volume", 20).over("ticker").alias("amihud"),
)
```

#### C. Cross-Sectional Factor Standardization & Sector Neutralization

```python
alpha_df = (
    panel_df.with_columns(
        (
            -pl.col("close").anta.ts_delta(5).over("symbol")
            * pl.col("volume").anta.ts_rank(20).over("symbol")
        ).alias("raw_alpha")
    )
    .with_columns(
        pl.col("raw_alpha")
        .anta.cs_winsorize(over="date", method="quantile", lower=0.01, upper=0.99)
        .anta.cs_zscore(over="date", clip_bound=3.0)
        .alias("clean_alpha")
    )
    .with_columns(
        pa.ts_scale(
            pa.neutralize("clean_alpha", by="sector", over="date", standardize=True),
            a=1.0,
            over="date",
        ).alias("portfolio_weight")
    )
)
```

#### D. Factor Evaluation (Rank IC / Quantiles) & 1-Line Vectorized Backtest

```python
# 1. Single-asset entry/exit crossover backtest & formatted tearsheet
bt_ema = (
    df.anta.ema(period=12, alias="ema_12")
    .anta.ema(period=26, alias="ema_26")
    .anta.backtest(
        entries=pl.col("ema_12").anta.crosses_above("ema_26"),
        exits=pl.col("ema_12").anta.crosses_below("ema_26"),
        fee_rate=0.0005,
        slippage_rate=0.0002,
        initial_capital=100_000.0,
    )
)
print(bt_ema.report())

# 2. Multi-asset cross-sectional factor Rank IC, 5-Quantiles & Dollar-Neutral Backtest
eval_df = panel_df.anta.alpha_012(group_by="symbol").anta.forward_returns(
    periods=(1, 5), group_by="symbol"
)
ic_stats = eval_df.anta.ic_summary("alpha_012", "fwd_ret_1", group_by="timestamp")
quantiles_df = eval_df.anta.quantile_analysis(
    "alpha_012", "fwd_ret_1", quantiles=5, group_by="timestamp"
)
```

---

## What can you model and compute?

| Area | Examples of functionality | API Docs |
| :--- | :--- | :---: |
| **Trend & Overlays** | Adaptive moving averages (`sma`, `ema`, `hma`, `kama`, `t3`, `alma`), `heikin_ashi`, `supertrend`, `ichimoku`, `adx`, `psar`, `pivot_points` | [Trend Docs](https://huuan26.github.io/polars-anta/indicators/trend/) |
| **Momentum Oscillators** | `rsi`, `macd`, `ppo`, `stochastic`, `stoch_rsi`, `squeeze_momentum`, `stc`, `tsi`, `fisher_transform`, `cci`, `williams_r`, `kst` | [Momentum Docs](https://huuan26.github.io/polars-anta/indicators/momentum/) |
| **Volatility & Channels** | `atr`, `natr`, `bollinger_bands`, `keltner_channels`, `donchian_channels`, `chandelier_exit`, `parkinson_volatility`, `garman_klass_volatility`, `ulcer_index` | [Volatility Docs](https://huuan26.github.io/polars-anta/indicators/volatility/) |
| **Volume & Money Flow** | `vwap`, `vwap_bands`, `obv`, `cmf`, `mfi`, `adl`, `klinger`, `force_index`, `nvi`, `pvi`, `vpci`, `vr` | [Volume Docs](https://huuan26.github.io/polars-anta/indicators/volume/) |
| **Statistics & Vectorized OLS** | Rolling `zscore`, `percentile_rank`, `skewness`, `kurtosis`, CAPM `rolling_beta`/`rolling_alpha`, closed-form `linear_regression_*` | [Stats Docs](https://huuan26.github.io/polars-anta/indicators/stats/) |
| **Alpha Factors & Microstructure** | Cross-sectional standardization (`cs_rank`, `cs_zscore`, `cs_winsorize`, `neutralize`), liquidity & price impact (`amihud_illiquidity`, `kyle_lambda`, `roll_spread`, `corwin_schultz_spread`), regime complexity (`hurst_exponent`, `fractal_dimension`, `entropy`), and formulaic alphas (`alpha_001`–`alpha_101`) | [Alpha Docs](https://huuan26.github.io/polars-anta/indicators/alpha/) |
| **Factor Evaluation & Backtesting** | `forward_returns`, `information_coefficient`, `ic_summary`, `quantile_analysis`, `factor_turnover`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `vectorized_backtest`, `compare_strategies` | [Eval & Backtest Docs](docs/indicators/evaluation-and-backtest.md) |
| **Data Engineering Utilities** | Multi-timeframe OHLCV aggregation (`resample_ohlcv`), schema & price integrity checks (`validate_ohlcv`), batch feature generator (`df.anta.add_all()`) | [Resampling Docs](https://huuan26.github.io/polars-anta/guides/resampling-and-dataframe-api/) |

---

## How `polars-anta` is organized

The library separates *data engineering*, *domain-specific indicator & alpha expressions*, *factor evaluation*, and *vectorized backtesting* into clean, self-contained modules so that each layer can be inspected and extended independently:

```text
polars_anta/
├── trend.py        # Moving averages, trend overlays, candlestick transforms & pivot systems
├── momentum.py     # Momentum oscillators, cycle detectors & divergence indicators
├── volatility.py   # OHLC volatility estimators, ATR trailing stops & price channels
├── volume.py       # Institutional VWAP bands, money flow & volume accumulation metrics
├── stats.py        # Rolling distributions, CAPM beta/alpha & closed-form OLS regression
├── alpha.py        # Cross-sectional factor ops, market microstructure & formulaic alphas
├── evaluation.py   # Alphalens-style Rank IC, ICIR, quantile analysis & risk expressions
├── backtest.py     # Loop-free vectorized backtester, crossover state machine & tearsheets
├── resample.py     # Multi-timeframe OHLCV resampling & structural integrity validation
└── namespace.py    # Dual .anta and .ta registration across Expr, DataFrame & LazyFrame
```

```mermaid
flowchart LR
    A["Raw Multi-Asset OHLCV<br/>(DataFrame / LazyFrame)"] --> B["1. Data Engineering<br/>validate_ohlcv() • resample()"]
    B --> C["2. Feature & Alpha Engine<br/>Native .anta / .ta Expressions<br/>(TA + Microstructure + Alphas)"]
    C --> D["3. Cross-Sectional Ops<br/>cs_winsorize → cs_zscore → neutralize"]
    D --> E["4. Factor Evaluation<br/>Rank IC • Quantiles • Turnover"]
    E --> F["5. Vectorized Backtesting<br/>entries/exits • Fees • Tearsheet<br/>Dollar-Neutral / Long-Only"]
```

---

## Documentation and learning resources

| Resource | Description |
| :--- | :--- |
| [**Online Documentation**](https://huuan26.github.io/polars-anta/) | Full documentation website with search, mathematical formulas, and usage guides |
| [**API Reference Overview**](docs/indicators/index.md) | Complete index of all supported functions, input arguments, and `pl.Struct` output schemas |
| [**Getting Started Guide**](docs/getting-started.md) | Introduction to Polars expressions, `.unnest()`, and fluent `.anta` / `.ta` chaining |
| [**Multi-Asset Universe Guide**](docs/guides/multi-asset.md) | Best practices for partitioning indicators and cross-sectional factors with `.over("ticker")` |
| [**Factor Evaluation & Backtesting Guide**](docs/indicators/evaluation-and-backtest.md) | Evaluating factor Rank IC / Quantiles and running `df.anta.backtest()` with transaction costs |
| [**Architecture & Vectorization**](docs/architecture.md) | Deep dive into closed-form OLS identities, Wilder smoothing recurrences, and stride kernels |
| [**Example Scripts**](examples/) | Standalone runnable Python scripts ([`quickstart.py`](examples/quickstart.py), [`04_backtest_showcase.py`](examples/04_backtest_showcase.py)) |
| [**Interactive Jupyter Notebooks**](#interactive-jupyter--colab-notebooks) | Hands-on tutorials runnable with 1 click in Google Colab |
| [**Benchmarks & Reproducibility**](docs/benchmarks.md) | 1,000,000-row benchmark methodology and reproduction script ([`benchmark_speed.py`](benchmarks/benchmark_speed.py)) |
| [**Roadmap**](ROADMAP.md) & [**Changelog**](CHANGELOG.md) | Release history, upcoming milestones, and version changes |

### Interactive Jupyter & Colab Notebooks

| Notebook | Topic & Workflow Covered | Launch in Browser |
| :--- | :--- | :---: |
| **[`quickstart_colab.ipynb`](examples/quickstart_colab.ipynb)** | 1-minute interactive demo covering expression syntax, 500-ticker scans, and live 112x speed benchmarks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huuan26/polars-anta/blob/master/examples/quickstart_colab.ipynb) |
| **[`01_quickstart.ipynb`](examples/01_quickstart.ipynb)** | Step-by-step introduction to `.anta` / `.ta` expressions, `pl.Struct` unnesting, and indicator plotting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huuan26/polars-anta/blob/master/examples/01_quickstart.ipynb) |
| **[`02_multi_asset_portfolio_scan.ipynb`](examples/02_multi_asset_portfolio_scan.ipynb)** | Screening a 500-ticker equity universe in ~50 ms using parallel `.over("ticker")` expressions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huuan26/polars-anta/blob/master/examples/02_multi_asset_portfolio_scan.ipynb) |
| **[`03_crypto_momentum_scan.ipynb`](examples/03_crypto_momentum_scan.ipynb)** | Cross-sectional cryptocurrency momentum, volatility filtering, and composite ranking workflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huuan26/polars-anta/blob/master/examples/03_crypto_momentum_scan.ipynb) |
| **[`04_backtest_and_alpha_eval.ipynb`](examples/04_backtest_and_alpha_eval.ipynb)** | 1-line entry/exit crossovers, formatted tearsheets, 5-quantile IC analysis, and dollar-neutral portfolio backtesting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huuan26/polars-anta/blob/master/examples/04_backtest_and_alpha_eval.ipynb) |

---

## Performance benchmark (1,000,000 rows)

<div align="center">

<img src="benchmarks/benchmark_speed.png" alt="polars-anta Performance Benchmark" width="900"/>

</div>

Benchmarks measured across **1,000,000 bars** (single-asset random walk) and a **500 tickers × 2,000 bars universe** (1,000,000 rows) on Polars' multi-threaded Rust engine vs. single-threaded Pandas:

| Operation / Indicator | Dataset Scale | Pandas Execution | `polars-anta` Execution | Measured Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression Slope (OLS)** | 1,000,000 bars | 2.3529 s | **0.0209 s** | **112.4x Faster** |
| **Multi-Asset RSI (14)** | 500 tickers × 2k bars | 0.4398 s | **0.0545 s** | **8.1x Faster** |
| **Multi-Asset SMA (20)** | 500 tickers × 2k bars | 0.1345 s | **0.0168 s** | **8.0x Faster** |
| **Z-Score Normalization** | 1,000,000 bars | 0.0450 s | **0.0267 s** | **1.7x Faster** |
| **Exponential Moving Average** | 1,000,000 bars | 0.0100 s | **0.0069 s** | **1.5x Faster** |
| **Simple Moving Average** | 1,000,000 bars | 0.0170 s | **0.0153 s** | **1.1x Faster** |

---

## API Reference & Indicator Catalog

All indicators, alpha primitives, and evaluation functions are exported at the top level (`import polars_anta as pa`) and registered on `.anta` and `.ta` namespaces. Click any module below to expand its complete function signature reference, or visit the [Online API Reference](https://huuan26.github.io/polars-anta/indicators/).

<details>
<summary><strong>1. Trend & Overlays</strong> — <code>polars_anta.trend</code> · <a href="docs/indicators/trend.md">Detailed Docs</a></summary>

<br/>

| Function Signature | Output Type / Struct Fields | Description |
| :--- | :--- | :--- |
| `sma(close, period=10)` | `Float64` | Simple Moving Average |
| `ema(close, period=10)` | `Float64` | Exponential Moving Average |
| `wma(close, period=10)` | `Float64` | Linearly Weighted Moving Average |
| `hma(close, period=10)` | `Float64` | Hull Moving Average (low-lag trend filter) |
| `dema(close, period=10)` | `Float64` | Double Exponential Moving Average |
| `tema(close, period=10)` | `Float64` | Triple Exponential Moving Average |
| `zlema(close, period=10)` | `Float64` | Zero-Lag Exponential Moving Average |
| `t3(close, period=5, vfactor=0.7)` | `Float64` | Tillson T3 Generalized DEMA Smoothing |
| `alma(close, period=9, offset=0.85, sigma=6.0)` | `Float64` | Arnaud Legoux Gaussian-Filtered Moving Average |
| `kama(close, period=10, fast_span=2, slow_span=30)` | `Float64` | Kaufman's Efficiency-Adaptive Moving Average |
| `mcginley_dynamic(close, period=14, k=0.6)` | `Float64` | McGinley Dynamic Volatility-Adjusted Moving Average |
| `vidya(close, period=9, cmo_period=9)` | `Float64` | Chande's Variable Index Dynamic Average |
| `heikin_ashi(open, high, low, close)` | `Struct(ha_open, ha_high, ha_low, ha_close)` | Vectorized Heikin-Ashi Candlestick Transform (via closed-form `ewm_mean(alpha=0.5)`) |
| `supertrend(high, low, close, period=10, multiplier=3.0)` | `Struct(supertrend, direction)` | ATR-Based Dynamic Trailing Volatility Stop |
| `psar(high, low, af_step=0.02, af_max=0.2)` | `Float64` | Welles Wilder's Parabolic Stop and Reverse (SAR) |
| `ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52, displacement=26)` | `Struct(tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span)` | Ichimoku Kinko Hyo Equilibrium Cloud |
| `adx(high, low, close, period=14)` | `Struct(adx, plus_di, minus_di)` | Average Directional Movement Index (Wilder's smoothing) |
| `vortex(high, low, close, period=14)` | `Struct(vi_plus, vi_minus)` | Vortex Positive & Negative Trend Movement Indicator |
| `aroon(high, low, period=25)` | `Struct(up, down, oscillator)` | Aroon Trend Age Indicator & Oscillator |
| `ssl_channel(high, low, close, period=10)` | `Struct(ssl_up, ssl_down, hlv)` | Semaphore Signal Level (SSL) Channel |
| `trend_intensity_index(close, period=30, signal_period=None)` | `Float64` or `Struct(tii, tii_signal)` | M.H. Pee's Trend Intensity Index (TII) |
| `chop(high, low, close, period=14)` | `Float64` | E.W. Dreiss Choppiness Index (Trend vs. Consolidation `[0, 100]`) |
| `mass_index(high, low, fast_period=9, slow_period=25)` | `Float64` | Donald Dorsey's Range Bulge Reversal Predictor |
| `center_of_gravity(close, period=10)` | `Float64` | John Ehlers' Center of Gravity (COG) Oscillator |
| `pivot_points(high, low, close)` | `Struct(pivot, r1, r2, r3, s1, s2, s3)` | Classic Floor Trader Pivot Points |
| `fibonacci_pivots(high, low, close)` | `Struct(pivot, r1, r2, r3, s1, s2, s3)` | Fibonacci Retracement/Extension Pivot Points (38.2%, 61.8%, 100%) |
| `camarilla_pivots(high, low, close)` | `Struct(pivot, r1..r4, s1..s4)` | Camarilla Intraday Mean-Reversion & Breakout Pivots |
| `woodie_pivots(open, high, low, close)` | `Struct(pivot, r1, r2, s1, s2)` | Woodie's Close-Weighted Pivot Points |

</details>

<details>
<summary><strong>2. Momentum Oscillators</strong> — <code>polars_anta.momentum</code> · <a href="docs/indicators/momentum.md">Detailed Docs</a></summary>

<br/>

| Function Signature | Output Type / Struct Fields | Description |
| :--- | :--- | :--- |
| `rsi(close, period=14)` | `Float64` | Relative Strength Index (Wilder's RMA smoothing) |
| `macd(close, fast_period=12, slow_period=26, signal_period=9)` | `Struct(macd, signal, histogram)` | Moving Average Convergence Divergence |
| `ppo(close, fast_period=12, slow_period=26, signal_period=9)` | `Struct(ppo, signal, histogram)` | Percentage Price Oscillator (scale-invariant MACD) |
| `stochastic(high, low, close, k_period=14, d_period=3, smooth_k=1)` | `Struct(k, d)` | Stochastic Oscillator (`%K`, `%D`) |
| `stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)` | `Struct(k, d)` | Chande & Kroll's Stochastic RSI |
| `squeeze_momentum(high, low, close, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5)` | `Struct(squeeze_on, squeeze_off, no_squeeze, momentum)` | John Carter's TTM Squeeze Volatility Compression & Linear Regression Momentum |
| `stc(close, fast_period=23, slow_period=50, cycle_period=10, factor=0.5)` | `Float64` | Doug Schaff's Trend Cycle (Double-Stochastic MACD Oscillator `[0, 100]`) |
| `tsi(close, long_period=25, short_period=13)` | `Float64` | William Blau's Double-Smoothed True Strength Index |
| `fisher_transform(high, low, period=10)` | `Struct(fisher, signal)` | John Ehlers' Gaussian Fisher Transform |
| `williams_r(high, low, close, period=14)` | `Float64` | Larry Williams' `%R` Overbought/Oversold Oscillator `[-100, 0]` |
| `cci(high, low, close, period=20, constant=0.015)` | `Float64` | Donald Lambert's Commodity Channel Index (Mean Absolute Deviation) |
| `roc(close, period=10)` | `Float64` | Percentage Rate of Change |
| `awesome_oscillator(high, low, fast_period=5, slow_period=34)` | `Float64` | Bill Williams' Midpoint Awesome Oscillator (AO) |
| `cmo(close, period=14)` | `Float64` | Tushar Chande's Momentum Oscillator `[-100, 100]` |
| `coppock(close, wma_period=10, fast_roc=11, slow_roc=14)` | `Float64` | Edwin Coppock's Long-Term Momentum Curve |
| `ultimate_oscillator(high, low, close, p1=7, p2=14, p3=28)` | `Float64` | Larry Williams' Multi-Timeframe Buying Pressure Oscillator |
| `rvi(open, high, low, close, period=10)` | `Struct(rvi, signal)` | John Ehlers' Symmetrically Weighted Relative Vigor Index |
| `kst(close, r1=10, r2=15, r3=20, r4=30, ...)` | `Struct(kst, signal)` | Martin Pring's "Know Sure Thing" Four-ROC Oscillator |
| `dpo(close, period=20)` | `Float64` | Detrended Price Oscillator (cycle isolator) |
| `bop(open, high, low, close)` | `Float64` | Igor Livshin's Balance of Market Power `[-1, 1]` |
| `cfo(close, period=14)` | `Float64` | Tushar Chande's OLS Forecast Oscillator (%) |

</details>

<details>
<summary><strong>3. Volatility & Channels</strong> — <code>polars_anta.volatility</code> · <a href="docs/indicators/volatility.md">Detailed Docs</a></summary>

<br/>

| Function Signature | Output Type / Struct Fields | Description |
| :--- | :--- | :--- |
| `true_range(high, low, close)` | `Float64` | Welles Wilder's Single-Bar True Range |
| `atr(high, low, close, period=14)` | `Float64` | Average True Range (Wilder's RMA smoothing) |
| `natr(high, low, close, period=14)` | `Float64` | Normalized Average True Range (`100 * ATR / Close`) |
| `bollinger_bands(close, period=20, num_std=2.0)` | `Struct(upper, middle, lower, bandwidth, percent_b)` | John Bollinger's Volatility Envelope, Bandwidth & `%B` |
| `bollinger_delta(close, period=20, num_std=2.0)` | `Float64` | Absolute Bollinger Band Spread ($2 \cdot k \cdot \sigma$) |
| `keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)` | `Struct(upper, middle, lower)` | Chester Keltner / Linda Raschke EMA-ATR Volatility Channels |
| `keltner_width(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)` | `Float64` | Normalized Keltner Channel Width (%) |
| `donchian_channels(high, low, period=20)` | `Struct(upper, middle, lower)` | Richard Donchian's Breakout High/Low Channels |
| `donchian_width(high, low, period=20)` | `Float64` | Normalized Donchian Channel Width (%) |
| `chandelier_exit(high, low, close, period=22, multiplier=3.0)` | `Struct(long_stop, short_stop)` | Chuck LeBeau's ATR Trailing Chandelier Exit Stops |
| `starc_bands(high, low, close, period=20, atr_period=15, multiplier=2.0)` | `Struct(upper, middle, lower)` | Manning Stoller's Average Range Channels (STARC) |
| `envelopes(close, period=20, percentage=0.025)` | `Struct(upper, middle, lower)` | Percentage Moving Average Envelopes |
| `stddev(close, period=20)` | `Float64` | Rolling Sample Standard Deviation |
| `ulcer_index(close, period=14)` | `Float64` | Peter Martin's Downside Drawdown Depth & Duration Risk Index |
| `historical_volatility(close, period=20, trading_days=252)` | `Float64` | Annualized Close-to-Close Log Return Volatility |
| `parkinson_volatility(high, low, period=20, trading_days=252)` | `Float64` | Parkinson (1980) Extreme-Value High-Low Range Volatility Estimator |
| `garman_klass_volatility(open, high, low, close, period=20, trading_days=252)` | `Float64` | Garman & Klass (1980) Minimum-Variance OHLC Volatility Estimator |
| `chaikin_volatility(high, low, period=10, roc_period=10)` | `Float64` | Marc Chaikin's High-Low Range Expansion Rate of Change |

</details>

<details>
<summary><strong>4. Volume & Money Flow</strong> — <code>polars_anta.volume</code> · <a href="docs/indicators/volume.md">Detailed Docs</a></summary>

<br/>

| Function Signature | Output Type / Struct Fields | Description |
| :--- | :--- | :--- |
| `obv(close, volume)` | `Float64` | Joe Granville's Cumulative On-Balance Volume |
| `vwap(high, low, close, volume, period=None)` | `Float64` | Volume Weighted Average Price (Cumulative or Rolling Window) |
| `vwap_bands(high, low, close, volume, period=None, mult1=1.0, mult2=2.0, mult3=3.0)` | `Struct(vwap, upper_1..3, lower_1..3)` | Institutional VWAP with Volume-Weighted Standard Deviation Bands |
| `vwma(close, volume, period=20)` | `Float64` | Volume Weighted Moving Average |
| `cmf(high, low, close, volume, period=20)` | `Float64` | Marc Chaikin's Money Flow (CLV-Weighted Volume Ratio) |
| `adl(high, low, close, volume)` | `Float64` | Chaikin Accumulation / Distribution Line |
| `mfi(high, low, close, volume, period=14)` | `Float64` | Quong & Soudack's Volume-Weighted RSI (Money Flow Index `[0, 100]`) |
| `eom(high, low, volume, period=14, scale=10000.0)` | `Float64` | Richard Arms' Ease of Movement (EMV) |
| `klinger(high, low, close, volume, fast_period=34, slow_period=55, signal_period=13)` | `Struct(klinger, signal)` | Stephen Klinger's Volume Force Oscillator |
| `nvi(close, volume)` | `Float64` | Norman Fosback's Negative Volume Index (Smart-Money Tracker) |
| `pvi(close, volume)` | `Float64` | Positive Volume Index (Crowd-Activity Tracker) |
| `pvt(close, volume)` | `Float64` | Cumulative Return-Weighted Price Volume Trend |
| `force_index(close, volume, period=13)` | `Float64` | Alexander Elder's EMA-Smoothed Force Index |
| `vpci(close, volume, short_period=5, long_period=20)` | `Float64` | Buff Dormeier's Volume Price Confirmation Indicator |
| `vr(close, volume, period=26)` | `Float64` | Up-Volume vs. Down-Volume Ratio (%) |

</details>

<details>
<summary><strong>5. Quantitative Statistics & Regression</strong> — <code>polars_anta.stats</code> · <a href="docs/indicators/stats.md">Detailed Docs</a></summary>

<br/>

| Function Signature | Output Type | Description |
| :--- | :--- | :--- |
| `zscore(close, period=20)` | `Float64` | Rolling Standardized Z-Score Anomaly Metric $(x_t - \mu_t) / \sigma_t$ |
| `percentile_rank(close, period=20)` | `Float64` | Vectorized Rolling Empirical Percentile Rank $[0.0, 1.0]$ |
| `skewness(close, period=20)` | `Float64` | Rolling Third Standardized Moment (Return Asymmetry) |
| `kurtosis(close, period=20)` | `Float64` | Rolling Fourth Standardized Moment (Excess Tail Risk) |
| `rolling_cov(x, y, period=20)` | `Float64` | Rolling Sample Covariance $\text{Cov}(X, Y)$ |
| `rolling_corr(x, y, period=20)` | `Float64` | Rolling Pearson Linear Correlation Coefficient $[-1.0, 1.0]$ |
| `rolling_beta(asset_ret, bench_ret, period=60)` | `Float64` | Rolling CAPM Systematic Market Beta $\text{Cov}(R_a, R_m) / \text{Var}(R_m)$ |
| `rolling_alpha(asset_ret, bench_ret, rf=0.0, period=60)` | `Float64` | Rolling Jensen's Risk-Adjusted Excess Alpha |
| `linear_regression_slope(close, period=14)` | `Float64` | Closed-Form Vectorized OLS Linear Regression Slope ($\beta_1$) |
| `linear_regression_intercept(close, period=14)` | `Float64` | Closed-Form Vectorized OLS Linear Regression Intercept ($\beta_0$) |
| `linear_regression_angle(close, period=14)` | `Float64` | Rolling OLS Regression Trend Angle in Degrees $\arctan(\beta_1) \cdot \frac{180}{\pi}$ |
| `linear_regression_forecast(close, period=14)` | `Float64` | Rolling OLS End-of-Window Fitted Value $\hat{y}_t = \beta_0 + \beta_1 (n - 1)$ |

</details>

<details>
<summary><strong>6. Alpha Factor & Microstructure Primitives</strong> — <code>polars_anta.alpha</code> · <a href="docs/indicators/alpha.md">Detailed Docs</a></summary>

<br/>

| Function Signature | Domain | Description |
| :--- | :--- | :--- |
| `cs_rank(expr, over=None, method="average", percentile=True)` | Cross-Sectional | Cross-sectional percentile rank $[0.0, 1.0]$ |
| `cs_zscore(expr, over=None, clip_bound=None)` | Cross-Sectional | Cross-sectional zero-mean unit-variance standardization with optional $z$-clipping |
| `cs_winsorize(expr, over=None, method="quantile", lower=0.01, upper=0.99, threshold=3.0)` | Cross-Sectional | Cross-sectional outlier clipping via `"quantile"`, `"zscore"`, or robust `"mad"` |
| `neutralize(expr, by, over=None, standardize=False)` | Cross-Sectional | Sector/industry/group demeaning & standardization |
| `ts_scale(expr, a=1.0, over=None)` | Cross-Sectional | L1 book-size normalization ($\sum |w_i| = a$) |
| `amihud_illiquidity(close, volume, period=20, scale=1e6)` | Microstructure | Amihud (2002) Illiquidity Ratio $\frac{1}{N}\sum \frac{|r_t|}{\text{DollarVolume}_t} \times \text{scale}$ |
| `roll_spread(close, period=20)` | Microstructure | Roll (1984) Serial Price-Change Covariance Effective Bid-Ask Spread $2\sqrt{\max(0, -\text{Cov}(\Delta P_t, \Delta P_{t-1}))}$ |
| `corwin_schultz_spread(high, low, period=20)` | Microstructure | Corwin & Schultz (2012) Two-Bar High-Low Range Effective Bid-Ask Spread Estimator |
| `kyle_lambda(close, volume, period=20, scale=1e6)` | Microstructure | Kyle (1985) Price-Impact Coefficient $\lambda$ via Rolling OLS on Signed Dollar Volume |
| `twap(high, low, close, period=20)` | Microstructure | Rolling Time-Weighted Average Price |
| `relative_spread(high, low, close)` | Microstructure | Intraday High-Low Relative Range Proxy $(H_t - L_t) / C_t$ |
| `hl_ratio(high, low)` | Microstructure | Log Intraday Volatility Expansion $\ln(H_t / L_t)$ |
| `co_ratio(close, open)` | Microstructure | Log Intraday Sentiment Drift $\ln(C_t / O_t)$ |
| `returns_divergence(close, volume, period=14)` | Microstructure | Rolling Price Return vs. Volume Change Divergence |
| `volume_weighted_momentum(close, volume, period=10)` | Microstructure | Price Rate of Change Amplified by Relative Volume Ratio |
| `ts_rank(close, period=10)` | Time-Series | Vectorized rolling percentile rank $[0.0, 1.0]$ |
| `ts_argmax(close, period=10)` | Time-Series | Vectorized bars elapsed since rolling window peak |
| `ts_argmin(close, period=10)` | Time-Series | Vectorized bars elapsed since rolling window trough |
| `decay_linear(close, period=10)` | Time-Series | Linearly decaying weighted moving average |
| `ts_delta(expr, period=1)` | Time-Series | Difference operator $x_t - x_{t-d}$ |
| `ts_delay(expr, period=1)` | Time-Series | Lag operator $x_{t-d}$ |
| `signed_power(expr, exponent=2.0)` | Time-Series | Sign-preserving power $\text{sign}(x) \cdot |x|^a$ |
| `ts_min(expr, period=10)` | Time-Series | Rolling window minimum |
| `ts_max(expr, period=10)` | Time-Series | Rolling window maximum |
| `ts_sum(expr, period=10)` | Time-Series | Rolling window sum |
| `ts_std(expr, period=10)` | Time-Series | Rolling sample standard deviation |
| `ts_product(expr, period=10)` | Time-Series | Vectorized rolling window product |
| `ts_correlation(x, y, period=10)` | Time-Series | Rolling Pearson correlation |
| `ts_covariance(x, y, period=10)` | Time-Series | Rolling sample covariance |
| `ts_zscore(close, period=20)` | Time-Series | Time-series rolling standardization $(x_t - \mu_t) / \sigma_t$ |
| `hurst_exponent(close, period=50)` | Regime / Complexity | Vectorized Rolling R/S Hurst Exponent $H \in [0, 1]$ (mean-reversion vs. trend persistence) |
| `fractal_dimension(high, low, period=20)` | Regime / Complexity | John Ehlers' Fractal Dimension Index $D \in [1, 2]$ (market noise vs. trend efficiency) |
| `entropy(close, period=20, bins=5)` | Regime / Complexity | Rolling Normalized Shannon Information Entropy $[0, 1]$ |
| `alpha_001` – `alpha_030` | Formulaic Alpha | Canonical formulaic alphas (#1 through #30, Kakushadze 2016) |
| `alpha_041(high, low, close, volume)` | Formulaic Alpha | Geometric High-Low Midpoint vs. VWAP |
| `alpha_054(open, high, low, close)` | Formulaic Alpha | Intraday Candlestick Power Ratio |
| `alpha_101(open, high, low, close)` | Formulaic Alpha | Normalized Intraday Body-to-Range Momentum |

</details>

<details>
<summary><strong>7. Factor Evaluation, Performance Metrics & Vectorized Backtesting</strong> — <code>polars_anta.evaluation</code>, <code>backtest</code> · <a href="docs/indicators/evaluation-and-backtest.md">Detailed Docs</a></summary>

<br/>

| Function Signature | Domain | Description |
| :--- | :--- | :--- |
| `forward_returns(df, close_col="close", periods=(1, 5, 10), group_by=None)` | Factor Evaluation | Multi-horizon forward returns $(P_{t+k} / P_t) - 1$ for factor testing |
| `information_coefficient(df, factor_col, return_col, group_by=None, method="spearman")` | Factor Evaluation | Cross-sectional Rank IC (`"spearman"`) or Normal IC (`"pearson"`) series |
| `ic_summary(df, factor_col, return_col, group_by=None, method="spearman")` | Factor Evaluation | Summary dictionary (`ic_mean`, `ic_std`, `icir`, `t_stat`, `hit_rate`, `n_periods`) |
| `quantile_analysis(df, factor_col, return_col, quantiles=5, group_by=None)` | Factor Evaluation | Quantile bucket returns, Sharpe, win rate, and cumulative performance ($Q_1 \dots Q_N$) |
| `factor_turnover(df, factor_col, asset_col="ticker", time_col="timestamp", lag=1)` | Factor Evaluation | Rank autocorrelation and L1 signal turnover across periods |
| `drawdown_series(returns_or_equity, is_returns=True)` | Performance Metric | Underwater percentage drawdown curve $\le 0.0$ |
| `max_drawdown(returns_or_equity, is_returns=True)` | Performance Metric | Scalar Maximum Drawdown (MDD) |
| `sharpe_ratio(returns, risk_free_rate=0.0, annualization=252.0)` | Performance Metric | Annualized Sharpe Ratio scalar expression |
| `sortino_ratio(returns, risk_free_rate=0.0, annualization=252.0)` | Performance Metric | Annualized Sortino Ratio using downside deviation |
| `calmar_ratio(returns, annualization=252.0)` | Performance Metric | Annualized Return divided by $|MDD|$ |
| `profit_factor(returns)` | Performance Metric | Gross Profit divided by $|Gross Loss|$ |
| `win_rate(returns, include_zeros=False)` | Performance Metric | Proportion of active periods with positive returns |
| `crosses_above(a, b, over=None)` / `crosses_below(a, b, over=None)` | Backtesting | Boolean crossover/crossunder detection against another series or scalar threshold |
| `signals_to_positions(entries, exits, ...)` | Backtesting | Loop-free `.forward_fill()` state machine converting entry/exit triggers to positions |
| `vectorized_backtest(df, signal="signal", ...)` / `df.anta.backtest(...)` | Backtesting | Vectorized backtester with `entries`/`exits`, `weight_scheme`, fees, slippage & tearsheet |
| `BacktestResult` / `compare_strategies(strategies)` | Backtesting | Result container (`.data`, `.summary`, `.report()`) & multi-strategy comparison table |

</details>

---

## Design philosophy

`polars-anta` aims to balance **mathematical clarity**, **cross-sectional expressiveness**, and **execution speed**:

1. **Keep quantitative formulas inspectable** — write indicators using closed-form algebraic identities and native Polars expressions (`pl.Expr`) rather than hiding logic behind compiled C extensions.
2. **Eliminate Python loops** — express recursive smoothing (Wilder's RMA, Heikin-Ashi) via exact `ewm_mean` recurrences and path-dependent rolling operators via zero-copy `sliding_window_view` kernels.
3. **Treat single-asset and multi-asset panels uniformly** — every expression composes naturally with `.over("ticker")` for time-series operations and `.over("date")` for cross-sectional ranking and neutralization.
4. **Prevent look-ahead bias by default** — backtesting utilities shift signals by `signal_lag=1` bar out of the box and account for proportional transaction fees and slippage.
5. **Provide reproducible examples and parity tests** — verify mathematical accuracy through comprehensive unit test suites and self-contained notebooks.

---

## Contributing

Contributions from quantitative researchers, developers, and finance students are welcome! You can help by implementing new indicators, expanding formulaic alphas, improving documentation, or adding tutorial notebooks.

- Browse our open [**Good First Issues & Help Wanted tasks**](https://github.com/huuan26/polars-anta/issues) to find a self-contained issue with full mathematical specifications.
- Read [**CONTRIBUTING.md**](CONTRIBUTING.md) for development setup, testing (`pytest -v`), and linting (`ruff check .`) guidelines.

---

## Project status and support

`polars-anta` is an actively developed open-source project. If you find a bug, need an indicator, or want to suggest a quantitative workflow example, please [open an issue](https://github.com/huuan26/polars-anta/issues) or join [GitHub Discussions](https://github.com/huuan26/polars-anta/discussions).

**Disclaimer:** `polars-anta` is provided for educational and research purposes without warranty of any kind. Quantitative models and backtest results should be independently verified before being used for live trading, investment, or risk-management decisions.

## Author

`polars-anta` was created and is maintained by **Phan Bui Huu An** ([@huuan26](https://github.com/huuan26) • [phanbuihuuan261103@gmail.com](mailto:phanbuihuuan261103@gmail.com)).

## License

`polars-anta` is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Explore the indicators • Run the Colab notebooks • Contribute to open quantitative finance**

[Documentation](https://huuan26.github.io/polars-anta/) • [API Reference](docs/indicators/index.md) • [Examples](examples/) • [GitHub Issues](https://github.com/huuan26/polars-anta/issues)

</div>
