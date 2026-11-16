<div align="center">

<img src="https://raw.githubusercontent.com/huuan26/polars-anta/master/assets/banner.png" alt="polars-anta banner" width="100%" />

**Vectorized signal engineering, cross-sectional factor conditioning, statistical falsification, and portfolio allocation in Polars.**

[![PyPI version](https://img.shields.io/pypi/v/polars-anta?color=0275d8)](https://pypi.org/project/polars-anta/)
[![Python versions](https://img.shields.io/pypi/pyversions/polars-anta)](https://pypi.org/project/polars-anta/)
[![CI](https://github.com/huuan26/polars-anta/actions/workflows/ci.yml/badge.svg)](https://github.com/huuan26/polars-anta/actions/workflows/ci.yml)
[![Docs](https://github.com/huuan26/polars-anta/actions/workflows/docs.yml/badge.svg)](https://huuan26.github.io/polars-anta/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[Quick Start](#quick-start) | [Workflows](#core-workflows) | [Verification & Case Studies](#verification--empirical-case-studies) | [API Overview](#api-overview) | [Resources](#documentation--examples)

</div>

---

`polars-anta` is a pure-Python quantitative finance library built on [Polars](https://pola.rs/) and NumPy (`polars>=1.0.0`, `numpy>=1.24.0`; no C compiler, SciPy, or CVXPY required). It exposes `241` functions (`246` exports across `15` modules) via top-level imports and `.anta` / `.ta` namespaces across four research stages:

| Research Stage | Core Functions | Method |
| :--- | :--- | :--- |
| **1. Signal & Bar Engineering** | `resample_volume_bars`, `stitch_futures_roll`, `frac_diff`, `vpin`, `ofi`, `alpha_001..101` | Native Polars expressions & zero-copy `sliding_window_view` |
| **2. Cross-Sectional Conditioning** | `cs_gaussianize`, `neutralize_panel`, `orthogonalize_panel`, `collinearity_summary` | Two-stage FWL group + risk projection & batched 3D Lowdin SVD |
| **3. Diagnostics & Falsification** | `analyze_alpha`, `marginal_alpha_report`, `falsify_factors`, `romano_wolf`, `pbo_cscv` | Newey-West HAC IC, turnover/breakeven bps, stationary bootstrap & FWER/FDR |
| **4. Allocation & Backtesting** | `ledoit_wolf_shrinkage`, `marchenko_pastur_denoise`, `solve_fista`, `combine_alphas_walkforward`, `vectorized_backtest` | RMT spectral denoising, joint box-simplex + L1 FISTA & horizon embargo |

---

## Quick start

```bash
python -m pip install polars-anta
# or with uv:
uv add polars-anta
```

### 1. Time-series and cross-sectional expressions in one query

Importing `polars_anta` registers `.anta` and `.ta` on `Expr`, `DataFrame`, and `LazyFrame`. Time-series operators compose with `.over("ticker")` and cross-sectional operators compose with `.over("date")`:

```python
import numpy as np
import polars as pl
import polars_anta as pa

rng = np.random.default_rng(7)
n_days, tickers = 30, ["AAPL", "MSFT", "NVDA", "XOM"]
n = n_days * len(tickers)
close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.015, (len(tickers), n_days)), axis=1)).ravel()
df = pl.DataFrame(
    {
        "date": np.tile(np.arange(n_days), len(tickers)),
        "ticker": np.repeat(tickers, n_days),
        "open": close * (1.0 + rng.normal(0.0, 0.003, n)),
        "high": close * 1.008,
        "low": close * 0.992,
        "close": close,
    }
)

out = df.with_columns(
    pl.col("close").anta.frac_diff(d=0.4, window=15).over("ticker").alias("ffd"),
    pl.col("open")
    .anta.yang_zhang_volatility("high", "low", "close", period=15)
    .over("ticker")
    .round(2)
    .alias("yz_vol"),
).with_columns(pl.col("ffd").anta.cs_gaussianize(over="date").round(3).alias("alpha_gauss"))
print(out.filter(pl.col("date") == 29).select("date", "ticker", "yz_vol", "alpha_gauss"))
```

```text
shape: (4, 4)
┌──────┬────────┬────────┬─────────────┐
│ date ┆ ticker ┆ yz_vol ┆ alpha_gauss │
│ ---  ┆ ---    ┆ ---    ┆ ---         │
│ i64  ┆ str    ┆ f64    ┆ f64         │
╞══════╪════════╪════════╪═════════════╡
│ 29   ┆ AAPL   ┆ 26.74  ┆ -1.15       │
│ 29   ┆ MSFT   ┆ 27.92  ┆ 1.15        │
│ 29   ┆ NVDA   ┆ 26.63  ┆ 0.319       │
│ 29   ┆ XOM    ┆ 23.25  ┆ -0.319      │
└──────┴────────┴────────┴─────────────┘
```

### 2. Multiple-testing factor falsification

Test a panel of candidate signals with Rank IC, Newey-West HAC t-statistics, Politis-Romano stationary bootstrap, Benjamini-Hochberg FDR, Romano-Wolf stepdown FWER, empirical-Bayes shrinkage, and OLS-CUSUM stability:

```python
import numpy as np
import polars as pl
import polars_anta as pa

rng = np.random.default_rng(42)
n_dates, n_tickers = 250, 40
panel = pl.DataFrame(
    {
        "date": np.repeat(np.arange(n_dates), n_tickers),
        "ticker": np.tile(np.arange(n_tickers), n_dates),
        "sig_true": rng.standard_normal(n_dates * n_tickers),
        "sig_noise_1": rng.standard_normal(n_dates * n_tickers),
        "sig_noise_2": rng.standard_normal(n_dates * n_tickers),
    }
).with_columns(
    (0.08 * pl.col("sig_true") + rng.standard_normal(n_dates * n_tickers)).alias("fwd_ret_5")
)

result = panel.anta.falsify_factors(
    factor_cols=["sig_true", "sig_noise_1", "sig_noise_2"],
    return_col="fwd_ret_5",
    group_by="date",
    horizon=5,
    n_boot=500,
    seed=42,
)
print(result.report())
```

```text
==========================================================================
                 POLARS-ANTA FACTOR FALSIFICATION REPORT
==========================================================================
Factors Tested        : 3
Common Periods Used   : 250
Raw p <= 0.05         : 1 (expected false under null: 0.15)
BH FDR Survivors      : 1
Romano-Wolf Survivors : 1
--------------------------------------------------------------------------
Verdict Breakdown     : robust=1 | decaying=0 | noise=2
--------------------------------------------------------------------------
Factor                   IC Mean     NW t     q_BH     p_RW   Post IC    Verdict
--------------------------------------------------------------------------
sig_true                  0.0732     8.41   0.0060   0.0020    0.0701     robust
sig_noise_2               0.0041     0.52   0.7565   0.7685    0.0039      noise
sig_noise_1               0.0026     0.26   0.7565   0.7685    0.0024      noise
==========================================================================
```

---

## Core workflows

### 1. Cross-sectional conditioning, walk-forward FISTA allocation & tearsheet

Neutralize sector and continuous risk exposures simultaneously via two-stage FWL, orthogonalize via batched 3D Lowdin SVD, allocate weights with Marchenko-Pastur RMT denoising and per-factor turnover cost deduction, and evaluate net margins (full runnable script in [`examples/07_cross_sectional_alpha_engineering.py`](examples/07_cross_sectional_alpha_engineering.py)):

```python
raw_cols = ["alpha_1", "alpha_2", "alpha_3"]

# 1. Rank-Gaussianize -> simultaneous FWL sector + beta neutralization -> Lowdin SVD:
neut = panel.with_columns(
    [pl.col(c).anta.cs_gaussianize(over="date").alias(f"{c}_g") for c in raw_cols]
).anta.neutralize_panel(
    [f"{c}_g" for c in raw_cols], time_col="date", risk_cols=["beta"], group_col="sector"
)
neut_cols = [f"{c}_g_neut" for c in raw_cols]
ortho = neut.anta.orthogonalize_panel(neut_cols, time_col="date", method="lowdin")

# 2. Screen incremental dimension (Delta N_eff, spanned R^2, and FWL Residual IC):
marg = ortho.anta.marginal_alpha_report(
    candidate_col="alpha_3_g_neut",
    benchmark_cols=["alpha_1_g_neut", "alpha_2_g_neut"],
    return_col="fwd_1",
    time_col="date",
)

# 3. Embargoed walk-forward allocation (RMT + factor-return cost drag + FISTA + EMA):
wf = ortho.anta.combine_alphas_walkforward(
    factor_cols=[f"{c}_ortho" for c in neut_cols],
    return_col="fwd_1",
    time_col="date",
    asset_col="ticker",
    lookback=60,
    horizon=1,
    signal_lag=1,
    objective_mode="factor_return",
    one_way_cost_bps=10.0,
    signal_ema_alpha=0.25,
    turnover_penalty=0.01,
    constraint="box",
    lower_bound=-1.0,
    upper_bound=1.0,
    cov_estimator="marchenko_pastur",
    ic_shrinkage="empirical_bayes",
)

# 4. Factor tearsheet (IC, half-life, turnover, margin bps, breakeven cost, ADV tiers):
diag = wf.data.anta.analyze_alpha(
    "composite_alpha", close_col="close", volume_col="volume", asset_col="ticker", time_col="date"
)
print(diag.report())
```

Sample output from `python examples/07_cross_sectional_alpha_engineering.py`:

```text
=== Alpha Diagnostic Report: composite_alpha ===
IC Mean: +0.0533 | ICIR: +0.619 | NW t-stat: +9.91 | Half-Life: inf
Gross Sharpe: +11.35 | Net Sharpe: +7.50 | Max DD: -0.81%
Turnover: 0.406 | Gross Margin: +29.58 bps | Breakeven Cost: +29.58 bps
Spread Attribution -> Long Leg: 48.8% | Short Leg: 51.2%
```

### 2. Intraday futures roll, information bars & microstructure

```python
cont_df = raw_contracts.anta.stitch_futures_roll(method="back_adjust")
vol_bars = cont_df.anta.resample_volume_bars(
    threshold=5_000,
    additional_aggs=[
        pl.col("trading_date").last(),
        pl.col("bid_price").last(),
        pl.col("bid_size").last(),
        pl.col("ask_price").last(),
        pl.col("ask_size").last(),
        pl.col("spot_close").last(),
        pl.col("days_to_expiry").last(),
        pl.col("open_interest").last(),
    ],
)

deriv_features = (
    vol_bars.anta.session_vwap(session_by="trading_date", unnest=True, prefix="svwap_")
    .with_columns(
        pl.col("close").anta.vpin("volume", period=20).alias("vpin_20"),
        pl.col("bid_price").anta.ofi("bid_size", "ask_price", "ask_size", period=5).alias("ofi_5"),
        pl.col("close")
        .anta.futures_basis("spot_close", "days_to_expiry", period=20)
        .alias("basis"),
        pl.col("close").anta.oi_regime("open_interest", period=1).alias("oi_state"),
    )
    .unnest("basis", "oi_state")
)
```

### 3. Quant ML labeling (FFD, Triple-Barrier, Trend-Scanning)

```python
ml_dataset = (
    df.with_columns(
        pl.col("close").anta.frac_diff(d=0.4, window=20).over("ticker").alias("close_ffd")
    )
    .anta.triple_barrier_labels(
        horizon=12,
        pt_mult=2.0,
        sl_mult=1.5,
        side="primary_signal",
        unnest=True,
        group_by="ticker",
    )
    .anta.trend_scanning_labels(min_window=5, max_window=20, unnest=True, group_by="ticker")
)
```

### 4. Vectorized backtesting, Sharpe inference (PSR/DSR) & trade ledger

```python
bt = (
    df.anta.ema(period=12, alias="ema_12")
    .anta.ema(period=26, alias="ema_26")
    .anta.backtest(
        entries=pl.col("ema_12").anta.crosses_above("ema_26"),
        exits=pl.col("ema_12").anta.crosses_below("ema_26"),
        execution_price="open",
        flat_mask=pl.col("is_atc_bar"),
        vol_target=0.15,
        no_trade_buffer=0.05,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        initial_capital=100_000.0,
    )
)
sr_tests = bt.data.select(
    pl.col("net_return").anta.probabilistic_sharpe_ratio(benchmark_sr=1.0).alias("psr"),
    pl.col("net_return").anta.deflated_sharpe_ratio(n_trials=50, trials_sr_std=0.4).alias("dsr"),
)
print(bt.trades())  # Round-trip ledger with PnL, holding_bars, MAE, and MFE
```

---

## Verification & empirical case studies

### 1. Solver numerical verification & 1,000,000-row throughput

Projections, proximal solvers, and walk-forward embargoes are checked against analytical bounds in [`benchmarks/alpha_engineering_validation.py`](benchmarks/alpha_engineering_validation.py) ([full output](docs/assets/alpha_engineering_validation.md)):

| Numerical Check | Observed Residual / Error | Bound |
| :--- | :---: | :---: |
| FWL Sector + Continuous Risk Orthogonality (`max \|X_t^T F_neut,t\|`) | `5.72e-13` | `< 1.00e-10` |
| Batched Lowdin Orthogonality (`\|\|F_t^T F_t / (N_t - 1) - I_K\|\|_F`) | `4.14e-15` | `< 1.00e-10` |
| Lowdin Permutation Equivariance (`max \|F_pi - F P\|`) | `3.55e-15` | `< 1.00e-11` |
| Joint Box-Simplex + L1 FISTA KKT Stationarity Residual | `5.09e-11` | `< 1.00e-07` |
| Embargo Prefix-Invariance Weight Leakage (`horizon=5, signal_lag=1`) | `0.00e+00` | `== 0.00e+00` |

<div align="center">

<img src="https://raw.githubusercontent.com/huuan26/polars-anta/master/benchmarks/benchmark_speed.png" alt="polars-anta performance benchmark" width="880"/>

</div>

Throughput on `1,000,000` single-asset bars and a `500 tickers x 2,000 bars` panel (`python benchmarks/benchmark_speed.py`, [methodology](docs/benchmarks.md)): **Linear regression slope `0.0209 s` vs pandas `2.3529 s` (`112.4x`)**, **Multi-asset RSI `0.0545 s` vs `0.4398 s` (`8.1x`)**, **Multi-asset SMA `0.0168 s` vs `0.1345 s` (`8.0x`)**.

### 2. Case study: 150 preregistered signals on Vietnamese equities (`HOSE`)

Every daily signal in `polars-anta` (`150` signals, `146` unique) was tested on the point-in-time top-150 liquid `HOSE` stocks (`2014-2021` discovery, `2022-2026` locked holdout, `60 bps` round-trip cost, preregistered in [`PREREGISTRATION.md`](research/factor_zoo/PREREGISTRATION.md) at commit [`4c1a4b5`](https://github.com/huuan26/polars-anta/commit/4c1a4b559d22a1367ce46b513ab7ff1079fff360)):

- **High collinearity**: `91/150` pass naive `p <= 0.05`, yet daily IC correlations reduce the effective independent dimension to **`N_eff = 6.00`**.
- **Out-of-sample IC persists, standalone net returns do not**: `71` survive Romano-Wolf FWER (`5%`), `64` remain stable in discovery, and **`38/64` replicate in the `2022-2026` holdout** (`97.6%` median IC retention), but **`0` survive `60 bps` round-trip long-only costs** (maximum standalone breakeven is `20.5 bps` round-trip; `90.0%` of holdout spread resides in the short leg). Full report: [`research/factor_zoo/REPORT.md`](research/factor_zoo/REPORT.md).

<div align="center">

<img src="research/factor_zoo/results/figures/01_falsification_funnel.png" alt="Factor Zoo Falsification Funnel on HOSE" width="860" />

</div>

### 3. Case study: Multi-market conditioning & turnover control (Crypto, US Equities, HOSE)

[`benchmarks/real_market_alpha_engineering.py`](benchmarks/real_market_alpha_engineering.py) evaluates 6 textbook daily OHLCV/orderflow formulas out of sample (`signal_lag = 1`, zero parameter tuning) across three real-world regimes ([full 4-stage report](docs/assets/real_market_alpha_engineering.md)):

- **Binance Spot Crypto (`N = 24` USDT pairs with `taker_buy_volume` orderflow, `2024-08` to `2026-11`, `10 bps` one-way)**: Raw Equal-Weight yields `Rank IC = +0.0452` (`NW t = +4.45`) and `+8.31 bps` gross margin (`-0.13` Net Sharpe at `10 bps` taker fee; positive below `8.3 bps` maker/VIP tiers). FWL + Lowdin Equal-Weight lifts pure cross-sectional Gross Sharpe from `+0.62` to `+0.86`, whereas 120-day rolling factor-timing lags fast altcoin rotations (`-0.95` Net Sharpe).
- **US Large-Cap Equities & Sector ETFs (`N = 30`, `2022-09` to `2026-11`, `10 bps` one-way)**: On a zero-edge public OHLCV baseline (`Raw IC = +0.0014`), unpenalized walk-forward churns `59.5%/day` (`-3.65` Net Sharpe); cost-aware RMT + FISTA cuts turnover **12x** (`0.595 -> 0.049`) and lifts Gross Margin from `-2.25` to **`+8.65 bps`** (`-0.04` Net Sharpe).
- **Vietnam `HOSE` Long-Only Holdout (`N <= 150`, `2022-01` to `2026-11`, `30 bps` one-way / `60 bps` RT)**: FWL beta + size neutralization + Lowdin lifts `NW t` from `+2.82` to `+4.25`, and cost-aware RMT + FISTA cuts daily turnover **4.2x** (`0.385 -> 0.091`), lifting Gross Margin by **`+27.58 bps`** (`-14.90 -> +12.68 bps`, `-0.17` vs `-1.78` Raw EW).

<details>
<summary>Expand full out-of-sample tables (3 real markets + controlled synthetic ablation)</summary>

<br/>

**Real-market out-of-sample comparison (`signal_lag = 1`):**

| Market & Regime | Pipeline Stage | Rank IC (NW t) | Daily Turnover | Gross Margin (bps) | Gross Sharpe | Net Sharpe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Crypto L/S** (`N=24`, `10 bps` OW) | `0. Raw Equal-Weight` | `+0.0452` (`+4.45`) | `0.436` | `+8.31` | `+0.62` | `-0.13` |
| **Crypto L/S** (`N=24`, `10 bps` OW) | `1. FWL + Lowdin (EW)` | `+0.0307` (`+4.04`) | `0.586` | `+6.00` | `+0.86` | `-0.58` |
| **Crypto L/S** (`N=24`, `10 bps` OW) | `3. Cost-Aware RMT + FISTA` | `-0.0047` (`-0.68`) | `0.206` | `-8.14` | `-0.43` | `-0.95` |
| **US Equities L/S** (`N=30`, `10 bps` OW) | `0. Raw Equal-Weight` | `+0.0014` (`+0.15`) | `0.294` | `-1.22` | `-0.10` | `-0.89` |
| **US Equities L/S** (`N=30`, `10 bps` OW) | `1. FWL + Lowdin (EW)` | `+0.0029` (`+0.55`) | `0.539` | `-1.96` | `-0.44` | `-2.69` |
| **US Equities L/S** (`N=30`, `10 bps` OW) | `3. Cost-Aware RMT + FISTA` | `+0.0003` (`+0.08`) | `0.049` | `+8.65` | `+0.24` | `-0.04` |
| **Vietnam HOSE L/O** (`N<=150`, `30 bps` OW) | `0. Raw Equal-Weight` | `+0.0138` (`+2.82`) | `0.385` | `-14.90` | `-0.59` | `-1.78` |
| **Vietnam HOSE L/O** (`N<=150`, `30 bps` OW) | `1. FWL + Lowdin (EW)` | `+0.0150` (`+4.25`) | `0.387` | `-5.74` | `-0.23` | `-1.41` |
| **Vietnam HOSE L/O** (`N<=150`, `30 bps` OW) | `3. Cost-Aware RMT + FISTA` | `+0.0043` (`+1.41`) | `0.091` | `+12.68` | `+0.13` | `-0.17` |

**Controlled synthetic ablation (planted factor structure, `N = 150, T = 600, 10 bps` OW):**

| Pipeline Stage | Rank IC | ICIR | Daily Turnover | Gross Margin (bps) | Gross Sharpe | Net Sharpe (`10 bps`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0. Raw Signals (Equal-Weight)` | `+0.0288` | `+0.222` | `1.218` | `+5.33` | `+3.57` | `-3.13` |
| `1. FWL + Lowdin Ortho (Equal-Weight)` | `+0.0311` | `+0.436` | `1.128` | `+6.50` | `+7.75` | `-4.16` |
| `2. Unpenalized Walk-Forward (Sample Cov)` | `+0.0269` | `+0.372` | `1.151` | `+5.66` | `+6.77` | `-5.16` |
| `3. Full Pipeline (RMT + Cost-Aware FISTA + EMA)` | `+0.0208` | `+0.365` | `0.457` | `+18.10` | `+9.26` | `+4.14` |

</details>

---

## API overview

Full function signatures and mathematical derivations are available in the [online API documentation](https://huuan26.github.io/polars-anta/indicators/).

| Area & Module(s) | Functions | Representative Exports | Docs |
| :--- | :---: | :--- | :--- |
| **Alpha Engineering** (`conditioning`, `diagnostics`, `covariance`, `allocation`) | 17 | `cs_gaussianize`, `neutralize_panel`, `orthogonalize_panel`, `lowdin_matrix`, `collinearity_summary`, `analyze_alpha`, `marginal_alpha_report`, `ledoit_wolf_shrinkage`, `marchenko_pastur_denoise`, `solve_fista`, `combine_alphas_walkforward` | [Alpha Engineering](https://huuan26.github.io/polars-anta/indicators/alpha-engineering/) |
| **Statistical Inference** (`inference`) | 17 | `stationary_bootstrap_indices`, `bootstrap_test`, `adjust_pvalues`, `romano_wolf`, `spa_test`, `pbo_cscv`, `empirical_bayes_shrinkage`, `oos_r2`, `clark_west_test`, `diebold_mariano_test`, `cusum_mean_test`, `breakeven_cost`, `falsify_factors` | [Inference](https://huuan26.github.io/polars-anta/indicators/inference/) |
| **Evaluation & Backtest** (`evaluation`, `backtest`) | 28 | `triple_barrier_labels`, `trend_scanning_labels`, `forward_returns`, `ic_summary`, `ic_decay`, `quantile_analysis`, `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `vectorized_backtest`, `extract_trades`, `parameter_sweep` | [Evaluation & Backtest](https://huuan26.github.io/polars-anta/indicators/evaluation-and-backtest/) |
| **Alpha & Microstructure** (`alpha`) | 71 | `cs_rank`, `cs_zscore`, `cs_winsorize`, `neutralize`, `orthogonalize`, `vpin`, `ofi`, `amihud_illiquidity`, `roll_spread`, `corwin_schultz_spread`, `kyle_lambda`, `futures_basis`, `oi_regime`, `hurst_exponent`, `alpha_001..101` | [Alpha](https://huuan26.github.io/polars-anta/indicators/alpha/) |
| **Statistics & OLS** (`stats`) | 15 | `frac_diff`, `zscore`, `percentile_rank`, `skewness`, `kurtosis`, `rolling_beta`, `rolling_alpha`, `rolling_cov`, `rolling_corr`, `linear_regression_*` (`slope`, `intercept`, `angle`, `forecast`, `r2`, `stderr`) | [Stats](https://huuan26.github.io/polars-anta/indicators/stats/) |
| **Information Bars & Roll** (`resample`) | 6 | `resample_ohlcv`, `resample_volume_bars`, `resample_dollar_bars`, `resample_imbalance_bars`, `stitch_futures_roll`, `validate_ohlcv` | [Resampling](https://huuan26.github.io/polars-anta/guides/resampling-and-dataframe-api/) |
| **Trend & Overlays** (`trend`) | 28 | `sma`, `ema`, `wma`, `hma`, `dema`, `tema`, `zlema`, `t3`, `alma`, `kama`, `mcginley_dynamic`, `vidya`, `heikin_ashi`, `supertrend`, `psar`, `ichimoku`, `adx`, `vortex`, `aroon`, `pivot_points` | [Trend](https://huuan26.github.io/polars-anta/indicators/trend/) |
| **Momentum Oscillators** (`momentum`) | 23 | `rsi`, `macd`, `ppo`, `trix`, `stochastic`, `stoch_rsi`, `squeeze_momentum`, `stc`, `tsi`, `fisher_transform`, `williams_r`, `cci`, `roc`, `awesome_oscillator`, `cmo`, `ultimate_oscillator` | [Momentum](https://huuan26.github.io/polars-anta/indicators/momentum/) |
| **Volatility & Channels** (`volatility`) | 20 | `atr`, `natr`, `true_range`, `bollinger_bands`, `keltner_channels`, `donchian_channels`, `chandelier_exit`, `parkinson_volatility`, `garman_klass_volatility`, `rogers_satchell_volatility`, `yang_zhang_volatility`, `ulcer_index` | [Volatility](https://huuan26.github.io/polars-anta/indicators/volatility/) |
| **Volume & Money Flow** (`volume`) | 16 | `vwap`, `vwap_bands`, `session_vwap`, `vwma`, `obv`, `cmf`, `adl`, `mfi`, `eom`, `klinger`, `nvi`, `pvi`, `pvt`, `force_index`, `vpci`, `vr` | [Volume](https://huuan26.github.io/polars-anta/indicators/volume/) |

---

## Documentation & examples

- **Guides & Benchmarks**: [Cross-Sectional Alpha Engineering](docs/indicators/alpha-engineering.md) | [Real-Market Benchmark](docs/assets/real_market_alpha_engineering.md) | [Factor Zoo Report](research/factor_zoo/REPORT.md) | [Architecture & Proofs](docs/architecture.md) | [Quant Theory](docs/guides/quant-theory-and-workflows.md) | [Multi-Asset Scaling](docs/guides/multi-asset.md) | [Changelog](CHANGELOG.md)
- **Scripts**: [`07_cross_sectional_alpha_engineering.py`](examples/07_cross_sectional_alpha_engineering.py) | [`04_backtest_showcase.py`](examples/04_backtest_showcase.py) | [`quickstart.py`](examples/quickstart.py)
- **Notebooks** ([Google Colab](https://colab.research.google.com/github/huuan26/polars-anta/blob/master/examples/quickstart_colab.ipynb)): [`quickstart_colab.ipynb`](examples/quickstart_colab.ipynb) | [`01_quickstart.ipynb`](examples/01_quickstart.ipynb) | [`02_multi_asset_portfolio_scan.ipynb`](examples/02_multi_asset_portfolio_scan.ipynb) | [`03_crypto_momentum_scan.ipynb`](examples/03_crypto_momentum_scan.ipynb) | [`04_backtest_and_alpha_eval.ipynb`](examples/04_backtest_and_alpha_eval.ipynb) | [`05_statistical_arbitrage_pairs_trading.ipynb`](examples/05_statistical_arbitrage_pairs_trading.ipynb) | [`06_multi_factor_sector_neutral_portfolio.ipynb`](examples/06_multi_factor_sector_neutral_portfolio.ipynb)

---

## Contributing & license

Contributions are welcome - see [CONTRIBUTING.md](CONTRIBUTING.md), [open issues](https://github.com/huuan26/polars-anta/issues), and [GitHub Discussions](https://github.com/huuan26/polars-anta/discussions).

**Disclaimer:** `polars-anta` is provided for educational and research purposes without warranty of any kind. Quantitative models and backtests must be independently verified before live trading or risk decisions.

Maintained by **Phan Bui Huu An** ([@huuan26](https://github.com/huuan26), [phanbuihuuan261103@gmail.com](mailto:phanbuihuuan261103@gmail.com)) under the **Apache License 2.0** ([LICENSE](LICENSE)).
