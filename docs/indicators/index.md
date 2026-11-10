# API Reference & Indicator Catalog

`polars-anta` provides **185+ high-performance technical indicators, market microstructure estimators, AFML machine learning primitives, institutional factor evaluation tools, and execution-realistic backtesting utilities** organized across eight specialized modules:

| Domain / Module | Count | Key Functionality | API Reference |
| :--- | :---: | :--- | :--- |
| **1. Trend & Overlays** (`polars_anta.trend`) | 28 | Moving averages (`sma`, `ema`, `hma`, `kama`, `t3`, `alma`), `heikin_ashi`, `supertrend`, `ichimoku`, `adx`, `psar`, `ssl_channel`, `pivot_points` | [Trend API →](trend.md) |
| **2. Momentum Oscillators** (`polars_anta.momentum`) | 21 | `rsi`, `macd`, `ppo`, `stochastic`, `stoch_rsi`, `squeeze_momentum`, `stc`, `tsi`, `fisher_transform`, `cci`, `williams_r`, `kst` | [Momentum API →](momentum.md) |
| **3. Volatility & Channels** (`polars_anta.volatility`) | 18 | `atr`, `natr`, `bollinger_bands`, `keltner_channels`, `donchian_channels`, `chandelier_exit`, `parkinson_volatility`, `garman_klass_volatility`, `ulcer_index` | [Volatility API →](volatility.md) |
| **4. Volume, Session VWAP & Flow** (`polars_anta.volume`) | 16 | `vwap`, `vwap_bands`, `session_vwap`, `obv`, `cmf`, `mfi`, `adl`, `klinger`, `force_index`, `nvi`, `pvi`, `vpci`, `vr` | [Volume API →](volume.md) |
| **5. Quantitative Statistics & FFD** (`polars_anta.stats`) | 13 | `frac_diff` (Fixed-Window Fractional Differentiation), `zscore`, `percentile_rank`, `skewness`, `kurtosis`, `rolling_cov`, `rolling_corr`, `rolling_beta`, `rolling_alpha`, closed-form OLS `linear_regression_*` | [Statistics API →](stats.md) |
| **6. Alpha, Microstructure & Futures** (`polars_anta.alpha`) | 67 | `cs_rank`, `cs_zscore`, `cs_winsorize`, `neutralize`, `orthogonalize`, `vpin`, `ofi`, `futures_basis`, `oi_regime`, `amihud_illiquidity`, `roll_spread`, `corwin_schultz_spread`, `kyle_lambda`, `hurst_exponent`, 33 WorldQuant Alphas | [Alpha & Microstructure API →](alpha.md) |
| **7. AFML Labeling, Evaluation & Backtest** (`evaluation`, `backtest`) | 25 | `triple_barrier_labels`, `trend_scanning_labels`, `ic_summary` (Newey-West HAC), `ic_decay`, `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `vectorized_backtest`, `BacktestResult.trades` (MAE/MFE), `parameter_sweep` | [Evaluation & Backtest API →](evaluation-and-backtest.md) |
| **8. Information Bars & Roll Stitching** (`polars_anta.resample`) | 6 | `resample_ohlcv`, `resample_volume_bars`, `resample_dollar_bars`, `resample_imbalance_bars`, `stitch_futures_roll`, `validate_ohlcv` | [Resampling & Bars Guide →](../guides/resampling-and-dataframe-api.md) |

---

## Architectural Module Map

```text
polars_anta/
├── trend.py        # Moving averages, adaptive filters, Heikin-Ashi & structural pivots
├── momentum.py     # Velocity oscillators, Gaussian Fisher transform & squeeze detectors
├── volatility.py   # Parkinson/Garman-Klass variance, Wilder ATR stops & price channels
├── volume.py       # Session-anchored VWAP, VWAP bands & institutional money flow
├── stats.py        # Rolling moments, CAPM beta/alpha, closed-form OLS & FFD (frac_diff)
├── alpha.py        # Cross-sectional ops, Gram-Schmidt orthogonalize, VPIN/OFI, futures_basis & 33 formulaic alphas
├── evaluation.py   # Triple-Barrier, Trend-Scanning, Rank IC/ICIR, Newey-West HAC, IC decay & PSR/DSR
├── backtest.py     # Execution-realistic vectorized backtester, MAE/MFE trade ledger & parameter_sweep
├── resample.py     # Time bars, Volume/Dollar/Imbalance bars, continuous futures roll & validation
└── namespace.py    # Dual .anta and .ta registration across pl.Expr, pl.DataFrame & pl.LazyFrame
```

---

## Three Interchangeable Invocation Patterns

Every indicator and quantitative primitive in `polars-anta` can be called via three equivalent interfaces:

```python
import polars as pl
import polars_anta as pa  # Registers .anta and .ta on Expr, DataFrame, and LazyFrame

# 1. Expression Namespace (.anta or .ta) — composable inside .select() / .with_columns()
df = df.with_columns(
    pl.col("close").anta.frac_diff(d=0.4, window=20).over("symbol").alias("close_ffd"),
    pl.col("close").ta.rsi(14).over("symbol").alias("rsi_14"),
)

# 2. DataFrame / LazyFrame Fluent Chaining — end-to-end research & backtesting pipelines
bt = (
    df.anta.validate_ohlcv(group_by="symbol")
    .anta.session_vwap(unnest=True, group_by="symbol")
    .anta.rsi(14, group_by="symbol")
    .anta.backtest(
        entries=pl.col("rsi_14").anta.crosses_above(30, over="symbol"),
        exits=pl.col("rsi_14").anta.crosses_below(70, over="symbol"),
        execution_price="open",
        vol_target=0.15,
        fee_rate=0.0005,
    )
)

# 3. Direct Functional Import — explicit top-level calls
decay_table = pa.ic_decay(df, factor_col="close_ffd", asset_col="symbol", group_by="timestamp")
```
