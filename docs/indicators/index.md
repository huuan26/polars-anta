# API Reference & Indicator Catalog

`polars-anta` provides high-performance technical indicators, cross-sectional alpha primitives, statistical regressions, factor evaluation tools, and vectorized backtesting utilities, organized into seven specialized modules:

| Domain / Module | Key Functionality | API Reference |
| :--- | :--- | :--- |
| **1. Trend & Overlays** (`polars_anta.trend`) | Moving averages (`sma`, `ema`, `hma`, `kama`, `t3`, `alma`), `heikin_ashi`, `supertrend`, `ichimoku`, `adx`, `psar`, `ssl_channel`, `pivot_points` | [Trend API →](trend.md) |
| **2. Momentum Oscillators** (`polars_anta.momentum`) | `rsi`, `macd`, `ppo`, `stochastic`, `stoch_rsi`, `squeeze_momentum`, `stc`, `tsi`, `fisher_transform`, `cci`, `williams_r`, `kst` | [Momentum API →](momentum.md) |
| **3. Volatility & Channels** (`polars_anta.volatility`) | `atr`, `natr`, `bollinger_bands`, `keltner_channels`, `donchian_channels`, `chandelier_exit`, `parkinson_volatility`, `garman_klass_volatility`, `ulcer_index` | [Volatility API →](volatility.md) |
| **4. Volume & Money Flow** (`polars_anta.volume`) | `vwap`, `vwap_bands`, `obv`, `cmf`, `mfi`, `adl`, `klinger`, `force_index`, `nvi`, `pvi`, `vpci`, `vr` | [Volume API →](volume.md) |
| **5. Quantitative Statistics** (`polars_anta.stats`) | `zscore`, `percentile_rank`, `skewness`, `kurtosis`, `rolling_cov`, `rolling_corr`, `rolling_beta`, `rolling_alpha`, closed-form OLS `linear_regression_*` | [Statistics API →](stats.md) |
| **6. Alpha & Microstructure** (`polars_anta.alpha`) | `cs_rank`, `cs_zscore`, `cs_winsorize`, `neutralize`, `amihud_illiquidity`, `roll_spread`, `corwin_schultz_spread`, `kyle_lambda`, `hurst_exponent`, formulaic alphas (`alpha_001`–`alpha_101`) | [Alpha & Microstructure API →](alpha.md) |
| **7. Evaluation & Backtesting** (`polars_anta.evaluation`, `backtest`) | `forward_returns`, `information_coefficient`, `ic_summary`, `quantile_analysis`, `factor_turnover`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `vectorized_backtest` | [Evaluation & Backtest API →](evaluation-and-backtest.md) |

---

## How `polars-anta` is Organized

The library separates **data engineering**, **indicator & alpha expressions**, **factor evaluation**, and **vectorized backtesting** into clean, inspectable modules:

```text
polars_anta/
├── trend.py        # Moving averages, trend overlays, candlestick transforms & pivots
├── momentum.py     # Momentum oscillators, cycle detectors & divergence indicators
├── volatility.py   # OHLC volatility estimators, ATR stops & price channels
├── volume.py       # Volume-weighted price, money flow & accumulation metrics
├── stats.py        # Rolling moments, CAPM beta/alpha & closed-form OLS regression
├── alpha.py        # Cross-sectional factor ops, market microstructure & formulaic alphas
├── evaluation.py   # Alphalens-style IC/IR, quantile monotonicity & risk metrics
├── backtest.py     # Loop-free vectorized backtester, crossover signals & tearsheets
├── resample.py     # Multi-timeframe OHLCV resampling & schema/integrity validation
└── namespace.py    # Dual .anta and .ta registration on Expr, DataFrame & LazyFrame
```

---

## Three Ways to Invoke Any Function

Every indicator and primitive in `polars-anta` can be called through three interchangeable interfaces:

```python
import polars as pl
import polars_anta as pa  # Registers .anta and .ta on Expr, DataFrame, and LazyFrame

# 1. Expression Namespace (.anta or .ta) — composable inside .select() / .with_columns()
df.with_columns(
    pl.col("close").anta.rsi(14).over("symbol").alias("rsi_14"),
    pl.col("close").ta.ema(20).over("symbol").alias("ema_20"),
)

# 2. DataFrame / LazyFrame Fluent Chaining — rapid end-to-end pipelines
df.anta.validate_ohlcv().anta.rsi(14).anta.macd(unnest=True).anta.backtest(
    entries=pl.col("rsi_14").anta.crosses_above(30),
    exits=pl.col("rsi_14").anta.crosses_below(70),
)

# 3. Direct Functional Import — explicit function calls
df.with_columns(pa.rsi("close", period=14).alias("rsi_14"))
```
