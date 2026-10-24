# Multi-Timeframe Resampling & DataFrame API

In quantitative workflows, analysts and traders often ingest tick or 1-minute market data and need to resample it into higher-timeframe bars (e.g., 5-minute, 15-minute, 1-hour, 1-day) before calculating technical features.

`polars-anta` provides:
1. **`resample_ohlcv`**: High-performance, zero-copy OHLCV aggregation powered by Polars' `group_by_dynamic`.
2. **`validate_ohlcv`**: Instant structural integrity validation (High >= Low, non-negative prices, no out-of-order timestamps).
3. **Fluent DataFrame & LazyFrame Namespaces**: Direct method chaining via `df.anta...` or `df.ta...` (and `lf.anta...` or `lf.ta...`).

---

## 1. Multi-Timeframe Resampling

### Single Asset Resampling

```python
import polars as pl
from polars_anta import resample_ohlcv

# Resample 1-minute bars to 1-hour bars:
df_1h = resample_ohlcv(df_1m, every="1h")
```

Under the hood, `resample_ohlcv` computes standard financial candle rules:
- **`open`**: First open price in the window
- **`high`**: Maximum high price in the window
- **`low`**: Minimum low price in the window
- **`close`**: Last close price in the window
- **`volume`**: Sum of volume in the window (if present)

### Multi-Ticker Universe Resampling

When handling universe data containing hundreds or thousands of tickers:

```python
# Resample multi-ticker data in parallel:
universe_1h = resample_ohlcv(
    universe_1m,
    every="1h",
    group_by="symbol",
    time_col="timestamp",
)
```

You can also pass multi-column groups, such as `group_by=["exchange", "symbol"]`.

### Custom Column Names and Additional Aggregations

```python
resampled = resample_ohlcv(
    df,
    every="15m",
    time_col="date_time",
    open_col="open_price",
    high_col="high_price",
    low_col="low_price",
    close_col="close_price",
    volume_col="turnover",
    additional_aggs=[
        pl.col("num_trades").sum().alias("trades_count"),
        pl.col("spread").mean().alias("avg_spread"),
    ],
)
```

---

## 2. OHLCV Data Validation

Before training machine learning models or deploying algorithmic strategies, validating data integrity is essential:

```python
from polars_anta import validate_ohlcv

# 1. Strict mode (default): returns original df if valid, raises ValueError on error
clean_df = validate_ohlcv(df)

# 2. Multi-ticker validation:
clean_universe = validate_ohlcv(universe_df, group_by="symbol")

# 3. Report mode: returns structured diagnostics dictionary without raising
report = validate_ohlcv(df, return_report=True)
if not report["valid"]:
    print("Errors found:", report["errors"])
    print("Metrics:", report["metrics"])
```

Checks performed:
- Schema types: price columns are numeric, timestamp is `Date` or `Datetime`.
- Candle consistency: `high >= low`, `high >= open`, `high >= close`, `low <= open`, `low <= close`.
- Values: prices and volume are non-negative.
- Timestamps: detects null timestamps and out-of-order timestamps per group.

---

## 3. Fluent DataFrame & LazyFrame API (`.anta` and `.ta`)

Both `.anta` and `.ta` namespaces are registered directly on `pl.DataFrame` and `pl.LazyFrame`.

### Method Chaining

```python
import polars as pl
import polars_anta

df_features = (
    df.anta.validate_ohlcv()
    .anta.resample("15m", group_by="symbol")
    .anta.sma(period=20)
    .anta.ema(period=50)
    .anta.rsi(period=14)
    .anta.macd(unnest=True)
    .anta.bollinger_bands(period=20, unnest=True)
)
```

### Struct Unnesting & Prefixes

Multi-output indicators (MACD, Bollinger Bands, Stochastic, Supertrend, etc.) can be kept as a `pl.Struct` or expanded into individual columns with clear prefixes:

```python
# Keep as struct column `bb`
df.anta.bollinger_bands(period=20, unnest=False)

# Unnest into `bb_upper`, `bb_middle`, `bb_lower`, `bb_bandwidth`, `bb_percent_b`
df.anta.bollinger_bands(period=20, unnest=True, prefix="bb_")
```

### Full Quant Suite in One Pass: `.add_all()`

Compute a comprehensive, standard set of technical indicators across trend, momentum, volatility, volume, and statistical alpha factors in a single parallelized pass:

```python
# Generates SMA, EMA, ADX, SuperTrend, RSI, MACD, Stoch, Williams %R, CCI, ROC,
# ATR, Bollinger Bands, Keltner Channels, OBV, VWAP, CMF, MFI, ZScore, TS Rank
features_df = df.anta.add_all(group_by="symbol")
```

### LazyFrame Out-of-Core Execution

The DataFrame namespace works seamlessly on `pl.LazyFrame`, enabling streaming execution for datasets larger than available RAM:

```python
lf_plan = (
    pl.scan_parquet("data/*.parquet")
    .anta.resample("1h", group_by="ticker")
    .anta.sma(20)
    .anta.rsi(14)
    .anta.add_all(group_by="ticker")
)

# Executes in multi-threaded Rust query engine
result = lf_plan.collect()
```
