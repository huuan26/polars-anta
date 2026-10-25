# Indicator Catalog (123 Indicators & Alpha Primitives)

`polars-anta` provides 123 high-performance financial indicators and alpha primitives categorized into six specialized domains:

| Category | Count | Key Features | Documentation |
| :--- | :---: | :--- | :--- |
| **Trend & Overlays** | **27** | Moving averages, adaptive tracking, SuperTrend, PSAR, SSL Channel, TII, Ichimoku, Pivots | [View Trend Docs](trend.md) |
| **Momentum Oscillators** | **20** | RSI, MACD, Stochastic, Squeeze Momentum, CMO, PPO, Coppock, Fisher, KST, UO | [View Momentum Docs](momentum.md) |
| **Volatility & Channels** | **18** | Bollinger, Keltner, Donchian, ATR, Parkinson, Garman-Klass, Envelopes | [View Volatility Docs](volatility.md) |
| **Volume Dynamics** | **15** | OBV, VWAP, VWAP Bands, VWMA, CMF, Klinger, NVI, PVI, Elder's Force Index | [View Volume Docs](volume.md) |
| **Quantitative Statistics** | **12** | Z-score, Skewness, Kurtosis, Rolling Cov/Corr, Rolling Beta/Alpha, OLS | [View Statistics Docs](stats.md) |
| **Alpha Factor Primitives** | **31** | WorldQuant 101 Alphas, cs_rank, neutralize, ts_scale, ts_delta, ts_delay, decay_linear, signed_power, ts_rank, Entropy, Fractal Dimension, Hurst, Alpha 1, 41, 54, 101 | [View Alpha Docs](alpha.md) |
| **Total Catalog** | **123** | Pure Polars Expressions (`pl.Expr`) with `.over("ticker")` support | |

---

## Unified Usage Syntax

All indicators can be invoked either via the chained expression syntax `.anta.<name>()` (or classic `.ta.<name>()`) or through functional imports:

```python
import polars as pl
import polars_anta

# Method 1: Chained syntax (.anta or .ta)
df.select(pl.col("close").anta.rsi(14))

# Method 2: Functional syntax
from polars_anta import rsi

df.select(rsi("close", 14))
```
