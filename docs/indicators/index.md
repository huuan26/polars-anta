# Indicator Catalog (106 Indicators)

`polars-anta` provides 106 high-performance financial indicators categorized into six specialized domains:

| Category | Count | Key Features | Documentation |
| :--- | :---: | :--- | :--- |
| **Trend & Overlays** | **27** | Moving averages, adaptive tracking, SuperTrend, PSAR, SSL Channel, TII, Ichimoku, Pivots | [View Trend Docs](trend.md) |
| **Momentum Oscillators** | **20** | RSI, MACD, Stochastic, Squeeze Momentum, CMO, PPO, Coppock, Fisher, KST, UO | [View Momentum Docs](momentum.md) |
| **Volatility & Channels** | **18** | Bollinger, Keltner, Donchian, ATR, Parkinson, Garman-Klass, Envelopes | [View Volatility Docs](volatility.md) |
| **Volume Dynamics** | **15** | OBV, VWAP, VWAP Bands, VWMA, CMF, Klinger, NVI, PVI, Elder's Force Index | [View Volume Docs](volume.md) |
| **Quantitative Statistics** | **12** | Z-score, Skewness, Kurtosis, Rolling Cov/Corr, Rolling Beta/Alpha, OLS | [View Statistics Docs](stats.md) |
| **Alpha Factor Primitives** | **14** | WorldQuant 101 Alphas, ts_rank, ts_argmax, Decay Linear, Entropy, Fractal Dimension, Hurst | [View Alpha Docs](alpha.md) |
| **Total Catalog** | **106** | Pure Polars Expressions (`pl.Expr`) with `.over("ticker")` support | |

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
