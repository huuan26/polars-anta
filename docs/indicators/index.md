# Indicator Catalog (131 Indicators & Alpha Primitives)

`polars-anta` provides 131 high-performance financial indicators and alpha primitives categorized into six specialized domains:

| Category | Count | Key Features | Documentation |
| :--- | :---: | :--- | :--- |
| **Trend & Overlays** | **28** | Moving averages, Heikin-Ashi, adaptive tracking, SuperTrend, PSAR, SSL Channel, TII, Ichimoku, Pivots | [View Trend Docs](trend.md) |
| **Momentum Oscillators** | **21** | RSI, MACD, Stochastic, Squeeze Momentum, STC, CMO, PPO, Coppock, Fisher, KST, UO | [View Momentum Docs](momentum.md) |
| **Volatility & Channels** | **18** | Bollinger, Keltner, Donchian, ATR, Parkinson, Garman-Klass, Envelopes | [View Volatility Docs](volatility.md) |
| **Volume Dynamics** | **15** | OBV, VWAP, VWAP Bands, VWMA, CMF, Klinger, NVI, PVI, Elder's Force Index | [View Volume Docs](volume.md) |
| **Quantitative Statistics** | **12** | Z-score, Skewness, Kurtosis, Rolling Cov/Corr, Rolling Beta/Alpha, OLS | [View Statistics Docs](stats.md) |
| **Alpha & Microstructure** | **37** | WorldQuant 101 Alphas, cs_rank, cs_zscore, cs_winsorize, Amihud, Roll/Corwin-Schultz Spread, Kyle's Lambda, Entropy, Fractal Dimension, Hurst | [View Alpha Docs](alpha.md) |
| **Total Catalog** | **131** | Pure Polars Expressions (`pl.Expr`) with `.over("ticker")` support | |

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
