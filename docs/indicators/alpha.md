# Alpha Factor & Microstructure Primitives (66 Primitives)

Inspired by Kakushadze (2015, *101 Formulaic Alphas*), WorldQuant, and modern quantitative factor investing, these building blocks allow quantitative analysts to compose complex multi-asset predictive alphas, market microstructure estimators, and portfolio weights natively in Polars expressions.

---

## 1. Cross-Sectional & Neutralization Operators

### `cs_rank(expr, over=None, method="average", percentile=True)`
Cross-Sectional Rank (WorldQuant Alpha primitive `rank(x)`).
Computes the rank of values across assets at each point in time. When `percentile=True`, scales the ranks between $1/N$ and $1.0$ (or $0.0$ to $1.0$).

### `cs_zscore(expr, over=None, clip_bound=None)`
Cross-Sectional Z-Score Standardization.
Standardizes an alpha signal across the asset universe at each timestamp ($(x - \mu_t) / \sigma_t$), with optional symmetric clipping into $[-\text{clip\_bound}, \text{clip\_bound}]$.

### `cs_winsorize(expr, over=None, method="quantile", lower=0.01, upper=0.99, threshold=3.0)`
Cross-Sectional Winsorization (Outlier Clipping).
Clips extreme cross-sectional outliers across the asset universe at each timestamp using empirical quantiles (`"quantile"`), standard deviation bounds (`"zscore"`), or Median Absolute Deviation (`"mad"`).

### `neutralize(expr, by, over=None, standardize=False)`
Group / Industry / Sector Neutralization (WorldQuant Alpha primitive `indneutralize(x, g)`).
Demeans an alpha signal within categorical groups (e.g. industry, sector), ensuring the resulting portfolio is dollar-neutral with respect to that classification. When `standardize=True`, also divides by the group standard deviation to yield z-scores.

### `ts_scale(expr, a=1.0, over=None)`
Scale Primitive (WorldQuant Alpha primitive `scale(x, a)`).
Rescales values such that the sum of their absolute values equals $a$: $a \cdot x / \sum |x|$. Commonly used to size long/short portfolio positions with targeted gross leverage.

---

## 2. Time-Series Primitives (WorldQuant 101 Alphas)

### `ts_rank(close, period=10)`
Time-series rank. Computes the rank of the current element within the lookback window, scaled from 0.0 (lowest) to 1.0 (highest).

### `ts_argmax(close, period=10)`
Number of bars ago the maximum price occurred in the lookback window (0 = today).

### `ts_argmin(close, period=10)`
Number of bars ago the minimum price occurred in the lookback window.

### `decay_linear(close, period=10)`
Linearly decayed moving average with weights $[1, 2, \dots, N]$. Essential building block for factor momentum decay.

### `ts_delta(expr, period=1)`
Time-series Delta (WorldQuant Alpha primitive `delta(x, d)`): $x_t - x_{t - d}$.

### `ts_delay(expr, period=1)`
Time-series Delay / Lag (WorldQuant Alpha primitive `delay(x, d)`): $x_{t - d}$.

### `signed_power(expr, exponent=2.0)`
Signed Power (WorldQuant Alpha primitive `signed_power(x, a)`): $\text{sign}(x) \cdot |x|^a$. Preserves direction while non-linearly expanding signal magnitudes.

### `ts_min(expr, period=10)`
Rolling time-series minimum over lookback window (WorldQuant Alpha primitive `ts_min(x, d)`).

### `ts_max(expr, period=10)`
Rolling time-series maximum over lookback window (WorldQuant Alpha primitive `ts_max(x, d)`).

### `ts_sum(expr, period=10)`
Rolling time-series sum over lookback window (WorldQuant Alpha primitive `ts_sum(x, d)`).

### `ts_std(expr, period=10)`
Rolling time-series standard deviation over lookback window (WorldQuant Alpha primitive `ts_std(x, d)`).

### `ts_correlation(x, y, period=10)`
Rolling Pearson correlation between $x$ and $y$ over lookback window (WorldQuant Alpha primitive `correlation(x, y, d)`).

### `ts_covariance(x, y, period=10)`
Rolling sample covariance between $x$ and $y$ over lookback window (WorldQuant Alpha primitive `covariance(x, y, d)`).

### `ts_product(expr, period=10)`
Rolling time-series product $\prod_{i=0}^{d-1} x_{t-i}$ over lookback window (WorldQuant Alpha primitive `product(x, d)`).

### `ts_zscore(close, period=20)`
Time-series rolling Z-score: $(x - \text{mean}) / \text{std}$.

---

## 3. Signal Transforms & Market Dynamics

### `returns_divergence(close, volume, period=14)`
Price vs Volume Returns Divergence: $\text{ZScore}(R_{price}) - \text{ZScore}(R_{vol})$. Identifies divergence between price movement and volume backing.

### `hl_ratio(high, low)`
Log intraday price expansion: $\ln(\text{High} / \text{Low})$.

### `co_ratio(close, open)`
Log intraday drift / conviction: $\ln(\text{Close} / \text{Open})$.

### `volume_weighted_momentum(close, volume, period=10)`
Price momentum amplified or dampened by relative trading volume:

$$\text{ROC}(P, period) \times \frac{\text{Volume}}{\text{SMA}(\text{Volume}, period)}$$

### `relative_spread(high, low, close)`
Normalized high-low spread relative to closing price: $(\text{High} - \text{Low}) / \text{Close}$.

### `twap(high, low, close, period=20)`
Time-Weighted Average Price. Rolling mean of typical price $(H + L + C) / 3$.

### `entropy(close, period=20, bins=5)`
Rolling Shannon Information Entropy of return distribution. Quantifies regime uncertainty: low entropy signifies strong directional trending, while high entropy indicates choppy market noise.

### `fractal_dimension(high, low, period=20)`
Ehlers' Fractal Dimension (FRAMA component). Measures the fractal roughness of price motion bounded between $1.0$ (smooth directional trend) and $2.0$ (random noise).

### `hurst_exponent(close, period=50)`
Rolling Hurst Exponent via Rescaled Range (R/S) analysis. Measures long-term memory: $H > 0.5$ indicates persistent trending, $H < 0.5$ indicates mean-reverting, and $H \approx 0.5$ indicates random walk.

---

## 4. Market Microstructure & Liquidity Estimators

### `amihud_illiquidity(close, volume, period=20, scale=1e6)`
Amihud (2002) Illiquidity Ratio. Measures price impact per unit of dollar trading volume over a rolling window:
$$\text{ILLIQ}_t = \frac{1}{N} \sum_{k=0}^{N-1} \frac{|R_{t-k}|}{P_{t-k} \cdot V_{t-k}} \times \text{scale}$$

### `roll_spread(close, period=20)`
Roll (1984) Implicit Effective Bid-Ask Spread Estimator. Estimates proportional bid-ask bounce from first-order serial autocovariance of log price changes:
$$S_{\text{Roll}} = 2 \sqrt{\max(0, -\text{Cov}(\Delta \ln P_t, \Delta \ln P_{t-1}))}$$

### `corwin_schultz_spread(high, low, period=20)`
Corwin & Schultz (2012) High-Low Effective Bid-Ask Spread Estimator. Separates transient bid-ask bounce from true volatility using 1-bar and 2-bar High/Low log ranges without requiring L2 orderbook data.

### `kyle_lambda(close, volume, period=20, scale=1e6)`
Kyle (1985) Price Impact Coefficient ($\lambda$). Estimates the rolling regression slope of price returns per unit of signed order flow ($\text{sign}(R_t) \cdot V_t$).

---

## 5. Canonical Formulaic Alphas (33 WorldQuant 101 Alphas)

`polars-anta` includes 33 pre-built institutional formulaic alphas from **Zura Kakushadze (2016), *101 Formulaic Alphas*** (`alpha_001` through `alpha_030`, `alpha_041`, `alpha_054`, and `alpha_101`). All formulaic alphas support both expression-level invocation and DataFrame/LazyFrame chaining (`df.anta.alpha_012(group_by="symbol")`).

| Function | Inputs Required | Key Mechanism / Description |
| :--- | :--- | :--- |
| `alpha_001` – `alpha_010` | `open`, `high`, `low`, `close`, `volume` | Volatility-conditional signed power rank, volume-price correlation reversals, open-VWAP divergence, and 5-day trend persistence |
| `alpha_011` – `alpha_020` | `open`, `high`, `low`, `close`, `volume` | VWAP-close extrema, momentum-volume reversal (`alpha_012`), rank covariance (`alpha_013`–`alpha_016`), and high-breakout ranking |
| `alpha_021` – `alpha_030` | `open`, `high`, `low`, `close`, `volume` | Bollinger/volume regime switches (`alpha_021`), 20-day high breakouts (`alpha_023`), volume-rank correlation (`alpha_026`), and 3-day sign momentum (`alpha_030`) |
| `alpha_041(high, low, close, volume)` | `high`, `low`, `close`, `volume` | Geometric High-Low midpoint vs. VWAP: $\sqrt{\text{High} \times \text{Low}} - \text{VWAP}$ |
| `alpha_054(open, high, low, close)` | `open`, `high`, `low`, `close` | Intraday candlestick power ratio: $-\frac{(\text{Low} - \text{Close}) \cdot \text{Open}^5}{(\text{Low} - \text{High}) \cdot \text{Close}^5 + 10^{-4}}$ |
| `alpha_101(open, high, low, close)` | `open`, `high`, `low`, `close` | Normalized intraday body-to-range momentum: $\frac{\text{Close} - \text{Open}}{(\text{High} - \text{Low}) + 10^{-4}}$ |
