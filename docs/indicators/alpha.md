# Alpha Factor Primitives & Formulaic Alphas (31 Primitives)

Inspired by Kakushadze (2015, *101 Formulaic Alphas*), WorldQuant, and modern quantitative factor investing, these building blocks allow quantitative analysts to compose complex multi-asset predictive alphas and portfolio weights natively in Polars expressions.

---

## 1. Cross-Sectional & Neutralization Operators

### `cs_rank(expr, over=None, method="average", percentile=True)`
Cross-Sectional Rank (WorldQuant Alpha primitive `rank(x)`).
Computes the rank of values across assets at each point in time. When `percentile=True`, scales the ranks between $1/N$ and $1.0$ (or $0.0$ to $1.0$).

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

## 4. Canonical Formulaic Alphas (Kakushadze 101 Alphas)

### `alpha_001(close, returns=None, over=None)`
WorldQuant Alpha #1:
$$\text{rank}(\text{ts\_argmax}(\text{signed\_power}(\text{returns} < 0 \;?\; \text{std}(\text{returns}, 20) : \text{close}, 2), 5)) - 0.5$$

### `alpha_041(high, low, close, volume)`
WorldQuant Alpha #41:
$$\sqrt{\text{High} \times \text{Low}} - \text{VWAP}$$

### `alpha_054(open, high, low, close)`
WorldQuant Alpha #54:
$$-\frac{(\text{Low} - \text{Close}) \times \text{Open}^5}{(\text{Low} - \text{High}) \times \text{Close}^5 + 10^{-4}}$$

### `alpha_101(open, high, low, close)`
WorldQuant Alpha #101:
$$\frac{\text{Close} - \text{Open}}{(\text{High} - \text{Low}) + 10^{-4}}$$
