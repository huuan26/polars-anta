# Alpha Factor Primitives (14 Indicators)

Inspired by WorldQuant 101 Alphas and modern factor investing literature, these building blocks allow quantitative analysts to compose complex multi-asset predictive alphas.

---

### `ts_rank(close, period=10)`
Time-series rank. Computes the rank of the current element within the lookback window, scaled from 0.0 (lowest) to 1.0 (highest).

### `ts_argmax(close, period=10)`
Number of bars ago the maximum price occurred in the lookback window (0 = today).

### `ts_argmin(close, period=10)`
Number of bars ago the minimum price occurred in the lookback window.

### `decay_linear(close, period=10)`
Linearly decayed moving average with weights $[1, 2, \dots, N]$. Essential building block for factor momentum decay.

### `ts_zscore(close, period=20)`
Time-series rolling Z-score. Standardizes factor values across time before cross-sectional neutralization.

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
