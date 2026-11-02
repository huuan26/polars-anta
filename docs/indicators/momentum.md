# Momentum Oscillators & Cycle Transforms (`polars_anta.momentum`)

The `polars_anta.momentum` module provides **21 velocity oscillators, double-smoothed momentum filters, Gaussian Fisher transforms, and volatility-squeeze cycle detectors**.

---

## 1. Bounded Relative Strength & Stochastic Oscillators

### `rsi(close, period=14)` & `cmo(close, period=14)`
Let $\Delta C_t = C_t - C_{t-1}$, $U_t = \max(\Delta C_t, 0)$, and $D_t = \max(-\Delta C_t, 0)$.
- **Wilder's Relative Strength Index (`rsi`)**: Uses Wilder's recursive smoothing $\bar{U}_t = \text{RMA}_n(U_t)$ and $\bar{D}_t = \text{RMA}_n(D_t)$:

$$
\text{RS}_t = \frac{\bar{U}_t}{\bar{D}_t}, \qquad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t} = 100 \frac{\bar{U}_t}{\bar{U}_t + \bar{D}_t} \in [0, 100]
$$

- **Chande Momentum Oscillator (`cmo`)**: Uses unsmoothed rolling sums over $n$ bars and scales symmetrically into $[-100, +100]$:

$$
\text{CMO}_t = 100 \frac{\sum_{i=0}^{n-1} U_{t-i} - \sum_{i=0}^{n-1} D_{t-i}}{\sum_{i=0}^{n-1} U_{t-i} + \sum_{i=0}^{n-1} D_{t-i}} \in [-100, +100]
$$

---

### `stochastic(high, low, close, k_period=14, d_period=3, smooth_k=1)`, `stoch_rsi(...)`, & `williams_r(high, low, close, period=14)`
Let $H_t^{(n)} = \max_{0 \le i < n} H_{t-i}$ and $L_t^{(n)} = \min_{0 \le i < n} L_{t-i}$:

$$
\%K_t^{\text{raw}} = 100 \frac{C_t - L_t^{(n)}}{H_t^{(n)} - L_t^{(n)}} \in [0, 100], \qquad \%R_t = -100 \frac{H_t^{(n)} - C_t}{H_t^{(n)} - L_t^{(n)}} = \%K_t^{\text{raw}} - 100 \in [-100, 0]
$$

- `stochastic` returns `Struct(k, d)` where $k = \text{SMA}_{n_k}(\%K^{\text{raw}})$ and $d = \text{SMA}_{n_d}(k)$ (using smoothing parameters `smooth_k` and `d_period`).
- `stoch_rsi` applies the Stochastic normalization directly to the $\text{RSI}_t$ series, returning `Struct(k, d)`.

---

## 2. Convergence/Divergence & Double-Smoothed Velocity Filters

### `macd(close, fast_period=12, slow_period=26, signal_period=9)` & `ppo(close, fast_period=12, slow_period=26, signal_period=9)`
- **MACD (`Struct(macd, signal, histogram)`)**:

$$
\text{MACD}_t = \text{EMA}_{12}(C_t) - \text{EMA}_{26}(C_t), \quad \text{Signal}_t = \text{EMA}_9(\text{MACD}_t), \quad \text{Hist}_t = \text{MACD}_t - \text{Signal}_t
$$

- **Percentage Price Oscillator (`ppo`, `Struct(ppo, signal, histogram)`)**: Normalizes MACD by $\text{EMA}_{26}(C_t)$ so momentum magnitudes can be compared across assets at different price levels:

$$
\text{PPO}_t = \frac{\text{EMA}_{12}(C_t) - \text{EMA}_{26}(C_t)}{\text{EMA}_{26}(C_t)} \times 100
$$

---

### `tsi(close, long_period=25, short_period=13)`
**William Blau's True Strength Index:**
Applies a cascaded double-EMA filter to both raw price change $\Delta C_t$ and absolute price change $|\Delta C_t|$:

$$
\text{TSI}_t = 100 \cdot \frac{\text{EMA}_{13}\!\left(\text{EMA}_{25}(\Delta C_t)\right)}{\text{EMA}_{13}\!\left(\text{EMA}_{25}(|\Delta C_t|)\right)} \in [-100, +100]
$$

Because the denominator normalizes by double-smoothed volatility, `tsi` acts as a high-signal-to-noise continuous Sharpe-like momentum oscillator.

---

## 3. Statistical Gaussian Transforms & Volatility Squeeze

### `fisher_transform(high, low, period=10)`
**John Ehlers' Fisher Transform (`Struct(fisher, trigger)`)**:
Empirical financial price series have approximately uniform or leptokurtic distributions within a rolling channel, meaning extreme readings do not cluster cleanly at turning points. Ehlers first normalizes midpoint price $M_t = (H_t + L_t)/2$ into $(-1, +1)$ (smoothed via $X_t$ and clipped to $(-0.999, +0.999)$) and applies the inverse hyperbolic tangent (**Fisher $z$-transformation**):

$$
F_t = \frac{1}{2} \ln\!\left(\frac{1 + X_t}{1 - X_t}\right) + \frac{1}{2} F_{t-1} = \operatorname{artanh}(X_t) + \frac{1}{2} F_{t-1}
$$

This maps bounded price position into an approximately **Gaussian normal distribution** with sharp, unambiguous probability spikes at cycle reversals.

---

### `squeeze_momentum(high, low, close, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5)`
**John Carter's Volatility Squeeze & Linear Regression Momentum (`Struct(momentum, squeeze_on, squeeze_off, no_squeeze)`)**:
1. **Volatility Compression Detection**:
   - `squeeze_on` (`Boolean`): $\text{BB}_{\text{upper}, t} < \text{KC}_{\text{upper}, t} \;\land\; \text{BB}_{\text{lower}, t} > \text{KC}_{\text{lower}, t}$ (Bollinger Bands contract completely inside Keltner Channels during low-volatility consolidation).
   - `squeeze_off` (`Boolean`): Bollinger Bands expand outside Keltner Channels, signaling an explosive volatility breakout.
2. **Directional Momentum Histogram**:
   Computes the deviation of $C_t$ from the average of the Donchian midline and SMA, smoothed via **closed-form OLS linear regression forecast** (`linear_regression_forecast`).

---

### `stc(close, fast_period=23, slow_period=50, cycle_period=10, smooth_period=3)`
**Doug Schaff's Trend Cycle Oscillator:**
Cascades two successive Stochastic-EMA cycle normalizations over the MACD line ($\text{MACD}_{23, 50}$), detecting cyclical turns significantly faster than standard MACD while remaining bounded in $[0, 100]$.

---

## 4. Multi-Horizon & Microstructure Momentum Oscillators

| Function Signature | Mathematical Definition | Output Type |
| :--- | :--- | :--- |
| `cci(high, low, close, period=20)` | $\frac{\text{TP}_t - \text{SMA}_n(\text{TP}_t)}{0.015 \cdot \text{MAD}_n(\text{TP}_t)}$ (Donald Lambert, 1980) | `Float64` |
| `roc(close, period=10)` | $\frac{C_t - C_{t-n}}{C_{t-n}} \times 100$ | `Float64` |
| `awesome_oscillator(high, low, fast=5, slow=34)` | $\text{SMA}_5\!\left(\frac{H_t + L_t}{2}\right) - \text{SMA}_{34}\!\left(\frac{H_t + L_t}{2}\right)$ | `Float64` |
| `ultimate_oscillator(high, low, close, short=7, medium=14, long=28)` | $100 \times \frac{4 A_7 + 2 A_{14} + A_{28}}{4 + 2 + 1}$, where $A_k = \frac{\sum_{i=0}^{k-1} \text{BP}_{t-i}}{\sum_{i=0}^{k-1} \text{TR}_{t-i}}$ | `Float64` |
| `coppock(close, wma_period=10, fast_roc=11, slow_roc=14)` | $\text{WMA}_{10}\!\left(\text{ROC}_{14}(C_t) + \text{ROC}_{11}(C_t)\right)$ | `Float64` |
| `kst(close, ...)` | Weighted sum of 4 smoothed ROCs ($10, 15, 20, 30$) & SMA signal | `Struct(kst, signal)` |
| `rvi(open, high, low, close, period=10)` | Symmetrically weighted $(C - O)$ vs. $(H - L)$ ratio | `Struct(rvi, signal)` |
| `dpo(close, period=20)` | $C_t - \text{SMA}_n(C)_{t - (\lfloor n/2 \rfloor + 1)}$ | `Float64` |
| `bop(open, high, low, close)` | $\frac{C_t - O_t}{H_t - L_t} \in [-1, +1]$ | `Float64` |
| `cfo(close, period=14)` | $\frac{C_t - \hat{y}_{\text{OLS}, t}}{C_t} \times 100$ | `Float64` |
