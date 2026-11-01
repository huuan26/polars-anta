# Trend Filters, Adaptive Moving Averages & Structural Pivots (`polars_anta.trend`)

The `polars_anta.trend` module provides **28 low-lag moving averages, adaptive digital filters, directional regime classifiers, volatility trailing stops, and structural pivot levels**.

---

## 1. Linear, Low-Lag & Gaussian Moving Averages

In digital signal processing (DSP), a moving average is a low-pass filter that attenuates high-frequency market noise at the cost of **phase lag**. `polars-anta` provides a full hierarchy of linear and zero-lag filters:

| Function Signature | Mathematical Definition | Phase Lag & Filter Characteristics |
| :--- | :--- | :--- |
| `sma(close, period=10)` | $\text{SMA}_t(n) = \frac{1}{n}\sum_{i=0}^{n-1} C_{t-i}$ | Equal weights; group delay of $\frac{n - 1}{2}$ bars. |
| `ema(close, period=10)` | $\text{EMA}_t(n) = \alpha C_t + (1 - \alpha)\text{EMA}_{t-1}(n), \quad \alpha = \frac{2}{n + 1}$ | Infinite impulse response (IIR) filter; center-of-mass lag of $\frac{n - 1}{2}$ bars with higher recent weight. |
| `wma(close, period=10)` | $\text{WMA}_t(n) = \frac{2}{n(n + 1)} \sum_{i=0}^{n-1} (n - i) C_{t-i}$ | Triangular weight profile; reduces group delay to $\frac{n - 1}{3}$ bars. |
| `hma(close, period=10)` | $\text{HMA}_t(n) = \text{WMA}_{\lfloor\sqrt{n}\rfloor}\!\left(2\,\text{WMA}_{\lfloor n/2 \rfloor}(C_t) - \text{WMA}_n(C_t)\right)$ | Alan Hull's algebraic lag cancellation; eliminates first-order linear phase lag while maintaining smoothness. |
| `dema(close, period=10)` | $\text{DEMA}_t = 2 E_1 - E_2$, where $E_1 = \text{EMA}_n(C), E_2 = \text{EMA}_n(E_1)$ | Patrick Mulloy's second-order error-correcting exponential filter. |
| `tema(close, period=10)` | $\text{TEMA}_t = 3 E_1 - 3 E_2 + E_3$, where $E_3 = \text{EMA}_n(E_2)$ | Third-order error cancellation for steep trend tracking. |
| `zlema(close, period=10)` | $\text{ZLEMA}_t = \text{EMA}_n\!\left(C_t + (C_t - C_{t - \lfloor(n-1)/2\rfloor})\right)$ | Pre-compensates input price by adding the momentum over the filter's half-window lag. |
| `t3(close, period=5, vfactor=0.7)` | $\text{T3}_t = c_1 E_6 + c_2 E_5 + c_3 E_4 + c_4 E_3$ ($6$-stage cascaded EMA) | Tim Tillson's generalized DEMA cascade; combines low phase lag with minimal overshoot. |
| `alma(close, period=9, offset=0.85, sigma=6.0)` | $\text{ALMA}_t = \frac{\sum_{i=0}^{n-1} w_i C_{t-n+1+i}}{\sum_{i=0}^{n-1} w_i}, \quad w_i = \exp\!\left(-\frac{(i - m)^2}{2 s^2}\right)$ | Arnaud Legoux's shifted Gaussian bell-curve convolution ($m = \text{offset}(n-1), s = n/\sigma$). |

---

## 2. Non-Linear Adaptive Moving Averages

Unlike fixed-period filters, **adaptive moving averages** dynamically modulate their smoothing constant $\alpha_t$ based on market efficiency or momentum:

### `kama(close, period=10, fast_period=2, slow_period=30)`
**Perry Kaufman's Adaptive Moving Average:**
Computes the **Efficiency Ratio ($\text{ER}_t \in [0, 1]$)** — the ratio of net directional displacement to total path length:

$$
\text{ER}_t = \frac{|C_t - C_{t-n}|}{\sum_{i=0}^{n-1} |C_{t-i} - C_{t-i-1}|}, \qquad \text{SC}_t = \left[ \text{ER}_t (\alpha_{\text{fast}} - \alpha_{\text{slow}}) + \alpha_{\text{slow}} \right]^2
$$

$$
\text{KAMA}_t = \text{KAMA}_{t-1} + \text{SC}_t (C_t - \text{KAMA}_{t-1})
$$

When price moves in a straight line ($\text{ER}_t \to 1$), $\text{KAMA}_t$ tracks closely at `fast_period`; during sideways chop ($\text{ER}_t \to 0$), $\text{SC}_t^2 \to \alpha_{\text{slow}}^2$, freezing the average flat to prevent whipsaws.

---

### `vidya(close, period=14, hist_period=None)` & `mcginley_dynamic(close, period=14)`
- **Tushar Chande's VIDYA**: Modulates $\alpha_t = \alpha_0 \cdot |\text{CMO}_t| / 100$ using the absolute Chande Momentum Oscillator.
- **John McGinley's Dynamic Indicator**: Adjusts smoothing inversely to the fourth power of price velocity:

$$
\text{MD}_t = \text{MD}_{t-1} + \frac{C_t - \text{MD}_{t-1}}{n \cdot (C_t / \text{MD}_{t-1})^4}
$$

---

## 3. Directional Strength, Regime Classifiers & Trailing Stops

### `adx(high, low, close, period=14)`
**J. Welles Wilder's Directional Movement System (`pl.Struct: ("adx", "plus_di", "minus_di")`)**:

$$
\text{+DM}_t = \max(H_t - H_{t-1}, 0)\,\mathbb{I}_{\{H_t - H_{t-1} > L_{t-1} - L_t\}}, \qquad \text{-DM}_t = \max(L_{t-1} - L_t, 0)\,\mathbb{I}_{\{L_{t-1} - L_t > H_t - H_{t-1}\}}
$$

$$
\text{+DI}_t = 100 \frac{\text{RMA}_n(\text{+DM}_t)}{\text{ATR}_t(n)}, \qquad \text{-DI}_t = 100 \frac{\text{RMA}_n(\text{-DM}_t)}{\text{ATR}_t(n)}
$$

$$
\text{DX}_t = 100 \frac{|\text{+DI}_t - \text{-DI}_t|}{\text{+DI}_t + \text{-DI}_t}, \qquad \text{ADX}_t = \text{RMA}_n(\text{DX}_t)
$$

---

### `chop(high, low, close, period=14)`
**E.W. Dreiss's Choppiness Index (Fractal Consolidation Filter):**

$$
\text{CHOP}_t(n) = 100 \cdot \frac{\log_{10}\!\left(\frac{\sum_{i=0}^{n-1} \text{TR}_{t-i}}{\max_{0 \le i < n} H_{t-i} - \min_{0 \le i < n} L_{t-i}}\right)}{\log_{10}(n)} \in [0, 100]
$$

- $\text{CHOP}_t > 61.8$: Market is range-bound / consolidating (Fibonacci retracement threshold).
- $\text{CHOP}_t < 38.2$: Strong directional trend is underway.

---

### `supertrend(high, low, close, period=10, multiplier=3.0)` & `psar(high, low, close=None, af_start=0.02, af_step=0.02, af_max=0.20)`
- **`supertrend`**: Stateful ATR ratchet stop around $\text{HL2}_t = (H_t + L_t)/2 \pm m \cdot \text{ATR}_t(n)$. Returns `Struct(supertrend, direction)` where `direction` is `+1` (bullish) or `-1` (bearish).
- **`psar` (Parabolic SAR)**: Wilder's accelerating stop-and-reverse parabola $\text{SAR}_{t+1} = \text{SAR}_t + \text{AF}_t (\text{EP}_t - \text{SAR}_t)$ where the acceleration factor $\text{AF}_t$ increments by `af_step` up to `af_max` whenever a new Extreme Point ($\text{EP}_t$) is reached.

---

### `vortex`, `aroon`, `mass_index`, `ssl_channel`, `trend_intensity_index`, `center_of_gravity`
| Function Signature | Output Type / Struct Fields | Mathematical Mechanism |
| :--- | :--- | :--- |
| `vortex(high, low, close, period=14)` | `Struct(vi_plus, vi_minus)` | $\text{VI}^\pm_t = \frac{\sum_{i=0}^{n-1} \vert H_{t-i} - L_{t-i-1}\vert }{\sum_{i=0}^{n-1} \text{TR}_{t-i}}$ (Botes & Siepman, 2010) |
| `aroon(high, low, period=14)` | `Struct(aroon_up, aroon_down, oscillator)` | $\text{AroonUp}_t = \frac{n - \text{bars\_since\_high}_n}{n} \times 100$ (Tushar Chande, 1995) |
| `mass_index(high, low, ema_period=9, sum_period=25)` | `Float64` | $\sum_{i=0}^{24} \frac{\text{EMA}_9(H - L)_{t-i}}{\text{EMA}_9(\text{EMA}_9(H - L))_{t-i}}$ (Donald Dorsey reversal bulge $> 27$) |
| `ssl_channel(high, low, close, period=10)` | `Struct(ssl_down, ssl_up)` | State-flipped SMA(High) and SMA(Low) channel lines |
| `trend_intensity_index(close, sma_period=60, period=30)` | `Float64` | M.H. Pee's percentage of positive deviations above $\text{SMA}_{60}$ $[0, 100]$ |
| `center_of_gravity(close, period=10)` | `Float64` | John Ehlers' zero-lag cycle oscillator $-\frac{\sum_{i=0}^{n-1}(i + 1)C_{t-i}}{\sum_{i=0}^{n-1} C_{t-i}}$ |

---

## 4. Cloud Overlays, Heikin-Ashi & Structural Pivot Points

| Function Signature | Output `pl.Struct` Schema | Formulation |
| :--- | :--- | :--- |
| `ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52, displacement=26)` | `Struct(tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span)` | Donchian midpoints over $9, 26, 52$ bars with forward/backward cloud displacement |
| `heikin_ashi(open, high, low, close)` | `Struct(ha_open, ha_high, ha_low, ha_close)` | Loop-free synthetic candlesticks via exact $\text{EMA}(\alpha=0.5)$ reduction |
| `pivot_points(high, low, close)` | `Struct(pivot, r1, r2, r3, s1, s2, s3)` | Classical Floor Trader pivots around $P = (H_{t-1} + L_{t-1} + C_{t-1})/3$ |
| `fibonacci_pivots(high, low, close)` | `Struct(pivot, r1, r2, r3, s1, s2, s3)` | $P \pm \{0.382, 0.618, 1.000\} \times (H_{t-1} - L_{t-1})$ |
| `camarilla_pivots(high, low, close)` | `Struct(pivot, r1, r2, r3, r4, s1, s2, s3, s4)` | Nick Scott's intraday mean-reversion/breakout levels $C_{t-1} \pm \frac{1.1}{2, 4, 6, 12}(H_{t-1} - L_{t-1})$ |
| `woodie_pivots(high, low, close)` | `Struct(pivot, r1, r2, s1, s2)` | Close-weighted pivot $P = (H_{t-1} + L_{t-1} + 2 C_{t-1})/4$ |
