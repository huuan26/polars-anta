# Volatility Estimators, ATR Stops & Price Channels (`polars_anta.volatility`)

The `polars_anta.volatility` module provides **18 volatility estimators, range-based efficient variance models, and dynamic price channels** used for position sizing, volatility targeting, breakout detection, and stop-loss placement.

---

## 1. High-Efficiency Range-Based Volatility Estimators

In continuous-time finance ($d\ln P_t = \sigma\,dW_t$), the standard Close-to-Close volatility estimator uses only two snapshots per bar ($C_{t-1}$ and $C_t$), discarding all intraday path information (a stock that surges $+5\%$ intraday and closes unchanged has zero close-to-close return).

`polars-anta` implements five classical annualized volatility estimators with increasing **statistical asymptotic efficiency** $\text{Eff}(\hat{\sigma}^2) = \frac{\text{Var}(\hat{\sigma}_{\text{CC}}^2)}{\text{Var}(\hat{\sigma}^2)}$:

| Estimator | Function Signature | Mathematical Formula (Annualized by $\sqrt{A}$, default $A = 252$) | Statistical Efficiency |
| :--- | :--- | :--- | :---: |
| **Close-to-Close** | `historical_volatility(close, period=20, trading_periods=252)` | $\hat{\sigma}_{\text{CC}, t} = \sqrt{A} \cdot \sqrt{\frac{1}{n - 1}\sum_{i=0}^{n-1} \left(r_{t-i} - \bar{r}_t\right)^2}, \quad r_t = \ln\frac{C_t}{C_{t-1}}$ | **$1.0\times$** (Baseline) |
| **Parkinson (1980)** | `parkinson_volatility(high, low, period=20, trading_periods=252)` | $\hat{\sigma}_{\text{P}, t} = \sqrt{A} \cdot \sqrt{\frac{1}{4 n \ln 2} \sum_{i=0}^{n-1} \left(\ln\frac{H_{t-i}}{L_{t-i}}\right)^2}$ | **$4.9\times$** |
| **Garman-Klass (1980)** | `garman_klass_volatility(open, high, low, close, period=20, trading_periods=252)` | $\hat{\sigma}_{\text{GK}, t} = \sqrt{A} \cdot \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} \left[\frac{1}{2}\left(\ln\frac{H_{t-i}}{L_{t-i}}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_{t-i}}{O_{t-i}}\right)^2\right]}$ | **$7.4\times$** |
| **Rogers-Satchell (1991)** | `rogers_satchell_volatility(open, high, low, close, period=20, trading_periods=252)` | $\hat{\sigma}_{\text{RS}, t} = \sqrt{A} \cdot \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} \left[\ln\frac{H_{t-i}}{C_{t-i}}\ln\frac{H_{t-i}}{O_{t-i}} + \ln\frac{L_{t-i}}{C_{t-i}}\ln\frac{L_{t-i}}{O_{t-i}}\right]}$ | **$6.0\times$** (Drift-independent) |
| **Yang-Zhang (2000)** | `yang_zhang_volatility(open, high, low, close, period=20, trading_periods=252)` | $\hat{\sigma}_{\text{YZ}, t} = \sqrt{A} \cdot \sqrt{\hat{\sigma}_o^2 + k\,\hat{\sigma}_c^2 + (1 - k)\,\hat{\sigma}_{\text{RS}}^2}, \quad k = \frac{0.34}{1.34 + \frac{n + 1}{n - 1}}$ | **$14.0\times$** (Drift & overnight jump robust) |

```python
import polars as pl
import polars_anta as pa

df = df.with_columns(
    pa.rogers_satchell_volatility("open", "high", "low", "close", period=20).alias("rs_vol_20"),
    pa.yang_zhang_volatility("open", "high", "low", "close", period=20).alias("yz_vol_20"),
)
```

!!! info "Quantitative Advantage of Garman-Klass"
    Because Garman-Klass achieves **$7.4\times$ higher statistical efficiency** than Close-to-Close volatility, a **10-bar Garman-Klass estimator** has lower estimation variance than a **70-bar Close-to-Close estimator**, enabling much faster regime adaptation in Volatility Targeting (`vol_target`) and Triple-Barrier scaling without increasing estimation noise.

---

## 2. True Range, Wilder's ATR & Trailing Volatility Stops

### `true_range(high, low, close)`, `atr(high, low, close, period=14)`, & `natr(high, low, close, period=14)`
**Mathematical Formulations (J. Welles Wilder Jr., 1978):**

$$
\text{TR}_t = \max\left(H_t - L_t,\, |H_t - C_{t-1}|,\, |L_t - C_{t-1}|\right)
$$

$$
\text{ATR}_t(n) = \text{RMA}_n(\text{TR}_t) = \frac{(n - 1)\,\text{ATR}_{t-1} + \text{TR}_t}{n}, \qquad \text{NATR}_t(n) = \frac{\text{ATR}_t(n)}{C_t} \times 100
$$

- **Quantitative Interpretation**: `true_range` accounts for overnight or inter-bar gaps ($|H_t - C_{t-1}|$ and $|L_t - C_{t-1}|$). While `atr` is denominated in raw price points (ideal for sizing point stops on VN30F1M), `natr` normalizes by price level so volatility can be ranked cross-sectionally across assets trading at different nominal prices.
- **Tech / Polars Execution**: Formulated as an exact `ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)` kernel in Rust.

---

### `chandelier_exit(high, low, close, period=22, multiplier=3.0)`
**Chuck LeBeau's ATR Trailing Stop (`pl.Struct: ("long_stop", "short_stop")`)**:

$$
\text{LongStop}_t = \max_{0 \le i < n} H_{t-i} - m \cdot \text{ATR}_t(n), \qquad \text{ShortStop}_t = \min_{0 \le i < n} L_{t-i} + m \cdot \text{ATR}_t(n)
$$

---

### `ulcer_index(close, period=14)`
**Peter Martin's Downside Drawdown Volatility (1987):**
Unlike standard deviation (`stddev`), which penalizes upside rallies and downside crashes equally, `ulcer_index` measures the root-mean-square depth and duration of rolling percentage drawdowns:

$$
R_t = \frac{C_t - \max_{0 \le i < n} C_{t-i}}{\max_{0 \le i < n} C_{t-i}} \times 100 \le 0, \qquad \text{UI}_t(n) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} R_{t-i}^2}
$$

---

### `chaikin_volatility(high, low, ema_period=10, roc_period=10)`
Measures the rate of expansion or contraction of the smoothed High-Low trading range:

$$
\text{EMA}_{\text{HL}, t} = \text{EMA}_{n_1}(H_t - L_t), \qquad \text{ChaikinVol}_t = \frac{\text{EMA}_{\text{HL}, t} - \text{EMA}_{\text{HL}, t - n_2}}{\text{EMA}_{\text{HL}, t - n_2}} \times 100
$$

---

## 3. Dynamic Volatility & Breakout Channels

| Function Signature | Midline ($M_t$) & Band Formulas ($U_t, L_t$) | Output Type / Struct Fields |
| :--- | :--- | :--- |
| `bollinger_bands(close, period=20, std_dev=2.0)` | $M_t = \text{SMA}_n(C_t)$, $U_t / L_t = M_t \pm k\hat{\sigma}_t$<br/>$\text{Bandwidth}_t = \frac{U_t - L_t}{M_t} \times 100$, $\%B_t = \frac{C_t - L_t}{U_t - L_t}$ | `Struct(upper, middle, lower, bandwidth, percent_b)` |
| `bollinger_delta(close, period=20, std_dev=2.0)` | $\Delta_{\text{BB}, t} = U_t - L_t = 2 k \hat{\sigma}_t$ | `Float64` |
| `keltner_channels(high, low, close, period=20, atr_period=10, multiplier=2.0)` | $M_t = \text{EMA}_{n}(\text{TP}_t)$, $U_t / L_t = M_t \pm m \cdot \text{ATR}_{n_{\text{atr}}}(t)$ | `Struct(upper, middle, lower)` |
| `keltner_width(high, low, close, period=20, atr_period=10, multiplier=2.0)` | $\text{KCWidth}_t = \frac{U_t - L_t}{M_t} = \frac{2 m \cdot \text{ATR}_t}{M_t}$ | `Float64` |
| `donchian_channels(high, low, period=20)` | $U_t = \max_{0 \le i < n} H_{t-i}$, $L_t = \min_{0 \le i < n} L_{t-i}$, $M_t = \frac{U_t + L_t}{2}$ | `Struct(upper, middle, lower)` |
| `donchian_width(high, low, period=20)` | $\text{DCWidth}_t = \frac{U_t - L_t}{M_t} \times 100$ | `Float64` |
| `starc_bands(high, low, close, sma_period=6, atr_period=15, multiplier=2.0)` | $M_t = \text{SMA}_{n_1}(C_t)$, $U_t / L_t = M_t \pm m \cdot \text{ATR}_{n_2}(t)$ | `Struct(upper, middle, lower)` |
| `envelopes(close, period=20, pct=0.025)` | $M_t = \text{SMA}_n(C_t)$, $U_t / L_t = M_t (1 \pm \text{pct})$ | `Struct(upper, middle, lower)` |
