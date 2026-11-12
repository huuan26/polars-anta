# Volume, Session VWAP & Money Flow (`polars_anta.volume`)

The `polars_anta.volume` module provides **16 volume-weighted price benchmarks, execution bands, and money-flow oscillators** designed to track accumulation/distribution and intraday fair value.

---

## 1. VWAP Benchmarks & Session Anchoring

### `vwap(high, low, close, volume, period=None)`
**Mathematical Formulation:**
Let Typical Price be $\text{TP}_t = \frac{H_t + L_t + C_t}{3}$.
- **Cumulative Mode (`period=None`, default)**:

$$
\text{VWAP}_t = \frac{\sum_{\tau=1}^t \text{TP}_\tau \cdot V_\tau}{\sum_{\tau=1}^t V_\tau}
$$

- **Rolling Window Mode (`period=N`)**:

$$
\text{VWAP}_t(N) = \frac{\sum_{k=0}^{N-1} \text{TP}_{t-k} \cdot V_{t-k}}{\sum_{k=0}^{N-1} V_{t-k}}
$$

---

### `vwap_bands(high, low, close, volume, period=None, mult1=1.0, mult2=2.0, mult3=3.0)`
**Mathematical Formulation (Volume-Weighted Standard Deviation Bands):**
Using the identity $\text{Var}_V(\text{TP}) = \mathbb{E}_V[\text{TP}^2] - \left(\mathbb{E}_V[\text{TP}]\right)^2$:

$$
\sigma_{\text{VWAP}, t} = \sqrt{\max\left(0,\, \frac{\sum \text{TP}_\tau^2 \cdot V_\tau}{\sum V_\tau} - \text{VWAP}_t^2\right)}
$$

$$
\text{Upper}_{k, t} = \text{VWAP}_t + m_k \cdot \sigma_{\text{VWAP}, t}, \qquad \text{Lower}_{k, t} = \text{VWAP}_t - m_k \cdot \sigma_{\text{VWAP}, t} \quad (k \in \{1, 2, 3\})
$$

Returns a `pl.Struct` with fields `("vwap", "upper_1", "lower_1", "upper_2", "lower_2", "upper_3", "lower_3")`.

---

### `session_vwap(high, low, close, volume, session_by=None, time_col="timestamp", mult1=1.0, mult2=2.0)`
**Session-Anchored VWAP with Deviation Z-Score:**
In intraday equity and index futures trading (such as **VN30F1M**), execution algorithms benchmark fills against the **Session VWAP** that resets at the start of every trading day (`pl.col(time_col).dt.date()` when `session_by=None`, or custom session identifiers such as morning/afternoon sessions).

$$
\text{VWAP}_{t \in S} = \frac{\sum_{\tau \in S, \tau \le t} \text{TP}_\tau V_\tau}{\sum_{\tau \in S, \tau \le t} V_\tau}, \qquad Z_{\text{VWAP}, t} = \frac{\text{TP}_t - \text{VWAP}_{t \in S}}{\sigma_{\text{VWAP}, t \in S}}
$$

- **Output `pl.Struct` Schema (`prefix="session_"` when `unnest=True`)**:
  `("vwap", "upper_1", "lower_1", "upper_2", "lower_2", "zscore")`.
- **Tech / Polars Execution**: Uses native Polars `.cum_sum().over(session_key)` window partitioning, resetting cumulative sums across thousands of trading sessions in parallel with zero loops.

---

### `vwma(close, volume, period=20)`
**Volume-Weighted Moving Average:**

$$
\text{VWMA}_t(n) = \frac{\sum_{i=0}^{n-1} C_{t-i} \cdot V_{t-i}}{\sum_{i=0}^{n-1} V_{t-i}}
$$

Comparing $\text{VWMA}_t(n) - \text{SMA}_t(n)$ reveals whether volume is concentrated on high-price bars (bullish volume participation) or low-price bars.

---

## 2. Accumulation, Distribution & Money Flow Oscillators

### `obv(close, volume)` & `pvt(close, volume)`
- **Joe Granville's On-Balance Volume (`obv`)**:

$$
\text{OBV}_t = \sum_{\tau=1}^t \text{sign}(C_\tau - C_{\tau-1}) \cdot V_\tau
$$

- **Price Volume Trend (`pvt`)**: Scales signed volume by proportional price return rather than a binary step function:

$$
\text{PVT}_t = \sum_{\tau=1}^t \left(\frac{C_\tau - C_{\tau-1}}{C_{\tau-1}}\right) V_\tau
$$

---

### `adl(high, low, close, volume)` & `cmf(high, low, close, volume, period=20)`
Both indicators rely on Marc Chaikin's **Close Location Value (CLV)**, which locates the closing price within the bar's High-Low range $[-1, +1]$:

$$
\text{CLV}_t = \frac{(C_t - L_t) - (H_t - C_t)}{H_t - L_t} = \frac{2C_t - L_t - H_t}{H_t - L_t} \in [-1, +1]
$$

- **Accumulation / Distribution Line (`adl`)**: $\text{ADL}_t = \sum_{\tau=1}^t \text{CLV}_\tau \cdot V_\tau$.
- **Chaikin Money Flow (`cmf`)**: $n$-period ratio of Money Flow Volume to total volume:

$$
\text{CMF}_t(n) = \frac{\sum_{i=0}^{n-1} \text{CLV}_{t-i} \cdot V_{t-i}}{\sum_{i=0}^{n-1} V_{t-i}} \in [-1, +1]
$$

---

### `mfi(high, low, close, volume, period=14)`
**Quong & Soudack's Money Flow Index (Volume-Weighted RSI):**
Let Raw Money Flow be $\text{RMF}_t = \text{TP}_t \cdot V_t$. Partitioning $\text{RMF}_t$ into positive flow ($\text{TP}_t > \text{TP}_{t-1}$) and negative flow ($\text{TP}_t < \text{TP}_{t-1}$) over window $n$:

$$
\text{MFR}_t = \frac{\sum_{i=0}^{n-1} \text{RMF}_{t-i}^+}{\sum_{i=0}^{n-1} \text{RMF}_{t-i}^-}, \qquad \text{MFI}_t = 100 - \frac{100}{1 + \text{MFR}_t} \in [0, 100]
$$

---

### `klinger(high, low, close, volume, fast_period=34, slow_period=55, signal_period=13)`
**Stephen Klinger's Volume Force Oscillator:**
Combines trend direction $T_t = \text{sign}(\text{TP}_t - \text{TP}_{t-1})$, daily range $dm_t = H_t - L_t$, and cumulative trend range $cm_t$ into Volume Force $\text{VF}_t = V_t \cdot \left|2\frac{dm_t}{cm_t} - 1\right| \cdot T_t \cdot 100$, then computes:

$$
\text{KVO}_t = \text{EMA}_{34}(\text{VF}_t) - \text{EMA}_{55}(\text{VF}_t), \qquad \text{Signal}_t = \text{EMA}_{13}(\text{KVO}_t)
$$

Returns `pl.Struct` with fields `("klinger", "signal")`.

---

### `eom`, `force_index`, `nvi`, `pvi`, `vpci`, `vr`
| Function Signature | Mathematical Definition | Quantitative Interpretation |
| :--- | :--- | :--- |
| `eom(high, low, volume, period=14, scale=10000.0)` | $\text{SMA}_n\left(\Delta\left(\frac{H_t + L_t}{2}\right) \cdot \frac{H_t - L_t}{V_t / \text{scale}}\right)$ | Richard Arms' Ease of Movement: high positive values mean price advances on light volume (low overhead resistance). |
| `force_index(close, volume, period=13)` | $\text{EMA}_n\left((C_t - C_{t-1}) \cdot V_t\right)$ | Alexander Elder's Force Index combining direction, extent of price change, and volume. |
| `nvi(close, volume)` / `pvi(close, volume)` | Updates cumulative return index only on bars where $V_t < V_{t-1}$ (`nvi`) or $V_t > V_{t-1}$ (`pvi`) | Norman Fosback's Smart-Money (`nvi`, active on quiet volume days) vs. Crowd-Activity (`pvi`) indices. |
| `vpci(close, volume, short_period=5, long_period=20)` | $\text{VPC}_t \times \text{VPR}_t \times \text{VM}_t$ | Buff Dormeier's Volume Price Confirmation Indicator measuring divergence between VWMA and SMA. |
| `vr(close, volume, period=26)` | $\frac{\text{AV}_n + \frac{1}{2}\text{CV}_n}{\text{BV}_n + \frac{1}{2}\text{CV}_n} \times 100$ | Up-Volume ($\text{AV}_n$) vs. Down-Volume ($\text{BV}_n$) Ratio. |
