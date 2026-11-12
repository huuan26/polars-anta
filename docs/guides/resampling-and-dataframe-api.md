# Time Bars, Information-Driven Bars & Continuous Futures Roll (`polars_anta.resample`)

In quantitative workflows, raw market feeds (tick data, Level-1 order book updates, or 1-minute candles) must be validated, stitched across futures contract expirations, and aggregated into chronological or activity-driven bars before feature extraction.

`polars-anta` provides six native Polars data engineering primitives in `polars_anta.resample`:

1. **`resample_ohlcv`**: Chronological multi-timeframe aggregation (`"5m"`, `"15m"`, `"1h"`, `"1d"`) via Polars' `group_by_dynamic`.
2. **`resample_volume_bars`**: Activity-synchronized Volume Bars emitted every $\Theta_V$ traded contracts/shares.
3. **`resample_dollar_bars`**: Notional-synchronized Dollar Bars emitted every $\Theta_D$ currency value traded.
4. **`resample_imbalance_bars`**: Order-flow-synchronized Tick / Volume Imbalance Bars (AFML) emitted whenever signed directional flow exceeds $\Theta_{\text{imb}}$.
5. **`stitch_futures_roll`**: Continuous futures contract splicing via additive (`"panama"`), multiplicative (`"ratio"`), or unadjusted (`"none"`) roll adjustment.
6. **`validate_ohlcv`**: High-speed structural integrity verification (`High >= Low`, non-negative prices, monotonic timestamps).

---

## 1. Information-Driven Bars (AFML Microstructure Sampling)

### `resample_volume_bars(df, threshold, ...)` & `resample_dollar_bars(df, threshold, ...)`
**Mathematical Formulation:**
Let $v_t = V_t$ for Volume Bars, or $v_t = C_t \cdot V_t$ for Dollar Bars. The cumulative activity bucket index is:

$$
b_t = \left\lfloor \frac{\left(\sum_{\tau=1}^t v_\tau\right)(1 - \epsilon)}{\Theta} \right\rfloor
$$

Grouping by `bar_id = b_t` aggregates standard OHLCV columns along with `tick_count` (number of raw observations merged into the bar) and `vwap` ($\sum C_\tau V_\tau / \sum V_\tau$).

```python
import polars_anta as pa

# Aggregate 1-minute VN30F1M bars into 5,000-contract Volume Bars
vol_bars = df.anta.resample_volume_bars(threshold=5_000, group_by="symbol")

# Aggregate into 500 Billion VND Notional Dollar Bars
dollar_bars = df.anta.resample_dollar_bars(threshold=5e11, group_by="symbol")
```

---

### `resample_imbalance_bars(df, threshold, mode="volume", ...)`
**Mathematical Formulation (López de Prado, 2018):**
Using the Lee-Ready tick rule $s_t = \text{sign}(C_t - C_{t-1})$ (forward-filled when $\Delta C_t = 0$), let signed flow be $\theta_t = s_t V_t$ (`mode="volume"`) or $\theta_t = s_t$ (`mode="tick"`). A new bar is triggered at bar $t$ whenever cumulative imbalance since the start of the current bar $t_0$ crosses `threshold`:

$$
\left| \sum_{\tau = t_0}^t \theta_\tau \right| \ge \Theta_{\text{imb}}
$$

Each output bar includes `bar_id`, `open`, `high`, `low`, `close`, `volume`, `imbalance` ($\sum_{\tau=t_0}^t \theta_\tau$), `tick_count`, and `vwap`.

---

## 2. Continuous Futures Contract Roll Stitching

### `stitch_futures_roll(df, contract_col="contract", method="panama", backward=True, ...)`
When constructing multi-year historical series for index futures (e.g., splicing `VN30F2601` $\to$ `VN30F2602` $\to$ `VN30F2603`), contract switches create artificial price gaps $\delta_k = O_{t_k, \text{new}} - C_{t_k - 1, \text{old}}$ at roll timestamps $t_k$.

| Method | Mathematical Adjustment (`backward=True`, anchoring latest contract to actual market price) | Preserved Invariant |
| :--- | :--- | :--- |
| **`method="panama"`** | $P_{t}^{\text{adj}} = P_t + \sum_{t_k > t} \left(O_{t_k, \text{new}} - C_{t_k - 1, \text{old}}\right)$ | Preserves exact **point differences ($\Delta P_t$)** and point-based ATR / stop distances. |
| **`method="ratio"`** | $P_{t}^{\text{adj}} = P_t \times \prod_{t_k > t} \left(\frac{O_{t_k, \text{new}}}{C_{t_k - 1, \text{old}}}\right)$ | Preserves exact **percentage returns ($r_t$)** and keeps historical prices strictly positive. |
| **`method="none"`** | $P_t^{\text{adj}} = P_t$ | Appends `roll_flag` and `roll_adjustment` columns for inspection without modifying OHLC. |

```python
continuous_df = df.anta.stitch_futures_roll(
    contract_col="contract",
    method="panama",
    backward=True,
    group_by="underlying",
)
```

---

## 3. Chronological Multi-Timeframe Resampling (`resample_ohlcv`)

```python
# Resample 1-minute multi-asset universe to 15-minute bars in parallel
df_15m = df.anta.resample(
    every="15m",
    group_by="symbol",
    time_col="timestamp",
    additional_aggs=[pl.col("open_interest").last().alias("open_interest")],
)
```

---

## 4. Structural Integrity Validation (`validate_ohlcv`)

Before training ML models or running backtests, `validate_ohlcv` checks four structural invariants in a single vectorized pass:
1. **Schema & Types**: Price/volume columns are numeric; timestamp is `pl.Date` or `pl.Datetime`.
2. **OHLC Envelope**: $H_t \ge \max(O_t, C_t, L_t)$ and $L_t \le \min(O_t, C_t, H_t)$.
3. **Positivity**: Prices are strictly positive ($> 0$) and volume is non-negative ($\ge 0$).
4. **Temporal Ordering**: No null timestamps and strictly non-decreasing timestamps within each `group_by` partition.
