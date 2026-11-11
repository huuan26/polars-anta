# Quantitative Theory & End-to-End Workflows

This guide presents the **quantitative finance theory**, **mathematical formulations**, and **production Polars implementations** across the four core pillars of alpha research in `polars-anta`:

1. **Pillar 1 — Derivatives & Intraday Market Microstructure**: Information-driven bar sampling, adverse selection & order flow toxicity (VPIN, OFI), session-anchored VWAP, index futures cost-of-carry basis, open interest regime classification, and continuous contract roll stitching.
2. **Pillar 2 — Quantitative Machine Learning (AFML Primitives)**: Memory-preserving stationarity via Fixed-Window Fractional Differentiation (FFD), dynamic Triple-Barrier labeling, Meta-Labeling, and multi-horizon OLS Trend-Scanning.
3. **Pillar 3 — Factor Validation & Risk Orthogonalization**: Cross-sectional preprocessing, Frisch-Waugh-Lovell multi-factor orthogonalization, Newey-West HAC $t$-statistics, Information Coefficient (IC) decay & Alpha Half-Life, and multiple-testing adjustments (Probabilistic & Deflated Sharpe Ratio).
4. **Pillar 4 — Execution-Realistic Backtesting & Excursion Analytics**: Next-Open / TWAP / VWAP execution models, intraday EOD session flattening, dynamic Volatility Targeting, no-trade turnover buffers, and Round-Trip Trade Ledgers with Maximum Adverse/Favorable Excursion (MAE/MFE).

---

## Pillar 1: Derivatives & Intraday Market Microstructure

### 1.1 Why Chronological Time Bars Fail in Intraday Markets

Standard chronological bars (e.g., 1-minute or 5-minute candles) sample the market at fixed wall-clock intervals regardless of trading activity. During quiet midday sessions, time bars oversample microstructure noise; during market open, macro announcements, or ATC auctions (such as on the **VN30F1M** index futures market), time bars undersample rapid price discovery. Consequently, time-sampled returns exhibit severe **serial correlation**, **heteroskedasticity**, and **leptokurtic fat tails** (Clark, 1973; López de Prado, 2018).

**Information-Driven Bars** synchronize sampling with the arrival of market activity:

- **Volume Bars (`resample_volume_bars`)**: Emit a new bar whenever cumulative traded contracts/shares reach threshold $\Theta_V$:

$$
b_t^{(V)} = \left\lfloor \frac{\sum_{\tau=1}^t V_\tau}{\Theta_V} \right\rfloor
$$

- **Dollar / Notional Bars (`resample_dollar_bars`)**: Emit a new bar whenever cumulative notional value $\sum_{\tau=1}^t P_\tau V_\tau$ crosses $\Theta_D$. Dollar bars automatically adapt to large secular price changes or inflation.
- **Volume / Tick Imbalance Bars (`resample_imbalance_bars`)**: Let $s_t = \text{sign}(\Delta P_t) \in \{-1, +1\}$ (propagated forward via the Lee-Ready tick rule when $\Delta P_t = 0$). The signed flow is $\theta_t = s_t V_t$ (or $s_t$ for tick imbalance). A new bar is emitted whenever cumulative directional imbalance since the previous bar reset exceeds $\Theta_{\text{imb}}$:

$$
\left| \sum_{\tau = t_0}^t \theta_\tau \right| \ge \Theta_{\text{imb}}
$$

!!! tip "Quantitative Property: Restoration of Return Normality"
    Empirically across equity index futures (E-mini S&P 500, VN30F1M), sampling in Volume or Dollar time subordinate to the trading clock (Ané & Geman, 2000) restores return distributions much closer to i.i.d. Gaussianity, significantly improving the stability of GARCH, Kalman filter, and gradient-boosted tree models.

```python
import polars as pl
import polars_anta as pa

# Convert 1-minute or tick data into 5,000-contract Volume Bars & Imbalance Bars
vol_bars = df.anta.resample_volume_bars(threshold=5_000, group_by="symbol")
imb_bars = df.anta.resample_imbalance_bars(threshold=1_500, mode="volume", group_by="symbol")
```

---

### 1.2 Order Flow Toxicity & Adverse Selection: VPIN & OFI

Market makers and algorithmic execution desks face **adverse selection risk** when trading against informed institutional flow.

#### Bulk Volume Classification VPIN (`vpin`)
Following Easley, López de Prado & O'Hara (2012), when tick-by-tick trade aggressor flags are unavailable or noisy, bar volume $V_t$ is partitioned into buyer-initiated ($V_t^B$) and seller-initiated ($V_t^S$) components using the standardized price change $Z_t = \frac{\Delta P_t}{\sigma_{\Delta P}}$:

$$
V_t^B = V_t \cdot \Phi\left(\frac{\Delta P_t}{\sigma_{\Delta P}}\right), \qquad V_t^S = V_t - V_t^B
$$

In `polars-anta`, $\Phi(z)$ is approximated via the smooth sigmoid $\left(1 + \exp(-1.702\,z)\right)^{-1}$ (which matches the standard normal CDF within $0.0095$ uniformly across $\mathbb{R}$). The **Volume-Synchronized Probability of Informed Trading** over window $N$ is:

$$
\text{VPIN}_t = \frac{\sum_{k=0}^{N-1} \left| V_{t-k}^B - V_{t-k}^S \right|}{\sum_{k=0}^{N-1} V_{t-k}} \in [0, 1]
$$

#### Order Flow Imbalance (`ofi`)
Following Cont, Kukanov & Stoikov (2014), short-horizon price changes are linearly driven by net supply/demand shifts at the best bid ($P_t^B, Q_t^B$) and best ask ($P_t^A, Q_t^A$):

$$
e_t^B = Q_t^B \mathbb{I}_{\{P_t^B \ge P_{t-1}^B\}} - Q_{t-1}^B \mathbb{I}_{\{P_t^B \le P_{t-1}^B\}}
$$

$$
e_t^A = Q_t^A \mathbb{I}_{\{P_t^A \le P_{t-1}^A\}} - Q_{t-1}^A \mathbb{I}_{\{P_t^A \ge P_{t-1}^A\}}
$$

$$
\text{OFI}_t(N) = \sum_{k=0}^{N-1} \left( e_{t-k}^B - e_{t-k}^A \right)
$$

---

### 1.3 Index Futures Cost-of-Carry, Open Interest Regimes & Continuous Roll Stitching

#### Futures Basis (`futures_basis`)
By no-arbitrage pricing, an index futures contract $F_t$ maturing at $T$ is linked to the underlying spot index $S_t$ (e.g., VN30F1M vs. VN30 cash index) by the cost-of-carry relation $F_t = S_t e^{(r - q)\tau}$. `futures_basis` computes:

$$
\text{Basis}_t = F_t - S_t, \qquad \text{Basis\%}_t = \frac{F_t - S_t}{S_t}, \qquad \text{Carry}_{\text{ann}, t} = \frac{F_t - S_t}{S_t} \cdot \frac{365}{\max(\text{DTE}_t, 1)}
$$

along with the rolling standardized anomaly score $\text{ZScore}(\text{Basis}_t, N)$ for index arbitrage and cash-and-carry strategies.

#### Open Interest 4-Phase Regime Classification (`oi_regime`)
Combining price change $\Delta P_t = P_t - P_{t-k}$ and Open Interest change $\Delta \text{OI}_t = \text{OI}_t - \text{OI}_{t-k}$ classifies futures market participation into four regimes:

| Price $\Delta P_t$ | Open Interest $\Delta \text{OI}_t$ | Regime Code | Interpretation |
| :---: | :---: | :---: | :--- |
| $> 0$ | $> 0$ | **`+2.0`** | **Long Buildup**: Fresh long positions opening; strong bullish trend conviction. |
| $> 0$ | $< 0$ | **`+1.0`** | **Short Covering**: Rally driven by shorts buying back to close; weaker follow-through. |
| $< 0$ | $> 0$ | **`-2.0`** | **Short Buildup**: Fresh short positions opening; strong bearish conviction. |
| $< 0$ | $< 0$ | **`-1.0`** | **Long Liquidation**: Decline driven by existing longs forced to exit/deleverage. |

#### Continuous Futures Roll Stitching (`stitch_futures_roll`)
When a front-month futures contract expires and rolls into the next maturity (`contract` changes from $C_k$ to $C_{k+1}$), raw price series exhibit artificial price jumps $\Delta_{\text{roll}} = P_{t, \text{new}}^{\text{open}} - P_{t-1, \text{old}}^{\text{close}}$ that corrupt moving averages, RSI, and backtest returns. `stitch_futures_roll` supports three adjustment modes:
1. **`method="panama"` (Additive Backward/Forward Adjustment)**: Preserves exact point/tick differences $\Delta P_t$ and point PnL by shifting historical contracts by the cumulative roll gap.
2. **`method="ratio"` (Multiplicative Adjustment)**: Preserves exact percentage returns $r_t = P_t / P_{t-1} - 1$ across contract rolls by scaling prices by the cumulative roll ratio $\prod (P_{\text{new}} / P_{\text{old}})$.

---

## Pillar 2: Quantitative Machine Learning (AFML Primitives)

### 2.1 The Stationarity vs. Memory Dilemma: Fractional Differentiation (`frac_diff`)

Raw log-prices $p_t = \ln P_t$ are non-stationary ($I(1)$), violating the i.i.d. assumptions of supervised machine learning estimators. However, taking first differences $\Delta p_t = (1 - B) p_t$ ($d = 1$) completely wipes out long-term trend and support/resistance memory.

Following López de Prado (2018, Chapter 5), the fractional differencing operator $(1 - B)^d$ for real order $d \in (0, 1)$ expands via the generalized binomial series:

$$
(1 - B)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-B)^k = \sum_{k=0}^{\infty} \omega_k B^k
$$

where the weights satisfy the exact recurrence:

$$
\omega_0 = 1, \qquad \omega_k = -\omega_{k-1} \frac{d - k + 1}{k} \quad (k \ge 1)
$$

In **Fixed-Window Fractional Differentiation (FFD)**, the filter is truncated at a fixed lookback window $W$:

$$
\tilde{X}_t^{(d, W)} = \sum_{k=0}^{W-1} \omega_k X_{t-k}
$$

Typically, $d \in [0.3, 0.5]$ achieves statistical stationarity (passing the Augmented Dickey-Fuller test) while retaining $> 90\%$ correlation with the original price level.

---

### 2.2 Dynamic Triple-Barrier Labeling & Meta-Labeling (`triple_barrier_labels`)

Fixed-horizon return labels $y_t = \text{sign}(P_{t+H} / P_t - 1)$ are flawed because they ignore the **path taken by price** between $t$ and $t+H$ (a position would have been stopped out intra-horizon) and use a static threshold regardless of volatility regimes.

[`triple_barrier_labels`](../indicators/evaluation-and-backtest.md) constructs three dynamic barriers for each bar $t$ based on local volatility $\sigma_t$:
1. **Upper Profit-Taking (PT) Barrier**: $U_t = +\max\left(m_{\text{PT}} \cdot \sigma_t,\, r_{\min}\right)$
2. **Lower Stop-Loss (SL) Barrier**: $L_t = -\max\left(m_{\text{SL}} \cdot \sigma_t,\, r_{\min}\right)$
3. **Vertical Expiration Barrier**: $t + H$ bars.

Let $\tau_{\text{PT}} = \min\{k \in \{1, \dots, H\} : r_{t, t+k}^{\text{dir}} \ge U_t\}$ and $\tau_{\text{SL}} = \min\{k \in \{1, \dots, H\} : r_{t, t+k}^{\text{dir}} \le L_t\}$. The first touched barrier determines:
- `label`: $+1.0$ (upper barrier hit first), $-1.0$ (lower barrier hit first), or $0.0$ (vertical horizon expired first).
- `touch_bars`: First-passage time $\tau = \min(\tau_{\text{PT}}, \tau_{\text{SL}}, H) \in \{1, \dots, H\}$.
- `realized_return`: Raw return $(P_{t + \tau} - P_t) / P_t$ at the barrier touch.
- `meta_label`: Binary $\{0.0, 1.0\}$ indicating whether the primary directional signal `side` ($\in \{-1, +1\}$) achieved its profit-taking target.

---

### 2.3 Multi-Horizon Trend-Scanning Labels (`trend_scanning_labels`)

Proposed by López de Prado (2020, *Machine Learning for Asset Managers*), **Trend-Scanning** eliminates arbitrary fixed lookahead horizons by fitting forward OLS regressions across a grid of candidate horizons $L \in [L_{\min}, L_{\max}]$:

$$
P_{t+i} = \beta_0(L) + \beta_1(L) \cdot i + \epsilon_i, \qquad i = 0, 1, \dots, L - 1
$$

For each candidate window $L$, the slope $t$-statistic is:

$$
t_{\hat{\beta}_1}(L) = \frac{\hat{\beta}_1(L)}{\text{SE}(\hat{\beta}_1(L))}, \qquad \text{SE}(\hat{\beta}_1(L)) = \sqrt{\frac{\text{SSR}(L) / (L - 2)}{S_{xx}(L)}}
$$

`trend_scanning_labels` selects the optimal horizon $L^*_t = \arg\max_{L} \left|t_{\hat{\beta}_1}(L)\right|$ using closed-form Polars expressions and returns `Struct(label, t_value, optimal_window, ret)`.

---

## Pillar 3: Factor Validation & Risk Orthogonalization

### 3.1 Newey-West HAC $t$-Statistics & Multi-Horizon IC Decay (`ic_summary`, `ic_decay`)

When evaluating an alpha factor against $h$-period forward returns ($h > 1$), overlapping return windows induce Moving Average $\text{MA}(h - 1)$ serial correlation in the cross-sectional Information Coefficient series $\{\text{IC}_t\}_{t=1}^T$. Standard i.i.d. $t$-statistics $\frac{\overline{\text{IC}}}{\hat{\sigma}_{\text{IC}} / \sqrt{T}}$ severely **overstate statistical significance**.

[`ic_summary`](../indicators/evaluation-and-backtest.md) computes the **Newey-West (1987) Heteroskedasticity and Autocorrelation Consistent (HAC)** long-run variance estimator using the Bartlett kernel up to lag $L$:

$$
\hat{\Omega}_{\text{NW}} = \hat{\gamma}_0 + 2 \sum_{j=1}^{L} \left(1 - \frac{j}{L + 1}\right) \hat{\gamma}_j, \qquad \hat{\gamma}_j = \frac{1}{T} \sum_{t=j+1}^{T} (\text{IC}_t - \overline{\text{IC}})(\text{IC}_{t-j} - \overline{\text{IC}})
$$

$$
t_{\text{NW}} = \frac{\overline{\text{IC}}}{\sqrt{\hat{\Omega}_{\text{NW}} / T}}
$$

[`ic_decay`](../indicators/evaluation-and-backtest.md) evaluates `ic_summary` across a vector of forward horizons $h \in (1, 2, 3, 5, 10, 20)$ (automatically setting `nw_lags = max(1, h - 1)`) and estimates the **Alpha Half-Life** via log-linear regression $\ln |\overline{\text{IC}}(h)| = a - \lambda h \implies t_{1/2} = \frac{\ln 2}{\hat{\lambda}}$.

---

### 3.2 Correcting Backtest Overfitting: Probabilistic & Deflated Sharpe Ratio

Standard Sharpe Ratio estimates ignore **non-normality** (negative skewness and excess kurtosis inflate tail risk) and **multiple-testing selection bias** (running $N$ parameter combinations guarantees finding a high in-sample Sharpe by chance).

#### Probabilistic Sharpe Ratio (`probabilistic_sharpe_ratio` — Bailey & López de Prado, 2012)
Given sample size $T$, non-annualized Sharpe $\widehat{SR}$, skewness $\hat{\gamma}_3$, and Pearson kurtosis $\hat{\gamma}_4$, the asymptotic standard error of $\widehat{SR}$ is:

$$
\hat{\sigma}(\widehat{SR}) = \sqrt{\frac{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{SR}^2}{T - 1}}
$$

The probability that the true Sharpe ratio exceeds benchmark $\text{SR}^{\ast}$ is:

$$
\text{PSR}(\text{SR}^{\ast}) = \Phi\left(\frac{\widehat{\text{SR}} - \text{SR}^{\ast}}{\hat{\sigma}(\widehat{\text{SR}})}\right) \in [0, 1]
$$

#### Deflated Sharpe Ratio (`deflated_sharpe_ratio` — Bailey & López de Prado, 2014)
When a researcher selects the best strategy out of $N$ independent trials (`n_trials`) whose annualized Sharpe ratios have standard deviation $\sigma_{\{SR\}}$, Extreme Value Theory (False Strategy Theorem) gives the expected maximum annualized Sharpe ratio under the null hypothesis of zero skill:

$$
\text{SR}^{\ast}_{\text{ann}}(N) = \sigma_{\{SR\}} \left[ (1 - \gamma) \Phi^{-1}\left(1 - \frac{1}{N}\right) + \gamma \Phi^{-1}\left(1 - \frac{1}{N \cdot e}\right) \right]
$$

where $\gamma \approx 0.5772156649$ is the Euler-Mascheroni constant. Plugging $\text{SR}^{\ast}_{\text{ann}}(N)$ into $\text{PSR}(\text{SR}^{\ast})$ yields the **Deflated Sharpe Ratio (DSR)**. A common threshold is $\text{DSR} \ge 0.95$.

---

## Pillar 4: Execution-Realistic Backtesting & Excursion Analytics

[`vectorized_backtest`](../indicators/evaluation-and-backtest.md) (`df.anta.backtest`) extends vectorized simulation with four execution controls and a full Round-Trip Trade Ledger:

```python
bt = df.anta.backtest(
    signal="alpha_signal",
    execution_price="open",  # Execute at next bar's Open (or 'twap' / 'vwap')
    flat_mask=pl.col("time") >= "14:25:00",  # Force 0 position at intraday EOD session close
    vol_target=0.15,  # 15% annualized dynamic volatility targeting
    vol_lookback=20,
    max_leverage=2.0,
    no_trade_buffer=0.05,  # Suppress tiny rebalances < 5% weight change
    fee_rate=0.0003,
    slippage_rate=0.0002,
)

# Inspect tearsheet and Round-Trip Trade Ledger with MAE / MFE
print(bt.report())
print(bt.trades())
```

### Interpreting MAE & MFE in `bt.trades()`
For every closed round-trip trade $k$, `bt.trades()` reports:
- **`mae` (Maximum Adverse Excursion $\le 0$)**: The worst intra-trade drawdown from entry price experienced before the trade closed. If winning trades consistently have $\text{MAE} > -1.2\%$, placing a hard stop-loss at $-1.5\%$ cuts tail losses without prematurely stopping out winners.
- **`mfe` (Maximum Favorable Excursion $\ge 0$)**: The highest unrealized profit reached during the trade's lifetime. Comparing `mfe` against final `gross_return` reveals how much profit is given back before exit.
