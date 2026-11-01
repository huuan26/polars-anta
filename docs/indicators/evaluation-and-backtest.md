# Factor Evaluation, AFML Labeling & Vectorized Backtesting (`evaluation`, `backtest`)

`polars-anta` provides an end-to-end institutional validation and simulation suite combining **AFML Path-Dependent Labeling**, **Alphalens-Style Cross-Sectional Factor Evaluation with Newey-West HAC Inference**, **Multiple-Testing Overfitting Controls (PSR & DSR)**, and an **Execution-Realistic Vectorized Backtester with Round-Trip MAE/MFE Trade Analytics**.

---

## 1. Quantitative Machine Learning Labeling (`polars_anta.evaluation`)

### `triple_barrier_labels(close, volatility=None, horizon=10, pt_mult=2.0, sl_mult=2.0, min_ret=0.0, vol_period=20, side=None)`
**Mathematical Formulation (López de Prado, 2018, Chapter 3):**
At each bar $t$ with local volatility $\sigma_t$ (defaulting to the $20$-bar rolling standard deviation of returns when `volatility=None`) and optional primary model directional prediction $s_t = \text{sign}(\text{side}_t) \in \{-1, +1\}$:

$$
r_{t, t+k}^{\text{raw}} = \frac{P_{t+k} - P_t}{P_t}, \qquad r_{t, t+k}^{\text{dir}} = s_t \cdot r_{t, t+k}^{\text{raw}} \quad (k = 1, \dots, H)
$$

The horizontal Profit-Taking ($U_t$) and Stop-Loss ($L_t$) thresholds are:

$$
U_t = \max\left(\text{pt\_mult} \cdot \sigma_t,\, \text{min\_ret}\right), \qquad L_t = \max\left(\text{sl\_mult} \cdot \sigma_t,\, \text{min\_ret}\right)
$$

Let $\tau_{\text{PT}} = \min\{k \le H : r_{t, t+k}^{\text{dir}} \ge U_t\}$ and $\tau_{\text{SL}} = \min\{k \le H : r_{t, t+k}^{\text{dir}} \le -L_t\}$ (with $\min \emptyset = H + 1$).

**Output `pl.Struct` Schema (`alias="triple_barrier"`, `prefix="tb_"` when `unnest=True`):**
- `label` (`Float64`): $+1.0$ if upper price barrier touched first, $-1.0$ if lower price barrier touched first, or $0.0$ if vertical expiration barrier $H$ reached first.
- `touch_bars` (`Int64`): First-passage time $\tau^* = \min(\tau_{\text{PT}}, \tau_{\text{SL}}, H) \in \{1, \dots, H\}$.
- `realized_return` (`Float64`): Raw price return $r_{t, t+\tau^*}^{\text{raw}}$ at the touched barrier.
- `meta_label` (`Float64`): Binary $\{0.0, 1.0\}$ equal to $1.0$ if and only if the trade in direction $s_t$ hit its Profit-Taking barrier ($\tau_{\text{PT}} < \tau_{\text{SL}}$ and $\tau_{\text{PT}} \le H$).

```python
df = df.anta.triple_barrier_labels(
    close="close",
    horizon=12,
    pt_mult=2.0,
    sl_mult=1.5,
    side="primary_signal",
    unnest=True,
    group_by="symbol",
)
```

---

### `trend_scanning_labels(close, min_window=5, max_window=20, step=1)`
**Mathematical Formulation (López de Prado, 2020):**
Fits forward OLS regressions $P_{t+i} = \hat{\beta}_0(L) + \hat{\beta}_1(L)\,i + \epsilon_i$ across all candidate horizons $L \in \{\text{min\_window}, \dots, \text{max\_window}\}$:

$$
t_{\hat{\beta}_1}(L) = \frac{\hat{\beta}_1(L)}{\sqrt{\frac{\text{SSR}(L) / (L - 2)}{\sum_{i=0}^{L-1}(i - \bar{i})^2}}}, \qquad L^*_t = \arg\max_{L} \left|t_{\hat{\beta}_1}(L)\right|
$$

**Output `pl.Struct` Schema (`alias="trend_scan"`, `prefix="ts_label_"` when `unnest=True`):**
- `label` (`Float64`): Trend regime direction $\text{sign}\left(t_{\hat{\beta}_1}(L^*_t)\right) \in \{-1.0, 0.0, +1.0\}$.
- `t_value` (`Float64`): Signed optimal OLS $t$-statistic $t_{\hat{\beta}_1}(L^*_t)$ (usable as sample weights in ML training).
- `optimal_window` (`Int64`): Optimal lookahead window length $L^*_t$.
- `ret` (`Float64`): Forward return $(P_{t + L^*_t - 1} / P_t) - 1$.

---

## 2. Institutional Factor Validation Suite (`polars_anta.evaluation`)

### `forward_returns(df, close_col="close", periods=(1, 5, 10), group_by=None, prefix="fwd_ret_")`
Appends multi-horizon forward return columns $r_{t \to t+p} = \frac{P_{t+p}}{P_t} - 1$ partitioned by `group_by`.

---

### `information_coefficient(df, factor_col, return_col, group_by=None, method="spearman")`
Computes the cross-sectional **Rank IC** (`method="spearman"`, default) or **Pearson IC** (`method="pearson"`) time series:

$$
\text{RankIC}_t = \text{Corr}\left(\text{rank}(F_{i, t}),\, \text{rank}(R_{i, t \to t+h})\right)
$$

---

### `ic_summary(df, factor_col, return_col, group_by=None, method="spearman", nw_lags=None)`
Computes institutional IC summary metrics including **Newey-West (1987) HAC autocorrelation-consistent $t$-statistics**:

$$
\hat{\Omega}_{\text{NW}} = \hat{\gamma}_0 + 2 \sum_{j=1}^{L} \left(1 - \frac{j}{L + 1}\right) \hat{\gamma}_j, \qquad \text{nw\_t\_stat} = \frac{\overline{\text{IC}}}{\sqrt{\hat{\Omega}_{\text{NW}} / T}}
$$

When `nw_lags=None`, automatically selects the Newey-West (1994) plug-in bandwidth $L = \max\left(1, \lfloor 4(T/100)^{2/9}\rfloor\right)$.

| Returned Key | Formula / Meaning | Institutional Benchmark |
| :--- | :--- | :--- |
| `ic_mean` | $\overline{\text{IC}} = \frac{1}{T}\sum_{t=1}^T \text{IC}_t$ | $\vert \overline{\text{IC}}\vert  \ge 0.03$ (Daily) |
| `ic_std` | Sample standard deviation $\hat{\sigma}_{\text{IC}}$ | Lower indicates regime stability |
| `icir` | Information Ratio of IC ($\overline{\text{IC}} / \hat{\sigma}_{\text{IC}}$) | $\vert \text{ICIR}\vert  \ge 0.50$ |
| `t_stat` | i.i.d. Student $t$-statistic $\overline{\text{IC}} / (\hat{\sigma}_{\text{IC}} / \sqrt{T})$ | $\vert t\vert  \ge 2.0$ ($h = 1$ only) |
| `nw_t_stat` | **Newey-West HAC $t$-statistic** $\overline{\text{IC}} / \sqrt{\hat{\Omega}_{\text{NW}} / T}$ | $\vert \text{nw\_t\_stat}\vert  \ge 2.0$ (robust to overlapping $h > 1$) |
| `hit_rate` | Fraction of periods with $\text{IC}_t > 0$ | $\ge 55\%$ |
| `n_periods` | Total valid cross-sectional timestamps $T$ | — |

---

### `ic_decay(df, factor_col, close_col="close", horizons=(1, 2, 3, 5, 10, 20), asset_col=None, group_by=None, method="spearman")`
Evaluates `ic_summary` (with `nw_lags = max(1, h - 1)`) across all forward horizons $h \in \text{horizons}$ and estimates the **Alpha Half-Life** (`half_life` column, in bars) via log-linear OLS regression $\ln |\overline{\text{IC}}(h)| = a - \lambda h \implies t_{1/2} = \frac{\ln 2}{\hat{\lambda}}$.

---

### `quantile_analysis(df, factor_col, return_col, quantiles=5, group_by=None, annualization=252.0)` & `factor_turnover(df, factor_col, asset_col="ticker", time_col="timestamp", lag=1)`
- `quantile_analysis`: Partitions assets into $Q$ cross-sectional factor buckets per timestamp and reports `count`, `mean_factor`, `mean_return`, `std_return`, `sharpe_ratio`, `win_rate`, and `cumulative_return` to verify monotonic return separation from $Q_1$ to $Q_Q$.
- `factor_turnover`: Computes cross-sectional rank autocorrelation $\text{Corr}(\text{rank}_{t}, \text{rank}_{t-\text{lag}})$ and L1 weight turnover $\sum_i |w_{i,t} - w_{i,t-\text{lag}}|$ to assess transaction cost drag before backtesting.

---

## 3. Strategy Performance & Overfitting-Adjusted Risk Expressions (`pl.Expr`)

All scalar risk metrics operate natively inside `df.select(...)`, `df.group_by("symbol").agg(...)`, or via `.anta` / `.ta` expression namespaces:

| Function Signature | Mathematical Definition |
| :--- | :--- |
| `drawdown_series(x, is_returns=True)` | $\text{DD}_t = \frac{E_t - \max_{s \le t} E_s}{\max_{s \le t} E_s} \le 0.0$ |
| `max_drawdown(x, is_returns=True)` | $\text{MDD} = \min_{t} \text{DD}_t$ |
| `sharpe_ratio(returns, risk_free_rate=0.0, annualization=252.0)` | $\text{SR}_{\text{ann}} = \frac{\mathbb{E}[r - r_f]}{\sigma(r - r_f)} \sqrt{N_{\text{ann}}}$ |
| `sortino_ratio(returns, risk_free_rate=0.0, annualization=252.0)` | $\text{Sortino}_{\text{ann}} = \frac{\mathbb{E}[r - r_f]}{\sqrt{\mathbb{E}[\min(0, r - r_f)^2]}} \sqrt{N_{\text{ann}}}$ |
| `calmar_ratio(returns, annualization=252.0)` | $\text{Calmar} = \frac{\mathbb{E}[r] \cdot N_{\text{ann}}}{\vert \text{MDD}\vert }$ |
| `profit_factor(returns)` | $\text{PF} = \frac{\sum_{r_t > 0} r_t}{\sum_{r_t < 0} \vert r_t\vert }$ |
| `win_rate(returns, include_zeros=False)` | $\text{WR} = \frac{\sum \mathbb{I}_{\{r_t > 0\}}}{\sum \mathbb{I}_{\{r_t \neq 0\}}}$ |
| `probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, ...)` | $\text{PSR}(SR^*) = \Phi\left(\frac{\widehat{SR} - SR^*}{\sqrt{\frac{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}{T - 1}}}\right) \in [0, 1]$ |
| `deflated_sharpe_ratio(returns, n_trials=10, trials_sr_std=0.5, ...)` | $\text{PSR}(SR^*_N)$ where $SR^*_N = \sigma_{\{SR\}}\left[(1-\gamma)\Phi^{-1}(1-\frac{1}{N}) + \gamma\Phi^{-1}(1-\frac{1}{Ne})\right]$ |

---

## 4. Execution-Realistic Vectorized Backtesting & MAE/MFE Trade Ledger (`polars_anta.backtest`)

### `vectorized_backtest(df, ...)` / `df.anta.backtest(...)`
Executes a single-asset or multi-asset vectorized backtest with full execution realism:
- **Look-Ahead Prevention (`signal_lag=1`)**: Shifts target weights by 1 bar before multiplying by bar returns.
- **Execution Price Benchmarks (`execution_price`)**:
  - `"close"` (default): Close-to-Close bar returns.
  - `"open"`: Next-Open to Open execution returns.
  - `"twap"`: OHLC average $(O_t + H_t + L_t + C_t)/4$ execution returns.
  - `"vwap"`: Typical price $(H_t + L_t + C_t)/3$ execution returns.
- **Intraday Session Flattening (`flat_mask`)**: Boolean expression forcing `position = 0.0` on matching bars (e.g., closing intraday VN30F1M positions prior to the 14:30 ATC auction).
- **Dynamic Volatility Targeting (`vol_target`, `vol_lookback=20`, `max_leverage=2.0`)**: Scales position size by $\min\left(\text{max\_leverage},\, \frac{\text{vol\_target}}{\hat{\sigma}_{t-1}\sqrt{N_{\text{ann}}}}\right)$.
- **No-Trade Turnover Buffer (`no_trade_buffer=0.0`)**: Suppresses minor position adjustments where $|w_t^{\text{target}} - w_{t-1}| \le \text{no\_trade\_buffer}$ unless exiting to `0.0`.

---

### `BacktestResult.trades()` & `extract_trades(bt_df, ...)`
Extracts a complete **Round-Trip Trade Ledger** with intra-trade excursion analytics:
- `trade_id`, `side` (`"LONG"` / `"SHORT"`), `entry_index`, `exit_index`, `entry_time`, `exit_time`, `holding_bars`, `avg_position`, `entry_price`, `exit_price`, `gross_return`, `cost`, `net_return`.
- **`mae` (Maximum Adverse Excursion $\le 0.0$)**: Worst intra-trade cumulative drawdown from entry price.
- **`mfe` (Maximum Favorable Excursion $\ge 0.0$)**: Highest intra-trade cumulative profit reached before exit.

---

### `parameter_sweep(df, signals, ...)` / `df.anta.parameter_sweep(...)`
Evaluates a dictionary of candidate parameter configurations `signals: Mapping[str, IntoExpr]` in a single call, computes the empirical cross-trial Sharpe dispersion $\sigma_{\{SR\}}$, and ranks all configurations with their multiple-testing **Deflated Sharpe Ratio (`deflated_sharpe_ratio`)**.
