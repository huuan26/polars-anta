# Factor Evaluation, Performance Metrics & Vectorized Backtesting (14 Tools)

`polars-anta` provides an institutional-grade suite for evaluating alpha factor predictive power (Alphalens-style **Information Coefficient**, **ICIR**, **Quantile Monotonicity**, **Turnover**) alongside scalar **Strategy Performance & Risk Expressions** and a loop-free **Vectorized Backtesting Engine** (`df.anta.backtest()`).

---

## 1. Alphalens-Style Factor Evaluation Suite (`polars_anta.evaluation`)

### `forward_returns(df, close_col="close", periods=(1, 5, 10), group_by=None, prefix="fwd_ret_")`
Computes multi-horizon forward returns $(P_{t+k} / P_t) - 1$ across single-asset or multi-asset (`group_by="symbol"`) DataFrames and LazyFrames.

```python
eval_df = df.anta.forward_returns(periods=(1, 5, 10), group_by="symbol")
```

### `information_coefficient(df, factor_col, return_col, group_by=None, method="spearman")`
Computes the cross-sectional **Rank IC** (`method="spearman"`, default) or **Normal IC** (`method="pearson"`) time series between an alpha factor column and a forward return column at each timestamp (`group_by="timestamp"`). Returns an eager `pl.DataFrame` with `[group_by, "ic", "n_obs"]`.

### `ic_summary(df, factor_col, return_col, group_by=None, method="spearman")`
Computes comprehensive summary statistics of a factor's Information Coefficient:
- `ic_mean`: Mean cross-sectional IC
- `ic_std`: Standard deviation of IC
- `icir`: Information Ratio of IC ($\text{IC}_{\text{mean}} / \text{IC}_{\text{std}}$)
- `t_stat`: Student's $t$-statistic testing $\text{IC} \neq 0$
- `hit_rate`: Proportion of timestamps with $\text{IC} > 0$
- `n_periods`: Number of cross-sectional periods evaluated

```python
stats = eval_df.anta.ic_summary("alpha_012", "fwd_ret_1", group_by="timestamp")
```

### `quantile_analysis(df, factor_col, return_col, quantiles=5, group_by=None, annualization=252.0)`
Partitions observations into $Q$ cross-sectional factor quantiles ($1 \dots Q$) per timestamp and computes per-bucket performance metrics (`count`, `mean_factor`, `mean_return`, `std_return`, `sharpe_ratio`, `win_rate`, `cumulative_return`) to verify factor monotonicity.

### `factor_turnover(df, factor_col, asset_col="ticker", time_col="timestamp", lag=1)`
Measures signal stability across time by computing cross-sectional rank autocorrelation (`rank_autocorr`) and L1 portfolio weight turnover (`turnover`) at each timestamp.

---

## 2. Strategy Performance & Risk Expressions (`pl.Expr`)

All performance metrics are implemented as native Polars expressions (`pl.Expr`) and can be used inside `df.select()`, `df.group_by("symbol").agg()`, or via `.anta` / `.ta` expression namespaces.

| Function Signature | Return Type | Mathematical Definition |
| :--- | :--- | :--- |
| `drawdown_series(returns_or_equity, is_returns=True)` | `pl.Expr` (Series) | Underwater percentage drawdown curve $(E_t - \max_{s \le t} E_s) / \max_{s \le t} E_s \le 0.0$ |
| `max_drawdown(returns_or_equity, is_returns=True)` | `pl.Expr` (Scalar) | Worst peak-to-trough percentage decline $\min_t \text{DD}_t$ |
| `sharpe_ratio(returns, risk_free_rate=0.0, annualization=252.0)` | `pl.Expr` (Scalar) | Annualized Sharpe Ratio $\frac{\mathbb{E}[r - r_f]}{\sigma(r - r_f)} \sqrt{N_{\text{ann}}}$ |
| `sortino_ratio(returns, risk_free_rate=0.0, annualization=252.0)` | `pl.Expr` (Scalar) | Annualized Sortino Ratio using downside deviation $\sqrt{\mathbb{E}[\min(0, r - r_f)^2]}$ |
| `calmar_ratio(returns, annualization=252.0)` | `pl.Expr` (Scalar) | Annualized Return divided by $\|\text{Max Drawdown}\|$ |
| `profit_factor(returns)` | `pl.Expr` (Scalar) | Gross Profit divided by $\|\text{Gross Loss}\|$ ($\sum r^+ / \sum |r^-|$) |
| `win_rate(returns, include_zeros=False)` | `pl.Expr` (Scalar) | Proportion of active periods with positive returns $r_t > 0$ |

---

## 3. Loop-Free Vectorized Backtesting (`polars_anta.backtest`)

### `crosses_above(a, b, over=None)` & `crosses_below(a, b, over=None)`
Returns a boolean Polars expression that evaluates to `True` on the exact bar where expression `a` crosses above (or below) expression or scalar threshold `b`.

```python
entries = pl.col("ema_12").anta.crosses_above("ema_26")
rsi_oversold = pl.col("rsi_14").anta.crosses_above(30)
```

### `signals_to_positions(entries, exits, short_entries=None, short_exits=None, over=None)`
Converts discrete boolean entry/exit triggers into a continuous position series in $\{-1.0, 0.0, +1.0\}$ using Polars' native `.forward_fill()` state machine with zero Python loops.

### `vectorized_backtest(df, ...)` / `df.anta.backtest(...)`
Executes a single-asset or multi-asset vectorized backtest with automatic look-ahead protection (`signal_lag=1`), proportional fee & slippage deduction (`turnover * (fee_rate + slippage_rate)`), cross-sectional portfolio weight schemes (`"raw"`, `"dollar_neutral"`, `"long_only"`, `"equal_weight"`), and Buy & Hold benchmark comparison.

```python
bt = (
    df.anta.ema(12, alias="ema_12")
    .anta.ema(26, alias="ema_26")
    .anta.backtest(
        entries=pl.col("ema_12").anta.crosses_above("ema_26"),
        exits=pl.col("ema_12").anta.crosses_below("ema_26"),
        fee_rate=0.0005,
        slippage_rate=0.0002,
        initial_capital=100_000.0,
    )
)
print(bt.report())
```

### `BacktestResult` & `compare_strategies(strategies)`
- `bt.data`: Polars DataFrame containing `position`, `turnover`, `asset_return`, `gross_return`, `cost`, `net_return`, `equity_curve`, `benchmark_equity`, and `drawdown`.
- `bt.summary`: Dictionary of 18 key performance and risk metrics.
- `bt.summary_df(transpose=False)`: Converts summary metrics into a Polars DataFrame.
- `bt.report()`: Formatted human-readable tearsheet DataFrame with units (`$`, `%`, `x`).
- `pa.compare_strategies({"Strat A": bt_a, "Strat B": bt_b})`: Side-by-side multi-strategy comparison table.
