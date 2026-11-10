# Quantitative Statistics, CAPM & Regression Transforms (`polars_anta.stats`)

The `polars_anta.stats` module provides **15 vectorized statistical, econometric, and stationarity primitives** engineered for factor standardization, systematic risk decomposition, rolling Ordinary Least Squares (OLS) regression, and memory-preserving fractional differentiation.

---

## 1. Rolling Moments & Non-Parametric Normalization

### `zscore(close, period=20)`
**Mathematical Formulation:**

$$
Z_t = \frac{x_t - \hat{\mu}_t}{\hat{\sigma}_t}, \qquad \hat{\mu}_t = \frac{1}{n}\sum_{i=0}^{n-1} x_{t-i}, \qquad \hat{\sigma}_t = \sqrt{\frac{1}{n - 1}\sum_{i=0}^{n-1}(x_{t-i} - \hat{\mu}_t)^2}
$$

- **Quantitative Interpretation**: Measures how many rolling standard deviations the current observation lies away from its trailing mean. Widely used in statistical arbitrage (Ornstein-Uhlenbeck mean-reversion triggers when $|Z_t| > 2.0$) and time-series feature normalization.
- **Tech / Polars Execution**: Pure native Rust `rolling_mean(period)` and `rolling_std(period)` with zero-division protection (`0.0` when $\hat{\sigma}_t = 0$).

---

### `percentile_rank(close, period=20)`
**Mathematical Formulation:**

$$
\text{PctRank}_t = \frac{1}{n - 1} \sum_{i=1}^{n-1} \left( \mathbb{I}_{\{x_{t-i} < x_t\}} + \frac{1}{2}\mathbb{I}_{\{x_{t-i} = x_t\}} \right) \in [0.0, 1.0]
$$

- **Quantitative Interpretation**: Non-parametric, outlier-resistant alternative to `zscore`. Because financial returns exhibit heavy tails where a single volatility shock can distort $\hat{\sigma}_t$ for $n$ subsequent bars, `percentile_rank` maps any fat-tailed distribution uniformly onto $[0, 1]$.
- **Tech / Polars Execution**: Vectorized via zero-copy `sliding_window_view` pairwise comparison against the rightmost column `windows[:, -1:]`.

---

### `skewness(close, period=20)` & `kurtosis(close, period=20)`
**Mathematical Formulations:**

$$
\hat{\gamma}_{3, t} = \frac{\frac{1}{n}\sum_{i=0}^{n-1}(x_{t-i} - \hat{\mu}_t)^3}{\left(\frac{1}{n}\sum_{i=0}^{n-1}(x_{t-i} - \hat{\mu}_t)^2\right)^{3/2}}, \qquad \hat{\gamma}_{4, t} = \frac{\frac{1}{n}\sum_{i=0}^{n-1}(x_{t-i} - \hat{\mu}_t)^4}{\left(\frac{1}{n}\sum_{i=0}^{n-1}(x_{t-i} - \hat{\mu}_t)^2\right)^{2}} - 3
$$

- **Quantitative Interpretation**:
  - **Skewness ($\hat{\gamma}_3$)**: Quantifies distributional asymmetry. Negative return skewness ($\hat{\gamma}_3 < 0$) warns of "crash risk" (frequent small gains punctuated by sharp drawdowns, typical of short-volatility or carry strategies).
  - **Excess Kurtosis ($\hat{\gamma}_4$)**: Quantifies tail thickness relative to a Gaussian distribution ($\gamma_4^{\text{normal}} = 0$). High excess kurtosis signals regime instability and elevated probability of $3\sigma+$ tail events.
- **Tech / Polars Execution**: Delegated directly to Polars' native streaming `rolling_skew(period)` and `rolling_kurtosis(period)` kernels in Rust.

---

## 2. Rolling Bivariate Moments & CAPM Decomposition

### `rolling_cov(x, y, period=20)` & `rolling_corr(x, y, period=20)`
**Mathematical Formulations:**

$$
\text{Cov}_t(X, Y) = \frac{1}{n - 1} \sum_{i=0}^{n-1} (x_{t-i} - \bar{x}_t)(y_{t-i} - \bar{y}_t), \qquad \rho_t(X, Y) = \frac{\text{Cov}_t(X, Y)}{\hat{\sigma}_{X, t}\,\hat{\sigma}_{Y, t}} \in [-1, 1]
$$

- **Tech / Polars Execution**: Computed via Polars' native `pl.rolling_cov(x, y, window_size=period)` and `pl.rolling_corr(x, y, window_size=period)` in Rust.

---

### `rolling_beta(asset_returns, benchmark_returns, period=60)` & `rolling_alpha(asset_returns, benchmark_returns, rf=0.0, period=60)`
**Mathematical Formulations (Sharpe-Lintner CAPM):**

$$
\hat{\beta}_t = \frac{\text{Cov}_t(R_a, R_m)}{\text{Var}_t(R_m)}, \qquad \hat{\alpha}_t = (\bar{R}_{a, t} - r_f) - \hat{\beta}_t (\bar{R}_{m, t} - r_f)
$$

- **Quantitative Interpretation**:
  - `rolling_beta` measures time-varying systematic exposure of an asset or strategy $R_a$ to the market benchmark $R_m$ (e.g., VN30 Index). It is the foundational input for constructing **Beta-Neutral** long/short portfolios ($w_i^{\text{hedged}} = w_i - \hat{\beta}_i w_m$) and passing into `orthogonalize`.
  - `rolling_alpha` (Jensen's Alpha) isolates idiosyncratic return per period unexplained by systematic market drift.

---

## 3. Closed-Form Rolling OLS Linear Regression Suite

Fitting $y_{t-n+1+i} = \beta_0 + \beta_1 \cdot i + \epsilon_i$ for $i \in \{0, 1, \dots, n - 1\}$ via exact WMA–SMA algebraic reduction (see [Architecture & Mathematical Derivations](../architecture.md#21-closed-form-rolling-ols-linear-regression-1124x-speedup)):

| Function Signature | Closed-Form Identity | Quantitative Use Case |
| :--- | :--- | :--- |
| `linear_regression_slope(close, period=14)` | $\hat{\beta}_1 = \frac{6}{n - 1}\left(\text{WMA}_n - \text{SMA}_n\right)$ | Velocity of price change per bar ($112.4\times$ faster than rolling `scipy.stats.linregress`). |
| `linear_regression_intercept(close, period=14)` | $\hat{\beta}_0 = 4\,\text{SMA}_n - 3\,\text{WMA}_n$ | Fitted value at the start of the rolling window ($i = 0$). |
| `linear_regression_forecast(close, period=14)` | $\hat{y}_t = \hat{\beta}_0 + \hat{\beta}_1(n - 1) = 3\,\text{WMA}_n - 2\,\text{SMA}_n$ | Zero-lag OLS end-of-window fair value estimate. |
| `linear_regression_angle(close, period=14)` | $\theta_t = \arctan(\hat{\beta}_1) \cdot \frac{180}{\pi}$ | Trend trajectory angle in degrees $(-90^\circ, +90^\circ)$. |
| `linear_regression_r2(close, period=14)` | $R^2_t = \text{clip}\left(\hat{\beta}_1^2 \cdot \frac{n(n + 1)}{12\,\text{Var}_s(y)},\, 0,\, 1\right)$ | Rolling OLS goodness-of-fit ($R^2 \in [0, 1]$) measuring trend linearity vs. noise. |
| `linear_regression_stderr(close, period=14)` | $s_{\hat{y}, t} = \sqrt{\frac{n - 1}{n - 2}\,\text{Var}_s(y)\,(1 - R^2_t)}$ | Rolling OLS standard error of the estimate ($\sqrt{\text{SSR}/(n-2)}$) in price units. |

```python
import polars as pl
import polars_anta as pa

df = df.with_columns(
    pl.col("close").anta.linear_regression_r2(period=14).alias("ols_r2_14"),
    pl.col("close").anta.linear_regression_stderr(period=14).alias("ols_se_14"),
)
```

---

## 4. Stationarity & Long-Memory Preservation (AFML)

### `frac_diff(close, d=0.4, window=20)`
**Mathematical Formulation (Fixed-Window Fractional Differentiation — López de Prado, 2018):**

$$
\tilde{X}_t^{(d, W)} = \sum_{k=0}^{W-1} \omega_k X_{t-k}, \qquad \omega_0 = 1, \qquad \omega_k = -\omega_{k-1} \frac{d - k + 1}{k} \quad (k = 1, \dots, W - 1)
$$

- **Quantitative Theory**:
  - When $d = 0.0$, $\omega_0 = 1$ and $\omega_k = 0$ ($\forall k \ge 1$), returning the raw non-stationary price series $X_t$.
  - When $d = 1.0$, $\omega_0 = 1, \omega_1 = -1$, and $\omega_k = 0$ ($\forall k \ge 2$), returning the standard first difference $X_t - X_{t-1}$ (which is stationary but erases trend memory).
  - For fractional orders $d \in (0, 1)$ (typically $d \in [0.3, 0.5]$), all weights $\omega_k$ ($k \ge 1$) are negative and decay slowly according to a power law $\omega_k \sim \frac{-d}{\Gamma(1-d)} k^{-d-1}$. This transformation removes unit-root non-stationarity so tree/neural ML models do not overfit to secular price levels, while preserving long-memory cointegration and support/resistance structure.
- **Tech / Polars Execution**: Precomputes the exact binomial weight vector `w_rev = weights[::-1]` once and executes a single BLAS matrix-vector product `sliding_window_view(vals, window) @ w_rev` inside `map_batches`, supporting multi-asset `.over("symbol")` partitioning.

```python
import polars as pl
import polars_anta as pa

df = df.with_columns(
    pl.col("close").anta.frac_diff(d=0.4, window=20).over("symbol").alias("close_ffd_04")
)
```
