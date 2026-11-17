# Cross-Sectional Alpha Engineering & Cost-Aware Allocation

`polars-anta` provides a zero-C, pure `Polars + NumPy` mathematical pipeline for turning noisy, correlated cross-sectional signals (`time x asset x K factors`) into risk-neutralized, symmetrically orthogonalized, and turnover-controlled portfolio weights.

---

## 1. Signal Conditioning & Orthogonalization (`polars_anta.conditioning`)

### Cross-Sectional Rank-Gaussianization (`cs_gaussianize`)
Maps cross-sectional percentile ranks $u_{i,t} = (r_{i,t} - 0.5) / N_t$ through Acklam's rational approximation of $\Phi^{-1}(u_{i,t})$ directly inside Polars expressions, suppressing heavy-tailed outliers before OLS projection or SVD.

### Two-Stage FWL Neutralization (`neutralize_panel`)
Removes simultaneous exposure to categorical groups $\mathbf{D}_t$ (e.g. sector/industry) and continuous risk characteristics $\mathbf{X}_t$ (e.g. market beta, log size, volatility). Both candidate factors and continuous risk regressors are first demeaned within `(time_col, *group_col)` on their shared valid support before Gram-Schmidt projection over `time_col`, guaranteeing exact simultaneous orthogonality:

$$
\mathbf{D}_t^T \mathbf{F}_{\text{neut}, t} = \mathbf{0}, \qquad \mathbf{X}_t^T \mathbf{F}_{\text{neut}, t} = \mathbf{0}
$$

### Batched Lowdin Symmetric Orthogonalization (`orthogonalize_panel`, `lowdin_matrix`)
Unlike sequential Gram-Schmidt (which distorts later factors more heavily than the first), Lowdin polar decomposition solves the minimum Frobenius-distortion problem on the Stiefel manifold:

$$
\min_{\widetilde{\mathbf{F}}_t} \lVert \mathbf{F}_t - \widetilde{\mathbf{F}}_t \rVert_F^2 \quad \text{s.t.} \quad \frac{1}{N_t - 1}\widetilde{\mathbf{F}}_t^T \widetilde{\mathbf{F}}_t = \mathbf{I}_K
$$

`orthogonalize_panel` aggregates all $K(K+1)/2$ cross-sectional Gram entries in a single Polars `group_by(time_col)` pass and computes batched 3D spectral inverse square roots with rank-deficiency truncation ($\lambda_{t,k} > \text{rcond} \cdot \lambda_{t,\max}$), avoiding per-timestamp Python loops and surviving duplicate or constant factor columns.

> **Subspace invariance requirement**: Because $\widetilde{\mathbf{F}}_t = \mathbf{Z}_t \mathbf{M}_t$ mixes all input columns linearly, prior sector/risk neutrality ($\mathbf{X}_t^T \mathbf{F}_t = \mathbf{0}$) is preserved if and only if **every** column in `factor_cols` has already been projected onto $\mathcal{N}(\mathbf{X}_t^T)$ via `neutralize_panel` before calling `orthogonalize_panel`.

| Function | Signature | Output |
| :--- | :--- | :--- |
| `cs_gaussianize` | `cs_gaussianize(expr, over=None, clip_bound=4.0)` | `pl.Expr` (`Float64`) |
| `lowdin_matrix` | `lowdin_matrix(mat, rcond=1e-10, standardize=True)` | `np.ndarray` (`2D` or `3D`) |
| `neutralize_panel` | `neutralize_panel(df, factor_cols, time_col="date", risk_cols=None, group_col=None, standardize=True, ridge=0.0, suffix="_neut")` | `DataFrame` or `LazyFrame` |
| `orthogonalize_panel` | `orthogonalize_panel(df, factor_cols, time_col="date", method="lowdin", rcond=1e-10, standardize=True, suffix="_ortho")` | `pl.DataFrame` |
| `collinearity_summary` | `collinearity_summary(df, factor_cols, time_col=None, rcond=1e-10)` | `pl.DataFrame` (`vif`, `effective_rank`, `condition_number`) |

---

## 2. Alphalens-Grade Diagnostics & Redundancy (`polars_anta.diagnostics`)

`analyze_alpha` evaluates predictive power alongside execution friction and universe concentration:

- **Predictive IC & Decay**: Cross-sectional Spearman/Pearson IC, Newey-West HAC $t$-statistic (`nw_lags = max(1, horizon - 1)`), and exponential IC half-life.
- **Turnover, Gross Margin & Breakeven Cost**: Simulates a causal `signal_lag`-shifted overlapping $h$-tranche book under `"dollar_neutral"`, `"long_only"`, or `"quantile_spread"` weights, reporting L1 turnover, gross margin (`bps` per unit turnover), breakeven one-way cost (`bps`), and unbalanced-panel-safe daily equal-weight spread attribution between the long and short legs.
- **Causal ADV Sub-Universe Split**: Ranks trailing median dollar volume (`close * volume`) up to bar $t$ to verify whether predictive power, tier turnover (`mean_turnover`), and per-trade margin (`gross_margin_bps`) survive in liquid large-caps rather than only illiquid small-caps.
- **Marginal Redundancy (`marginal_alpha_report`)**: Tests a candidate signal against an existing factor library, reporting `max_abs_corr`, participation-ratio gain $\Delta N_{\text{eff}}$, cross-sectional `spanned_r2`, and FWL-orthogonalized `residual_ic_mean` / `residual_nw_t_stat`.

| Function / Class | Signature | Output |
| :--- | :--- | :--- |
| `analyze_alpha` | `analyze_alpha(df, factor_col, return_col=None, close_col="close", volume_col="volume", asset_col="ticker", time_col="date", horizon=1, signal_lag=1, quantiles=5, weight_scheme="dollar_neutral", one_way_cost_bps=10.0, ...)` | `AlphaDiagnosticReport` |
| `marginal_alpha_report` | `marginal_alpha_report(df, candidate_col, benchmark_cols, return_col, time_col="date", horizon=1, ic_method="spearman")` | `dict[str, float]` |

---

## 3. Covariance Shrinkage & RMT Denoising (`polars_anta.covariance`)

| Function | Description |
| :--- | :--- |
| `ledoit_wolf_shrinkage(data, target="constant_correlation")` | Loop-free BLAS analytical Ledoit-Wolf (2004) optimal shrinkage estimator returning `(cov, delta)`. |
| `marchenko_pastur_denoise(data_or_cov, n_obs=None, method="constant_residual")` | Random Matrix Theory denoiser estimating the Marchenko-Pastur bulk edge $\lambda_+ = \hat{\sigma}^2(1 + \sqrt{K/T})^2$ via fixed-point iteration and replacing bulk noise eigenvalues while preserving trace and unit correlation diagonal. |
| `exponential_covariance(data, half_life=60.0)` | Unbiased exponentially weighted covariance projected onto the SPD cone. |
| `sample_covariance(data, ddof=1)` | Finite-row sample covariance guaranteed to return 2D `(K, K)` shape even for $K = 1$. |
| `nearest_spd(matrix, eps=1e-8)` | Higham spectral projection onto Symmetric Positive Definite matrices. |

---

## 4. Cost-Aware FISTA Allocation (`polars_anta.allocation`)

### Exact Joint Proximal Operator
`solve_fista` minimizes the turnover-penalized quadratic objective:

$$
\min_{\mathbf{w} \in \mathcal{C}} \; -\hat{\boldsymbol{\mu}}^T \mathbf{w} + \frac{\gamma}{2} \mathbf{w}^T \boldsymbol{\Omega} \mathbf{w} + \frac{\lambda_2}{2} \lVert \mathbf{w} \rVert_2^2 + \lambda_1 \lVert \mathbf{w} - \mathbf{w}_{\text{prev}} \rVert_1
$$

Instead of sequential composition (which breaks sparsity on the simplex), `solve_fista` evaluates the exact Lagrangian joint proximal operator $w_k(\nu^\star) = \text{clip}(\text{prox}_{\tau |\cdot - w_{\text{prev},k}|}(z_k - \nu^\star), l_k, u_k)$ via 1D dual bisection with O'Donoghue-Candes gradient restarts and reports the exact proximal-gradient KKT residual.

### Strictly Causal Embargo & Asset-Cost Walk-Forward (`combine_alphas_walkforward`)
When forecasting $h$-bar forward returns with execution lag $d$, realized observations from the most recent $\text{embargo} = \max(0, h + d - 1)$ periods overlap future prices. `combine_alphas_walkforward` enforces this embargo automatically (`end_excl = idx - embargo`), generating live portfolio weights on the latest bar even when recent forward returns are `null` with zero prefix-invariance leakage.

Key execution-aware controls:

- **`objective_mode`**: Optimize cross-sectional Rank/Linear ICs (`"ic"`) or realized dollar-neutral factor portfolio returns (`"factor_return"`).
- **`one_way_cost_bps` + `asset_col`**: Deducts each factor's historical stock-level rebalancing drag $c \cdot \bar{\tau}_{k,t}$ before Empirical Bayes shrinkage and FISTA optimization, shifting capital away from high-churn signals toward persistent alphas.
- **`signal_ema_alpha`**: Applies identical causal per-asset exponential smoothing across estimation, `composite_alpha`, and `equal_weight_alpha` (isolating the pure effect of FISTA + RMT while preserving un-mutated input factor columns in `panel`).
- **`normalize_composite` & `min_weight_norm`**: Cross-sectionally Z-scores `composite_alpha` when `normalize_composite=True` (default), automatically outputting `0.0` whenever $\sum_k |w_{k,t}| \le \text{min\_weight\_norm}$ (`1e-8`) to prevent numerical noise amplification during cash/flat regimes.

```python
import numpy as np
import polars as pl
import polars_anta as pa

rng = np.random.default_rng(42)
n_dates, n_tickers = 120, 40
panel = pl.DataFrame(
    {
        "date": np.repeat(np.arange(n_dates), n_tickers),
        "ticker": [f"EQ{i:02d}" for i in np.tile(np.arange(n_tickers), n_dates)],
        "sector": [
            ["Tech", "Fin", "Health", "Ind"][i % 4] for i in np.tile(np.arange(n_tickers), n_dates)
        ],
        "beta": rng.uniform(0.7, 1.3, size=n_dates * n_tickers),
        "close": 100.0 + np.cumsum(rng.normal(0, 1.0, size=n_dates * n_tickers)),
        "volume": rng.uniform(1e5, 1e6, size=n_dates * n_tickers),
        "a1": rng.standard_normal(n_dates * n_tickers),
        "a2": rng.standard_normal(n_dates * n_tickers),
    }
).anta.forward_returns(close_col="close", periods=(1,), group_by="ticker", prefix="fwd_")

ortho = (
    panel.with_columns([pl.col(c).anta.cs_gaussianize(over="date").alias(c) for c in ("a1", "a2")])
    .anta.neutralize_panel(["a1", "a2"], time_col="date", risk_cols=["beta"], group_col="sector")
    .anta.orthogonalize_panel(["a1_neut", "a2_neut"], time_col="date", method="lowdin")
)

wf = ortho.anta.combine_alphas_walkforward(
    factor_cols=["a1_neut_ortho", "a2_neut_ortho"],
    return_col="fwd_1",
    time_col="date",
    asset_col="ticker",
    lookback=40,
    horizon=1,
    signal_lag=1,
    objective_mode="factor_return",
    one_way_cost_bps=10.0,
    signal_ema_alpha=0.25,
    turnover_penalty=0.01,
    constraint="box",
    lower_bound=-1.0,
    upper_bound=1.0,
    cov_estimator="marchenko_pastur",
    ic_shrinkage="empirical_bayes",
)
print(wf.summary())
```
