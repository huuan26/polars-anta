# Cross-Sectional Alpha Engineering - Mathematical & Ablation Validation

Generated automatically by `benchmarks/alpha_engineering_validation.py`.

## Environment

- **Python**: `3.14.1` (`Windows 11`)
- **Polars**: `2.1.0` | **NumPy**: `2.5.4` | **polars-anta**: `0.5.0`
- **Random Seed**: `202605` | **Total Benchmark Runtime**: `2.16 s`

## Table 1 - Exact Algebraic, KKT & Causal Prefix-Invariance Checks

| Mathematical Invariant | Observed Error / Residual | Acceptance Bound |
| :--- | :---: | :---: |
| FWL Simultaneous Sector + Continuous Risk Exposure $\max |\mathbf{X}_t^T \mathbf{F}_{\text{neut},t}|$ | `5.72e-13` | `< 1.00e-10` |
| Batched Lowdin Orthogonality $\|\widetilde{\mathbf{F}}_t^T \widetilde{\mathbf{F}}_t / (N_t - 1) - \mathbf{I}_K\|_F$ | `4.14e-15` | `< 1.00e-10` |
| Lowdin Permutation Equivariance $\max |\widetilde{\mathbf{F}}_{\pi} - \widetilde{\mathbf{F}}\mathbf{P}|$ | `3.55e-15` | `< 1.00e-11` |
| Joint Box-Simplex + $L_1$ FISTA KKT Stationarity Residual | `5.09e-11` | `< 1.00e-07` |
| Causal Embargo Prefix-Invariance Weight Leakage ($h=5, d=1$) | `0.00e+00` | `== 0.00e+00` |

## Table 2 - Covariance Conditioning Monte Carlo ($T = 80, K = 24, R = 100$)

| Estimator | Mean Frobenius Error vs. $\mathbf{C}_{\text{true}}$ | SD(Error) | Mean Condition Number $\kappa_2(\widehat{\mathbf{C}})$ |
| :--- | :---: | :---: | :---: |
| Sample | `2.466` | `0.285` | `44.68` |
| Ledoit-Wolf | `2.331` | `0.336` | `18.36` |
| Marchenko-Pastur (RMT) | `2.093` | `0.332` | `12.26` |

## Table 3 - Dollar-Neutral Long-Short Walk-Forward Ablation ($N = 150, T = 600$, `10 bps` Cost)

| Pipeline Stage | Rank IC | ICIR | Daily L1 Turnover | Gross Margin (`bps`) | Gross Sharpe | Net Sharpe (`10 bps`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0. Raw Signals (Equal-Weight) | `+0.0288` | `+0.222` | `1.218` | `+5.33` | `+3.57` | `-3.13` |
| 1. FWL + Lowdin Ortho (Equal-Weight) | `+0.0311` | `+0.436` | `1.128` | `+6.50` | `+7.75` | `-4.16` |
| 2. Unpenalized Walk-Forward (Sample Cov) | `+0.0269` | `+0.372` | `1.151` | `+5.66` | `+6.77` | `-5.16` |
| 3. Full Pipeline (RMT + EB + FISTA + Decay) | `+0.0397` | `+0.541` | `0.409` | `+22.61` | `+9.73` | `+5.42` |
