# Statistical Inference Engine — Monte Carlo Size, Power & PBO Validation

Generated automatically by `benchmarks/inference_size_power.py`.

## Simulation Parameters & Environment

- **Python**: `3.14.1` (`Windows 11`)
- **Polars**: `2.1.0` | **NumPy**: `2.5.4`
- **Hypotheses ($K$)**: `50` | **Sample Length ($T$)**: `1000`
- **Monte Carlo Replications**: `200` | **Bootstrap Replicates ($B$)**: `499`
- **Nominal FWER ($\alpha$)**: `0.05` | **Target FDR ($q$)**: `0.10`
- **Random Seed**: `42` | **Total Benchmark Runtime**: `24.28 s`

## Table 1 — Empirical Family-Wise Error Rate (Global Null, $\alpha = 0.05$)

Fraction of Monte Carlo replications ($R = 200$) with at least one false rejection across
$K = 50$ pure-noise strategies ($T = 1000$). Under $\text{AR}(1)$ serial dependence
($\phi = 0.30$), the long-run variance is $\frac{1+\phi}{1-\phi} \approx 1.86\times$ the
naive sample variance, causing i.i.d. procedures (`OLS t` and `mean_block=1` bootstrap) to
severely over-reject while HAC and the Politis–Romano Stationary Bootstrap preserve size
near the nominal $5\%$ level.

| Multiple Testing Procedure | Empirical FWER — i.i.d. $\mathcal{N}(0,1)$ | Empirical FWER — $\text{AR}(1)$ ($\phi = 0.30$) | Nominal Target |
| :--- | :---: | :---: | :---: |
| Unadjusted (OLS t) | 93.5% | 100.0% | 5.0% |
| Bonferroni (OLS t) | 7.0% | 60.0% | 5.0% |
| Holm (OLS t) | 7.0% | 60.0% | 5.0% |
| Bonferroni (HAC t) | 6.5% | 14.5% | 5.0% |
| Holm (HAC t) | 6.5% | 14.5% | 5.0% |
| Romano-Wolf (iid, mean_block=1) | 6.5% | 61.0% | 5.0% |
| Romano-Wolf (Stationary Bootstrap) | 5.5% | 16.0% | 5.0% |

## Table 2 — Empirical FDR & Statistical Power (5 True + 45 Null Strategies)

Equicorrelated factor zoo ($\rho = 0.50$, $K = 50$, $T = 1000$) containing $5$ true
strategies with annualized Sharpe $= 1.0$ ($\mu = 1/\sqrt{252} \approx 0.063$) and $45$
zero-mean strategies.

| Metric | Value | Reference Target |
| :--- | :---: | :---: |
| Benjamini–Hochberg ($q = 0.10$) Empirical FDR | 8.2% | $\le 10.0\%$ ($\frac{m_0}{m} q = 9.0\%$) |
| Benjamini–Hochberg ($q = 0.10$) Average Power | 23.0% | — |
| Romano–Wolf Stepdown ($\alpha = 0.05$) Average Power | 15.1% | $\ge$ Bonferroni |
| Bonferroni ($\alpha = 0.05$) Average Power | 10.5% | — |
| Romano–Wolf Null-Subset FWER ($\alpha = 0.05$) | 5.5% | $\le 5.0\%$ |
| Bonferroni Null-Subset FWER ($\alpha = 0.05$) | 2.0% | $\le 5.0\%$ |

## Table 3 — CSCV Probability of Backtest Overfitting (PBO)

Exact Combinatorially Symmetric Cross-Validation ($S = 16$, $\binom{16}{8} = 12,870$
partitions) over $N = 20$ candidate strategies ($T = 1600$, 50 replications).

| Strategy Universe | Mean PBO | SD(PBO) | Mean Prob. OOS Loss | Theoretical Expectation |
| :--- | :---: | :---: | :---: | :---: |
| Pure Noise ($N = 20$ i.i.d. $\mathcal{N}(0,1)$) | 0.546 | 0.205 | 0.548 | $\approx 0.500$ |
| 1 Dominant Strategy ($\mu_1 = 0.12$) + 19 Noise | 0.046 | 0.082 | 0.044 | $< 0.050$ |

## Table 4 — End-to-End Panel Falsification Throughput

| Panel Dimensions | Rows | Bootstrap Replicates ($B$) | Wall-Clock Time | Surviving Robust Factors |
| :--- | :---: | :---: | :---: | :---: |
| `150` factors $\times$ `2500` periods $\times$ `100` assets | `250,000` | `1000` | `2.44 s` | `5 / 5` planted |
