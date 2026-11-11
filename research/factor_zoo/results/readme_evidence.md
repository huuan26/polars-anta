To demonstrate why multiple-testing correction and transaction-cost accounting are mandatory in quantitative research, we executed a pre-specified analysis plan ([`PREREGISTRATION.md`](research/factor_zoo/PREREGISTRATION.md), commit `4c1a4b559d22a1367ce46b513ab7ff1079fff360`, registry SHA-256 `8aa3c11b97810083b875084d38cfda6df1994de38ec9b077faeaf46dc31c948c`; see [Deviations & Timeline](research/factor_zoo/REPORT.md#31-deviations-from-the-analysis-plan--timeline)) testing all **150** daily-OHLCV-compatible signals (**146** unique after accounting for **4** exact duplicate pairs) in `polars-anta` at their **default parameters** on the point-in-time top-150 liquid Ho Chi Minh Stock Exchange (`HOSE`) equity universe (`2014-01-01`–`2021-12-31` discovery, `2022-01-01`–`2026-11-10` locked holdout, confirmatory horizon $h = 5$ open-to-open trading days):

#### Track A — Statistical Predictability Funnel (Cross-Sectional Rank IC, $h = 5$ Days)

| Stage | Signals | Share of Registry | Notes |
| :--- | ---: | ---: | :--- |
| **A1. Tested in Registry** | **150** | `100.0%` | 146 unique signals + 4 exact duplicate pairs ($N_{\text{eff}} = 6.00$) |
| **A2. Benjamini–Hochberg FDR $q \le 0.10$** | **94** | `62.7%` | Raw Stationary Bootstrap $p \le 0.05$: **91** (NW $|t| \ge 1.96$: **95**; $\approx 7.5$ under i.i.d. noise) |
| **A3. Romano–Wolf Stepdown FWER $p_{\text{RW}} \le 0.05$** | **71** | `47.3%` | Joint studentized $\max$-$t$ bootstrap across all 150 IC series |
| **A4. Structurally Stable (`robust` Verdict)** | **64** | `42.7%` | RW $p \le 0.05$ and non-decaying (+ **16** `fdr_only`, **14** `decaying`, **56** `noise`) |
| **A5. Replicate in Locked Holdout (`2022-01-01`–`2026-11-10`)** | **38** | `25.3%` | **38** of **64** (`59.4%`) pass Holdout RW $p \le 0.05$ (**49** or `76.6%` pass BH $q \le 0.10$; **54** or `84.4%` retain sign) |

#### Track B — Economic Implementability Funnel (Long-Only Q5 vs. Equal-Weight Universe, HOSE Costs)

| Stage | Signals | Share of Registry | Share of 80 Candidates |
| :--- | ---: | ---: | ---: |
| **B1. Non-Decaying Discovery Survivors Evaluated** | **80** | `53.3%` | `100.0%` (**64** `robust` + **16** `fdr_only`) |
| **B2. Positive Net Excess Return After HOSE Costs (`60 bps` RT)** | **12** | `8.0%` | `15.0%` |
| **B3. Unadjusted One-Sided Bootstrap $p \le 0.05$ on Net Excess** | **2** | `1.3%` | `2.5%` (`trix_15`, `tsi`) |
| **B4. Survive Benjamini–Hochberg FDR $q \le 0.10$ Across Cost Stage** | **0** | `0.0%` | `0.0%` |
| **B5. Confirmed Post-Cost Survivors in Locked Holdout** | **0** | `0.0%` | `0.0%` |

<div align="center">

<img src="research/factor_zoo/results/figures/01_falsification_funnel.png" alt="Factor Zoo Falsification Funnel on HOSE" width="860" />

</div>

- **Why 91 of 150 pass raw $p \le 0.05$**: Spectral decomposition of the $150 \times 150$ daily Spearman IC correlation matrix shows an eigenvalue participation ratio of **$N_{\text{eff}} = 6.00$** effective independent tests ($\lambda_1 = 55.21$; the top 5 principal components explain **66.6%** of cross-signal IC variance).
- **Predictability vs. Standalone Implementability**: In the locked `2022–2026` holdout, **38 of the 64** non-decaying `robust` discovery signals (`59.4%`) replicate their cross-sectional Rank IC under Romano–Wolf FWER control ($p_{\text{RW}} \le 0.05$, with **97.6%** median IC retention and $r = 0.77$ discovery-vs-holdout IC correlation across all 150 signals). However, when traded as standalone 5-day rebalanced long-only top-quintile (Q5) portfolios against the equal-weight universe under retail `HOSE` execution frictions (`60 bps` round-trip brokerage, slippage & sell tax), the top-6 $|t_{\text{NW}}|$ fast reversal signals face `63.8%–87.2%` one-way turnover every 5 days (`-6.7–20.5 bps` breakeven cost, or `5.7–20.5 bps` across the 5 with positive gross Q5 excess), while the only 2 slower momentum signals passing unadjusted $p \le 0.05$ (`trix_15`, `tsi`) fail BH correction ($q = 1.00$) and decay in holdout (illustrative, $n = 2$).
- **Exploratory Post-Holdout Diagnostics ([`REPORT.md` §8.3](research/factor_zoo/REPORT.md#83-exploratory-post-holdout-analyses))**: Orthogonalizing all 150 signals against 5-day return reversal (`rev_5_rank`) raises $N_{\text{eff}}$ from **6.00** to **9.37** and leaves **91** `robust` discovery signals (**40** replicating in Holdout under RW $p \le 0.05$); Q1–Q5 decomposition shows that **65.5%** of the top-10 signals' $\text{Q5} - \text{Q1}$ spread in Discovery (**90.0%** in Holdout) originates in Q1 short-leg underperformance; and **0** standalone 5-day Q5 strategies survive BH $q \le 0.10$ at any cost $\ge 30\text{ bps}$ round-trip in Discovery or $\ge 10\text{ bps}$ in Holdout. Read the full study in [**`research/factor_zoo/REPORT.md`**](research/factor_zoo/REPORT.md).
