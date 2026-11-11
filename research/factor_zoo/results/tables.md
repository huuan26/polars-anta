# Generated Factor Zoo Tables (Auto-rendered from JSON/CSV)

## Funnel Table (Two-Track Falsification)

### Track A — Statistical Predictability Funnel (Cross-Sectional Rank IC, $h = 5$ Days)

| Stage | Signals | Share of Registry | Notes |
| :--- | ---: | ---: | :--- |
| **A1. Tested in Registry** | **150** | `100.0%` | 146 unique signals + 4 exact duplicate pairs ($N_{\text{eff}} = 6.00$) |
| **A2. Benjamini–Hochberg FDR $q \le 0.10$** | **94** | `62.7%` | Raw Stationary Bootstrap $p \le 0.05$: **91** (NW $|t| \ge 1.96$: **95**; $\approx 7.5$ under i.i.d. noise) |
| **A3. Romano–Wolf Stepdown FWER $p_{\text{RW}} \le 0.05$** | **71** | `47.3%` | Joint studentized $\max$-$t$ bootstrap across all 150 IC series |
| **A4. Structurally Stable (`robust` Verdict)** | **64** | `42.7%` | RW $p \le 0.05$ and non-decaying (+ **16** `fdr_only`, **14** `decaying`, **56** `noise`) |
| **A5. Replicate in Locked Holdout (`2022-01-01`–`2026-11-10`)** | **38** | `25.3%` | **38** of **64** (`59.4%`) pass Holdout RW $p \le 0.05$ (**49** or `76.6%` pass BH $q \le 0.10$; **54** or `84.4%` retain sign) |

### Track B — Economic Implementability Funnel (Long-Only Q5 vs. Equal-Weight Universe, HOSE Costs)

| Stage | Signals | Share of Registry | Share of 80 Candidates |
| :--- | ---: | ---: | ---: |
| **B1. Non-Decaying Discovery Survivors Evaluated** | **80** | `53.3%` | `100.0%` (**64** `robust` + **16** `fdr_only`) |
| **B2. Positive Net Excess Return After HOSE Costs (`60 bps` RT)** | **12** | `8.0%` | `15.0%` |
| **B3. Unadjusted One-Sided Bootstrap $p \le 0.05$ on Net Excess** | **2** | `1.3%` | `2.5%` (`trix_15`, `tsi`) |
| **B4. Survive Benjamini–Hochberg FDR $q \le 0.10$ Across Cost Stage** | **0** | `0.0%` | `0.0%` |
| **B5. Confirmed Post-Cost Survivors in Locked Holdout** | **0** | `0.0%` | `0.0%` |

## Family Breakdown (Discovery 2014–2021, h = 5)

| Family | Tested | Raw p ≤ 0.05 | BH q ≤ 0.10 | RW p ≤ 0.05 | Robust | FDR-Only | Decaying | Median |IC| | Median |Posterior IC| |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trend` | 28 | 18 | 19 | 16 | 16 | 3 | 0 | 0.0236 | 0.0220 |
| `momentum` | 22 | 15 | 15 | 9 | 9 | 6 | 0 | 0.0158 | 0.0146 |
| `volatility` | 20 | 12 | 12 | 9 | 3 | 0 | 9 | 0.0183 | 0.0172 |
| `volume` | 15 | 8 | 8 | 4 | 4 | 4 | 0 | 0.0109 | 0.0104 |
| `stats` | 13 | 7 | 7 | 5 | 5 | 0 | 2 | 0.0102 | 0.0098 |
| `microstructure` | 5 | 2 | 2 | 1 | 1 | 0 | 1 | 0.0114 | 0.0105 |
| `alpha` | 47 | 29 | 31 | 27 | 26 | 3 | 2 | 0.0162 | 0.0155 |

## Top Positive Net Excess Return Signals in Cost Stage (Discovery 2014–2021)

| Factor | Verdict | Gross Excess (5d) | L1 Turnover (5d) | Net Excess (% Ann.) | Net IR (Ann.) | Raw Bootstrap p | BH q | Breakeven Cost (Round-Trip bps) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

### Cost-Stage Descriptive Ranges (Auto-derived from `cost_stage.csv`)

- **Top-6 $|t_{\text{NW}}|$ fast reversal signals** (`returns_divergence_14, hma_10, tema_10, zlema_10, true_range_norm, alpha_018`): 5-day L1 turnover `1.28–1.74` (`63.8%–87.2%` one-way replacement), annualized cost drag `19.3%–26.4%/yr`, breakeven round-trip cost `-6.7–20.5 bps` (`5.7–20.5 bps` across the 5 positive-gross-excess signals; `alpha_018` has negative long-only Q5 gross excess `-0.0005` and breakeven `-6.7 bps`).
- **All 68 negative-net-excess candidates**: 5-day L1 turnover `0.27–1.83` (`13.5%–91.4%` one-way), annualized cost drag `4.1%–27.6%/yr`, breakeven round-trip cost `-23.8–53.1 bps`.
- **All 12 positive-net-excess candidates**: 5-day L1 turnover `0.28–0.79` (`14.1%–39.5%` one-way), annualized cost drag `4.3%–11.9%/yr`, breakeven round-trip cost `68.1–129.8 bps`.

| `trix_15` | `robust` | 0.0031 | 0.48 | +8.37% | 0.71 | 0.0170 | 1.00 | 129.8 |
| `tsi` | `robust` | 0.0032 | 0.58 | +7.65% | 0.70 | 0.0320 | 1.00 | 112.7 |
| `kst` | `robust` | 0.0026 | 0.49 | +5.84% | 0.51 | 0.0555 | 1.00 | 107.1 |
| `awesome_osc` | `robust` | 0.0030 | 0.68 | +4.67% | 0.41 | 0.1069 | 1.00 | 87.4 |
| `frac_diff_04` | `robust` | 0.0026 | 0.56 | +4.61% | 0.50 | 0.1439 | 1.00 | 92.5 |
| `dpo_20` | `fdr_only` | 0.0030 | 0.70 | +4.56% | 0.39 | 0.1414 | 1.00 | 86.0 |
| `tii_30` | `robust` | 0.0022 | 0.57 | +2.57% | 0.33 | 0.1949 | 1.00 | 77.9 |
| `adx_14` | `fdr_only` | 0.0022 | 0.62 | +1.90% | 0.18 | 0.3003 | 1.00 | 72.1 |
| `coppock` | `robust` | 0.0027 | 0.78 | +1.79% | 0.16 | 0.3238 | 1.00 | 69.1 |
| `nvi_roc_20` | `robust` | 0.0027 | 0.79 | +1.64% | 0.18 | 0.3048 | 1.00 | 68.2 |
| `squeeze_mom` | `robust` | 0.0026 | 0.78 | +1.58% | 0.14 | 0.3373 | 1.00 | 68.1 |
| `alpha_005` | `robust` | 0.0011 | 0.28 | +1.21% | 0.14 | 0.3923 | 1.00 | 76.9 |

## Top 12 Discovery Survivors & Locked Holdout Replication (2022–2026, h = 5)

| Factor | Family | Disc IC | Posterior IC | Disc NW t | Disc RW p | Breakeven (RT bps) | Holdout IC | Holdout NW t | Holdout RW p | Retention Ratio | CUSUM p |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `returns_divergence_14` | `alpha` | -0.0276 | -0.0272 | -12.04 | 0.0005 | 5.7 | -0.0146 | -4.46 | 0.0010 | 0.53 | 0.004 |
| `hma_10` | `trend` | -0.0406 | -0.0392 | -11.01 | 0.0005 | 9.4 | -0.0387 | -6.26 | 0.0005 | 0.95 | 0.757 |
| `tema_10` | `trend` | -0.0421 | -0.0405 | -10.92 | 0.0005 | 20.5 | -0.0355 | -5.34 | 0.0005 | 0.84 | 0.656 |
| `zlema_10` | `trend` | -0.0402 | -0.0388 | -10.86 | 0.0005 | 9.9 | -0.0374 | -6.10 | 0.0005 | 0.93 | 0.506 |
| `true_range_norm` | `volatility` | -0.0472 | -0.0450 | -10.71 | 0.0005 | 17.8 | -0.0486 | -5.15 | 0.0005 | 1.03 | 0.538 |
| `alpha_018` | `alpha` | 0.0327 | 0.0319 | 10.59 | 0.0005 | -6.7 | 0.0453 | 9.92 | 0.0005 | 1.38 | 0.002 |
| `pivot_classic` | `trend` | -0.0370 | -0.0358 | -10.19 | 0.0005 | 6.1 | -0.0376 | -6.37 | 0.0005 | 1.02 | 0.627 |
| `pivot_fibonacci` | `trend` | -0.0370 | -0.0358 | -10.19 | 0.0005 | 6.1 | -0.0376 | -6.37 | 0.0005 | 1.02 | 0.627 |
| `pivot_woodie` | `trend` | -0.0360 | -0.0349 | -10.18 | 0.0005 | 8.3 | -0.0366 | -6.33 | 0.0005 | 1.01 | 0.657 |
| `alpha_023` | `alpha` | 0.0310 | 0.0301 | 9.47 | 0.0005 | 12.3 | 0.0332 | 6.83 | 0.0005 | 1.07 | 0.330 |
| `tsi` | `momentum` | 0.0369 | 0.0338 | 6.24 | 0.0005 | 112.7 | -0.0011 | -0.10 | 1.0000 | -0.03 | 0.009 |
| `trix_15` | `momentum` | 0.0313 | 0.0284 | 4.95 | 0.0010 | 129.8 | -0.0089 | -0.75 | 0.9930 | -0.28 | 0.013 |

## Appendix A. Excluded Library Exports (`10` Export Groups)

| Export / Function Group | Family / Module | Exclusion Reason |
| :--- | :--- | :--- |
| `ofi` | `microstructure` | Requires L1 order-book bid_price, bid_size, ask_price, ask_size. |
| `futures_basis` | `microstructure` | Requires derivatives futures_close, spot_close, and days_to_expiry. |
| `oi_regime` | `microstructure` | Requires futures open_interest series not present in cash equities. |
| `session_vwap` | `volume` | Intraday session-anchored VWAP requiring sub-daily bars. |
| `rolling_beta` | `stats` | Bivariate rolling regression requiring an external benchmark return column. |
| `rolling_alpha` | `stats` | Bivariate CAPM intercept requiring an external benchmark return column. |
| `cs_rank / cs_zscore / cs_winsorize / neutralize / orthogonalize` | `alpha` | Cross-sectional conditioning operators (not standalone time-series signals). |
| `ts_scale / ts_delta / ts_delay / signed_power / ts_min / ts_max / ts_sum / ts_std / ts_product / ts_correlation / ts_covariance` | `alpha` | Low-level algebraic DSL building blocks used inside alpha_001..alpha_101. |
| `resample_* / stitch_futures_roll / validate_ohlcv` | `resample` | Data engineering, bar aggregation, and validation utilities. |
| `evaluation.* / backtest.* / inference.*` | `evaluation/backtest/inference` | Target return generators, evaluation metrics, backtesting engine, and inference tests. |
