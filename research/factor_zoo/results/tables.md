# Generated Factor Zoo Tables (Auto-rendered from JSON/CSV)

## Funnel Table

| Stage | Signals |
| :--- | ---: |
| Tested (pre-registered registry) | 150 |
| Raw p < 0.05 (Stationary Bootstrap / Newey–West) | 91 *(≈ 7.5 expected by chance)* |
| Benjamini–Hochberg q ≤ 0.10 | 94 |
| Romano–Wolf FWER ≤ 0.05 | 71 (64 non-decaying `robust`) |
| Survive costs (long-only, VN fees & tax, BH q ≤ 0.10) | 0 (2 at raw p ≤ 0.05) |
| Confirmed in locked holdout (2022-01-01–2026-11-10) | 0 (38 statistical IC survivors) |

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
