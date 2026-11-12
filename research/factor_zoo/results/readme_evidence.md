We ran every daily signal in `polars-anta` (**150** signals, **146** unique) through the library's own falsification pipeline on the most liquid stocks of the Ho Chi Minh Stock Exchange (`HOSE`). The analysis plan was committed to Git before any results were computed, and the `2022-2026` holdout was evaluated only after the discovery results were committed.

> **Why Vietnam?** `HOSE` is a retail-driven emerging market with daily price limits, no short selling for most investors and high trading costs. It is far less studied than US equities, which makes it a demanding out-of-sample test for signals that were designed elsewhere.

| Question | Result |
| :--- | ---: |
| Signals tested | **150** |
| Effectively independent tests among them | $N_{\text{eff}} = 6.00$ |
| Significant after multiple-testing correction (Romano-Wolf, 5% FWER) | **71** |
| Significant and stable over the discovery period (`2014-2021`) | **64** |
| Still significant in the locked holdout (`2022-2026`) | **38** of 64 |
| Significant long-only excess return after `60 bps` round-trip costs | **0** |

<div align="center">

<img src="research/factor_zoo/results/figures/01_falsification_funnel.png" alt="Factor Zoo Falsification Funnel on HOSE" width="860" />

</div>

**What this means**

- **Most signals are variations of the same effect.** 91 of 150 signals pass a naive $p \le 0.05$ test, but their daily ICs move together so closely that they amount to only $N_{\text{eff}} = 6.00$ independent tests. Much of this common component is short-term (5-day) reversal: removing it raises $N_{\text{eff}}$ from 6.00 to 9.37.
- **The predictive power holds up out of sample.** 38 of the 64 stable discovery signals remain significant in the holdout, with a median IC retention of 97.6%.
- **But it does not pay for itself.** The strongest signals replace 63.8% to 87.2% of the portfolio every 5 days and break even at 20.5 bps round-trip at best, well below realistic costs. In addition, 65.5% of the top-10 signals' long-short spread in discovery (90.0% in holdout) comes from the short leg, which long-only investors cannot capture.

Read the full study in [`research/factor_zoo/REPORT.md`](research/factor_zoo/REPORT.md).

<details markdown="1">
<summary>Study design, reproducibility and full results tables</summary>

| Design choice | Value |
| :--- | :--- |
| Universe | Point-in-time top-150 most liquid `HOSE` stocks |
| Signals | 150 signals at default parameters (146 unique; 4 exact duplicate pairs disclosed) |
| Discovery sample | `2014-01-01` to `2021-12-31` |
| Locked holdout | `2022-01-01` to `2026-11-10` |
| Forecast horizon | 5 trading days, open to open |
| Execution costs | `60 bps` round-trip (brokerage, slippage and sell tax) |
| Analysis plan | [`PREREGISTRATION.md`](research/factor_zoo/PREREGISTRATION.md), committed in [`4c1a4b5`](https://github.com/huuan26/polars-anta/commit/4c1a4b559d22a1367ce46b513ab7ff1079fff360) |
| Signal registry checksum | SHA-256 recorded in `PREREGISTRATION.md` and verified by the pipeline |
| Deviations from the plan | [Deviations and Timeline](research/factor_zoo/REPORT.md#31-deviations-from-the-analysis-plan--timeline) |

**Track A: statistical predictability** (cross-sectional Rank IC, 5-day horizon)

| Stage | Signals | Share | Notes |
| :--- | ---: | ---: | :--- |
| A1. Tested | **150** | `100.0%` | 146 unique signals |
| A2. Benjamini-Hochberg FDR $q \le 0.10$ | **94** | `62.7%` | Raw bootstrap $p \le 0.05$: **91** ($\approx 7.5$ expected by chance) |
| A3. Romano-Wolf FWER $p \le 0.05$ | **71** | `47.3%` | Joint bootstrap across all 150 IC series |
| A4. Stable (`robust` verdict) | **64** | `42.7%` | Plus **16** `fdr_only`, **14** `decaying`, **56** `noise` |
| A5. Replicate in holdout | **38** | `25.3%` | **38** of **64** pass holdout RW $p \le 0.05$; **54** keep their sign |

**Track B: economic implementability** (long-only top quintile vs. equal-weight universe)

| Stage | Signals | Share of 80 Candidates |
| :--- | ---: | ---: |
| B1. Candidates evaluated (`robust` + `fdr_only`) | **80** | `100.0%` |
| B2. Positive net excess return after costs | **12** | `15.0%` |
| B3. Unadjusted one-sided $p \le 0.05$ | **2** | `2.5%` (`trix_15`, `tsi`) |
| B4. Benjamini-Hochberg FDR $q \le 0.10$ | **0** | `0.0%` |
| B5. Confirmed in holdout | **0** | `0.0%` |

A cost sweep from 0 to 60 bps round-trip finds **0** significant long-only strategies at $\ge 30$ bps in discovery and at $\ge 10$ bps in holdout ([`REPORT.md` Section 8.3](research/factor_zoo/REPORT.md#83-exploratory-post-holdout-analyses)).

</details>
