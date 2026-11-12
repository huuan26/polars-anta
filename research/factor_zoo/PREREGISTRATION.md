# Pre-Registration: Testing the Technical & Formulaic Factor Zoo on Vietnamese Equities (HOSE)

- **Study Title**: *How Many Technical and Formulaic Alphas Survive Multiple-Testing and Transaction-Cost Falsification on Vietnamese Equities?*
- **Pre-Registration Date**: `2026-11-10`
- **Signal Registry Count ($K$)**: `150`
- **Signal Registry SHA-256**: `8aa3c11b97810083b875084d38cfda6df1994de38ec9b077faeaf46dc31c948c`
- **Frozen Configuration File**: [`research/factor_zoo/config.py`](config.py)
- **Frozen Registry File**: [`research/factor_zoo/registry.py`](registry.py)

---

## 1. Primary Null Hypothesis ($H_0$)

We test the joint composite null hypothesis across all $K = 150$ daily-OHLCV-compatible signals shipped in `polars-anta`:

$$
H_{0,k}: \mathbb{E}\!\left[\text{IC}^{\text{Spearman}}_{k,t}(h=5)\right] = 0 \qquad \forall\, k \in \{1, \dots, 150\}
$$

and the economic null hypothesis that any signal surviving statistical multiple-testing corrections fails to generate positive net excess returns over the equal-weighted investable universe after realistic Ho Chi Minh Stock Exchange (`HOSE`) transaction costs, securities transfer tax, and execution slippage under a long-only constraint.

---

## 2. Market Data & Point-in-Time Universe Specification

1. **Market & Frequency**: Common equities listed on the Ho Chi Minh Stock Exchange (`HOSE`), daily OHLCV bars (`warmup_start = 2012-01-01`, `discovery_start = 2014-01-01`).
2. **Pre-Registered Data Cleaning (`research/factor_zoo/data.py`)**:
   - **Step 1 (OHLCV Structural Validation)**: Audit via `polars_anta.validate_ohlcv` and remove any row violating $H_t \ge \max(O_t, C_t, L_t)$, $L_t \le \min(O_t, C_t, H_t)$, $O_t, H_t, L_t, C_t > 0$, or $V_t \ge 0$.
   - **Step 2 (Non-Trading Sessions)**: Remove sessions with `volume <= 0` prior to indicator evaluation.
   - **Step 3 (Price-Limit Jump Filter)**: Remove daily close-to-close return observations exceeding the HOSE $\pm 7.0\%$ daily price limit plus a $0.5\%$ tolerance ($|r_t| > 7.5\%$) to guard against unadjusted corporate-action discontinuities or feed errors.
3. **Point-in-Time Investable Universe (`in_universe`)**:
   On each trading day $t$, a stock $i$ enters the cross-sectional universe if and only if (using strictly information available up to close of day $t$):
   - Cumulative trading history up to day $t$ is at least **250 sessions**,
   - Closing price $C_{i,t} \ge \mathbf{5{,}000\text{ VND}}$,
   - 60-session rolling median daily traded value $\text{Median}_{s \in [t-59, t]}(C_{i,s} \cdot V_{i,s})$ ranks in the **top 150** among eligible stocks on day $t$.

---

## 3. Signal Registry & Scale-Free Normalization

- **Coverage**: All 150 daily-OHLCV-compatible indicator, microstructure, statistical, and formulaic alpha primitives in `polars-anta`, evaluated at their **default library parameters** (zero hyperparameter tuning):
  - `trend`: 28 signals
  - `momentum`: 22 signals
  - `volatility`: 20 signals
  - `volume`: 15 signals
  - `stats`: 13 signals
  - `microstructure`: 5 signals
  - `alpha`: 47 signals (14 time-series/regime primitives + 33 WorldQuant formulaic alphas)
- **Mandatory Scale-Free Rule**: Price-level overlays (moving averages, bands, stops, pivots) are mapped to dimensionless ratios relative to `close` (e.g., `close / sma(close, 10) - 1`, `percent_b`, `(close - supertrend) / close`). Multi-line `pl.Struct` outputs select one primary component as documented in `registry.py`.
- **Cross-Sectional Standardization**: After computing each raw signal per ticker (`.over("ticker")`), signals are standardized cross-sectionally within the point-in-time universe on each date $t$ via percentile ranking `cs_rank(..., over="date")`.
- **Two-Sided Testing**: All IC tests are two-sided (`alternative="two-sided"`), avoiding ex-ante sign assumptions or double-counting flipped signals.

---

## 4. Pre-Registered Statistical & Economic Design

| Design Element | Pre-Registered Specification |
| :--- | :--- |
| **Target Return ($r_{t,h}$)** | Next-day **open-to-open** return $r_{t,h} = \frac{\text{open}_{t+1+h}}{\text{open}_{t+1}} - 1$ (signal observed at $\text{close}_t$, executed at $\text{open}_{t+1}$; zero look-ahead) |
| **Confirmatory Horizon** | **$h = 5$** trading days ($h = 1$ and $h = 20$ are reported strictly as exploratory) |
| **Discovery Sample** | `2014-01-01` through `2021-12-31` |
| **Locked Holdout Sample** | `2022-01-01` through end of sample (`2026-11-10`) |
| **Primary Metric** | Daily cross-sectional Spearman Rank IC (`falsify_factors(..., method="spearman", horizon=5)`) |
| **Autocorrelation & Resampling** | Newey–West HAC lag $L = 4$; Politis–Romano Stationary Bootstrap with $\text{mean\_block} = \max(5, T^{1/3})$, $B = 2{,}000$, `seed = 20261110` |
| **Multiple-Testing Control** | Benjamini–Hochberg FDR at $q = 0.10$ and Romano–Wolf stepdown FWER at $\alpha = 0.05$ jointly across all $K = 150$ signals |
| **Effect-Size Shrinkage** | Efron–Morris / Jensen–Kelly–Pedersen Empirical-Bayes shrinkage with `prior="zero"` |
| **Effective Independent Tests ($N_{\text{eff}}$)** | Eigenvalue participation ratio $N_{\text{eff}} = \frac{(\sum_{k=1}^{K} \lambda_k)^2}{\sum_{k=1}^{K} \lambda_k^2}$ of the discovery IC correlation matrix |
| **Transaction-Cost Stage** | Evaluated on all discovery survivors (`verdict in {"robust", "fdr_only"}`): long-only top-quintile (Q5, oriented by discovery $\text{sign}(\overline{\text{IC}}_k)$), equal-weighted, rebalanced every $5$ trading days vs. equal-weighted universe benchmark |
| **Vietnamese Equity Cost Schedule** | Brokerage fee `0.15%`/side + execution slippage `0.10%`/side + statutory sell tax `0.10%` on gross sales (equivalent to `0.30%` per unit of one-way L1 weight turnover $\frac{1}{2}\sum_i |w_{i,t} - w_{i,t-1}|$) |
| **Post-Cost Survival Criterion** | Net active return over equal-weight universe has Stationary Bootstrap $p \le 0.05$ (`alternative="greater"`) with BH adjustment ($q \le 0.10$) across cost-stage candidates, plus positive `breakeven_cost` exceeding the `30 bps` one-way turnover cost hurdle |
| **Locked Holdout Evaluation** | Evaluated **only** after discovery is frozen: out-of-sample Rank IC, retention ratio $\overline{\text{IC}}_{\text{holdout}} / \overline{\text{IC}}_{\text{discovery}}$, Romano–Wolf adjusted $p$-value on survivors, and full-sample `cusum_mean_test` |

---

## 5. Verdict Classification Rules (`falsify_factors`)

Each factor in the discovery sample receives one of four mutually exclusive verdicts (priority `decaying > robust > fdr_only > noise`):

1. **`robust`**: `p_rw <= 0.05` (survives Romano–Wolf FWER control across all 150 signals) and not `decaying`.
2. **`fdr_only`**: `q_bh <= 0.10` and `p_rw > 0.05` and not `decaying`.
3. **`decaying`**: (`p_rw <= 0.05` or `q_bh <= 0.10`) and `p_cusum <= 0.05` and $|\overline{\text{IC}}_{\text{second half}}| < |\overline{\text{IC}}_{\text{first half}}|$.
4. **`noise`**: `p_rw > 0.05` and `q_bh > 0.10`.

---

## 6. Commitments & Prohibited Researcher Degrees of Freedom

- We **will not** tune indicator lookback windows, smoothing multipliers, or thresholds.
- We **will not** alter `config.py`, `registry.py`, or this `PREREGISTRATION.md` after unlocking the holdout period (`run.py --stage holdout --unlock-holdout` enforces a programmatic Git cleanliness and commit-history guard on all three files).
- If **zero** signals survive Romano–Wolf FWER control, transaction costs, or the locked holdout sample, **zero survivors is reported as the primary empirical finding**.
