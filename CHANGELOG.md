# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Statistical Inference & Falsification (`polars_anta.inference`)**: Added a zero-SciPy econometric inference and factor falsification engine: Politis–Romano (1994) `stationary_bootstrap_indices` and `bootstrap_test`, multiple-testing corrections `adjust_pvalues` (Bonferroni, Holm, Benjamini–Hochberg, Benjamini–Yekutieli), Romano–Wolf (2005, 2016) stepdown `romano_wolf`, Hansen (2005) / White (2000) `spa_test`, Bailey et al. (2017) CSCV `pbo_cscv`, Efron–Morris / Jensen–Kelly–Pedersen (2023) `empirical_bayes_shrinkage`, out-of-sample forecast evaluation (`oos_r2`, `clark_west_test`, `diebold_mariano_test`), Ploberger–Krämer (1992) `cusum_mean_test`, `breakeven_cost` / `breakeven_cost_from_backtest`, panel cross-sectional `ic_matrix` and `long_short_returns`, `FalsificationResult` / `falsify_factors`, and `sweep_robustness` (with `.anta` / `.ta` DataFrame and LazyFrame namespace methods and Monte Carlo size/power validation in `benchmarks/inference_size_power.py`).
- **Volatility (`polars_anta.volatility`)**: Added `rogers_satchell_volatility` (Rogers & Satchell, 1991 drift-independent OHLC estimator) and `yang_zhang_volatility` (Yang & Zhang, 2000 minimum-variance overnight/intraday estimator) (PR #36).
- **Statistics (`polars_anta.stats`)**: Added closed-form rolling OLS goodness-of-fit `linear_regression_r2` ($R^2 \in [0, 1]$) and residual dispersion `linear_regression_stderr` (PR #37).
- **Momentum (`polars_anta.momentum`)**: Added Jack Hutson's Triple Exponential Average `trix` (`Struct(trix, signal)`) (PR #33).
- **Testing**: Added `tests/test_doc_snippets.py` to syntax-check all `python` code blocks in `README.md` and `docs/**/*.md`, and added Python 3.14 to the CI test matrix.

### Fixed
- **Statistics (`polars_anta.stats`)**: Added `bias: bool = True` parameter to `skewness()` and `kurtosis()` to support unbiased sample moment estimation (PR #34).
- **Documentation**: Fixed syntax error in the README factor-validation snippet, reconciled public function counts with `__all__` (202 functions), documented the pandas `rolling().apply()` baseline behind the 112.4x OLS benchmark speedup, and replaced unverifiable marketing claims with factual descriptions.

---

## [0.3.1] - 2026-11-01

### Added
- **Microstructure, Information Bars & Futures (`polars_anta.resample`, `polars_anta.alpha`, `polars_anta.volume`)**:
  - Information-driven bar sampling: `resample_volume_bars`, `resample_dollar_bars`, and `resample_imbalance_bars`.
  - Continuous futures contract roll stitching (`stitch_futures_roll` with `"backward"`, `"ratio"`, and `"panama"` methods).
  - Intraday `session_vwap`, order flow toxicity (`vpin`, `ofi`), `futures_basis`, and 4-state `oi_regime`.
- **AFML Stationarity & Labeling (`polars_anta.stats`, `polars_anta.evaluation`)**:
  - Fixed-Window Fractional Differentiation (`frac_diff`).
  - Dynamic path-dependent `triple_barrier_labels` (with meta-labeling support) and `trend_scanning_labels`.
- **Cross-Sectional Orthogonalization & Factor Evaluation (`polars_anta.alpha`, `polars_anta.evaluation`)**:
  - Modified Gram-Schmidt multi-factor OLS residualization (`orthogonalize`).
  - Newey-West HAC autocorrelation-consistent $t$-statistics in `ic_summary`, multi-horizon `ic_decay` with exponential half-life estimation, `probabilistic_sharpe_ratio` (PSR), and `deflated_sharpe_ratio` (DSR).
- **Execution-Realistic Backtesting (`polars_anta.backtest`)**:
  - Next-bar execution prices (`execution_price="open" | "twap" | "vwap" | "close"`), session flatten masks (`flat_mask`), target volatility scaling (`vol_target`), hysteresis no-trade buffers (`no_trade_buffer`), round-trip trade extraction (`extract_trades` / `bt.trades()`) with MAE/MFE, and `parameter_sweep`.
- **Documentation & Governance**:
  - Added mathematical and architectural documentation (`docs/architecture.md`, `docs/guides/quant-theory-and-workflows.md`), `AGENTS.md`, and `CODEOWNERS`.

---

## [0.3.0] - 2026-10-28

### Added
- **Formulaic Alphas (`polars_anta.alpha`)**:
  - Expanded formulaic alphas with `alpha_002` through `alpha_030` (33 implemented alphas in total: `alpha_001`–`alpha_030`, `alpha_041`, `alpha_054`, `alpha_101`).
  - Added cross-sectional outlier clipping (`cs_winsorize`) and standardization (`cs_zscore`), plus liquidity and spread estimators (`amihud_illiquidity`, `roll_spread`, `corwin_schultz_spread`, `kyle_lambda`).
- **Trend & Momentum (`polars_anta.trend`, `polars_anta.momentum`)**:
  - Added vectorized `heikin_ashi` candlestick transform and Schaff Trend Cycle (`stc`).
- **Factor Evaluation & Vectorized Backtesting (`polars_anta.evaluation`, `polars_anta.backtest`)**:
  - Added `forward_returns`, `information_coefficient`, `ic_summary`, `quantile_analysis`, `factor_turnover`, performance metrics (`sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `drawdown_series`, `profit_factor`, `win_rate`), `crosses_above` / `crosses_below`, `signals_to_positions`, `vectorized_backtest`, `BacktestResult`, and `compare_strategies`.
  - Added `examples/03_crypto_momentum_scan.ipynb`, `examples/04_backtest_and_alpha_eval.ipynb`, and `examples/04_backtest_showcase.py`.

---

## [0.2.0] - 2026-10-25

### Added
- **Multi-Timeframe Resampling (`resample_ohlcv`)**:
  - High-performance, zero-copy candle resampling powered natively by Polars' `group_by_dynamic`.
  - Supports single-asset and multi-ticker universes via `group_by="symbol"` or multi-column `group_by=["exchange", "symbol"]`.
  - Standard financial aggregation rules: open (first), high (max), low (min), close (last), volume (sum).
  - Supports custom column names and additional aggregation expressions (`additional_aggs`).
  - Seamlessly compatible with both eager `pl.DataFrame` and lazy `pl.LazyFrame`.
- **OHLCV Data Validation (`validate_ohlcv`)**:
  - Deep schema and candle integrity verification: `high >= low`, `high >= open`, `high >= close`, `low <= open`, `low <= close`.
  - Value checks for non-negative prices and volume, null values, and out-of-order timestamp detection per group.
  - Supports chainable strict mode (returns `df`) or diagnostic report mode (`return_report=True`).
- **Fluent DataFrame & LazyFrame Namespaces (`TechnicalAnalysisFrame`)**:
  - Registered under both `.anta` and `.ta` on `pl.DataFrame` and `pl.LazyFrame` via `@pl.api.register_dataframe_namespace` and `@pl.api.register_lazyframe_namespace`.
  - Fluent method chaining: `df.anta.validate_ohlcv().anta.resample("15m").anta.sma(20).anta.rsi(14).anta.macd(unnest=True)`.
  - Struct unnesting and prefixed column extraction for multi-component indicators.
  - `.add_indicators(*exprs, group_by=...)`: batch append one or more indicators in a single vectorized pass.
  - `.add_all(...)`: one-liner generating a comprehensive standard quant feature suite (trend, momentum, volatility, volume, alpha stats) computed in parallel inside Polars' query engine.
- **WorldQuant 101 Alphas & Cross-Sectional Primitives**:
  - `cs_rank`: Cross-sectional ranking with configurable tie-breaking (`average`, `min`, `max`, `dense`, `ordinal`) and percentile normalization.
  - `neutralize`: Sector / industry group neutralization (demeaning and z-scoring), ensuring signals are dollar-neutral across categories.
  - `ts_scale`: L1 gross leverage scaling ($\sum |w_i| = a$).
  - `ts_delta` & `ts_delay`: Lagged difference and shift operators.
  - `signed_power`: $\text{sign}(x) \cdot |x|^a$. Direction-preserving non-linear scaling.
  - `ts_min`, `ts_max`, `ts_sum`, `ts_std`: Rolling lookback extrema and moments.
  - `ts_correlation`, `ts_covariance`, `ts_product`: Rolling Pearson correlation, covariance, and cumulative product.
  - Canonical WorldQuant 101 Formulaic Alphas: `alpha_001`, `alpha_041`, `alpha_054`, `alpha_101`.
- **New Technical Indicators in Core Catalog** (expanding library to 123 indicators):
  - Trend: `ssl_channel`, `trend_intensity_index` (TII).
  - Momentum: `squeeze_momentum` (John Carter / LazyBear).
  - Volatility & Volume: `vwap_bands`.
  - Alpha Dynamics: `fractal_dimension` (Ehlers FRAMA), `hurst_exponent` (Rescaled Range R/S).
- **PEP 561 Type Completeness**:
  - Added `py.typed` marker package data to `src/polars_anta/` for full static typing and IDE autocompletion support.

### Changed & Optimized
- **Recursive Indicator Speedup**: Replaced Python `Series.to_list()` iterations with SIMD-accelerated NumPy C-buffers (`s.to_numpy()`) in recursive indicators (`psar`, `supertrend`, `kama`, `vidya`, `mcginley_dynamic`, `entropy`, `ts_rank`, `ts_argmax`, `ts_argmin`), yielding 3x-10x execution speedups.
- **Polars 2.0 Compatibility**: Updated dynamic grouping to use `group_by` instead of deprecated `by` parameter.
- **Test Suite & Coverage**: Expanded test suite to **152 unit tests (100% pass rate)** with **96% code coverage** (`namespace.py` 100%, `resample.py` 100%, `alpha.py` 98%).
- **Package Metadata**: Promoted `Development Status` classifier to `4 - Beta` in `pyproject.toml`.

---

## [0.1.0] - 2026-10-22

### Added
- Initial release of `polars-anta`.
- 102 core technical indicators across Trend, Momentum, Volatility, Volume, and Quantitative Statistics.
- Dual expression namespaces `.anta` and `.ta` on `pl.col(...)`.
- Multi-asset parallelization via Polars' native `.over("ticker")` window context.
