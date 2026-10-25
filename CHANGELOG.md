# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
