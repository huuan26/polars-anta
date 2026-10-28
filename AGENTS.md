# polars-anta Repository Instructions for AI Agents & Contributors

This is the canonical instruction file for coding agents (Cursor, Claude, Copilot, Antigravity, etc.) and developers working in this repository. All automated contributions and Pull Requests must strictly conform to these invariants and verification requirements.

---

## 🗺️ Repository Map

- `src/polars_anta/`: Core library package (pure Polars, zero-C dependencies).
  - `alpha.py`: WorldQuant 101 Formulaic Alphas (`alpha_001`–`alpha_030`, `alpha_041`, `alpha_054`, `alpha_101`), cross-sectional operators (`cs_rank`, `cs_zscore`, `cs_winsorize`, `neutralize`), and time-series primitives.
  - `trend.py`: Trend-following indicators (SMA, EMA, HMA, KAMA, SuperTrend, Ichimoku, ADX, PSAR, SSL Channel, Pivots).
  - `momentum.py`: Momentum oscillators (RSI, MACD, Stochastic, Squeeze Momentum, STC, CCI, Williams %R, TSI).
  - `volatility.py`: Volatility estimators & bands (ATR, Bollinger Bands, Keltner, Donchian, Parkinson, Garman-Klass, Ulcer Index).
  - `volume.py`: Volume & money flow analytics (VWAP, VWAP Bands, OBV, CMF, MFI, ADL, Klinger, Force Index).
  - `stats.py`: Vectorized OLS regression (`slope`, `intercept`, `forecast`, `angle`) and rolling distribution statistics (`beta`, `alpha`, `skewness`, `kurtosis`).
  - `evaluation.py`: Alphalens-style factor evaluation (`forward_returns`, `information_coefficient`, `ic_summary`, `quantile_analysis`, `factor_turnover`) and strategy performance metrics (`sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `profit_factor`, `win_rate`).
  - `backtest.py`: Lightweight loop-free vectorized backtesting engine, crossover expressions (`crosses_above`, `crosses_below`), entry/exit state machine (`signals_to_positions`), portfolio weight schemes (`dollar_neutral`, `long_only`, `equal_weight`), tearsheets (`report()`), and `compare_strategies()`.
  - `resample.py`: Multi-timeframe OHLCV resampling (`resample_ohlcv`) and structural data integrity validation (`validate_ohlcv`).
  - `namespace.py`: Fluent Polars Expression (`pl.Expr.anta` / `.ta`), DataFrame, and LazyFrame namespaces.
  - `core.py`: Internal expression parsers (`parse_expr`) and type definitions (`IntoExpr`).
- `tests/`: Comprehensive Pytest test suite and numerical parity verification.
- `examples/`: Standalone demonstration scripts and 1-click Google Colab notebooks.
- `.github/`: CI workflows (`ci.yml`, `release.yml`, `docs.yml`), issue templates, and governance (`CODEOWNERS`).

---

## 🔒 Architectural & Implementation Invariants

Every agent generating or modifying code in this repository **MUST** adhere to the following non-negotiable invariants:

### 1. Zero C-Dependencies & Pure Polars Engine
- **NEVER** introduce C/C++ extensions, Cython, Numba, or foreign compile-time build dependencies (e.g., `gcc`, `clang`, `msvc`, `CMake`).
- All algorithms must be implemented as native Polars expressions (`pl.Expr`), or accelerated via contiguous NumPy matrix operations (`numpy.lib.stride_tricks.sliding_window_view`) where path-dependent rolling operations require it.

### 2. Zero Row-by-Row Python Loops
- **NEVER** use Python `for` loops, `.iter_rows()`, `.apply()`, or `map_elements()` to iterate across price bars or timestamps.
- Use closed-form algebraic expressions, native Polars scan/cumulative operators (`cum_prod()`, `cum_sum()`, `cum_max()`), exponential smoothing (`ewm_mean()`), or `.forward_fill()` state propagation.

### 3. Window & Multi-Asset Invariant (`.over()`)
- Every indicator and alpha expression **MUST** evaluate correctly within Polars window contexts (`.over("symbol")` or `.over(["exchange", "symbol"])`).
- Never write expressions that assume a single global time-series or fail when executed across multi-ticker panel datasets.

### 4. Dual Namespace Parity (`.anta` & `.ta`)
- Every indicator or feature exposed on `ExprAntaNamespace` or `TechnicalAnalysisFrame` must be registered identically under both `.anta` and `.ta` namespaces.
- Top-level exports in `polars_anta/__init__.py` and the `__all__` list must be kept in complete 1:1 synchronization.

### 5. Typed `pl.Struct` Outputs for Multi-Line Indicators
- Multi-line indicators (e.g., MACD, Bollinger Bands, Keltner Channels, SuperTrend, Ichimoku) **MUST** return a strongly-typed `pl.Struct` expression with explicit field names and provide `unnest=True` support when called on DataFrames/LazyFrames.

### 6. Backtesting Look-Ahead Bias Prevention
- In `vectorized_backtest()` or `df.anta.backtest()`, signals must default to execution lag `signal_lag=1` (shifting by 1 bar over `group_by`) to ensure no look-ahead bias.
- State transitions (Long/Flat/Short) must be handled via `signals_to_positions` using pure Polars forward filling.

---

## 🛡️ Governance & PR Review Authority

- **Code Ownership**: As formalized in [`.github/CODEOWNERS`](.github/CODEOWNERS), **Phan Bùi Hữu Ân ([@huuan26](https://github.com/huuan26))** is the sole creator, maintainer, and lead architect of `polars-anta`.
- **Review Requirement**: All Pull Requests targeting `master` require review and final approval from `@huuan26`.
- **No Unauthorized Breaking Changes**: Do not deprecate, rename, or alter existing public API function signatures or return types without explicit guidance and consensus from `@huuan26`.

---

## 🧪 Verification & PR Quality Checklist

Before submitting or recommending any Pull Request, agents must execute and pass the following quality checks:

```bash
# 1. Run the complete test suite (must pass 100%)
python -m pytest tests -v

# 2. Check code linting with Ruff (zero errors or warnings)
ruff check .

# 3. Check formatting with Ruff (zero unformatted files)
ruff format --check .
```

### Pull Request Expectations:
- **Unit Tests Required**: Any new indicator, factor, or feature must include dedicated unit tests in `tests/` covering:
  1. Standard numeric correctness.
  2. Edge cases (flat prices, `NaN` / `null` handling, all zeros).
  3. Multi-asset `.over("symbol")` window evaluation.
  4. Namespace parity (`pl.col().anta.*`, `df.anta.*`, `lf.ta.*`).
- **Code Coverage**: Code coverage across core modules (`namespace.py`, `backtest.py`, `evaluation.py`, `resample.py`) must remain $\ge 95\%$.
- **Documentation**: If adding new functions or parameters, update `README.md` tables, docstrings, and `polars_anta/__init__.py`.
