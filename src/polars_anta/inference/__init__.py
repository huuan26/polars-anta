"""Statistical inference and factor falsification engine for quantitative research.

Provides Polars-native / vectorized NumPy econometric procedures specifically
tailored to financial time series and panel factor selection:

1. **Dependence-aware resampling & reality checks**:
   - :func:`stationary_bootstrap_indices` & :func:`bootstrap_test`
   - :func:`romano_wolf`
   - :func:`spa_test`
   - :func:`adjust_pvalues`
2. **Backtest overfitting & selection bias**:
   - :func:`pbo_cscv`
   - :func:`empirical_bayes_shrinkage`
3. **Out-of-sample predictability & structural stability**:
   - :func:`oos_r2` & :func:`clark_west_test`
   - :func:`diebold_mariano_test`
   - :func:`cusum_mean_test`
   - :func:`breakeven_cost` & :func:`breakeven_cost_from_backtest`
4. **Panel factor falsification**:
   - :func:`ic_matrix`, :func:`long_short_returns`, :func:`falsify_factors`,
     :class:`FalsificationResult`, and :func:`sweep_robustness`.
"""

from __future__ import annotations

from ._bootstrap import (
    _bootstrap_means as _bootstrap_means,
)
from ._bootstrap import (
    adjust_pvalues,
    bootstrap_test,
    romano_wolf,
    spa_test,
    stationary_bootstrap_indices,
)
from ._overfitting import (
    breakeven_cost,
    breakeven_cost_from_backtest,
    clark_west_test,
    cusum_mean_test,
    diebold_mariano_test,
    empirical_bayes_shrinkage,
    oos_r2,
    pbo_cscv,
)
from ._panel import (
    FalsificationResult,
    falsify_factors,
    ic_matrix,
    long_short_returns,
    sweep_robustness,
)
from ._validation import (
    _normal_cdf as _normal_cdf,
)
from ._validation import (
    _to_1d_array as _to_1d_array,
)
from ._validation import (
    _to_matrix as _to_matrix,
)
from ._validation import (
    _two_sided_normal_p as _two_sided_normal_p,
)

__all__ = [
    "stationary_bootstrap_indices",
    "bootstrap_test",
    "adjust_pvalues",
    "romano_wolf",
    "spa_test",
    "pbo_cscv",
    "empirical_bayes_shrinkage",
    "oos_r2",
    "clark_west_test",
    "diebold_mariano_test",
    "cusum_mean_test",
    "breakeven_cost",
    "breakeven_cost_from_backtest",
    "ic_matrix",
    "long_short_returns",
    "falsify_factors",
    "FalsificationResult",
    "sweep_robustness",
]
