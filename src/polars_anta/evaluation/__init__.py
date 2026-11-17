"""Factor evaluation (IC, Rank IC, Quantile Analysis) and strategy performance metrics for Polars."""

from __future__ import annotations

from ._factors import (
    _newey_west_lrv as _newey_west_lrv,
)
from ._factors import (
    _nw_auto_lag as _nw_auto_lag,
)
from ._factors import (
    factor_turnover,
    forward_returns,
    ic_decay,
    ic_summary,
    information_coefficient,
    quantile_analysis,
)
from ._labeling import (
    trend_scanning_labels,
    triple_barrier_labels,
)
from ._metrics import (
    calmar_ratio,
    conditional_var,
    deflated_sharpe_ratio,
    drawdown_series,
    max_drawdown,
    omega_ratio,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
    win_rate,
)

__all__ = [
    "drawdown_series",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "profit_factor",
    "win_rate",
    "value_at_risk",
    "conditional_var",
    "omega_ratio",
    "tail_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "forward_returns",
    "information_coefficient",
    "ic_summary",
    "ic_decay",
    "quantile_analysis",
    "factor_turnover",
    "triple_barrier_labels",
    "trend_scanning_labels",
]
