"""Lightweight vectorized backtesting engine and tearsheet generator for Polars."""

from __future__ import annotations

from ._engine import parameter_sweep, vectorized_backtest
from ._signals import crosses_above, crosses_below, signals_to_positions
from ._trades import BacktestResult, compare_strategies, extract_trades

__all__ = [
    "BacktestResult",
    "vectorized_backtest",
    "extract_trades",
    "parameter_sweep",
    "crosses_above",
    "crosses_below",
    "signals_to_positions",
    "compare_strategies",
]
