"""Discrete signal triggers and position state-machine expressions for Polars."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..core import IntoExpr, parse_expr


def _parse_expr_or_scalar(val: IntoExpr | float | int) -> pl.Expr:
    """Convert column name, Polars expression, or numeric threshold into a Polars expression."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return pl.lit(float(val))
    return parse_expr(val)


def _cross_pair(
    a: IntoExpr,
    b: IntoExpr | float | int,
    over: str | Sequence[str] | None,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Build current and 1-bar lagged expressions for crossing detection."""
    ea = parse_expr(a).cast(pl.Float64)
    eb = _parse_expr_or_scalar(b).cast(pl.Float64)
    prev_a = ea.shift(1).over(over) if over is not None else ea.shift(1)
    is_scalar = isinstance(b, (int, float)) and not isinstance(b, bool)
    prev_b = eb if is_scalar else (eb.shift(1).over(over) if over is not None else eb.shift(1))
    return ea, eb, prev_a, prev_b


def crosses_above(
    a: IntoExpr,
    b: IntoExpr | float | int,
    over: str | Sequence[str] | None = None,
) -> pl.Expr:
    """Return a boolean expression that is True on the exact bar where ``a`` crosses above ``b``.

    Supports crossing another indicator column/expression or a constant scalar threshold
    (e.g., ``pl.col('rsi_14').anta.crosses_above(30)``).

    :param a: Fast series or indicator expression/column name.
    :param b: Slow series, indicator expression/column name, or numeric threshold.
    :param over: Optional grouping column(s) for multi-asset panels.
    :return: Boolean Polars expression.
    """
    ea, eb, prev_a, prev_b = _cross_pair(a, b, over)
    return ((ea > eb) & (prev_a <= prev_b)).fill_null(False)


def crosses_below(
    a: IntoExpr,
    b: IntoExpr | float | int,
    over: str | Sequence[str] | None = None,
) -> pl.Expr:
    """Return a boolean expression that is True on the exact bar where ``a`` crosses below ``b``.

    Supports crossing another indicator column/expression or a constant scalar threshold
    (e.g., ``pl.col('rsi_14').anta.crosses_below(70)``).

    :param a: Fast series or indicator expression/column name.
    :param b: Slow series, indicator expression/column name, or numeric threshold.
    :param over: Optional grouping column(s) for multi-asset panels.
    :return: Boolean Polars expression.
    """
    ea, eb, prev_a, prev_b = _cross_pair(a, b, over)
    return ((ea < eb) & (prev_a >= prev_b)).fill_null(False)


def signals_to_positions(
    entries: IntoExpr,
    exits: IntoExpr,
    short_entries: IntoExpr | None = None,
    short_exits: IntoExpr | None = None,
    over: str | Sequence[str] | None = None,
) -> pl.Expr:
    """Convert discrete entry/exit boolean triggers into a continuous position series.

    Uses Polars' native ``forward_fill()`` state propagation with zero Python loops:
    - Holds ``+1.0`` from ``entries=True`` until ``exits=True`` (which resets to ``0.0``).
    - When ``short_entries`` is also supplied, holds ``-1.0`` from ``short_entries=True``
      until ``short_exits=True`` (or ``entries=True``).

    :param entries: Boolean column/expression triggering a long entry (``+1.0``).
    :param exits: Boolean column/expression triggering a long exit (``0.0``).
    :param short_entries: Optional boolean column/expression triggering a short entry (``-1.0``).
    :param short_exits: Optional boolean column/expression triggering a short exit (``0.0``).
    :param over: Optional symbol/ticker grouping column(s) for multi-asset panels.
    :return: Continuous position Polars expression in ``{-1.0, 0.0, 1.0}``.
    """
    ent = parse_expr(entries).cast(pl.Boolean).fill_null(False)
    ext = parse_expr(exits).cast(pl.Boolean).fill_null(False)

    long_state = (
        pl.when(ext & ~ent).then(0.0).when(ent).then(1.0).otherwise(pl.lit(None, dtype=pl.Float64))
    )
    long_pos = (
        long_state.forward_fill().over(over).fill_null(0.0)
        if over is not None
        else long_state.forward_fill().fill_null(0.0)
    )

    if short_entries is None:
        return long_pos

    s_ent = parse_expr(short_entries).cast(pl.Boolean).fill_null(False)
    s_ext = (
        parse_expr(short_exits).cast(pl.Boolean).fill_null(False)
        if short_exits is not None
        else ent
    )
    short_state = (
        pl.when(s_ext & ~s_ent)
        .then(0.0)
        .when(s_ent)
        .then(-1.0)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    short_pos = (
        short_state.forward_fill().over(over).fill_null(0.0)
        if over is not None
        else short_state.forward_fill().fill_null(0.0)
    )
    return (long_pos + short_pos).clip(-1.0, 1.0)
