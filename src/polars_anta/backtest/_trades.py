"""Round-trip trade ledger extraction, MAE/MFE analytics, and BacktestResult container."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl

_EMPTY_TRADE_SCHEMA: dict[str, pl.DataType] = {
    "trade_id": pl.Int64(),
    "side": pl.String(),
    "entry_index": pl.Int64(),
    "exit_index": pl.Int64(),
    "holding_bars": pl.Int64(),
    "avg_position": pl.Float64(),
    "entry_price": pl.Float64(),
    "exit_price": pl.Float64(),
    "gross_return": pl.Float64(),
    "cost": pl.Float64(),
    "net_return": pl.Float64(),
    "mae": pl.Float64(),
    "mfe": pl.Float64(),
}


def extract_trades(
    bt_df: pl.DataFrame,
    price_col: str | None = "close",
    time_col: str | None = None,
    group_by: str | Sequence[str] | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
) -> pl.DataFrame:
    """Extract a Round-Trip Trade Ledger with MAE and MFE from a backtest DataFrame.

    Identifies contiguous non-zero position regimes in ``bt_df`` using native Polars
    state-transition expressions and aggregates trade-level metrics including holding
    duration, gross/net returns, **Maximum Adverse Excursion (MAE)**, and
    **Maximum Favorable Excursion (MFE)**.

    By default, MAE and MFE are measured along the bar-close cumulative gross return path
    (``_cum_gross``). When both ``high_col`` and ``low_col`` are supplied and present in
    ``bt_df`` alongside ``price_col``, MAE and MFE incorporate intrabar high/low excursions
    relative to the prior reference price scaled by active ``position``.

    :param bt_df: Backtest DataFrame containing ``position``, ``gross_return``, ``cost``,
        and ``net_return`` columns.
    :param price_col: Optional price column name in ``bt_df`` for reporting ``entry_price``
        and ``exit_price`` (default ``'close'``).
    :param time_col: Optional timestamp/date column name for reporting ``entry_time`` and ``exit_time``.
    :param group_by: Optional symbol/ticker column(s) when ``bt_df`` is an unaggregated multi-asset panel.
    :param high_col: Optional bar high price column name for intrabar-aware MAE/MFE.
    :param low_col: Optional bar low price column name for intrabar-aware MAE/MFE.
    :return: Polars DataFrame with one row per round-trip trade.
    """
    if len(bt_df) == 0 or "position" not in bt_df.columns:
        return pl.DataFrame(schema=_EMPTY_TRADE_SCHEMA)

    gb_cols: list[str] | None = None
    if group_by is not None:
        cand = [group_by] if isinstance(group_by, str) else list(group_by)
        if all(c in bt_df.columns for c in cand):
            gb_cols = cand

    has_price = price_col is not None and price_col in bt_df.columns
    has_time = time_col is not None and time_col in bt_df.columns
    has_intrabar = (
        has_price
        and high_col is not None
        and low_col is not None
        and high_col in bt_df.columns
        and low_col in bt_df.columns
    )

    pos_sign = pl.col("position").sign().fill_null(0.0)
    prev_sign = (
        pos_sign.shift(1, fill_value=0.0).over(gb_cols)
        if gb_cols is not None
        else pos_sign.shift(1, fill_value=0.0)
    )

    prep_exprs: list[pl.Expr] = [
        pl.int_range(0, pl.len()).cast(pl.Int64).alias("_row_idx"),
        pos_sign.alias("_pos_sign"),
        (pos_sign != prev_sign).cast(pl.Int64).cum_sum().alias("_raw_seg"),
    ]
    if has_price and price_col is not None:
        p_expr = pl.col(price_col).cast(pl.Float64)
        prev_p = (
            p_expr.shift(1).over(gb_cols).fill_null(p_expr)
            if gb_cols is not None
            else p_expr.shift(1).fill_null(p_expr)
        )
        prep_exprs.extend([prev_p.alias("_entry_ref_price"), p_expr.alias("_bar_price")])
    else:
        prep_exprs.extend(
            [
                pl.lit(None, dtype=pl.Float64).alias("_entry_ref_price"),
                pl.lit(None, dtype=pl.Float64).alias("_bar_price"),
            ]
        )

    active = bt_df.with_columns(prep_exprs).filter(pl.col("_pos_sign") != 0.0)
    if len(active) == 0:
        return pl.DataFrame(schema=_EMPTY_TRADE_SCHEMA)

    seg_keys = [*gb_cols, "_raw_seg"] if gb_cols is not None else ["_raw_seg"]
    active = active.with_columns(
        [
            ((1.0 + pl.col("gross_return")).cum_prod().over(seg_keys) - 1.0).alias("_cum_gross"),
            ((1.0 + pl.col("net_return")).cum_prod().over(seg_keys) - 1.0).alias("_cum_net"),
        ]
    )
    if has_intrabar and high_col is not None and low_col is not None:
        prev_eq = (1.0 + pl.col("_cum_gross")).shift(1, fill_value=1.0).over(seg_keys)
        ref_p = pl.col("_entry_ref_price")
        hi_step = pl.col("position") * ((pl.col(high_col).cast(pl.Float64) / ref_p) - 1.0)
        lo_step = pl.col("position") * ((pl.col(low_col).cast(pl.Float64) / ref_p) - 1.0)
        intra_hi = (
            pl.when(ref_p > 0.0)
            .then(prev_eq * (1.0 + hi_step) - 1.0)
            .otherwise(pl.col("_cum_gross"))
        )
        intra_lo = (
            pl.when(ref_p > 0.0)
            .then(prev_eq * (1.0 + lo_step) - 1.0)
            .otherwise(pl.col("_cum_gross"))
        )
        active = active.with_columns(
            [
                pl.min_horizontal(pl.col("_cum_gross"), intra_hi, intra_lo).alias("_bar_mae_path"),
                pl.max_horizontal(pl.col("_cum_gross"), intra_hi, intra_lo).alias("_bar_mfe_path"),
            ]
        )
    else:
        active = active.with_columns(
            [
                pl.col("_cum_gross").alias("_bar_mae_path"),
                pl.col("_cum_gross").alias("_bar_mfe_path"),
            ]
        )

    agg_exprs: list[pl.Expr] = [
        pl.when(pl.col("_pos_sign").first() > 0.0)
        .then(pl.lit("LONG"))
        .otherwise(pl.lit("SHORT"))
        .alias("side"),
        pl.col("_row_idx").first().alias("entry_index"),
        pl.col("_row_idx").last().alias("exit_index"),
    ]
    if has_time and time_col is not None:
        agg_exprs.extend(
            [
                pl.col(time_col).first().alias("entry_time"),
                pl.col(time_col).last().alias("exit_time"),
            ]
        )
    agg_exprs.extend(
        [
            pl.len().cast(pl.Int64).alias("holding_bars"),
            pl.col("position").abs().mean().alias("avg_position"),
            pl.col("_entry_ref_price").first().alias("entry_price"),
            pl.col("_bar_price").last().alias("exit_price"),
            pl.col("_cum_gross").last().alias("gross_return"),
            pl.col("cost").sum().alias("cost"),
            pl.col("_cum_net").last().alias("net_return"),
            pl.min_horizontal(pl.col("_bar_mae_path").min(), pl.lit(0.0)).alias("mae"),
            pl.max_horizontal(pl.col("_bar_mfe_path").max(), pl.lit(0.0)).alias("mfe"),
        ]
    )

    trades_df = (
        active.group_by(seg_keys)
        .agg(agg_exprs)
        .sort("entry_index")
        .drop("_raw_seg")
        .with_row_index("trade_id", offset=1)
        .with_columns(pl.col("trade_id").cast(pl.Int64))
    )
    return trades_df


@dataclass(frozen=True)
class BacktestResult:
    """Container holding vectorized backtest time-series data, trade ledger, and tearsheet summary.

    :ivar data: Polars DataFrame containing ``position``, ``turnover``, ``asset_return``,
        ``gross_return``, ``cost``, ``net_return``, ``equity_curve``, ``benchmark_equity``,
        and ``drawdown``.
    :ivar summary: Dictionary of key performance, risk, and round-trip trade metrics.
    :ivar trade_ledger: Polars DataFrame of round-trip trades with MAE and MFE analytics.
    """

    data: pl.DataFrame
    summary: dict[str, float]
    trade_ledger: pl.DataFrame = field(default_factory=pl.DataFrame)

    def trades(self) -> pl.DataFrame:
        """Return the Round-Trip Trade Ledger DataFrame (with entry/exit, net_return, MAE, MFE)."""
        if len(self.trade_ledger) > 0 or "trade_id" in self.trade_ledger.columns:
            return self.trade_ledger
        return extract_trades(self.data)

    def summary_df(self, transpose: bool = False) -> pl.DataFrame:
        """Convert the performance summary dictionary into a Polars DataFrame.

        :param transpose: If False (default), returns a 1-row wide DataFrame with one column
            per metric; if True, returns a 2-column ``['metric', 'value']`` DataFrame.
        :return: Polars DataFrame.
        """
        if transpose:
            return pl.DataFrame(
                {
                    "metric": list(self.summary.keys()),
                    "value": [float(v) for v in self.summary.values()],
                }
            )
        return pl.DataFrame([self.summary])

    def report(self) -> pl.DataFrame:
        """Return a human-readable formatted tearsheet DataFrame with units (%, $, ratios)."""
        s = self.summary
        rows = [
            ("Initial Capital", f"${s.get('initial_capital', 0.0):,.2f}"),
            ("Final Equity", f"${s.get('final_equity', 0.0):,.2f}"),
            ("Total Return", f"{s.get('total_return', 0.0) * 100.0:+.2f}%"),
            ("Benchmark Return", f"{s.get('benchmark_return', 0.0) * 100.0:+.2f}%"),
            ("Excess Return (vs Benchmark)", f"{s.get('excess_return', 0.0) * 100.0:+.2f}%"),
            ("Annualized Return", f"{s.get('annualized_return', 0.0) * 100.0:+.2f}%"),
            ("Annualized Volatility", f"{s.get('annualized_volatility', 0.0) * 100.0:.2f}%"),
            ("Sharpe Ratio", f"{s.get('sharpe_ratio', 0.0):.2f}"),
            ("Probabilistic Sharpe Ratio (PSR)", f"{s.get('psr', 0.0) * 100.0:.1f}%"),
            ("Sortino Ratio", f"{s.get('sortino_ratio', 0.0):.2f}"),
            ("Calmar Ratio", f"{s.get('calmar_ratio', 0.0):.2f}"),
            ("Max Drawdown", f"{s.get('max_drawdown', 0.0) * 100.0:.2f}%"),
            ("Win Rate", f"{s.get('win_rate', 0.0) * 100.0:.2f}%"),
            ("Profit Factor", f"{s.get('profit_factor', 0.0):.2f}"),
            ("Exposure Time", f"{s.get('exposure', 0.0) * 100.0:.1f}%"),
            ("Round-Trip Trades", f"{int(s.get('n_round_trips', 0.0)):,d}"),
            ("Trade Win Rate", f"{s.get('trade_win_rate', 0.0) * 100.0:.2f}%"),
            ("Avg Holding Bars", f"{s.get('avg_holding_bars', 0.0):.1f}"),
            (
                "Avg Trade MAE / MFE",
                f"{s.get('avg_mae', 0.0) * 100.0:.2f}% / {s.get('avg_mfe', 0.0) * 100.0:+.2f}%",
            ),
            ("Total Trades (Rebalances)", f"{int(s.get('n_trades', 0.0)):,d}"),
            ("Total Turnover", f"{s.get('total_turnover', 0.0):.2f}x"),
            ("Total Cost (Fee + Slippage)", f"{s.get('total_cost', 0.0) * 100.0:.3f}%"),
            ("Bars Evaluated", f"{int(s.get('n_bars', 0.0)):,d}"),
        ]
        return pl.DataFrame(
            {
                "metric": [r[0] for r in rows],
                "value": [r[1] for r in rows],
            }
        )

    def __getitem__(self, key: str) -> Any:
        if key in self.summary:
            return self.summary[key]
        return self.data[key]


def compare_strategies(
    strategies: Mapping[str, BacktestResult],
) -> pl.DataFrame:
    """Compare multiple :class:`BacktestResult` objects side-by-side in a single DataFrame.

    :param strategies: Dictionary mapping strategy names to their :class:`BacktestResult`.
    :return: Polars DataFrame with one row per strategy and key performance columns.
    """
    if not strategies:
        raise ValueError("strategies mapping must contain at least one BacktestResult")

    rows: list[dict[str, Any]] = []
    for name, res in strategies.items():
        row: dict[str, Any] = {"strategy": name}
        row.update(res.summary)
        rows.append(row)
    return pl.DataFrame(rows)
