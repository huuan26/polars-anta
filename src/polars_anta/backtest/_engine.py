"""Lightweight vectorized backtesting engine and parameter sweep for Polars."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from ..core import IntoExpr, check_period, check_positive, parse_expr
from ..evaluation import (
    calmar_ratio,
    deflated_sharpe_ratio,
    drawdown_series,
    max_drawdown,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from ._helpers import (
    _apply_no_trade_buffer,
    _empty_summary,
    _normalize_signal_weights,
    _resolve_exec_price_expr,
)
from ._signals import signals_to_positions
from ._trades import BacktestResult, extract_trades


def vectorized_backtest(
    df: pl.DataFrame | pl.LazyFrame,
    signal: IntoExpr = "signal",
    close_col: str = "close",
    return_col: str | None = None,
    group_by: str | Sequence[str] | None = None,
    time_col: str | None = None,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    signal_lag: int = 1,
    initial_capital: float = 1.0,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
    compound: bool = True,
    aggregate_portfolio: bool = True,
    entries: IntoExpr | None = None,
    exits: IntoExpr | None = None,
    short_entries: IntoExpr | None = None,
    short_exits: IntoExpr | None = None,
    weight_scheme: Literal["raw", "dollar_neutral", "long_only", "equal_weight"] = "raw",
    execution_price: Literal["close", "open", "twap", "vwap"] = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    flat_mask: IntoExpr | None = None,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    no_trade_buffer: float = 0.0,
    regularize_calendar: bool = False,
) -> BacktestResult:
    """Run a fast, loop-free vectorized backtest over single-asset or multi-asset Polars frames.

    Automatically shifts signals by ``signal_lag`` bars to eliminate look-ahead bias,
    supports realistic execution prices (``'close'``, ``'open'``, ``'twap'``, ``'vwap'``),
    intraday EOD session flattening (``flat_mask``), dynamic Volatility Targeting (``vol_target``),
    and no-trade turnover buffers (``no_trade_buffer``). Deducts transaction fees and slippage
    proportional to position turnover ``|position[t] - position[t-1]|``, constructs cumulative
    strategy & benchmark equity curves, and generates a Round-Trip Trade Ledger with MAE/MFE.

    :param df: Input Polars DataFrame or LazyFrame.
    :param signal: Signal or portfolio weight column name or Polars expression (default ``'signal'``).
        Supports boolean flags, discrete signals ``{-1, 0, 1}``, or continuous weights.
        Ignored when ``entries`` and ``exits`` are explicitly provided.
    :param close_col: Name of close price column used to compute asset returns when
        ``return_col`` is None (default ``'close'``).
    :param return_col: Optional pre-computed asset return column.
    :param group_by: Optional symbol/ticker column(s) for multi-asset universes.
    :param time_col: Optional timestamp/date column. When both ``group_by`` and ``time_col``
        are provided and ``aggregate_portfolio=True``, aggregates portfolio returns per timestamp.
    :param fee_rate: One-way proportional commission rate (e.g., ``0.0005`` for 5 bps, default 0.0).
    :param slippage_rate: One-way proportional slippage rate (e.g., ``0.0002`` for 2 bps, default 0.0).
    :param signal_lag: Execution lag in bars (default 1, preventing look-ahead bias).
    :param initial_capital: Starting portfolio equity (default 1.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :param risk_free_rate: Annualized risk-free rate for Sharpe/Sortino calculations (default 0.0).
    :param compound: When True (default), compounds returns multiplicatively; when False,
        accumulates returns additively.
    :param aggregate_portfolio: When True and both ``group_by`` and ``time_col`` are set,
        aggregates asset-level returns into a single portfolio time series ordered by ``time_col``.
    :param entries: Optional boolean column/expression triggering a long entry (``+1.0``).
        Requires ``exits`` to be set.
    :param exits: Optional boolean column/expression triggering a long exit (``0.0``).
    :param short_entries: Optional boolean column/expression triggering a short entry (``-1.0``).
    :param short_exits: Optional boolean column/expression triggering a short exit (``0.0``).
    :param weight_scheme: Cross-sectional portfolio weighting scheme when ``time_col`` is provided:
        ``'raw'`` (default, uses signal directly), ``'dollar_neutral'`` (demeans per timestamp and
        scales gross exposure to 1.0), ``'long_only'`` (scales positive signals to sum to 1.0),
        or ``'equal_weight'`` (assigns equal weight ``1/N`` to assets with ``signal > 0``).
    :param execution_price: Price benchmark used to compute bar returns when ``return_col=None``:
        ``'close'`` (default), ``'open'`` (next-open execution), ``'twap'`` (OHLC average),
        or ``'vwap'`` (typical price ``(H+L+C)/3``).
    :param open_col: Open price column name when ``execution_price`` is ``'open'`` or ``'twap'``.
    :param high_col: High price column name when ``execution_price`` is ``'twap'`` or ``'vwap'``.
    :param low_col: Low price column name when ``execution_price`` is ``'twap'`` or ``'vwap'``.
    :param flat_mask: Optional boolean column/expression forcing ``position = 0.0`` on matching bars
        (e.g., EOD intraday flattening at 14:25 for VN30F1M futures).
    :param vol_target: Optional target annualized portfolio volatility (e.g., ``0.15`` for 15%).
        Scales positions dynamically by ``min(max_leverage, vol_target / realized_vol)``.
    :param vol_lookback: Rolling window length in bars for realized volatility estimation (default 20).
    :param max_leverage: Maximum position scaling multiplier when ``vol_target`` is active (default 2.0).
    :param no_trade_buffer: Minimum position change threshold ``|target - prev_pos|`` required to
        trigger a rebalance trade (default 0.0).
    :param regularize_calendar: When True, reindexes onto the dense ``(time_col, group_by)`` grid
        before computing returns and ``signal_lag`` shifts so halted/missing bars hold positions cleanly.
    :return: :class:`BacktestResult` containing ``.data`` DataFrame, ``.summary`` dict, and ``.trades()``.
    """
    if fee_rate < 0.0 or slippage_rate < 0.0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")
    if signal_lag < 0:
        raise ValueError("signal_lag must be a non-negative integer")
    check_positive(initial_capital, "initial_capital")
    check_positive(annualization, "annualization")
    if weight_scheme not in ("raw", "dollar_neutral", "long_only", "equal_weight"):
        raise ValueError(
            "weight_scheme must be one of 'raw', 'dollar_neutral', 'long_only', 'equal_weight'"
        )
    if execution_price not in ("close", "open", "twap", "vwap"):
        raise ValueError("execution_price must be one of 'close', 'open', 'twap', 'vwap'")
    if (entries is not None) ^ (exits is not None):
        raise ValueError("both entries and exits must be provided together")
    if vol_target is not None:
        check_positive(vol_target, "vol_target")
        check_period(vol_lookback, 2, "vol_lookback", inclusive=True)
        check_positive(max_leverage, "max_leverage")
    if no_trade_buffer < 0.0:
        raise ValueError("no_trade_buffer must be non-negative")

    gb_cols: list[str] | None = None
    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)

    if regularize_calendar and (gb_cols is None or time_col is None):
        raise ValueError("regularize_calendar=True requires both group_by and time_col")

    if (
        entries is not None
        and exits is not None
        and gb_cols is not None
        and time_col is not None
        and aggregate_portfolio
        and weight_scheme == "raw"
    ):
        warnings.warn(
            "Using weight_scheme='raw' with multi-asset entries/exits and aggregate_portfolio=True "
            "sums unnormalized +/-1.0 positions across active assets (scaling gross leverage with N). "
            "Pass weight_scheme='dollar_neutral', 'long_only', or 'equal_weight' to normalize weights.",
            UserWarning,
            stacklevel=2,
        )

    total_cost_rate = float(fee_rate) + float(slippage_rate)

    if entries is not None and exits is not None:
        raw_sig_expr = signals_to_positions(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            over=gb_cols,
        )
    else:
        raw_sig_expr = parse_expr(signal)

    sig_expr = raw_sig_expr.cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    step0 = _normalize_signal_weights(
        df.with_columns(sig_expr.alias("_anta_sig")),
        weight_scheme=weight_scheme,
        time_col=time_col,
    )
    exec_price_expr = _resolve_exec_price_expr(
        execution_price, open_col, high_col, low_col, close_col
    )

    drop_temp_cols = ["_anta_sig"]
    if regularize_calendar and gb_cols is not None and time_col is not None:
        step0 = step0.with_columns(
            [
                pl.lit(True).alias("_anta_has_bar"),
                (
                    pl.col(return_col).cast(pl.Float64)
                    if return_col is not None
                    else exec_price_expr
                ).alias("_anta_cal_val"),
            ]
        )
        grid = step0.select(time_col).unique().join(step0.select(gb_cols).unique(), how="cross")
        step0 = (
            grid.join(step0, on=[*gb_cols, time_col], how="left")
            .sort([*gb_cols, time_col])
            .with_columns(
                [
                    pl.col("_anta_has_bar").fill_null(False),
                    pl.col("_anta_sig").forward_fill().over(gb_cols).fill_null(0.0),
                ]
            )
        )
        if return_col is not None:
            ret_expr = pl.col("_anta_cal_val").fill_null(0.0).fill_nan(0.0)
        else:
            ret_expr = (
                pl.col("_anta_cal_val")
                .forward_fill()
                .over(gb_cols)
                .pct_change()
                .over(gb_cols)
                .fill_null(0.0)
                .fill_nan(0.0)
            )
        base_sig = pl.col("_anta_sig")
        raw_pos = (
            base_sig.shift(signal_lag, fill_value=0.0).over(gb_cols) if signal_lag > 0 else base_sig
        )
        pos_expr = (
            pl.when(pl.col("_anta_has_bar"))
            .then(raw_pos)
            .otherwise(None)
            .forward_fill()
            .over(gb_cols)
            .fill_null(0.0)
            .fill_nan(0.0)
        )
        drop_temp_cols.extend(["_anta_has_bar", "_anta_cal_val"])
    else:
        if return_col is not None:
            ret_expr = pl.col(return_col).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
        else:
            ret_expr = (
                exec_price_expr.pct_change().over(gb_cols)
                if gb_cols is not None
                else exec_price_expr.pct_change()
            )
        base_sig = pl.col("_anta_sig")
        pos_expr = base_sig.shift(signal_lag, fill_value=0.0) if signal_lag > 0 else base_sig
        if gb_cols is not None and signal_lag > 0:
            pos_expr = base_sig.shift(signal_lag, fill_value=0.0).over(gb_cols)
        ret_expr = ret_expr.fill_null(0.0).fill_nan(0.0)
        pos_expr = pos_expr.fill_null(0.0).fill_nan(0.0)

    if flat_mask is not None:
        f_mask = parse_expr(flat_mask).cast(pl.Boolean).fill_null(False)
        pos_expr = pl.when(f_mask).then(0.0).otherwise(pos_expr)

    step1 = step0.with_columns([pos_expr.alias("position"), ret_expr.alias("asset_return")]).drop(
        drop_temp_cols
    )

    if vol_target is not None:
        r_std = pl.col("asset_return").rolling_std(window_size=vol_lookback, min_samples=2).shift(1)
        if gb_cols is not None:
            r_std = r_std.over(gb_cols)
        ann_realized_vol = r_std * math.sqrt(float(annualization))
        vol_scalar = (
            pl.when(ann_realized_vol.is_not_null() & (ann_realized_vol > 1e-8))
            .then((float(vol_target) / ann_realized_vol).clip(0.0, float(max_leverage)))
            .otherwise(1.0)
        )
        scaled_pos = pl.col("position") * vol_scalar
        if flat_mask is not None:
            f_mask = parse_expr(flat_mask).cast(pl.Boolean).fill_null(False)
            scaled_pos = pl.when(f_mask).then(0.0).otherwise(scaled_pos)
        step1 = step1.with_columns(scaled_pos.alias("position"))

    if no_trade_buffer > 0.0:
        step1 = step1.with_columns(
            _apply_no_trade_buffer(pl.col("position"), no_trade_buffer, over=gb_cols).alias(
                "position"
            )
        )

    prev_pos_expr = (
        pl.col("position").shift(1, fill_value=0.0).over(gb_cols)
        if gb_cols is not None
        else pl.col("position").shift(1, fill_value=0.0)
    )
    step3 = (
        step1.with_columns(
            [
                (pl.col("position") - prev_pos_expr).abs().alias("turnover"),
                (pl.col("position") * pl.col("asset_return")).alias("gross_return"),
            ]
        )
        .with_columns((pl.col("turnover") * total_cost_rate).alias("cost"))
        .with_columns((pl.col("gross_return") - pl.col("cost")).alias("net_return"))
    )

    if gb_cols is not None and time_col is not None and aggregate_portfolio:
        working = (
            step3.group_by(time_col)
            .agg(
                [
                    pl.col("position").sum().alias("position"),
                    pl.col("turnover").sum().alias("turnover"),
                    pl.col("asset_return").mean().alias("asset_return"),
                    pl.col("gross_return").sum().alias("gross_return"),
                    pl.col("cost").sum().alias("cost"),
                    pl.col("net_return").sum().alias("net_return"),
                ]
            )
            .sort(time_col)
        )
    else:
        working = step3

    if compound:
        eq_expr = float(initial_capital) * (1.0 + pl.col("net_return")).cum_prod()
        bench_eq_expr = float(initial_capital) * (1.0 + pl.col("asset_return")).cum_prod()
    else:
        eq_expr = float(initial_capital) * (1.0 + pl.col("net_return").cum_sum())
        bench_eq_expr = float(initial_capital) * (1.0 + pl.col("asset_return").cum_sum())

    if gb_cols is not None and not (time_col is not None and aggregate_portfolio):
        eq_expr = eq_expr.over(gb_cols)
        bench_eq_expr = bench_eq_expr.over(gb_cols)
        dd_expr = drawdown_series(pl.col("equity_curve"), is_returns=False).over(gb_cols)
    else:
        dd_expr = drawdown_series(pl.col("equity_curve"), is_returns=False)

    final_frame = working.with_columns(
        [eq_expr.alias("equity_curve"), bench_eq_expr.alias("benchmark_equity")]
    ).with_columns(dd_expr.alias("drawdown"))

    bt_df = final_frame.collect() if isinstance(final_frame, pl.LazyFrame) else final_frame
    n_bars = len(bt_df)
    if n_bars == 0:
        return BacktestResult(
            data=bt_df,
            summary=_empty_summary(initial_capital),
            trade_ledger=extract_trades(bt_df, price_col=close_col, time_col=time_col),
        )

    trades_df = extract_trades(
        bt_df,
        price_col=open_col if execution_price == "open" else close_col,
        time_col=time_col,
        group_by=gb_cols
        if not (gb_cols is not None and time_col is not None and aggregate_portfolio)
        else None,
    )
    n_rt = len(trades_df)
    t_win_rate = float((trades_df["net_return"] > 0.0).mean() or 0.0) if n_rt > 0 else 0.0
    avg_hold = float(trades_df["holding_bars"].mean() or 0.0) if n_rt > 0 else 0.0
    avg_mae = float(trades_df["mae"].mean() or 0.0) if n_rt > 0 else 0.0
    avg_mfe = float(trades_df["mfe"].mean() or 0.0) if n_rt > 0 else 0.0

    metrics_row = bt_df.select(
        [
            pl.col("equity_curve").last().alias("final_equity"),
            pl.col("benchmark_equity").last().alias("final_bench"),
            pl.col("net_return").mean().alias("mean_ret"),
            pl.col("net_return").std().fill_null(0.0).alias("std_ret"),
            sharpe_ratio(
                "net_return", risk_free_rate=risk_free_rate, annualization=annualization
            ).alias("sharpe"),
            probabilistic_sharpe_ratio(
                "net_return",
                benchmark_sr=0.0,
                risk_free_rate=risk_free_rate,
                annualization=annualization,
            ).alias("psr"),
            sortino_ratio(
                "net_return", risk_free_rate=risk_free_rate, annualization=annualization
            ).alias("sortino"),
            calmar_ratio("net_return", annualization=annualization).alias("calmar"),
            max_drawdown("equity_curve", is_returns=False).alias("mdd"),
            win_rate("net_return", include_zeros=False).alias("win_rate"),
            profit_factor("net_return").alias("profit_factor"),
            (pl.col("position").abs() > 1e-12).cast(pl.Float64).mean().alias("exposure"),
            (pl.col("turnover") > 1e-12).cast(pl.Float64).sum().alias("n_trades"),
            pl.col("turnover").sum().alias("total_turnover"),
            pl.col("cost").sum().alias("total_cost"),
        ]
    ).to_dicts()[0]

    final_eq = float(metrics_row["final_equity"])
    final_bench = float(metrics_row["final_bench"])
    total_ret = (final_eq / float(initial_capital)) - 1.0
    bench_ret = (final_bench / float(initial_capital)) - 1.0
    summary_dict: dict[str, float] = {
        "initial_capital": float(initial_capital),
        "final_equity": final_eq,
        "total_return": total_ret,
        "benchmark_return": bench_ret,
        "excess_return": total_ret - bench_ret,
        "annualized_return": float(metrics_row["mean_ret"] or 0.0) * float(annualization),
        "annualized_volatility": float(metrics_row["std_ret"] or 0.0)
        * math.sqrt(float(annualization)),
        "sharpe_ratio": float(metrics_row["sharpe"] or 0.0),
        "psr": float(metrics_row["psr"] or 0.0),
        "sortino_ratio": float(metrics_row["sortino"] or 0.0),
        "calmar_ratio": float(metrics_row["calmar"] or 0.0),
        "max_drawdown": float(metrics_row["mdd"] or 0.0),
        "win_rate": float(metrics_row["win_rate"] or 0.0),
        "profit_factor": float(metrics_row["profit_factor"] or 0.0),
        "exposure": float(metrics_row["exposure"] or 0.0),
        "n_round_trips": float(n_rt),
        "trade_win_rate": t_win_rate,
        "avg_holding_bars": avg_hold,
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "n_trades": float(metrics_row["n_trades"] or 0.0),
        "total_turnover": float(metrics_row["total_turnover"] or 0.0),
        "total_cost": float(metrics_row["total_cost"] or 0.0),
        "n_bars": float(n_bars),
    }
    return BacktestResult(data=bt_df, summary=summary_dict, trade_ledger=trades_df)


def parameter_sweep(
    df: pl.DataFrame | pl.LazyFrame,
    signals: Mapping[str, IntoExpr],
    close_col: str = "close",
    return_col: str | None = None,
    group_by: str | Sequence[str] | None = None,
    time_col: str | None = None,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    signal_lag: int = 1,
    initial_capital: float = 1.0,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
    weight_scheme: Literal["raw", "dollar_neutral", "long_only", "equal_weight"] = "raw",
    execution_price: Literal["close", "open", "twap", "vwap"] = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    flat_mask: IntoExpr | None = None,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    no_trade_buffer: float = 0.0,
    regularize_calendar: bool = False,
) -> pl.DataFrame:
    """Evaluate a grid of candidate signals/parameter configurations and compute Deflated Sharpe Ratios.

    Runs ``vectorized_backtest`` across all configurations in ``signals``, measures the empirical
    cross-trial Sharpe dispersion $\\sigma_{\\{SR\\}}$, and computes the multiple-testing
    **Deflated Sharpe Ratio (`deflated_sharpe_ratio`)** with ``n_trials = len(signals)`` so
    quantitative researchers can distinguish genuine parameter stability from backtest overfitting.

    :param df: Input Polars DataFrame or LazyFrame.
    :param signals: Mapping from parameter/strategy name to signal column name or Polars expression.
    :param close_col: Close price column name (default ``'close'``).
    :param return_col: Optional pre-computed return column.
    :param group_by: Optional symbol/ticker column(s).
    :param time_col: Optional timestamp/date column.
    :param fee_rate: One-way commission rate (default 0.0).
    :param slippage_rate: One-way slippage rate (default 0.0).
    :param signal_lag: Execution lag in bars (default 1).
    :param initial_capital: Starting portfolio equity (default 1.0).
    :param annualization: Trading periods per year (default 252.0).
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param weight_scheme: Portfolio weighting scheme (default ``'raw'``).
    :param execution_price: Execution price benchmark (``'close'``, ``'open'``, ``'twap'``, ``'vwap'``).
    :param open_col: Open price column name.
    :param high_col: High price column name.
    :param low_col: Low price column name.
    :param flat_mask: Optional boolean column/expression forcing ``position = 0.0``.
    :param vol_target: Optional target annualized volatility.
    :param vol_lookback: Rolling window length for volatility targeting (default 20).
    :param max_leverage: Maximum leverage multiplier (default 2.0).
    :param no_trade_buffer: Minimum position change threshold to rebalance (default 0.0).
    :param regularize_calendar: Reindex onto dense ``(time_col, group_by)`` grid when True.
    :return: Polars DataFrame of parameter sweep results sorted by ``sharpe_ratio`` descending.
    """
    if not signals:
        raise ValueError("signals mapping must contain at least one strategy expression")

    eager_df = df.collect() if isinstance(df, pl.LazyFrame) else df
    bt_kwargs: dict[str, Any] = {
        "close_col": close_col,
        "return_col": return_col,
        "group_by": group_by,
        "time_col": time_col,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "signal_lag": signal_lag,
        "initial_capital": initial_capital,
        "annualization": annualization,
        "risk_free_rate": risk_free_rate,
        "weight_scheme": weight_scheme,
        "execution_price": execution_price,
        "open_col": open_col,
        "high_col": high_col,
        "low_col": low_col,
        "flat_mask": flat_mask,
        "vol_target": vol_target,
        "vol_lookback": vol_lookback,
        "max_leverage": max_leverage,
        "no_trade_buffer": no_trade_buffer,
        "regularize_calendar": regularize_calendar,
    }
    results = {
        name: vectorized_backtest(eager_df, signal=sig_expr, **bt_kwargs)
        for name, sig_expr in signals.items()
    }
    sharpes = [res.summary["sharpe_ratio"] for res in results.values()]

    n_trials = len(signals)
    sr_std = float(np.std(sharpes, ddof=1)) if n_trials > 1 else 0.5
    if sr_std <= 1e-8:
        sr_std = 0.5

    rows: list[dict[str, Any]] = []
    for name, bt_res in results.items():
        if len(bt_res.data) > 0:
            dsr_val = float(
                bt_res.data.select(
                    deflated_sharpe_ratio(
                        "net_return",
                        n_trials=n_trials,
                        trials_sr_std=sr_std,
                        risk_free_rate=risk_free_rate,
                        annualization=annualization,
                    )
                ).item()
                or 0.0
            )
        else:
            dsr_val = 0.0

        row: dict[str, Any] = {"strategy": name, "deflated_sharpe_ratio": dsr_val}
        row.update(bt_res.summary)
        rows.append(row)

    return pl.DataFrame(rows).sort("sharpe_ratio", descending=True)
