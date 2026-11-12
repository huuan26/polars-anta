"""Multi-timeframe OHLCV resampling and data validation for Polars DataFrames and LazyFrames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy as np
import polars as pl

from .core import check_positive

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)

__all__ = [
    "resample_ohlcv",
    "resample_volume_bars",
    "resample_dollar_bars",
    "resample_imbalance_bars",
    "stitch_futures_roll",
    "validate_ohlcv",
]

_METRIC_ERROR_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("high_lt_low", "{n} rows where high < low"),
    ("high_lt_open", "{n} rows where high < open"),
    ("high_lt_close", "{n} rows where high < close"),
    ("low_gt_open", "{n} rows where low > open"),
    ("low_gt_close", "{n} rows where low > close"),
    ("negative_open", "{n} rows with negative open price"),
    ("negative_high", "{n} rows with negative high price"),
    ("negative_low", "{n} rows with negative low price"),
    ("negative_close", "{n} rows with negative close price"),
    ("negative_volume", "{n} rows with negative volume"),
    ("null_open", "{n} null values in open"),
    ("null_high", "{n} null values in high"),
    ("null_low", "{n} null values in low"),
    ("null_close", "{n} null values in close"),
    ("null_volume", "{n} null values in volume"),
    ("null_timestamp", "{n} null values in timestamp"),
    ("unsorted_timestamp", "{n} out-of-order timestamps"),
)


def _get_schema(df: FrameType) -> pl.Schema | dict[str, Any]:
    """Return LazyFrame schema without materialization or DataFrame schema directly."""
    return df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema


def _resolve_group_and_sort(
    df: FrameType,
    schema: pl.Schema | dict[str, Any],
    group_by: str | Sequence[str] | None,
    time_col: str | None,
    sort: bool,
) -> tuple[FrameType, list[str]]:
    """Validate group_by columns and optionally sort chronologically per group."""
    gb_cols: list[str] = []
    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        for g_col in gb_cols:
            if g_col not in schema:
                raise ValueError(f"Group-by column '{g_col}' not found in schema.")

    has_time = time_col is not None and time_col in schema
    if sort and (gb_cols or has_time):
        sort_keys = gb_cols + ([time_col] if has_time and time_col is not None else [])
        return df.sort(sort_keys), gb_cols
    return df, gb_cols


def _prepare_information_bar_frame(
    df: FrameType,
    threshold: float,
    label: str,
    close_col: str,
    volume_col: str | None,
    group_by: str | Sequence[str] | None,
    time_col: str | None,
    sort: bool,
    *,
    require_volume: bool = True,
) -> tuple[FrameType, pl.Schema | dict[str, Any], list[str]]:
    """Validate parameters and schema shared by volume, dollar, and imbalance bar samplers."""
    check_positive(threshold, "threshold")
    if label not in ("left", "right"):
        raise ValueError("label must be either 'left' or 'right'")

    schema = _get_schema(df)
    if close_col not in schema:
        raise ValueError(f"Required close column '{close_col}' not found in schema.")
    if require_volume:
        if volume_col is None or volume_col not in schema:
            raise ValueError(f"Required volume column '{volume_col}' not found in schema.")
    elif volume_col is not None and volume_col not in schema:
        raise ValueError(f"Specified volume column '{volume_col}' not found in schema.")

    working_df, gb_cols = _resolve_group_and_sort(df, schema, group_by, time_col, sort)
    return working_df, schema, gb_cols


def _bucket_id_from_cumulative(cum_expr: pl.Expr, threshold: float) -> pl.Expr:
    """Assign integer bar bucket IDs from cumulative volume or notional flow."""
    return (
        ((cum_expr * (1.0 - 1e-12)) / float(threshold))
        .floor()
        .clip(lower_bound=0.0)
        .fill_null(0.0)
        .cast(pl.Int64)
        .alias("_anta_bar_id")
    )


def validate_ohlcv(
    df: FrameType,
    time_col: str | None = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str | None = "volume",
    group_by: str | Sequence[str] | None = None,
    check_data: bool = True,
    return_report: bool = False,
) -> FrameType | dict[str, Any]:
    """Validate OHLCV schema, structural integrity, and price consistency.

    Performs schema validation and data-integrity checks:
    - Verifies required price columns exist and have numeric types.
    - Verifies temporal index column exists and has Date/Datetime type.
    - Verifies High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close.
    - Verifies prices and volume are non-negative.
    - Detects null values in critical columns and out-of-order timestamps.

    :param df: Input Polars DataFrame or LazyFrame.
    :param time_col: Name of timestamp/date column, or None to skip temporal checks.
    :param open_col: Name of open price column (default 'open').
    :param high_col: Name of high price column (default 'high').
    :param low_col: Name of low price column (default 'low').
    :param close_col: Name of close price column (default 'close').
    :param volume_col: Name of volume column, or None to skip volume checks.
    :param group_by: Symbol/asset grouping column(s) for multi-ticker universes.
    :param check_data: When True, executes data-level range and null checks.
    :param return_report: When True, returns a dict report instead of raising or returning df.
    :return: Original DataFrame/LazyFrame if valid, or validation report dictionary if return_report=True.
    :raises ValueError: If validation fails and return_report is False.
    """
    schema = _get_schema(df)
    errors: list[str] = []

    # 1. Schema checks
    for role, col in [
        ("open", open_col),
        ("high", high_col),
        ("low", low_col),
        ("close", close_col),
    ]:
        if col not in schema:
            errors.append(f"Missing required {role} column '{col}'")
        elif not schema[col].is_numeric():
            errors.append(f"Column '{col}' must be numeric (got {schema[col]})")

    if time_col is not None:
        if time_col not in schema:
            errors.append(f"Missing required timestamp column '{time_col}'")
        elif not schema[time_col].is_temporal():
            errors.append(
                f"Timestamp column '{time_col}' must be Date or Datetime (got {schema[time_col]})"
            )

    has_volume = False
    if volume_col is not None:
        if volume_col in schema:
            has_volume = True
            if not schema[volume_col].is_numeric():
                errors.append(
                    f"Volume column '{volume_col}' must be numeric (got {schema[volume_col]})"
                )
        elif volume_col != "volume":
            errors.append(f"Specified volume column '{volume_col}' not found")

    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        for col in gb_cols:
            if col not in schema:
                errors.append(f"Group-by column '{col}' not found")

    if errors:
        if return_report:
            return {"valid": False, "errors": errors, "metrics": {}}
        raise ValueError(f"OHLCV schema validation failed: {'; '.join(errors)}")

    # 2. Data checks
    metrics: dict[str, int] = {}
    if check_data:
        check_exprs: list[pl.Expr] = [
            (pl.col(high_col) < pl.col(low_col)).fill_null(False).sum().alias("high_lt_low"),
            (pl.col(high_col) < pl.col(open_col)).fill_null(False).sum().alias("high_lt_open"),
            (pl.col(high_col) < pl.col(close_col)).fill_null(False).sum().alias("high_lt_close"),
            (pl.col(low_col) > pl.col(open_col)).fill_null(False).sum().alias("low_gt_open"),
            (pl.col(low_col) > pl.col(close_col)).fill_null(False).sum().alias("low_gt_close"),
            (pl.col(open_col) < 0).fill_null(False).sum().alias("negative_open"),
            (pl.col(high_col) < 0).fill_null(False).sum().alias("negative_high"),
            (pl.col(low_col) < 0).fill_null(False).sum().alias("negative_low"),
            (pl.col(close_col) < 0).fill_null(False).sum().alias("negative_close"),
            pl.col(open_col).is_null().sum().alias("null_open"),
            pl.col(high_col).is_null().sum().alias("null_high"),
            pl.col(low_col).is_null().sum().alias("null_low"),
            pl.col(close_col).is_null().sum().alias("null_close"),
        ]

        if has_volume and volume_col is not None:
            check_exprs.extend(
                [
                    (pl.col(volume_col) < 0).fill_null(False).sum().alias("negative_volume"),
                    pl.col(volume_col).is_null().sum().alias("null_volume"),
                ]
            )

        if time_col is not None:
            unsorted = pl.col(time_col) < pl.col(time_col).shift(1)
            if group_by is not None:
                unsorted = unsorted.over(group_by)
            check_exprs.extend(
                [
                    pl.col(time_col).is_null().sum().alias("null_timestamp"),
                    unsorted.fill_null(False).sum().alias("unsorted_timestamp"),
                ]
            )

        eval_df = df.select(check_exprs)
        eval_res = (
            eval_df.collect().to_dicts()[0]
            if isinstance(eval_df, pl.LazyFrame)
            else eval_df.to_dicts()[0]
        )
        metrics = {k: (int(v) if v is not None else 0) for k, v in eval_res.items()}

        for key, template in _METRIC_ERROR_TEMPLATES:
            count = metrics.get(key, 0)
            if count > 0:
                errors.append(template.format(n=count))

    if errors:
        if return_report:
            return {"valid": False, "errors": errors, "metrics": metrics}
        raise ValueError(f"OHLCV validation failed: {'; '.join(errors)}")

    if return_report:
        return {"valid": True, "errors": [], "metrics": metrics}

    return df


def resample_ohlcv(
    df: FrameType,
    every: str,
    time_col: str = "timestamp",
    group_by: str | Sequence[str] | None = None,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str | None = "volume",
    closed: Literal["left", "right"] = "left",
    label: Literal["left", "right", "datapoint"] = "left",
    additional_aggs: Sequence[pl.Expr] | None = None,
    sort: bool = True,
) -> FrameType:
    """Resample lower-timeframe OHLCV data into higher-timeframe bars.

    Computes standard financial candle aggregation:
    - Open: first open price in the window.
    - High: maximum high price in the window.
    - Low: minimum low price in the window.
    - Close: last close price in the window.
    - Volume: sum of volume in the window (if volume_col is present).

    Supports both single-asset and multi-ticker universes (`group_by="symbol"`).
    Compatible with both eager Polars `DataFrame` and lazy `LazyFrame`.

    :param df: Input Polars DataFrame or LazyFrame.
    :param every: Resampling interval string (e.g., '5m', '15m', '1h', '4h', '1d').
    :param time_col: Name of timestamp column (must be Date or Datetime).
    :param group_by: Symbol/asset grouping column(s) for multi-ticker universes.
    :param open_col: Column name for open price.
    :param high_col: Column name for high price.
    :param low_col: Column name for low price.
    :param close_col: Column name for close price.
    :param volume_col: Column name for volume (or None to omit volume).
    :param closed: Define which side of the temporal interval is closed ('left' or 'right').
    :param label: Define which interval boundary assigns the timestamp label ('left', 'right', 'datapoint').
    :param additional_aggs: Optional sequence of custom Polars aggregation expressions to compute.
    :param sort: When True, ensures inputs and outputs are sorted chronologically (per group).
    :return: Resampled DataFrame or LazyFrame of the same type as input.
    """
    schema = _get_schema(df)

    for col_name, role in [
        (time_col, "timestamp"),
        (open_col, "open"),
        (high_col, "high"),
        (low_col, "low"),
        (close_col, "close"),
    ]:
        if col_name not in schema:
            raise ValueError(f"Required {role} column '{col_name}' not found in schema.")

    if not schema[time_col].is_temporal():
        raise ValueError(
            f"Timestamp column '{time_col}' must be Date or Datetime (got {schema[time_col]})."
        )

    working_df, gb_cols = _resolve_group_and_sort(df, schema, group_by, time_col, sort)
    sort_keys = gb_cols + [time_col] if gb_cols else [time_col]

    aggs: list[pl.Expr] = [
        pl.col(open_col).first().alias(open_col),
        pl.col(high_col).max().alias(high_col),
        pl.col(low_col).min().alias(low_col),
        pl.col(close_col).last().alias(close_col),
    ]

    if volume_col is not None:
        if volume_col in schema:
            aggs.append(pl.col(volume_col).sum().alias(volume_col))
        elif volume_col != "volume":
            raise ValueError(f"Specified volume column '{volume_col}' not found in schema.")

    if additional_aggs:
        aggs.extend(additional_aggs)

    if gb_cols:
        group_arg: str | list[str] = gb_cols[0] if len(gb_cols) == 1 else gb_cols
        resampled = working_df.group_by_dynamic(
            time_col,
            every=every,
            group_by=group_arg,
            closed=closed,
            label=label,
        ).agg(aggs)
    else:
        resampled = working_df.group_by_dynamic(
            time_col,
            every=every,
            closed=closed,
            label=label,
        ).agg(aggs)

    return resampled.sort(sort_keys) if sort else resampled


def _build_information_bar_aggs(
    schema: pl.Schema | dict[str, Any],
    time_col: str | None,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str | None,
    label: Literal["left", "right"],
    additional_aggs: Sequence[pl.Expr] | None,
    extra_cols: Sequence[pl.Expr] | None = None,
) -> list[pl.Expr]:
    """Construct standard OHLCV + VWAP + observation count aggregations for information-driven bars."""
    aggs: list[pl.Expr] = []
    if time_col is not None and time_col in schema:
        time_expr = (
            pl.col(time_col).first().alias(time_col)
            if label == "left"
            else pl.col(time_col).last().alias(time_col)
        )
        aggs.append(time_expr)

    src_open = open_col if open_col in schema else close_col
    src_high = high_col if high_col in schema else close_col
    src_low = low_col if low_col in schema else close_col

    aggs.extend(
        [
            pl.col(src_open).first().alias(open_col),
            pl.col(src_high).max().alias(high_col),
            pl.col(src_low).min().alias(low_col),
            pl.col(close_col).last().alias(close_col),
        ]
    )

    if volume_col is not None and volume_col in schema:
        v_sum = pl.col(volume_col).sum()
        pv_sum = (pl.col(close_col).cast(pl.Float64) * pl.col(volume_col).cast(pl.Float64)).sum()
        aggs.extend(
            [
                v_sum.alias(volume_col),
                pl.when(v_sum > 0.0)
                .then(pv_sum / v_sum)
                .otherwise(pl.col(close_col).mean())
                .alias("vwap"),
            ]
        )

    if extra_cols:
        aggs.extend(extra_cols)

    aggs.append(pl.len().cast(pl.Int64).alias("n_obs"))

    if additional_aggs:
        aggs.extend(additional_aggs)

    return aggs


def resample_volume_bars(
    df: FrameType,
    threshold: float,
    volume_col: str = "volume",
    time_col: str | None = "timestamp",
    group_by: str | Sequence[str] | None = None,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    label: Literal["left", "right"] = "right",
    additional_aggs: Sequence[pl.Expr] | None = None,
    sort: bool = True,
) -> FrameType:
    """Resample tick or lower-timeframe OHLCV data into Information-Driven Volume Bars.

    Samples a new bar every time cumulative traded volume reaches ``threshold`` units
    (López de Prado, *Advances in Financial Machine Learning*). Accepts both full OHLCV
    bars and raw tick data (where only ``close_col`` and ``volume_col`` are present).

    :param df: Input Polars DataFrame or LazyFrame.
    :param threshold: Volume threshold per bar (must be strictly positive).
    :param volume_col: Column name for volume (default ``'volume'``).
    :param time_col: Optional timestamp column name (default ``'timestamp'``).
    :param group_by: Optional symbol/ticker column(s) for multi-asset universes.
    :param open_col: Column name for open price (falls back to ``close_col`` if absent).
    :param high_col: Column name for high price (falls back to ``close_col`` if absent).
    :param low_col: Column name for low price (falls back to ``close_col`` if absent).
    :param close_col: Column name for close/trade price (default ``'close'``).
    :param label: Which boundary timestamp to assign to the bar: ``'right'`` (close time, default)
        or ``'left'`` (open time).
    :param additional_aggs: Optional custom Polars aggregation expressions.
    :param sort: When True, sorts chronologically per group before bucketing.
    :return: Volume-bar DataFrame or LazyFrame with ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``vwap``, and ``n_obs`` columns.
    """
    working_df, schema, gb_cols = _prepare_information_bar_frame(
        df, threshold, label, close_col, volume_col, group_by, time_col, sort, require_volume=True
    )

    v_clean = pl.col(volume_col).cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0)
    cum_v = v_clean.cum_sum().over(gb_cols) if gb_cols else v_clean.cum_sum()
    aggs = _build_information_bar_aggs(
        schema, time_col, open_col, high_col, low_col, close_col, volume_col, label, additional_aggs
    )

    return (
        working_df.with_columns(_bucket_id_from_cumulative(cum_v, threshold))
        .group_by([*gb_cols, "_anta_bar_id"], maintain_order=True)
        .agg(aggs)
        .drop("_anta_bar_id")
    )


def resample_dollar_bars(
    df: FrameType,
    threshold: float,
    close_col: str = "close",
    volume_col: str = "volume",
    multiplier: float = 1.0,
    time_col: str | None = "timestamp",
    group_by: str | Sequence[str] | None = None,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    label: Literal["left", "right"] = "right",
    additional_aggs: Sequence[pl.Expr] | None = None,
    sort: bool = True,
) -> FrameType:
    """Resample tick or lower-timeframe OHLCV data into Information-Driven Dollar Value Bars.

    Samples a new bar every time cumulative traded notional value
    ``Close * Volume * multiplier`` reaches ``threshold`` currency units. Particularly
    effective for index futures (e.g., VN30F1M with ``multiplier=100_000``) and assets
    experiencing large price appreciation/depreciation.

    :param df: Input Polars DataFrame or LazyFrame.
    :param threshold: Notional dollar/currency value threshold per bar (must be > 0).
    :param close_col: Column name for close/trade price (default ``'close'``).
    :param volume_col: Column name for volume (default ``'volume'``).
    :param multiplier: Contract point multiplier (default 1.0; e.g. 100,000 for VN30F1M).
    :param time_col: Optional timestamp column name (default ``'timestamp'``).
    :param group_by: Optional symbol/ticker column(s) for multi-asset universes.
    :param open_col: Column name for open price (falls back to ``close_col`` if absent).
    :param high_col: Column name for high price (falls back to ``close_col`` if absent).
    :param low_col: Column name for low price (falls back to ``close_col`` if absent).
    :param label: ``'right'`` (bar close timestamp, default) or ``'left'`` (bar open timestamp).
    :param additional_aggs: Optional custom Polars aggregation expressions.
    :param sort: When True, sorts chronologically per group before bucketing.
    :return: Dollar-bar DataFrame or LazyFrame with ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``vwap``, ``dollar_volume``, and ``n_obs`` columns.
    """
    check_positive(threshold, "threshold")
    check_positive(multiplier, "multiplier")
    working_df, schema, gb_cols = _prepare_information_bar_frame(
        df, threshold, label, close_col, volume_col, group_by, time_col, sort, require_volume=True
    )

    dvol_expr = (
        pl.col(close_col).cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0)
        * pl.col(volume_col).cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0)
        * float(multiplier)
    )
    cum_dvol = dvol_expr.cum_sum().over(gb_cols) if gb_cols else dvol_expr.cum_sum()
    aggs = _build_information_bar_aggs(
        schema,
        time_col,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col,
        label,
        additional_aggs,
        extra_cols=[pl.col("_anta_dvol").sum().alias("dollar_volume")],
    )

    return (
        working_df.with_columns(
            [dvol_expr.alias("_anta_dvol"), _bucket_id_from_cumulative(cum_dvol, threshold)]
        )
        .group_by([*gb_cols, "_anta_bar_id"], maintain_order=True)
        .agg(aggs)
        .drop("_anta_bar_id")
    )


def resample_imbalance_bars(
    df: FrameType,
    threshold: float,
    close_col: str = "close",
    volume_col: str | None = "volume",
    time_col: str | None = "timestamp",
    group_by: str | Sequence[str] | None = None,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    label: Literal["left", "right"] = "right",
    additional_aggs: Sequence[pl.Expr] | None = None,
    sort: bool = True,
) -> FrameType:
    """Resample tick or lower-timeframe OHLCV data into Tick or Volume Imbalance Bars (TIB / VIB).

    Classifies each trade/bar direction $b_t \\in \\{-1, +1\\}$ via the Lee-Ready tick rule
    (propagating the previous sign when $\\Delta P_t = 0$) and samples a new bar whenever
    the absolute cumulative signed imbalance $|\\sum \\theta_t| \\ge \\text{threshold}$, where
    $\\theta_t = b_t \\cdot v_t$ (Volume Imbalance Bars when ``volume_col`` is provided) or
    $\\theta_t = b_t$ (Tick Imbalance Bars when ``volume_col=None``).

    :param df: Input Polars DataFrame or LazyFrame.
    :param threshold: Absolute signed imbalance threshold triggering a bar close (must be > 0).
    :param close_col: Column name for close/trade price (default ``'close'``).
    :param volume_col: Column name for volume (default ``'volume'`` for VIB, or ``None`` for TIB).
    :param time_col: Optional timestamp column name (default ``'timestamp'``).
    :param group_by: Optional symbol/ticker column(s) for multi-asset universes.
    :param open_col: Column name for open price (falls back to ``close_col`` if absent).
    :param high_col: Column name for high price (falls back to ``close_col`` if absent).
    :param low_col: Column name for low price (falls back to ``close_col`` if absent).
    :param label: ``'right'`` (bar close timestamp, default) or ``'left'`` (bar open timestamp).
    :param additional_aggs: Optional custom Polars aggregation expressions.
    :param sort: When True, sorts chronologically per group before bucketing.
    :return: Imbalance-bar DataFrame or LazyFrame with ``open``, ``high``, ``low``, ``close``,
        ``imbalance``, and ``n_obs`` (plus ``volume`` and ``vwap`` if ``volume_col`` is present).
    """
    working_df, schema, gb_cols = _prepare_information_bar_frame(
        df, threshold, label, close_col, volume_col, group_by, time_col, sort, require_volume=False
    )

    c = pl.col(close_col).cast(pl.Float64)
    diff = c.diff().over(gb_cols) if gb_cols else c.diff()
    raw_sign = pl.when(diff > 0.0).then(1.0).when(diff < 0.0).then(-1.0).otherwise(None)
    tick_sign = (
        raw_sign.forward_fill().over(gb_cols).fill_null(1.0)
        if gb_cols
        else raw_sign.forward_fill().fill_null(1.0)
    )

    theta_expr = (
        tick_sign * pl.col(volume_col).cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0)
        if volume_col is not None
        else tick_sign
    )

    thresh_val = float(threshold)
    _reset_ufunc = np.frompyfunc(
        lambda acc, x: 0.0 if abs(acc + x) >= thresh_val else acc + x, 2, 1
    )

    def _compute_imbalance_bar_ids(s: pl.Series) -> pl.Series:
        vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(vals)
        if n == 0:
            return pl.Series(np.empty(0, dtype=np.int64), dtype=pl.Int64)
        padded = np.empty(n + 1, dtype=object)
        padded[0] = 0.0
        padded[1:] = vals
        acc = _reset_ufunc.accumulate(padded, dtype=object)[1:].astype(np.float64)
        triggered = (acc == 0.0) & (np.abs(vals) > 0.0)
        shifted = np.empty(n, dtype=np.int64)
        shifted[0] = 0
        shifted[1:] = triggered[:-1].astype(np.int64)
        return pl.Series(np.cumsum(shifted), dtype=pl.Int64)

    bar_id_expr = theta_expr.map_batches(_compute_imbalance_bar_ids, return_dtype=pl.Int64)
    if gb_cols:
        bar_id_expr = bar_id_expr.over(gb_cols)

    aggs = _build_information_bar_aggs(
        schema,
        time_col,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col,
        label,
        additional_aggs,
        extra_cols=[pl.col("_anta_theta").sum().alias("imbalance")],
    )

    return (
        working_df.with_columns(
            [theta_expr.alias("_anta_theta"), bar_id_expr.alias("_anta_bar_id")]
        )
        .group_by([*gb_cols, "_anta_bar_id"], maintain_order=True)
        .agg(aggs)
        .drop("_anta_bar_id")
    )


def stitch_futures_roll(
    df: FrameType,
    contract_col: str = "contract",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    method: Literal["back_adjust", "forward_adjust", "ratio"] = "back_adjust",
    roll_measure: Literal["open_to_close", "close_to_close"] = "open_to_close",
    group_by: str | Sequence[str] | None = None,
    time_col: str | None = "timestamp",
    sort: bool = True,
) -> FrameType:
    """Stitch discrete futures contracts into a continuous roll-adjusted price series.

    Eliminates artificial expiration roll gaps (e.g., VN30F1M third-Thursday monthly rolls)
    using pure Polars cumulative expressions:
    - ``'back_adjust'`` (Panama Canal method): preserves current live contract prices
      unchanged and shifts historical contracts by cumulative roll gaps.
    - ``'forward_adjust'``: preserves earliest contract prices unchanged and shifts
      subsequent contracts by cumulative roll gaps.
    - ``'ratio'``: multiplicatively scales historical contracts by cumulative price ratios,
      preserving percentage returns and strictly positive prices.

    :param df: Input Polars DataFrame or LazyFrame containing a contract identifier column.
    :param contract_col: Column identifying the active futures contract (e.g. ``'VN30F2610'``).
    :param open_col: Column name for open price (default ``'open'``).
    :param high_col: Column name for high price (default ``'high'``).
    :param low_col: Column name for low price (default ``'low'``).
    :param close_col: Column name for close price (default ``'close'``).
    :param method: Adjustment method: ``'back_adjust'`` (default), ``'forward_adjust'``, or ``'ratio'``.
    :param roll_measure: How the roll gap is measured on the transition bar:
        ``'open_to_close'`` (``Open_new[t] - Close_old[t-1]``, default) or
        ``'close_to_close'`` (``Close_new[t] - Close_old[t-1]``).
    :param group_by: Optional root symbol column(s) for multi-instrument futures panels.
    :param time_col: Optional timestamp column used when ``sort=True``.
    :param sort: When True, sorts chronologically per group before computing roll adjustments.
    :return: Continuous roll-adjusted DataFrame or LazyFrame with an added ``roll_adjustment`` column.
    """
    if method not in ("back_adjust", "forward_adjust", "ratio"):
        raise ValueError("method must be one of 'back_adjust', 'forward_adjust', or 'ratio'")
    if roll_measure not in ("open_to_close", "close_to_close"):
        raise ValueError("roll_measure must be either 'open_to_close' or 'close_to_close'")

    schema = _get_schema(df)
    if contract_col not in schema:
        raise ValueError(f"Required contract column '{contract_col}' not found in schema.")
    if close_col not in schema:
        raise ValueError(f"Required close column '{close_col}' not found in schema.")

    working_df, gb_cols = _resolve_group_and_sort(df, schema, group_by, time_col, sort)

    price_cols = [c for c in (open_col, high_col, low_col, close_col) if c in schema]
    ref_new_col = (
        open_col if (roll_measure == "open_to_close" and open_col in schema) else close_col
    )

    prev_contract = (
        pl.col(contract_col).shift(1).over(gb_cols) if gb_cols else pl.col(contract_col).shift(1)
    )
    prev_close = (
        pl.col(close_col).cast(pl.Float64).shift(1).over(gb_cols)
        if gb_cols
        else pl.col(close_col).cast(pl.Float64).shift(1)
    )
    is_roll = (
        pl.col(contract_col).is_not_null()
        & prev_contract.is_not_null()
        & (pl.col(contract_col) != prev_contract)
    )

    if method in ("back_adjust", "forward_adjust"):
        step_gap = (
            pl.when(is_roll)
            .then(pl.col(ref_new_col).cast(pl.Float64) - prev_close)
            .otherwise(0.0)
            .fill_null(0.0)
        )
        cum_gap = step_gap.cum_sum().over(gb_cols) if gb_cols else step_gap.cum_sum()
        if method == "back_adjust":
            total_gap = step_gap.sum().over(gb_cols) if gb_cols else step_gap.sum()
            adj_expr = (total_gap - cum_gap).alias("roll_adjustment")
        else:
            adj_expr = (-cum_gap).alias("roll_adjustment")

        return working_df.with_columns(adj_expr).with_columns(
            [
                (pl.col(col).cast(pl.Float64) + pl.col("roll_adjustment")).alias(col)
                for col in price_cols
            ]
        )

    step_ratio = (
        pl.when(is_roll & (prev_close > 0.0) & (pl.col(ref_new_col).cast(pl.Float64) > 0.0))
        .then(pl.col(ref_new_col).cast(pl.Float64) / prev_close)
        .otherwise(1.0)
        .fill_null(1.0)
    )
    log_r = step_ratio.log()
    cum_log_r = log_r.cum_sum().over(gb_cols) if gb_cols else log_r.cum_sum()
    total_log_r = log_r.sum().over(gb_cols) if gb_cols else log_r.sum()
    factor_expr = (total_log_r - cum_log_r).exp().alias("roll_adjustment")

    return working_df.with_columns(factor_expr).with_columns(
        [
            (pl.col(col).cast(pl.Float64) * pl.col("roll_adjustment")).alias(col)
            for col in price_cols
        ]
    )
