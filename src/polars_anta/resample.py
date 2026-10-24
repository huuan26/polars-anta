"""Multi-timeframe OHLCV resampling and data validation for Polars DataFrames and LazyFrames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import polars as pl

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)


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
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    errors: list[str] = []

    # 1. Schema checks
    price_cols = [("open", open_col), ("high", high_col), ("low", low_col), ("close", close_col)]
    for role, col in price_cols:
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
            check_exprs.append(
                (pl.col(volume_col) < 0).fill_null(False).sum().alias("negative_volume")
            )
            check_exprs.append(pl.col(volume_col).is_null().sum().alias("null_volume"))

        if time_col is not None:
            check_exprs.append(pl.col(time_col).is_null().sum().alias("null_timestamp"))
            if group_by is not None:
                check_exprs.append(
                    (pl.col(time_col) < pl.col(time_col).shift(1))
                    .over(group_by)
                    .fill_null(False)
                    .sum()
                    .alias("unsorted_timestamp")
                )
            else:
                check_exprs.append(
                    (pl.col(time_col) < pl.col(time_col).shift(1))
                    .fill_null(False)
                    .sum()
                    .alias("unsorted_timestamp")
                )

        eval_df = df.select(check_exprs)
        if isinstance(eval_df, pl.LazyFrame):
            eval_res = eval_df.collect().to_dicts()[0]
        else:
            eval_res = eval_df.to_dicts()[0]

        for k, v in eval_res.items():
            metrics[k] = int(v) if v is not None else 0

        if metrics.get("high_lt_low", 0) > 0:
            errors.append(f"{metrics['high_lt_low']} rows where high < low")
        if metrics.get("high_lt_open", 0) > 0:
            errors.append(f"{metrics['high_lt_open']} rows where high < open")
        if metrics.get("high_lt_close", 0) > 0:
            errors.append(f"{metrics['high_lt_close']} rows where high < close")
        if metrics.get("low_gt_open", 0) > 0:
            errors.append(f"{metrics['low_gt_open']} rows where low > open")
        if metrics.get("low_gt_close", 0) > 0:
            errors.append(f"{metrics['low_gt_close']} rows where low > close")
        if metrics.get("negative_open", 0) > 0:
            errors.append(f"{metrics['negative_open']} rows with negative open price")
        if metrics.get("negative_high", 0) > 0:
            errors.append(f"{metrics['negative_high']} rows with negative high price")
        if metrics.get("negative_low", 0) > 0:
            errors.append(f"{metrics['negative_low']} rows with negative low price")
        if metrics.get("negative_close", 0) > 0:
            errors.append(f"{metrics['negative_close']} rows with negative close price")
        if metrics.get("negative_volume", 0) > 0:
            errors.append(f"{metrics['negative_volume']} rows with negative volume")
        if metrics.get("null_open", 0) > 0:
            errors.append(f"{metrics['null_open']} null values in open")
        if metrics.get("null_high", 0) > 0:
            errors.append(f"{metrics['null_high']} null values in high")
        if metrics.get("null_low", 0) > 0:
            errors.append(f"{metrics['null_low']} null values in low")
        if metrics.get("null_close", 0) > 0:
            errors.append(f"{metrics['null_close']} null values in close")
        if metrics.get("null_volume", 0) > 0:
            errors.append(f"{metrics['null_volume']} null values in volume")
        if metrics.get("null_timestamp", 0) > 0:
            errors.append(f"{metrics['null_timestamp']} null values in timestamp")
        if metrics.get("unsorted_timestamp", 0) > 0:
            errors.append(f"{metrics['unsorted_timestamp']} out-of-order timestamps")

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
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema

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

    # Validate group_by columns
    gb_cols: list[str] = []
    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        for g_col in gb_cols:
            if g_col not in schema:
                raise ValueError(f"Group-by column '{g_col}' not found in schema.")

    # Sort if requested
    sort_keys = gb_cols + [time_col] if gb_cols else [time_col]
    working_df = df.sort(sort_keys) if sort else df

    # Build candle aggregations
    aggs: list[pl.Expr] = [
        pl.col(open_col).first().alias(open_col),
        pl.col(high_col).max().alias(high_col),
        pl.col(low_col).min().alias(low_col),
        pl.col(close_col).last().alias(close_col),
    ]

    # Handle volume
    if volume_col is not None:
        if volume_col in schema:
            aggs.append(pl.col(volume_col).sum().alias(volume_col))
        elif volume_col != "volume":
            raise ValueError(f"Specified volume column '{volume_col}' not found in schema.")

    # Custom additional aggregations
    if additional_aggs:
        aggs.extend(additional_aggs)

    # Perform dynamic grouping (Polars 2.0 uses group_by parameter)
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

    if sort:
        resampled = resampled.sort(sort_keys)

    return resampled
