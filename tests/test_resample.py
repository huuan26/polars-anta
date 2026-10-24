"""Unit tests for OHLCV resampling and data integrity validation."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from polars_anta.resample import resample_ohlcv, validate_ohlcv


def test_resample_single_asset_basic(sample_time_ohlcv_df: pl.DataFrame):
    """Test standard 5m resampling of 1m OHLCV data for a single asset."""
    res = resample_ohlcv(sample_time_ohlcv_df, every="5m")

    # 30 1-minute rows resampled to 5-minute bars -> 6 bars
    assert len(res) == 6
    assert res.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    # Verify first 5m bucket (rows 0..4)
    first_bucket = sample_time_ohlcv_df.slice(0, 5)
    row0 = res.row(0, named=True)
    assert row0["open"] == first_bucket["open"][0]
    assert row0["high"] == first_bucket["high"].max()
    assert row0["low"] == first_bucket["low"].min()
    assert row0["close"] == first_bucket["close"][-1]
    assert row0["volume"] == first_bucket["volume"].sum()


def test_resample_multi_asset(multi_asset_time_ohlcv_df: pl.DataFrame):
    """Test multi-ticker dynamic grouping resampling."""
    res = resample_ohlcv(multi_asset_time_ohlcv_df, every="10m", group_by="ticker")

    # 3 tickers * (30 minutes / 10m) = 9 rows
    assert len(res) == 9
    assert "ticker" in res.columns
    tickers = res["ticker"].unique().sort().to_list()
    assert tickers == ["AAPL", "GOOG", "MSFT"]

    # Verify AAPL first 10m bar (rows 0..9)
    aapl_orig = multi_asset_time_ohlcv_df.filter(pl.col("ticker") == "AAPL").slice(0, 10)
    aapl_res = res.filter(pl.col("ticker") == "AAPL").row(0, named=True)

    assert aapl_res["open"] == aapl_orig["open"][0]
    assert aapl_res["high"] == aapl_orig["high"].max()
    assert aapl_res["low"] == aapl_orig["low"].min()
    assert aapl_res["close"] == aapl_orig["close"][-1]
    assert aapl_res["volume"] == aapl_orig["volume"].sum()


def test_resample_multi_column_group_by():
    """Test resampling when grouped by multiple columns (e.g., exchange and symbol)."""
    start = datetime(2024, 1, 1, 9, 30)
    rows = []
    for exchange, symbol in [("NASDAQ", "AAPL"), ("NYSE", "IBM")]:
        for i in range(10):
            rows.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timestamp": start + timedelta(minutes=i),
                    "open": 100.0 + i,
                    "high": 102.0 + i,
                    "low": 99.0 + i,
                    "close": 101.0 + i,
                    "volume": 500,
                }
            )
    df = pl.DataFrame(rows)
    res = resample_ohlcv(df, every="5m", group_by=["exchange", "symbol"])

    # 2 symbols * 2 bars = 4 rows
    assert len(res) == 4
    assert "exchange" in res.columns
    assert "symbol" in res.columns


def test_resample_lazyframe(sample_time_ohlcv_df: pl.DataFrame):
    """Test resampling on a Polars LazyFrame."""
    lf = sample_time_ohlcv_df.lazy()
    res_lf = resample_ohlcv(lf, every="15m")
    assert isinstance(res_lf, pl.LazyFrame)

    res_df = res_lf.collect()
    assert len(res_df) == 2
    assert "open" in res_df.columns
    assert "close" in res_df.columns


def test_resample_custom_column_names():
    """Test resampling with custom column names."""
    start = datetime(2024, 1, 1, 9, 30)
    df = pl.DataFrame(
        {
            "time": [start + timedelta(minutes=i) for i in range(10)],
            "o": [10.0 + i for i in range(10)],
            "h": [12.0 + i for i in range(10)],
            "l": [9.0 + i for i in range(10)],
            "c": [11.0 + i for i in range(10)],
            "v": [100.0] * 10,
        }
    )
    res = resample_ohlcv(
        df,
        every="5m",
        time_col="time",
        open_col="o",
        high_col="h",
        low_col="l",
        close_col="c",
        volume_col="v",
    )
    assert len(res) == 2
    assert res.columns == ["time", "o", "h", "l", "c", "v"]


def test_resample_no_volume(sample_time_ohlcv_df: pl.DataFrame):
    """Test resampling when volume column is omitted."""
    df_no_vol = sample_time_ohlcv_df.drop("volume")
    res = resample_ohlcv(df_no_vol, every="5m", volume_col=None)
    assert "volume" not in res.columns
    assert len(res) == 6


def test_resample_additional_aggs(sample_time_ohlcv_df: pl.DataFrame):
    """Test custom additional aggregation expressions during resampling."""
    df_extra = sample_time_ohlcv_df.with_columns(pl.lit(1).alias("trades"))
    res = resample_ohlcv(
        df_extra,
        every="5m",
        additional_aggs=[pl.col("trades").sum().alias("total_trades")],
    )
    assert "total_trades" in res.columns
    assert res["total_trades"].to_list() == [5, 5, 5, 5, 5, 5]


def test_resample_invalid_inputs(sample_time_ohlcv_df: pl.DataFrame):
    """Test input validation exceptions for resampling."""
    # Missing time col
    with pytest.raises(ValueError, match="Required timestamp column"):
        resample_ohlcv(sample_time_ohlcv_df, every="5m", time_col="non_existent")

    # Missing open col
    with pytest.raises(ValueError, match="Required open column"):
        resample_ohlcv(sample_time_ohlcv_df, every="5m", open_col="non_existent")

    # Non-temporal timestamp
    df_bad_time = sample_time_ohlcv_df.with_columns(
        pl.col("timestamp").dt.to_string("%Y-%m-%d %H:%M:%S")
    )
    with pytest.raises(ValueError, match="must be Date or Datetime"):
        resample_ohlcv(df_bad_time, every="5m")

    # Missing group_by col
    with pytest.raises(ValueError, match="Group-by column"):
        resample_ohlcv(sample_time_ohlcv_df, every="5m", group_by="missing_symbol")

    # Specified volume col not found
    with pytest.raises(ValueError, match="Specified volume column"):
        resample_ohlcv(sample_time_ohlcv_df, every="5m", volume_col="missing_vol")


def test_validate_ohlcv_valid(
    sample_time_ohlcv_df: pl.DataFrame, multi_asset_time_ohlcv_df: pl.DataFrame
):
    """Test validation on valid OHLCV data."""
    # Strict mode returns df unchanged
    validated = validate_ohlcv(sample_time_ohlcv_df)
    assert len(validated) == len(sample_time_ohlcv_df)

    # Multi-ticker validation with group_by
    multi_val = validate_ohlcv(multi_asset_time_ohlcv_df, group_by="ticker")
    assert len(multi_val) == len(multi_asset_time_ohlcv_df)

    # Report mode returns valid=True and 0 metrics
    report = validate_ohlcv(sample_time_ohlcv_df, return_report=True)
    assert report["valid"] is True
    assert len(report["errors"]) == 0
    assert report["metrics"]["high_lt_low"] == 0
    assert report["metrics"]["unsorted_timestamp"] == 0


def test_validate_ohlcv_schema_errors():
    """Test schema validation errors."""
    df_missing_high = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "open": [10.0],
            "low": [9.0],
            "close": [10.5],
        }
    )
    with pytest.raises(ValueError, match="Missing required high column"):
        validate_ohlcv(df_missing_high)

    report = validate_ohlcv(df_missing_high, return_report=True)
    assert report["valid"] is False
    assert any("high" in e for e in report["errors"])

    # Non-temporal timestamp
    df_str_time = df_missing_high.with_columns(
        pl.lit(12.0).alias("high"), pl.col("timestamp").dt.to_string("%Y-%m-%d")
    )
    with pytest.raises(ValueError, match="must be Date or Datetime"):
        validate_ohlcv(df_str_time)

    # Missing timestamp column
    with pytest.raises(ValueError, match="Missing required timestamp column"):
        validate_ohlcv(df_missing_high, time_col="missing_time")

    # Non-numeric price column
    df_str_open = df_missing_high.with_columns(
        pl.lit(12.0).alias("high"), pl.lit("invalid").alias("open")
    )
    with pytest.raises(ValueError, match="must be numeric"):
        validate_ohlcv(df_str_open)

    # Non-numeric volume column
    df_str_vol = df_missing_high.with_columns(
        pl.lit(12.0).alias("high"), pl.lit("not_num").alias("volume")
    )
    with pytest.raises(ValueError, match="Volume column 'volume' must be numeric"):
        validate_ohlcv(df_str_vol)

    # Non-existent specified volume column
    with pytest.raises(ValueError, match="Specified volume column 'turnover' not found"):
        validate_ohlcv(
            df_missing_high.with_columns(pl.lit(12.0).alias("high")), volume_col="turnover"
        )

    # Missing group_by column
    with pytest.raises(ValueError, match="Group-by column 'ticker' not found"):
        validate_ohlcv(df_missing_high.with_columns(pl.lit(12.0).alias("high")), group_by="ticker")


def test_validate_ohlcv_price_inconsistencies():
    """Test detection of high < low, high < open, low > close, negative prices, etc."""
    df_bad = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 9, 31)],
            "open": [100.0, 105.0],
            "high": [98.0, 110.0],  # row 0 has high < low (98 < 99)
            "low": [99.0, 108.0],  # row 1 has low > close (108 > 102)
            "close": [101.0, 102.0],
            "volume": [1000, -50],  # row 1 has negative volume
        }
    )

    with pytest.raises(ValueError, match="OHLCV validation failed"):
        validate_ohlcv(df_bad)

    report = validate_ohlcv(df_bad, return_report=True)
    assert report["valid"] is False
    assert report["metrics"]["high_lt_low"] == 1
    assert report["metrics"]["low_gt_close"] == 1
    assert report["metrics"]["negative_volume"] == 1


def test_validate_ohlcv_nulls_and_unsorted():
    """Test null detection and unsorted timestamp detection."""
    df_nulls = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 9, 29)],
            "open": [100.0, None],
            "high": [105.0, 106.0],
            "low": [95.0, 96.0],
            "close": [102.0, 103.0],
            "volume": [100, 200],
        }
    )
    report = validate_ohlcv(df_nulls, return_report=True)
    assert report["valid"] is False
    assert report["metrics"]["null_open"] == 1
    assert report["metrics"]["unsorted_timestamp"] == 1


def test_validate_ohlcv_all_negative_and_null_metrics():
    """Test all negative price metrics and null columns metrics."""
    df_negs = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 30), None],
            "open": [-10.0, 10.0],
            "high": [-5.0, None],
            "low": [-20.0, None],
            "close": [-15.0, None],
            "volume": [100, None],
        }
    )
    report = validate_ohlcv(df_negs, return_report=True)
    assert report["valid"] is False
    assert report["metrics"]["negative_open"] == 1
    assert report["metrics"]["negative_high"] == 1
    assert report["metrics"]["negative_low"] == 1
    assert report["metrics"]["negative_close"] == 1
    assert report["metrics"]["null_high"] == 1
    assert report["metrics"]["null_low"] == 1
    assert report["metrics"]["null_close"] == 1
    assert report["metrics"]["null_volume"] == 1
    assert report["metrics"]["null_timestamp"] == 1

    # Test check_data=False only validates schema
    df_no_check_data = validate_ohlcv(df_negs, check_data=False)
    assert len(df_no_check_data) == 2


def test_validate_ohlcv_lazy(sample_time_ohlcv_df: pl.DataFrame):
    """Test validation on Polars LazyFrame."""
    lf = sample_time_ohlcv_df.lazy()
    res = validate_ohlcv(lf)
    assert isinstance(res, pl.LazyFrame)
    assert len(res.collect()) == len(sample_time_ohlcv_df)
