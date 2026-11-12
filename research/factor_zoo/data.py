"""Point-in-time universe construction and OHLCV cleaning for the Factor Zoo study."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from polars_anta.resample import validate_ohlcv
from research.factor_zoo.config import DATA_CACHE_PATH, DEFAULT_CONFIG, StudyConfig
from research.factor_zoo.fetch_data import build_cache


def load_panel(
    config: StudyConfig = DEFAULT_CONFIG,
    cache_path: Path = DATA_CACHE_PATH,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Load, validate, clean, and tag the point-in-time HOSE daily equity panel.

    Cleaning pipeline (Section 1.4 of pre-registration):
    1. Run ``polars_anta.validate_ohlcv(..., return_report=True)`` and drop rows violating
       structural candle invariants (``high >= max(open, close, low)``, ``low <= min(open, close, high)``,
       strictly positive prices, non-negative volume, and non-null values).
    2. Drop zero-volume sessions (``volume <= 0``) from each ticker's trading series.
    3. Flag and drop daily close-to-close return jumps exceeding the HOSE daily price limit plus
       tolerance (``|r_t| > 0.07 + 0.005 = 0.075``).
    4. Construct the point-in-time investable universe (``in_universe``) per date $t$:
       - ``hist_bars >= 250`` sessions up to $t$,
       - ``close >= 5,000`` VND,
       - 60-session rolling median traded value ``(close * volume)`` in the top 150 on date $t$.
    5. Compute open-to-open forward returns ``fwd_ret_1``, ``fwd_ret_5``, ``fwd_ret_20``:
       ``r_{t,h} = open_{t+1+h} / open_{t+1} - 1``.
    """
    raw_df, fetch_meta = build_cache(config=config, cache_path=cache_path)
    raw_df = raw_df.sort(["ticker", "date"])
    n_raw = int(raw_df.height)

    # Step 1: Audit with polars_anta.validate_ohlcv and filter structural violations
    val_report = validate_ohlcv(
        raw_df,
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        volume_col="volume",
        time_col="date",
        group_by="ticker",
        return_report=True,
    )
    valid_candle_mask = (
        pl.col("open").is_not_null()
        & pl.col("high").is_not_null()
        & pl.col("low").is_not_null()
        & pl.col("close").is_not_null()
        & pl.col("volume").is_not_null()
        & (pl.col("open") > 0.0)
        & (pl.col("high") > 0.0)
        & (pl.col("low") > 0.0)
        & (pl.col("close") > 0.0)
        & (pl.col("volume") >= 0.0)
        & (pl.col("high") >= pl.max_horizontal("open", "close", "low"))
        & (pl.col("low") <= pl.min_horizontal("open", "close", "high"))
    )
    df_step1 = raw_df.filter(valid_candle_mask)
    n_after_step1 = int(df_step1.height)
    n_dropped_invalid_ohlcv = n_raw - n_after_step1

    # Step 2: Drop zero-volume (non-trading) days
    df_step2 = df_step1.filter(pl.col("volume") > 0.0)
    n_after_step2 = int(df_step2.height)
    n_dropped_zero_volume = n_after_step1 - n_after_step2

    # Step 3: Flag & drop daily close-to-close return jumps > HOSE limit + 0.5% tolerance (7.5%)
    max_abs_ret = config.hose_price_limit + config.price_limit_tolerance
    df_step3_all = df_step2.sort(["ticker", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("_ret_1d")
    )
    df_step3 = df_step3_all.filter(
        pl.col("_ret_1d").is_null() | (pl.col("_ret_1d").abs() <= max_abs_ret)
    ).drop("_ret_1d")
    n_after_step3 = int(df_step3.height)
    n_dropped_price_limit = n_after_step2 - n_after_step3

    # Step 4: Point-in-time universe & open-to-open forward returns
    horizons = (config.primary_horizon, *config.exploratory_horizons)
    fwd_exprs = [
        (
            pl.col("open").shift(-(1 + h)).over("ticker") / pl.col("open").shift(-1).over("ticker")
            - 1.0
        ).alias(f"fwd_ret_{h}")
        for h in sorted(set(horizons))
    ]

    df_enriched = (
        df_step3.sort(["ticker", "date"])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("ticker").alias("hist_bars"),
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap_proxy"),
            (pl.col("close") * pl.col("volume"))
            .rolling_median(window_size=config.adv_window, min_samples=config.adv_window)
            .over("ticker")
            .alias("adv_60"),
            *fwd_exprs,
        )
        .with_columns(
            (
                (pl.col("hist_bars") >= config.min_history_bars)
                & (pl.col("close") >= config.min_close_vnd)
                & pl.col("adv_60").is_not_null()
            ).alias("_eligible")
        )
        .with_columns(
            pl.when(pl.col("_eligible"))
            .then(pl.col("adv_60"))
            .otherwise(None)
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("_liq_rank")
        )
        .with_columns(
            (pl.col("_eligible") & (pl.col("_liq_rank") <= config.top_n_liquid)).alias(
                "in_universe"
            )
        )
        .drop(["_eligible", "_liq_rank"])
        .sort(["date", "ticker"])
    )

    # Compute descriptive statistics for post-2014 study period
    study_df = df_enriched.filter(
        (pl.col("date") >= config.discovery_start) & pl.col("in_universe")
    )
    daily_counts = study_df.group_by("date").len().sort("date")
    disc_df = study_df.filter(pl.col("date") <= config.discovery_end)
    hold_df = study_df.filter(pl.col("date") >= config.holdout_start)

    disc_daily = disc_df.group_by("date").len()
    hold_daily = hold_df.group_by("date").len()

    summary: dict[str, object] = {
        "exchange": config.exchange,
        "fetch_metadata": fetch_meta,
        "validation_errors_count": len(val_report.get("errors", []))
        if isinstance(val_report, dict)
        else 0,
        "validation_metrics": val_report.get("metrics", {}) if isinstance(val_report, dict) else {},
        "raw_rows": n_raw,
        "rows_dropped_invalid_ohlcv": n_dropped_invalid_ohlcv,
        "rows_dropped_zero_volume": n_dropped_zero_volume,
        "rows_dropped_price_limit": n_dropped_price_limit,
        "cleaned_rows_total": n_after_step3,
        "study_universe_rows": int(study_df.height),
        "unique_tickers_raw": int(raw_df["ticker"].n_unique()),
        "unique_tickers_cleaned": int(df_step3["ticker"].n_unique()),
        "unique_tickers_universe": int(study_df["ticker"].n_unique()),
        "date_min_raw": str(raw_df["date"].min()),
        "date_max_raw": str(raw_df["date"].max()),
        "date_min_study": str(study_df["date"].min()),
        "date_max_study": str(study_df["date"].max()),
        "total_study_days": int(daily_counts.height),
        "discovery_days": int(disc_daily.height),
        "discovery_rows": int(disc_df.height),
        "discovery_unique_tickers": int(disc_df["ticker"].n_unique()),
        "holdout_days": int(hold_daily.height),
        "holdout_rows": int(hold_df.height),
        "holdout_unique_tickers": int(hold_df["ticker"].n_unique()),
        "stocks_per_day_min": int(daily_counts["len"].min() or 0),
        "stocks_per_day_median": float(daily_counts["len"].median() or 0.0),
        "stocks_per_day_max": int(daily_counts["len"].max() or 0),
    }
    return df_enriched, summary


if __name__ == "__main__":
    panel_df, info = load_panel()
    print(json.dumps(info, indent=2))
