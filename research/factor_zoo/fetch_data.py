"""Public daily OHLCV downloader for HOSE equities (vnstock-compatible public endpoints).

Fetches adjusted daily OHLCV series from 2012-01-01 onwards for all 3-letter HOSE equity
tickers (both currently listed and delisted where available) and writes the local cache
``research/data/vn_daily.parquet`` (gitignored).
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

from research.factor_zoo.config import DATA_CACHE_PATH, DEFAULT_CONFIG, StudyConfig

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) polars-anta-research/0.4"


def fetch_hose_tickers() -> tuple[list[str], dict[str, int]]:
    """Fetch 3-letter HOSE stock codes (including delisted records) from public registry."""
    url = "https://api-finfo.vndirect.com.vn/v4/stocks?q=type:stock~floor:HOSE&size=1000"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    status_counts: dict[str, int] = {}
    tickers: list[str] = []

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for item in payload.get("data", []):
            code = str(item.get("code", "")).strip().upper()
            status = str(item.get("status", "unknown")).strip().lower()
            if len(code) == 3 and code.isalpha():
                tickers.append(code)
                status_counts[status] = status_counts.get(status, 0) + 1
    except Exception:
        # Fallback to SSI iBoard HOSE directory
        ssi_url = "https://iboard-query.ssi.com.vn/stock/exchange/hose"
        req_ssi = urllib.request.Request(
            ssi_url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req_ssi, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for item in payload.get("data", []):
            code = str(item.get("stockSymbol", "")).strip().upper()
            if len(code) == 3 and code.isalpha():
                tickers.append(code)
                status_counts["listed"] = status_counts.get("listed", 0) + 1

    unique_tickers = sorted(set(tickers))
    return unique_tickers, status_counts


def _fetch_single_ticker(
    ticker: str,
    start_ts: int,
    end_ts: int,
) -> pl.DataFrame | None:
    """Download daily OHLCV for a single ticker from Entrade or VNDIRECT dchart."""
    urls = [
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from={start_ts}&to={end_ts}&resolution=1D&symbol={ticker}",
        f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={ticker}&from={start_ts}&to={end_ts}",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            timestamps = data.get("t", [])
            if not timestamps:
                continue
            df = pl.DataFrame(
                {
                    "ts": pl.Series(timestamps, dtype=pl.Int64),
                    "open": pl.Series(data["o"], dtype=pl.Float64),
                    "high": pl.Series(data["h"], dtype=pl.Float64),
                    "low": pl.Series(data["l"], dtype=pl.Float64),
                    "close": pl.Series(data["c"], dtype=pl.Float64),
                    "volume": pl.Series(data["v"], dtype=pl.Float64),
                }
            )
            # Convert unix seconds (UTC+7 trading day) to Date and scale 1,000-VND quotes to VND
            med_close = float(df["close"].median() or 0.0)
            price_mult = 1000.0 if 0.0 < med_close < 1000.0 else 1.0
            df = (
                df.with_columns(
                    pl.from_epoch(pl.col("ts") + 7 * 3600, time_unit="s").dt.date().alias("date"),
                    pl.lit(ticker).alias("ticker"),
                    (pl.col("open") * price_mult).alias("open"),
                    (pl.col("high") * price_mult).alias("high"),
                    (pl.col("low") * price_mult).alias("low"),
                    (pl.col("close") * price_mult).alias("close"),
                )
                .select(["date", "ticker", "open", "high", "low", "close", "volume"])
                .unique(subset=["date", "ticker"], keep="last")
                .sort("date")
            )
            return df
        except urllib.error.HTTPError:
            continue
        except Exception:
            time.sleep(0.05)
            continue
    return None


def build_cache(
    config: StudyConfig = DEFAULT_CONFIG,
    cache_path: Path = DATA_CACHE_PATH,
    force: bool = False,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Build or load ``research/data/vn_daily.parquet``."""
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and not force:
        df = pl.read_parquet(cache_path)
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {"fetched_tickers": int(df["ticker"].n_unique())}
        )
        return df, meta

    tickers, status_counts = fetch_hose_tickers()
    start_dt = dt.datetime.combine(config.warmup_start, dt.time(0, 0), tzinfo=dt.timezone.utc)
    start_ts = int(start_dt.timestamp())
    end_ts = int(time.time()) + 86400

    frames: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_single_ticker, sym, start_ts, end_ts): sym for sym in tickers}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is not None and res.height > 0:
                frames.append(res)

    if not frames:
        raise RuntimeError("Failed to download any HOSE ticker history.")

    panel = (
        pl.concat(frames, how="vertical")
        .filter(pl.col("date") >= config.warmup_start)
        .sort(["date", "ticker"])
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(cache_path)

    meta_info = {
        "directory_total_tickers": len(tickers),
        "directory_listed": int(status_counts.get("listed", 0)),
        "directory_delisted": int(status_counts.get("delisted", 0)),
        "fetched_tickers": int(panel["ticker"].n_unique()),
        "raw_rows": int(panel.height),
    }
    meta_path.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")
    return panel, meta_info


if __name__ == "__main__":
    df_out, meta_out = build_cache()
    print(f"Cached {df_out.height:,} rows across {df_out['ticker'].n_unique()} HOSE tickers.")
    print("Metadata:", json.dumps(meta_out, indent=2))
