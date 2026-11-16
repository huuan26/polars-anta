"""Real-market out-of-sample cross-sectional alpha engineering benchmark across 3 regimes.

Evaluates the 4-stage `polars-anta` pipeline (`conditioning` -> `diagnostics` -> `covariance` ->
`allocation`) out of sample on three real-world market datasets cached in gitignored `research/data/`:
1. Global Crypto Dollar-Neutral Long-Short (`data-api.binance.vision`, 24 liquid USDT pairs,
   real `taker_buy_volume` orderflow, `10 bps` one-way execution cost, 365-day annualization).
2. US Equities & Sector ETFs Dollar-Neutral Long-Short (`stooq.com` primary with automatic fallback
   to `query1.finance.yahoo.com` split/dividend-adjusted daily OHLCV when Stooq DNS/rate-limit is
   unavailable, 30 liquid S&P large-caps & Sector SPDRs across 6 sector groups, `10 bps` one-way cost).
3. Vietnam Emerging-Market Long-Only (`HOSE` point-in-time top-150 universe from
   `research/data/vn_daily.parquet`, Locked Holdout `2022-01-01` to `2026-11-10`, `30 bps` one-way /
   `60 bps` round-trip execution cost).

Generates `docs/assets/real_market_alpha_engineering.md` directly from script execution.

Run from the repository root:
    python benchmarks/real_market_alpha_engineering.py
"""

from __future__ import annotations

import io
import json
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.factor_zoo.data import load_panel as load_hose_panel  # noqa: E402

import polars_anta as pa  # noqa: E402

DATA_DIR = REPO_ROOT / "research" / "data"
CRYPTO_CACHE = DATA_DIR / "binance_crypto_24_1d.parquet"
US_CACHE = DATA_DIR / "us_equities_30_1d.parquet"
US_META_CACHE = DATA_DIR / "us_equities_30_1d.meta.json"

CRYPTO_UNIVERSE: dict[str, str] = {
    "BTCUSDT": "Layer1",
    "ETHUSDT": "Layer1",
    "SOLUSDT": "Layer1",
    "ADAUSDT": "Layer1",
    "AVAXUSDT": "Layer1",
    "DOTUSDT": "Layer1",
    "TRXUSDT": "Layer1",
    "NEARUSDT": "Layer1",
    "ATOMUSDT": "Layer1",
    "ETCUSDT": "Layer1",
    "ICPUSDT": "Layer1",
    "ALGOUSDT": "Layer1",
    "UNIUSDT": "DeFi",
    "AAVEUSDT": "DeFi",
    "INJUSDT": "DeFi",
    "LINKUSDT": "Infra",
    "FILUSDT": "Infra",
    "HBARUSDT": "Infra",
    "VETUSDT": "Infra",
    "BNBUSDT": "ExchangePay",
    "XRPUSDT": "ExchangePay",
    "LTCUSDT": "ExchangePay",
    "BCHUSDT": "ExchangePay",
    "DOGEUSDT": "ExchangePay",
}

US_UNIVERSE: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "XLK": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "XLF": "Financials",
    "LLY": "HealthCare",
    "UNH": "HealthCare",
    "JNJ": "HealthCare",
    "ABBV": "HealthCare",
    "XLV": "HealthCare",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "XLE": "Energy",
    "CAT": "Industrials",
    "GE": "Industrials",
    "UNP": "Industrials",
    "XLI": "Industrials",
    "AMZN": "Consumer",
    "HD": "Consumer",
    "COST": "Consumer",
    "PG": "Consumer",
    "KO": "Consumer",
    "XLP": "Consumer",
}


def _http_get(url: str, timeout: float = 12.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "polars-anta/0.5.0 (Empirical Benchmark)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_or_fetch_binance_crypto(limit: int = 1000) -> pl.DataFrame:
    """Fetch 24 liquid Binance USDT pairs with real taker_buy_volume orderflow."""
    if CRYPTO_CACHE.is_file():
        return pl.read_parquet(CRYPTO_CACHE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    frames: list[pl.DataFrame] = []
    for sym, sec in CRYPTO_UNIVERSE.items():
        q = urllib.parse.urlencode({"symbol": sym, "interval": "1d", "limit": limit})
        raw = json.loads(
            _http_get(f"https://data-api.binance.vision/api/v3/klines?{q}").decode("utf-8")
        )
        closed = [r for r in raw if int(r[6]) <= now_ms]
        frames.append(
            pl.DataFrame(
                {
                    "open_ms": [int(r[0]) for r in closed],
                    "ticker": [sym] * len(closed),
                    "sector": [sec] * len(closed),
                    "open": [float(r[1]) for r in closed],
                    "high": [float(r[2]) for r in closed],
                    "low": [float(r[3]) for r in closed],
                    "close": [float(r[4]) for r in closed],
                    "volume": [float(r[5]) for r in closed],
                    "quote_volume": [float(r[7]) for r in closed],
                    "trade_count": [float(r[8]) for r in closed],
                    "taker_buy_volume": [float(r[9]) for r in closed],
                }
            ).with_columns(pl.from_epoch("open_ms", time_unit="ms").dt.date().alias("date"))
        )

    df = (
        pl.concat(frames)
        .drop("open_ms")
        .filter((pl.col("close") > 0.0) & (pl.col("volume") > 0.0))
        .sort(["ticker", "date"])
    )
    pa.validate_ohlcv(df, time_col="date", group_by="ticker", return_report=True)
    df.write_parquet(CRYPTO_CACHE)
    return df


def _fetch_stooq_one(sym: str, sec: str) -> pl.DataFrame:
    raw = _http_get(f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d", timeout=6.0)
    head = raw[:120].decode("utf-8", errors="ignore").lower()
    if not head or "exceeded" in head or "no data" in head:
        raise RuntimeError(f"Stooq unavailable for {sym}")
    df = pl.read_csv(io.BytesIO(raw), try_parse_dates=True)
    df = df.rename({c: c.strip().lower() for c in df.columns})
    return (
        df.with_columns(
            pl.col("date").cast(pl.Date),
            pl.lit(sym).alias("ticker"),
            pl.lit(sec).alias("sector"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        .sort("date")
        .tail(1260)
        .select(["date", "ticker", "sector", "open", "high", "low", "close", "volume"])
    )


def _fetch_yahoo_one(sym: str, sec: str) -> pl.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5y&interval=1d"
    payload = json.loads(_http_get(url).decode("utf-8"))["chart"]["result"][0]
    ts = payload["timestamp"]
    q = payload["indicators"]["quote"][0]
    adj = payload["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    raw_df = pl.DataFrame(
        {
            "ts": ts,
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "adjclose": adj,
            "volume": q["volume"],
        }
    ).drop_nulls()
    ratio = pl.col("adjclose").cast(pl.Float64) / pl.col("close").cast(pl.Float64)
    return (
        raw_df.with_columns(
            pl.from_epoch("ts", time_unit="s").dt.date().alias("date"),
            pl.lit(sym).alias("ticker"),
            pl.lit(sec).alias("sector"),
            (pl.col("open").cast(pl.Float64) * ratio).alias("open"),
            (pl.col("high").cast(pl.Float64) * ratio).alias("high"),
            (pl.col("low").cast(pl.Float64) * ratio).alias("low"),
            pl.col("adjclose").cast(pl.Float64).alias("close"),
            pl.col("volume").cast(pl.Float64),
        )
        .unique(subset=["date"], keep="last")
        .sort("date")
        .select(["date", "ticker", "sector", "open", "high", "low", "close", "volume"])
    )


def load_or_fetch_us_equities() -> tuple[pl.DataFrame, str]:
    """Fetch 30 US large-cap equities & Sector ETFs (Stooq primary, Yahoo v8 fallback)."""
    if US_CACHE.is_file() and US_META_CACHE.is_file():
        meta = json.loads(US_META_CACHE.read_text(encoding="utf-8"))
        return pl.read_parquet(US_CACHE), str(meta.get("source", "cached"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    used_fallback = False
    stooq_ok = True
    for sym, sec in US_UNIVERSE.items():
        if stooq_ok:
            try:
                frames.append(_fetch_stooq_one(sym, sec))
                continue
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError):
                stooq_ok = False
                used_fallback = True
        frames.append(_fetch_yahoo_one(sym, sec))

    source_note = (
        "stooq.com -> fallback query1.finance.yahoo.com (split/dividend-adjusted)"
        if used_fallback
        else "stooq.com"
    )
    df = (
        pl.concat(frames)
        .filter(
            (pl.col("close") > 0.0)
            & (pl.col("volume") > 0.0)
            & (pl.col("high") >= pl.max_horizontal("open", "close", "low"))
            & (pl.col("low") <= pl.min_horizontal("open", "close", "high"))
        )
        .sort(["ticker", "date"])
    )
    pa.validate_ohlcv(df, time_col="date", group_by="ticker", return_report=True)
    df.write_parquet(US_CACHE)
    US_META_CACHE.write_text(json.dumps({"source": source_note}, indent=2), encoding="utf-8")
    return df, source_note


def _add_causal_market_beta(df: pl.DataFrame) -> pl.DataFrame:
    """Compute strictly causal 60-bar rolling beta against equal-weight universe return."""
    with_ret = df.sort(["ticker", "date"]).with_columns(
        pl.col("close").pct_change().over("ticker").alias("_r1")
    )
    with_mkt = with_ret.with_columns(pl.col("_r1").mean().over("date").alias("_rm"))
    var_m = pa.ts_std("_rm", 60).over("ticker").pow(2)
    cov_im = pa.ts_covariance("_r1", "_rm", 60).over("ticker")
    return with_mkt.with_columns(
        pl.when(var_m > 1e-12).then(cov_im / var_m).otherwise(1.0).fill_null(1.0).alias("beta_60")
    )


def evaluate_four_stages(
    panel: pl.DataFrame,
    raw_cols: list[str],
    risk_cols: list[str],
    group_col: str | None,
    weight_scheme: Literal["dollar_neutral", "long_only"],
    cost_bps: float,
    annualization: float,
    lookback_fast: int,
    lookback_slow: int,
    ema_alpha: float,
    oos_start_date: date,
) -> list[dict[str, float | str]]:
    """Run identical out-of-sample evaluation across all 4 pipeline stages."""
    work = pa.forward_returns(
        panel, close_col="close", periods=(1,), group_by="ticker", prefix="fwd_"
    )

    # Stage 0: Raw z-scored equal-weight composite
    s0 = work.with_columns(
        [pa.cs_zscore(c, over="date").alias(f"{c}_z") for c in raw_cols]
    ).with_columns(pl.mean_horizontal([pl.col(f"{c}_z") for c in raw_cols]).alias("sig_s0"))

    # Stage 1: Gaussianize -> Simultaneous FWL Neutralization -> Batched Lowdin Orthogonalization
    g_cols = [f"{c}_g" for c in raw_cols]
    cond = work.with_columns([pa.cs_gaussianize(c, over="date").alias(f"{c}_g") for c in raw_cols])
    neut = pa.neutralize_panel(
        cond,
        factor_cols=g_cols,
        time_col="date",
        risk_cols=risk_cols,
        group_col=group_col,
    )
    neut_cols = [f"{c}_neut" for c in g_cols]
    ortho = pa.orthogonalize_panel(neut, factor_cols=neut_cols, time_col="date", method="lowdin")
    ortho_cols = [f"{c}_ortho" for c in neut_cols]
    ortho = ortho.with_columns(pl.mean_horizontal([pl.col(c) for c in ortho_cols]).alias("sig_s1"))

    # Stage 2: Unpenalized Walk-Forward FISTA (short noisy window, Sample Cov, zero cost awareness)
    wf_s2 = pa.combine_alphas_walkforward(
        ortho,
        factor_cols=ortho_cols,
        return_col="fwd_1",
        time_col="date",
        asset_col="ticker",
        lookback=lookback_fast,
        horizon=1,
        signal_lag=1,
        objective_mode="ic",
        one_way_cost_bps=0.0,
        risk_aversion=1.0,
        turnover_penalty=0.0,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        cov_estimator="sample",
        composite_col="sig_s2",
    )

    # Stage 3: Full Pipeline (RMT + Factor-Return Asset-Cost Drag + EB + FISTA L1 + Causal EMA)
    wf_s3 = pa.combine_alphas_walkforward(
        ortho,
        factor_cols=ortho_cols,
        return_col="fwd_1",
        time_col="date",
        asset_col="ticker",
        lookback=lookback_slow,
        horizon=1,
        signal_lag=1,
        objective_mode="factor_return",
        one_way_cost_bps=cost_bps,
        signal_ema_alpha=ema_alpha,
        risk_aversion=2.5,
        turnover_penalty=0.01,
        constraint="box",
        lower_bound=-1.0,
        upper_bound=1.0,
        cov_estimator="marchenko_pastur",
        ic_shrinkage="empirical_bayes",
        no_trade_buffer=0.02,
        composite_col="sig_s3",
    )

    stages = [
        ("0. Raw Signals (Equal-Weight)", s0.filter(pl.col("date") >= oos_start_date), "sig_s0"),
        (
            "1. FWL + Lowdin Ortho (Equal-Weight)",
            ortho.filter(pl.col("date") >= oos_start_date),
            "sig_s1",
        ),
        (
            "2. Unpenalized Walk-Forward (Sample Cov)",
            wf_s2.data.filter(pl.col("date") >= oos_start_date),
            "sig_s2",
        ),
        (
            "3. Cost-Aware Pipeline (RMT + Cost Drag + FISTA + EMA)",
            wf_s3.data.filter(pl.col("date") >= oos_start_date),
            "sig_s3",
        ),
    ]
    rows: list[dict[str, float | str]] = []
    for name, frame, col in stages:
        rep = pa.analyze_alpha(
            frame,
            factor_col=col,
            return_col=None,
            close_col="close",
            volume_col="volume",
            asset_col="ticker",
            time_col="date",
            horizon=1,
            signal_lag=1,
            weight_scheme=weight_scheme,
            one_way_cost_bps=cost_bps,
            annualization=annualization,
        )
        s = rep.summary()
        rows.append(
            {
                "stage": name,
                "ic_mean": s["ic_mean"],
                "icir": s["icir"],
                "nw_t": s["nw_t_stat"],
                "turnover": s["mean_turnover"],
                "gross_margin_bps": s["gross_margin_bps"],
                "breakeven_bps": s["breakeven_cost_bps"],
                "gross_sharpe": s["gross_sharpe"],
                "net_sharpe": s["net_sharpe"],
            }
        )
    return rows


def build_crypto_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str], date]:
    base = _add_causal_market_beta(df)
    imb = (2.0 * pl.col("taker_buy_volume") - pl.col("volume")) / pl.col("volume")
    feat = (
        base.sort(["ticker", "date"])
        .with_columns(
            imb.rolling_mean(window_size=10).over("ticker").alias("taker_imb_10"),
            imb.rolling_mean(window_size=5).over("ticker").alias("_imb5"),
            pa.roc("close", 5).over("ticker").alias("_roc5"),
            (pa.roc("close", 20).over("ticker") / pa.stddev("_r1", 20).over("ticker")).alias(
                "sharpe_20"
            ),
            (-pa.roc("close", 3).over("ticker")).alias("rev_3"),
            pa.cmf("high", "low", "close", "quote_volume", 20).over("ticker").alias("cmf_20"),
            (-pa.stddev("_r1", 20).over("ticker")).alias("low_vol_20"),
        )
        .with_columns(
            (pa.cs_zscore("_imb5", over="date") - pa.cs_zscore("_roc5", over="date")).alias(
                "flow_div_5"
            )
        )
    )
    cols = ["taker_imb_10", "flow_div_5", "sharpe_20", "rev_3", "cmf_20", "low_vol_20"]
    dates = feat.select("date").unique().sort("date")["date"].to_list()
    return feat, cols, dates[180]


def build_us_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str], date]:
    base = _add_causal_market_beta(df)
    body = (pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-9)
    feat = base.sort(["ticker", "date"]).with_columns(
        (
            pl.col("close").shift(5).over("ticker") / pl.col("close").shift(60).over("ticker") - 1.0
        ).alias("mom_60_skip5"),
        (-pa.roc("close", 5).over("ticker")).alias("rev_5"),
        (pa.roc("close", 20).over("ticker") / pa.stddev("_r1", 20).over("ticker")).alias(
            "sharpe_20"
        ),
        pa.cmf("high", "low", "close", "volume", 20).over("ticker").alias("cmf_20"),
        body.rolling_mean(window_size=5).over("ticker").alias("body_5"),
        (-pa.amihud_illiquidity("close", "volume", 20).over("ticker")).alias("liq_20"),
    )
    cols = ["mom_60_skip5", "rev_5", "sharpe_20", "cmf_20", "body_5", "liq_20"]
    dates = feat.select("date").unique().sort("date")["date"].to_list()
    return feat, cols, dates[200]


def build_hose_features() -> tuple[pl.DataFrame, list[str], date]:
    full_df, _ = load_hose_panel()
    sub = full_df.filter((pl.col("date") >= date(2020, 9, 1)) & pl.col("in_universe"))
    base = _add_causal_market_beta(sub)
    vwap_20 = pa.vwap("high", "low", "close", "volume", period=20).over("ticker")
    feat = base.sort(["ticker", "date"]).with_columns(
        pl.col("adv_60").log().alias("log_adv_60"),
        (
            pl.col("close").shift(5).over("ticker") / pl.col("close").shift(60).over("ticker") - 1.0
        ).alias("mom_60_skip5"),
        (-pa.roc("close", 5).over("ticker")).alias("rev_5"),
        pa.cmf("high", "low", "close", "volume", 20).over("ticker").alias("cmf_20"),
        (-pa.stddev("_r1", 20).over("ticker")).alias("low_vol_20"),
        (-(pl.col("close") / vwap_20 - 1.0)).alias("vwap_rev_20"),
        pa.amihud_illiquidity("close", "volume", 20).over("ticker").alias("amihud_20"),
    )
    cols = ["mom_60_skip5", "rev_5", "cmf_20", "low_vol_20", "vwap_rev_20", "amihud_20"]
    return feat, cols, date(2022, 1, 1)


def _fmt_table(rows: list[dict[str, float | str]], cost_label: str) -> list[str]:
    out = [
        f"| Pipeline Stage | Rank IC | NW $t$ | Daily Turnover | Gross Margin (`bps`) | Gross Sharpe | Net Sharpe ({cost_label}) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        out.append(
            f"| {r['stage']} | `{r['ic_mean']:+.4f}` | `{r['nw_t']:+.2f}` | "
            f"`{r['turnover']:.3f}` | `{r['gross_margin_bps']:+.2f}` | "
            f"`{r['gross_sharpe']:+.2f}` | `{r['net_sharpe']:+.2f}` |"
        )
    return out


def main() -> None:
    t0 = time.perf_counter()
    crypto_raw = load_or_fetch_binance_crypto()
    us_raw, us_src = load_or_fetch_us_equities()

    c_df, c_cols, c_oos = build_crypto_features(crypto_raw)
    c_res = evaluate_four_stages(
        c_df,
        c_cols,
        risk_cols=["beta_60"],
        group_col="sector",
        weight_scheme="dollar_neutral",
        cost_bps=10.0,
        annualization=365.0,
        lookback_fast=45,
        lookback_slow=120,
        ema_alpha=0.20,
        oos_start_date=c_oos,
    )

    u_df, u_cols, u_oos = build_us_features(us_raw)
    u_res = evaluate_four_stages(
        u_df,
        u_cols,
        risk_cols=["beta_60"],
        group_col="sector",
        weight_scheme="dollar_neutral",
        cost_bps=10.0,
        annualization=252.0,
        lookback_fast=45,
        lookback_slow=120,
        ema_alpha=0.20,
        oos_start_date=u_oos,
    )

    h_df, h_cols, h_oos = build_hose_features()
    h_res = evaluate_four_stages(
        h_df,
        h_cols,
        risk_cols=["beta_60", "log_adv_60"],
        group_col=None,
        weight_scheme="long_only",
        cost_bps=30.0,
        annualization=252.0,
        lookback_fast=45,
        lookback_slow=120,
        ema_alpha=0.12,
        oos_start_date=h_oos,
    )
    elapsed = time.perf_counter() - t0

    c_max = str(c_df["date"].max())
    u_max = str(u_df["date"].max())
    h_max = str(h_df["date"].max())
    lines = [
        "# Real-Market Empirical Validation - Global Crypto, US Equities & Vietnam HOSE",
        "",
        "Generated automatically by `benchmarks/real_market_alpha_engineering.py` (`signal_lag = 1` across all markets).",
        "",
        "## Environment & Data Provenance",
        "",
        f"- **Python**: `{platform.python_version()}` | **Polars**: `{pl.__version__}` | **NumPy**: `{np.__version__}` | **polars-anta**: `{pa.__version__}` | **Runtime**: `{elapsed:.2f} s`",
        f"- **Market A (Crypto Long-Short)**: `data-api.binance.vision` (`24` liquid USDT pairs, real `taker_buy_volume` orderflow, OOS `{c_oos}` to `{c_max}`, `10 bps` one-way cost)",
        f"- **Market B (US Equities & Sector ETFs Long-Short)**: `{us_src}` (`30` S&P large-caps & Sector SPDRs across `6` sector groups, OOS `{u_oos}` to `{u_max}`, `10 bps` one-way cost)",
        f"- **Market C (Vietnam HOSE Long-Only)**: Point-in-time top-150 liquid `HOSE` stocks (`research/data/vn_daily.parquet`, Locked Holdout `{h_oos}` to `{h_max}`, `30 bps` one-way / `60 bps` round-trip cost)",
        "",
        f"## Market A - Global Crypto Dollar-Neutral Long-Short (`{c_oos}` to `{c_max}`, `N = 24`, `10 bps` One-Way)",
        "",
        *_fmt_table(c_res, "`10 bps`"),
        "",
        f"## Market B - US Equities & Sector ETFs Dollar-Neutral Long-Short (`{u_oos}` to `{u_max}`, `N = 30`, `10 bps` One-Way)",
        "",
        *_fmt_table(u_res, "`10 bps`"),
        "",
        f"## Market C - Vietnam HOSE Long-Only Locked Holdout (`{h_oos}` to `{h_max}`, `N <= 150`, `30 bps` One-Way / `60 bps` RT)",
        "",
        *_fmt_table(h_res, "`30 bps` OW"),
        "",
    ]
    out_path = REPO_ROOT / "docs" / "assets" / "real_market_alpha_engineering.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
