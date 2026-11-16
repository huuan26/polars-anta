# Real-Market Empirical Validation - Global Crypto, US Equities & Vietnam HOSE

Generated automatically by `benchmarks/real_market_alpha_engineering.py` (`signal_lag = 1` across all markets).

## Environment & Data Provenance

- **Python**: `3.14.1` | **Polars**: `2.1.0` | **NumPy**: `2.5.4` | **polars-anta**: `0.5.0` | **Runtime**: `73.25 s`
- **Market A (Crypto Long-Short)**: `data-api.binance.vision` (`24` liquid USDT pairs, real `taker_buy_volume` orderflow, OOS `2024-08-19` to `2026-11-15`, `10 bps` one-way cost)
- **Market B (US Equities & Sector ETFs Long-Short)**: `stooq.com -> fallback query1.finance.yahoo.com (split/dividend-adjusted)` (`30` S&P large-caps & Sector SPDRs across `6` sector groups, OOS `2022-09-02` to `2026-11-16`, `10 bps` one-way cost)
- **Market C (Vietnam HOSE Long-Only)**: Point-in-time top-150 liquid `HOSE` stocks (`research/data/vn_daily.parquet`, Locked Holdout `2022-01-01` to `2026-11-10`, `30 bps` one-way / `60 bps` round-trip cost)

## Market A - Global Crypto Dollar-Neutral Long-Short (`2024-08-19` to `2026-11-15`, `N = 24`, `10 bps` One-Way)

| Pipeline Stage | Rank IC | NW $t$ | Daily Turnover | Gross Margin (`bps`) | Gross Sharpe | Net Sharpe (`10 bps`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0. Raw Signals (Equal-Weight) | `+0.0452` | `+4.45` | `0.436` | `+8.31` | `+0.62` | `-0.13` |
| 1. FWL + Lowdin Ortho (Equal-Weight) | `+0.0307` | `+4.04` | `0.586` | `+6.00` | `+0.86` | `-0.58` |
| 2. Unpenalized Walk-Forward (Sample Cov) | `+0.0117` | `+1.70` | `0.676` | `-0.98` | `-0.17` | `-1.94` |
| 3. Cost-Aware Pipeline (RMT + Cost Drag + FISTA + EMA) | `-0.0047` | `-0.68` | `0.206` | `-8.14` | `-0.43` | `-0.95` |

## Market B - US Equities & Sector ETFs Dollar-Neutral Long-Short (`2022-09-02` to `2026-11-16`, `N = 30`, `10 bps` One-Way)

| Pipeline Stage | Rank IC | NW $t$ | Daily Turnover | Gross Margin (`bps`) | Gross Sharpe | Net Sharpe (`10 bps`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0. Raw Signals (Equal-Weight) | `+0.0014` | `+0.15` | `0.294` | `-1.22` | `-0.10` | `-0.89` |
| 1. FWL + Lowdin Ortho (Equal-Weight) | `+0.0029` | `+0.55` | `0.539` | `-1.96` | `-0.44` | `-2.69` |
| 2. Unpenalized Walk-Forward (Sample Cov) | `-0.0082` | `-1.63` | `0.595` | `-2.25` | `-0.67` | `-3.65` |
| 3. Cost-Aware Pipeline (RMT + Cost Drag + FISTA + EMA) | `+0.0003` | `+0.08` | `0.049` | `+8.65` | `+0.24` | `-0.04` |

## Market C - Vietnam HOSE Long-Only Locked Holdout (`2022-01-01` to `2026-11-10`, `N <= 150`, `30 bps` One-Way / `60 bps` RT)

| Pipeline Stage | Rank IC | NW $t$ | Daily Turnover | Gross Margin (`bps`) | Gross Sharpe | Net Sharpe (`30 bps` OW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0. Raw Signals (Equal-Weight) | `+0.0138` | `+2.82` | `0.385` | `-14.90` | `-0.59` | `-1.78` |
| 1. FWL + Lowdin Ortho (Equal-Weight) | `+0.0150` | `+4.25` | `0.387` | `-5.74` | `-0.23` | `-1.41` |
| 2. Unpenalized Walk-Forward (Sample Cov) | `+0.0187` | `+5.30` | `0.482` | `+1.68` | `+0.08` | `-1.41` |
| 3. Cost-Aware Pipeline (RMT + Cost Drag + FISTA + EMA) | `+0.0043` | `+1.41` | `0.091` | `+12.68` | `+0.13` | `-0.17` |
