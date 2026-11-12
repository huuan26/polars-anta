from __future__ import annotations

import polars as pl
import pytest

import polars_anta  # noqa: F401 - triggers namespace registration
from polars_anta.evaluation import (
    calmar_ratio,
    conditional_var,
    drawdown_series,
    factor_turnover,
    forward_returns,
    ic_summary,
    information_coefficient,
    max_drawdown,
    omega_ratio,
    profit_factor,
    quantile_analysis,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
    win_rate,
)


def test_performance_metrics_expressions():
    """Test drawdown_series, max_drawdown, sharpe, sortino, calmar, profit_factor, and win_rate."""
    df = pl.DataFrame(
        {
            "ret": [0.02, -0.01, 0.03, -0.02, 0.04, 0.0, 0.01, -0.005],
            "equity": [100.0, 110.0, 99.0, 105.0, 93.5, 115.0, 120.0, 114.0],
        }
    )

    res = df.select(
        [
            drawdown_series("ret", is_returns=True).alias("dd_ret"),
            drawdown_series("equity", is_returns=False).alias("dd_eq"),
        ]
    )
    assert (res["dd_ret"] <= 0.0).all()
    assert (res["dd_eq"] <= 0.0).all()
    # Worst drop on equity is from 110.0 to 93.5 -> (93.5 - 110.0) / 110.0 = -0.15
    assert res["dd_eq"].min() == pytest.approx(-0.15)

    scalars = df.select(
        [
            max_drawdown("equity", is_returns=False).alias("mdd_eq"),
            max_drawdown("ret", is_returns=True).alias("mdd_ret"),
            sharpe_ratio("ret", risk_free_rate=0.0, annualization=252.0).alias("sharpe"),
            sortino_ratio("ret", risk_free_rate=0.0, annualization=252.0).alias("sortino"),
            calmar_ratio("ret", annualization=252.0).alias("calmar"),
            profit_factor("ret").alias("pf"),
            win_rate("ret", include_zeros=False).alias("wr_active"),
            win_rate("ret", include_zeros=True).alias("wr_all"),
        ]
    ).to_dicts()[0]

    assert scalars["mdd_eq"] == pytest.approx(-0.15)
    assert scalars["mdd_ret"] < 0.0
    assert scalars["sharpe"] > 0.0
    assert scalars["sortino"] > scalars["sharpe"]
    assert scalars["calmar"] > 0.0
    # Gross profit = 0.02 + 0.03 + 0.04 + 0.01 = 0.10; Gross loss = 0.01 + 0.02 + 0.005 = 0.035
    assert scalars["pf"] == pytest.approx(0.10 / 0.035)
    # 4 wins out of 7 active non-zero bars
    assert scalars["wr_active"] == pytest.approx(4.0 / 7.0)
    # 4 wins out of 8 total bars
    assert scalars["wr_all"] == pytest.approx(4.0 / 8.0)

    # Namespace parity on Expr and Frame
    ns_scalars = df.select(
        [
            pl.col("ret").anta.drawdown_series(is_returns=True).min().alias("dd_min"),
            pl.col("equity").ta.max_drawdown(is_returns=False).alias("mdd_eq"),
            pl.col("ret").anta.sharpe_ratio(annualization=252.0).alias("sharpe"),
            pl.col("ret").ta.sortino_ratio(annualization=252.0).alias("sortino"),
            pl.col("ret").anta.calmar_ratio(annualization=252.0).alias("calmar"),
            pl.col("ret").ta.profit_factor().alias("pf"),
            pl.col("ret").anta.win_rate().alias("wr_active"),
        ]
    ).to_dicts()[0]

    assert ns_scalars["mdd_eq"] == pytest.approx(scalars["mdd_eq"])
    assert ns_scalars["sharpe"] == pytest.approx(scalars["sharpe"])
    assert ns_scalars["sortino"] == pytest.approx(scalars["sortino"])
    assert ns_scalars["calmar"] == pytest.approx(scalars["calmar"])
    assert ns_scalars["pf"] == pytest.approx(scalars["pf"])
    assert ns_scalars["wr_active"] == pytest.approx(scalars["wr_active"])

    df_with_dd = df.anta.drawdown_series(col="equity", is_returns=False, alias="dd_col")
    assert "dd_col" in df_with_dd.columns
    assert df_with_dd["dd_col"].min() == pytest.approx(-0.15)


def test_performance_metrics_edge_cases():
    """Test flat returns, zero losses, and invalid parameter errors."""
    flat_df = pl.DataFrame({"ret": [0.0, 0.0, 0.0]})
    flat_res = flat_df.select(
        [
            sharpe_ratio("ret").alias("s"),
            sortino_ratio("ret").alias("so"),
            calmar_ratio("ret").alias("c"),
            profit_factor("ret").alias("pf"),
            win_rate("ret").alias("wr"),
        ]
    ).to_dicts()[0]
    assert flat_res["s"] == 0.0
    assert flat_res["so"] == 0.0
    assert flat_res["c"] == 0.0
    assert flat_res["pf"] == 0.0
    assert flat_res["wr"] == 0.0

    with pytest.raises(ValueError, match="annualization must be a positive"):
        sharpe_ratio("ret", annualization=0.0)
    with pytest.raises(ValueError, match="annualization must be a positive"):
        sortino_ratio("ret", annualization=-10.0)
    with pytest.raises(ValueError, match="annualization must be a positive"):
        calmar_ratio("ret", annualization=0.0)


def test_factor_evaluation_suite():
    """Test forward_returns, information_coefficient, ic_summary, quantile_analysis, and factor_turnover."""
    # Construct a multi-asset cross-sectional dataset where factor strongly predicts forward return
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    rows = []
    for d_idx, dt in enumerate(dates):
        for t_idx, sym in enumerate(tickers):
            signal = float(t_idx + 1) + 0.1 * d_idx
            close_price = 100.0 * (1.0 + 0.01 * (t_idx + 1) * (d_idx + 1))
            fwd_1 = 0.002 * (t_idx + 1) + (0.0005 if d_idx % 2 == 0 else -0.0002)
            rows.append(
                {
                    "date": dt,
                    "ticker": sym,
                    "close": close_price,
                    "alpha": signal,
                    "manual_ret": fwd_1,
                }
            )

    df = pl.DataFrame(rows)

    # 1. Forward returns via function and Frame namespace
    fwd_df = df.anta.forward_returns(close_col="close", periods=(1, 2), group_by="ticker")
    assert "fwd_ret_1" in fwd_df.columns
    assert "fwd_ret_2" in fwd_df.columns

    with pytest.raises(ValueError, match="periods must contain at least one"):
        forward_returns(df, periods=())
    with pytest.raises(ValueError, match="each forward return period must be a positive"):
        forward_returns(df, periods=(0,))

    # 2. Information Coefficient (Spearman & Pearson, grouped and global)
    ic_spearman = df.anta.information_coefficient(
        factor_col="alpha", return_col="manual_ret", group_by="date", method="spearman"
    )
    assert len(ic_spearman) == 5
    # Since alpha and manual_ret are strictly monotonic per date, Rank IC must be 1.0
    for val in ic_spearman["ic"].to_list():
        assert val == pytest.approx(1.0)

    ic_pearson = information_coefficient(
        df.lazy(), factor_col="alpha", return_col="manual_ret", method="pearson"
    )
    assert len(ic_pearson) == 1
    assert ic_pearson["ic"][0] > 0.95

    with pytest.raises(ValueError, match="method must be either"):
        information_coefficient(df, "alpha", "manual_ret", method="invalid")  # type: ignore[arg-type]

    # 3. IC Summary
    summary = df.ta.ic_summary(
        factor_col="alpha", return_col="manual_ret", group_by="date", method="spearman"
    )
    assert summary["ic_mean"] == pytest.approx(1.0)
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert summary["n_periods"] == 5.0

    # Empty IC summary edge case
    empty_df = df.filter(pl.col("alpha") > 999999.0)
    empty_sum = ic_summary(empty_df, factor_col="alpha", return_col="manual_ret")
    assert empty_sum["n_periods"] == 0.0
    assert empty_sum["ic_mean"] == 0.0

    # 4. Quantile Analysis
    q_df = df.anta.quantile_analysis(
        factor_col="alpha", return_col="manual_ret", quantiles=5, group_by="date"
    )
    assert len(q_df) == 5
    assert q_df["quantile"].to_list() == [1, 2, 3, 4, 5]
    # Monotonicity check: Q5 mean return > Q1 mean return
    assert q_df["mean_return"][-1] > q_df["mean_return"][0]

    q_global = quantile_analysis(
        df.lazy(), factor_col="alpha", return_col="manual_ret", quantiles=2
    )
    assert len(q_global) == 2

    with pytest.raises(ValueError, match="quantiles must be an integer greater than or equal to 2"):
        quantile_analysis(df, "alpha", "manual_ret", quantiles=1)

    # 5. Factor Turnover
    to_df = df.ta.factor_turnover(factor_col="alpha", asset_col="ticker", time_col="date", lag=1)
    assert len(to_df) == 4  # 5 dates minus 1 lag
    # Since asset rankings never change across dates, rank autocorrelation is 1.0
    for ac in to_df["rank_autocorr"].to_list():
        assert ac == pytest.approx(1.0)

    with pytest.raises(ValueError, match="lag must be a positive integer"):
        factor_turnover(df, "alpha", lag=0)


def test_triple_barrier_labels() -> None:
    from polars_anta import triple_barrier_labels

    # Construct deterministic paths:
    # - From index 0 (100.0): jumps to 105.0 at k=1 (+5% >= +3% PT) -> label = +1.0, touch_bars = 1
    # - From index 2 (100.0): drops to 96.0 at k=1 (-4% <= -3% SL) -> label = -1.0, touch_bars = 1
    # - From index 4 (100.0): stays within [99.0, 101.0] for 3 bars -> vertical barrier (label = 0.0, touch_bars = 3)
    df = pl.DataFrame(
        {
            "close": [100.0, 105.0, 100.0, 96.0, 100.0, 100.5, 99.8, 100.4, 100.0],
            "side": [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        }
    )

    out = df.anta.triple_barrier_labels(
        close="close",
        volatility=0.015,  # upper = +3%, lower = -3% with pt_mult=2, sl_mult=2
        horizon=3,
        pt_mult=2.0,
        sl_mult=2.0,
        side="side",
        unnest=True,
    )
    assert {"tb_label", "tb_touch_bars", "tb_realized_return", "tb_meta_label"}.issubset(
        set(out.columns)
    )

    # Bar 0: +5% at k=1 hits PT (+1.0), side=+1.0 -> meta_label = 1.0
    assert out["tb_label"][0] == pytest.approx(1.0)
    assert out["tb_touch_bars"][0] == 1
    assert out["tb_realized_return"][0] == pytest.approx(0.05)
    assert out["tb_meta_label"][0] == pytest.approx(1.0)

    # Bar 2: -4% at k=1 hits SL (-1.0), side=-1.0 (short model was right!) -> meta_label = 1.0
    assert out["tb_label"][2] == pytest.approx(-1.0)
    assert out["tb_touch_bars"][2] == 1
    assert out["tb_realized_return"][2] == pytest.approx(-0.04)
    assert out["tb_meta_label"][2] == pytest.approx(1.0)

    # Bar 4: stays within +/-3% for k=1..3 -> vertical barrier (label=0.0, touch_bars=3, ret=+0.004)
    assert out["tb_label"][4] == pytest.approx(0.0)
    assert out["tb_touch_bars"][4] == 3
    assert out["tb_realized_return"][4] == pytest.approx(0.004)
    assert out["tb_meta_label"][4] == pytest.approx(0.0)

    # Default rolling volatility (volatility=None) and Expr namespace (.ta)
    df_long = pl.DataFrame(
        {
            "ticker": ["VN30F1M"] * 12,
            "close": [100.0 + i * 0.5 + ((-1) ** i) * 0.2 for i in range(12)],
        }
    )
    expr_out = df_long.select(
        pl.col("close")
        .ta.triple_barrier_labels(horizon=2, vol_period=3, min_ret=0.001)
        .over("ticker")
        .alias("tb")
    ).unnest("tb")
    assert expr_out["label"].null_count() < len(df_long)

    # Disable upper or lower barrier (pt_mult=0 or sl_mult=0) and custom volatility column
    only_sl = (
        df.with_columns(pl.lit(0.015).alias("vol_col"))
        .select(
            triple_barrier_labels(
                "close", volatility="vol_col", horizon=2, pt_mult=0.0, sl_mult=2.0
            ).alias("tb")
        )
        .unnest("tb")
    )
    # Bar 0 (+5% at k=1, 0% at k=2) does not trigger upper barrier because pt_mult=0
    assert only_sl["label"][0] == pytest.approx(0.0)
    assert only_sl["label"][2] == pytest.approx(-1.0)

    only_pt = df.select(
        triple_barrier_labels("close", volatility=0.015, horizon=2, pt_mult=2.0, sl_mult=0.0).alias(
            "tb"
        )
    ).unnest("tb")
    assert only_pt["label"][0] == pytest.approx(1.0)
    assert only_pt["label"][2] == pytest.approx(0.0)

    # Validation checks
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        triple_barrier_labels("close", horizon=0)
    with pytest.raises(ValueError, match="pt_mult and sl_mult must be non-negative"):
        triple_barrier_labels("close", pt_mult=-1.0)
    with pytest.raises(ValueError, match="min_ret must be non-negative"):
        triple_barrier_labels("close", min_ret=-0.01)
    with pytest.raises(ValueError, match="vol_period must be an integer greater than 1"):
        triple_barrier_labels("close", vol_period=1)
    with pytest.raises(ValueError, match="scalar volatility must be a positive number"):
        triple_barrier_labels("close", volatility=0.0)


def test_trend_scanning_labels() -> None:
    from polars_anta import trend_scanning_labels

    # First segment: exponential uptrend -> +1.0 with positive t-value
    # Second segment: exponential downtrend -> -1.0 with negative t-value
    up_prices = [100.0 * (1.015**i) for i in range(12)]
    down_prices = [up_prices[-1] * (0.98**i) for i in range(1, 12)]
    df = pl.DataFrame(
        {
            "ticker": ["VN30F1M"] * (len(up_prices) + len(down_prices)),
            "close": up_prices + down_prices,
        }
    )

    out = df.anta.trend_scanning_labels(
        close="close",
        min_window=4,
        max_window=8,
        step=2,
        unnest=True,
        group_by="ticker",
    )
    assert {
        "ts_label_label",
        "ts_label_t_value",
        "ts_label_optimal_window",
        "ts_label_ret",
    }.issubset(set(out.columns))

    # Start of uptrend (bar 0): strongly positive t-stat and +1.0 label
    assert out["ts_label_label"][0] == pytest.approx(1.0)
    assert out["ts_label_t_value"][0] > 10.0
    assert out["ts_label_optimal_window"][0] in (4, 6, 8)
    assert out["ts_label_ret"][0] > 0.0

    # Start of downtrend (bar 11): strongly negative t-stat and -1.0 label
    assert out["ts_label_label"][11] == pytest.approx(-1.0)
    assert out["ts_label_t_value"][11] < -10.0
    assert out["ts_label_optimal_window"][11] in (4, 6, 8)
    assert out["ts_label_ret"][11] < 0.0

    # Expr namespace (.ta)
    expr_res = df.select(
        pl.col("close").ta.trend_scanning_labels(min_window=4, max_window=6).alias("ts")
    ).unnest("ts")
    assert expr_res["label"][0] == pytest.approx(1.0)

    # Validation checks
    with pytest.raises(
        ValueError, match="min_window must be an integer greater than or equal to 3"
    ):
        trend_scanning_labels("close", min_window=2, max_window=5)
    with pytest.raises(ValueError, match="max_window must be greater than or equal to min_window"):
        trend_scanning_labels("close", min_window=6, max_window=5)
    with pytest.raises(ValueError, match="step must be a positive integer"):
        trend_scanning_labels("close", min_window=4, max_window=8, step=0)


def test_probabilistic_and_deflated_sharpe_ratio() -> None:
    from polars_anta import deflated_sharpe_ratio, probabilistic_sharpe_ratio

    # Strong positive strategy vs negative strategy over 60 bars
    pos_rets = [0.008 + ((-1) ** i) * 0.003 for i in range(60)]
    neg_rets = [-0.005 + ((-1) ** i) * 0.004 for i in range(60)]
    df = pl.DataFrame({"pos": pos_rets, "neg": neg_rets, "flat": [0.0] * 60})

    res = df.select(
        [
            pl.col("pos").anta.probabilistic_sharpe_ratio(benchmark_sr=0.0).alias("psr_pos"),
            pl.col("neg").ta.probabilistic_sharpe_ratio(benchmark_sr=0.0).alias("psr_neg"),
            probabilistic_sharpe_ratio("flat", benchmark_sr=0.0).alias("psr_flat"),
            pl.col("pos").anta.deflated_sharpe_ratio(n_trials=1, trials_sr_std=0.5).alias("dsr_n1"),
            pl.col("pos").ta.deflated_sharpe_ratio(n_trials=50, trials_sr_std=0.8).alias("dsr_n50"),
        ]
    )

    assert res["psr_pos"][0] > 0.99
    assert res["psr_neg"][0] < 0.01
    assert res["psr_flat"][0] == pytest.approx(0.0)
    # When n_trials=1, DSR equals PSR(benchmark_sr=0.0)
    assert res["dsr_n1"][0] == pytest.approx(res["psr_pos"][0])
    # Penalizing for 50 trials raises the hurdle SR and lowers DSR
    assert 0.0 <= res["dsr_n50"][0] <= res["dsr_n1"][0]

    with pytest.raises(ValueError, match="annualization must be a positive"):
        probabilistic_sharpe_ratio("pos", annualization=0.0)
    with pytest.raises(ValueError, match="n_trials must be a positive integer"):
        deflated_sharpe_ratio("pos", n_trials=0)
    with pytest.raises(ValueError, match="trials_sr_std must be non-negative"):
        deflated_sharpe_ratio("pos", trials_sr_std=-0.1)
    with pytest.raises(ValueError, match="annualization must be a positive"):
        deflated_sharpe_ratio("pos", annualization=0.0)


def test_newey_west_ic_summary_and_ic_decay() -> None:
    from polars_anta import ic_decay, ic_summary

    # Build a multi-asset panel across 15 dates x 5 tickers where factor predicts future prices
    rows = []
    for d_idx in range(1, 16):
        date_str = f"2026-01-{d_idx:02d}"
        for t_idx in range(1, 6):
            ticker = f"T{t_idx}"
            # Positive drift proportional to t_idx with slight period variation in rank
            base_price = 100.0 * ((1.0 + 0.005 * t_idx) ** d_idx)
            alpha_val = float(t_idx) + (1.5 if (d_idx % 4 == 0 and t_idx == 2) else 0.0)
            rows.append(
                {"date": date_str, "ticker": ticker, "close": base_price, "alpha": alpha_val}
            )

    panel = pl.DataFrame(rows)
    panel_fwd = panel.anta.forward_returns(close_col="close", periods=(1, 2), group_by="ticker")

    summary = panel_fwd.anta.ic_summary(
        factor_col="alpha", return_col="fwd_ret_1", group_by="date", nw_lags=2
    )
    assert "nw_t_stat" in summary
    assert summary["ic_mean"] > 0.8
    assert summary["nw_t_stat"] != 0.0

    with pytest.raises(ValueError, match="nw_lags must be a non-negative integer"):
        ic_summary(panel_fwd, "alpha", "fwd_ret_1", group_by="date", nw_lags=-1)

    decay_df = panel.ta.ic_decay(
        factor_col="alpha",
        close_col="close",
        horizons=(1, 2, 3),
        asset_col="ticker",
        group_by="date",
    )
    assert len(decay_df) == 3
    assert {
        "horizon",
        "ic_mean",
        "ic_std",
        "icir",
        "t_stat",
        "nw_t_stat",
        "hit_rate",
        "n_periods",
        "half_life",
    }.issubset(set(decay_df.columns))
    assert decay_df["horizon"].to_list() == [1, 2, 3]
    assert decay_df["half_life"][0] > 0.0

    # Single horizon edge case for ic_decay
    single_h = ic_decay(panel, "alpha", horizons=(1,), asset_col="ticker", group_by="date")
    assert len(single_h) == 1

    with pytest.raises(ValueError, match="horizons must contain at least one"):
        ic_decay(panel, "alpha", horizons=())
    with pytest.raises(ValueError, match="each horizon in horizons must be a positive"):
        ic_decay(panel, "alpha", horizons=(0,))


def _tail_returns() -> pl.DataFrame:
    """Seven returns whose empirical quantiles are easy to hand-check."""
    return pl.DataFrame({"ret": [-0.05, -0.02, -0.01, 0.0, 0.01, 0.03, 0.04]})


def _scalar(df: pl.DataFrame, expr: pl.Expr, name: str = "v"):
    return df.select(expr.alias(name))[name][0]


def test_value_at_risk_matches_manual_linear_quantile():
    df = _tail_returns()
    # sorted: 0.05 * (7 - 1) = 0.3 -> between -0.05 and -0.02
    expected = -0.05 + 0.3 * (-0.02 - -0.05)

    assert _scalar(df, value_at_risk("ret", cutoff=0.05)) == pytest.approx(expected)
    assert value_at_risk("ret", cutoff=0.05) is not None


def test_conditional_var_is_mean_of_tail_below_var():
    df = _tail_returns()
    # only -0.05 sits at or below the VaR level
    assert _scalar(df, conditional_var("ret", cutoff=0.05)) == pytest.approx(-0.05)

    wider = _scalar(df, conditional_var("ret", cutoff=0.30))
    tail = [v for v in df["ret"].to_list() if v <= _scalar(df, value_at_risk("ret", cutoff=0.30))]
    assert wider == pytest.approx(sum(tail) / len(tail))


def test_conditional_var_never_above_var():
    df = pl.DataFrame({"ret": [((i * 37) % 11 - 5) / 100.0 for i in range(60)]})
    for cutoff in (0.01, 0.05, 0.10, 0.25):
        var = _scalar(df, value_at_risk("ret", cutoff=cutoff))
        cvar = _scalar(df, conditional_var("ret", cutoff=cutoff))
        assert cvar <= var + 1e-12


def test_omega_ratio_symmetric_series_is_one():
    df = _tail_returns()
    # upside 0.01 + 0.03 + 0.04 = 0.08, downside 0.05 + 0.02 + 0.01 = 0.08
    assert _scalar(df, omega_ratio("ret")) == pytest.approx(1.0)


def test_omega_ratio_rewards_positive_skew():
    df = pl.DataFrame({"ret": [-0.01, -0.01, 0.02, 0.05, 0.08]})
    assert _scalar(df, omega_ratio("ret")) > 2.0


def test_omega_ratio_respects_annualized_threshold():
    df = pl.DataFrame({"ret": [0.0001, 0.0002, 0.0003]})
    # threshold = 0.05 / 252 ~ 1.98e-4, so only the last bar clears it
    assert _scalar(df, omega_ratio("ret", risk_free_rate=0.05, annualization=252.0)) > 0.0


def test_tail_ratio_matches_manual_quantiles():
    df = _tail_returns()
    upper = 0.03 + 0.7 * (0.04 - 0.03)
    lower = -0.05 + 0.3 * (-0.02 - -0.05)

    assert _scalar(df, tail_ratio("ret", cutoff=0.05)) == pytest.approx(abs(upper) / abs(lower))


def test_tail_ratio_is_above_one_for_right_skewed_series():
    df = pl.DataFrame({"ret": [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.10]})
    assert _scalar(df, tail_ratio("ret", cutoff=0.10)) > 1.0


def test_tail_ratio_zero_lower_tail_reports_inf():
    df = pl.DataFrame({"ret": [0.0, 0.0, 0.0, 0.01, 0.02]})
    assert _scalar(df, tail_ratio("ret", cutoff=0.05)) == float("inf")


def test_tail_metrics_handle_all_zero_series_without_nan():
    df = pl.DataFrame({"ret": [0.0] * 10})
    values = df.select(
        value_at_risk("ret").alias("var"),
        conditional_var("ret").alias("cvar"),
        omega_ratio("ret").alias("omega"),
        tail_ratio("ret").alias("tail"),
    ).row(0)

    assert values == (0.0, 0.0, 0.0, 0.0)


def test_tail_metrics_handle_empty_series_without_nan():
    df = pl.DataFrame({"ret": []}, schema={"ret": pl.Float64})
    values = df.select(
        value_at_risk("ret").alias("var"),
        conditional_var("ret").alias("cvar"),
    ).row(0)

    assert values == (None, None)


def test_tail_metrics_reject_invalid_cutoff():
    for bad in (0.0, 1.0, -0.05, 1.5):
        with pytest.raises(ValueError, match="cutoff must be a probability"):
            value_at_risk("ret", cutoff=bad)
        with pytest.raises(ValueError, match="cutoff must be a probability"):
            conditional_var("ret", cutoff=bad)
        with pytest.raises(ValueError, match="cutoff must be a probability"):
            tail_ratio("ret", cutoff=bad)


def test_omega_ratio_rejects_invalid_annualization():
    with pytest.raises(ValueError, match="annualization must be a positive"):
        omega_ratio("ret", annualization=0.0)


def test_tail_metrics_are_registered_on_namespaces():
    df = _tail_returns()
    out = df.select(
        pl.col("ret").anta.value_at_risk().alias("var"),
        pl.col("ret").ta.conditional_var().alias("cvar"),
        pl.col("ret").anta.omega_ratio().alias("omega"),
        pl.col("ret").ta.tail_ratio().alias("tail"),
    ).row(0)

    assert out[0] < 0.0
    assert out[1] <= out[0]
    assert out[2] > 0.0
    assert out[3] > 0.0


def test_tail_metrics_are_exported_from_package():
    import polars_anta as pa

    for name in ("value_at_risk", "conditional_var", "omega_ratio", "tail_ratio"):
        assert hasattr(pa, name)
        assert name in pa.__all__
