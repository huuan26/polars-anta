from __future__ import annotations

import polars as pl
import pytest

import polars_anta  # noqa: F401 - triggers namespace registration
from polars_anta.evaluation import (
    calmar_ratio,
    drawdown_series,
    factor_turnover,
    forward_returns,
    ic_summary,
    information_coefficient,
    max_drawdown,
    profit_factor,
    quantile_analysis,
    sharpe_ratio,
    sortino_ratio,
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
