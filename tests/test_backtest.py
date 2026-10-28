from __future__ import annotations

import polars as pl
import pytest

import polars_anta  # noqa: F401 - triggers namespace registration
from polars_anta.backtest import BacktestResult, vectorized_backtest


def test_single_asset_vectorized_backtest():
    """Test single-asset vectorized backtest with fees, slippage, and signal lag."""
    df = pl.DataFrame(
        {
            "close": [100.0, 102.0, 104.04, 103.0, 105.06],
            "signal": [1.0, 1.0, -1.0, 0.0, 1.0],
        }
    )

    res = df.anta.backtest(
        signal="signal",
        close_col="close",
        fee_rate=0.001,
        slippage_rate=0.0005,
        signal_lag=1,
        initial_capital=1000.0,
    )
    assert isinstance(res, BacktestResult)
    bt = res.data

    # Expected lagged positions: [0.0, 1.0, 1.0, -1.0, 0.0]
    assert bt["position"].to_list() == [0.0, 1.0, 1.0, -1.0, 0.0]
    # Expected turnover: |0-0|=0, |1-0|=1, |1-1|=0, |-1-1|=2, |0 - (-1)|=1
    assert bt["turnover"].to_list() == [0.0, 1.0, 0.0, 2.0, 1.0]
    # Total turnover = 4.0; total cost = 4.0 * 0.0015 = 0.006
    assert res["total_turnover"] == pytest.approx(4.0)
    assert res["total_cost"] == pytest.approx(0.006)
    assert res["initial_capital"] == 1000.0
    assert res["final_equity"] > 1000.0
    assert res["total_return"] > 0.0

    # Check __getitem__ for both summary keys and data columns
    assert res["n_bars"] == 5.0
    assert len(res["equity_curve"]) == 5

    # Summary DataFrame (wide and transposed)
    wide_df = res.summary_df(transpose=False)
    assert len(wide_df) == 1
    assert "sharpe_ratio" in wide_df.columns

    long_df = res.summary_df(transpose=True)
    assert list(long_df.columns) == ["metric", "value"]
    assert len(long_df) == len(res.summary)


def test_multi_asset_portfolio_backtest():
    """Test multi-asset backtest with and without portfolio aggregation, LazyFrame, and additive mode."""
    df = pl.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"] * 2,
            "ticker": ["AAPL"] * 4 + ["MSFT"] * 4,
            "close": [100.0, 102.0, 104.0, 106.0, 200.0, 198.0, 196.0, 194.0],
            "weight": [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            "pre_ret": [0.0, 0.02, 0.0196, 0.0192, 0.0, -0.01, -0.0101, -0.0102],
        }
    )

    # 1. Aggregated portfolio across tickers per timestamp (via LazyFrame and .ta namespace)
    port_res = df.lazy().ta.backtest(
        signal="weight",
        return_col="pre_ret",
        group_by="ticker",
        time_col="timestamp",
        fee_rate=0.0005,
        signal_lag=1,
        compound=False,
        aggregate_portfolio=True,
    )
    assert len(port_res.data) == 4
    assert port_res["total_return"] > 0.0
    assert port_res["win_rate"] == pytest.approx(1.0)

    # 2. Unaggregated per-asset backtest (aggregate_portfolio=False)
    asset_res = vectorized_backtest(
        df,
        signal=pl.col("weight"),
        close_col="close",
        group_by="ticker",
        time_col="timestamp",
        signal_lag=0,
        aggregate_portfolio=False,
    )
    assert len(asset_res.data) == 8

    # 3. Empty DataFrame edge case
    empty_df = df.filter(pl.col("close") < 0.0)
    empty_res = empty_df.anta.backtest(signal="weight")
    assert empty_res["n_bars"] == 0.0
    assert empty_res["total_return"] == 0.0


def test_backtest_validation_errors():
    """Test input validation errors in vectorized_backtest."""
    df = pl.DataFrame({"close": [100.0, 101.0], "signal": [1.0, 1.0]})

    with pytest.raises(ValueError, match="fee_rate and slippage_rate must be non-negative"):
        df.anta.backtest(fee_rate=-0.01)
    with pytest.raises(ValueError, match="signal_lag must be a non-negative integer"):
        df.anta.backtest(signal_lag=-1)
    with pytest.raises(ValueError, match="initial_capital must be a positive"):
        df.anta.backtest(initial_capital=0.0)
    with pytest.raises(ValueError, match="annualization must be a positive"):
        df.anta.backtest(annualization=-252.0)
    with pytest.raises(ValueError, match="weight_scheme must be one of"):
        df.anta.backtest(weight_scheme="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both entries and exits must be provided together"):
        df.anta.backtest(entries=pl.col("close") > 100.0)


def test_crossovers_and_entry_exit_backtest():
    """Test crosses_above, crosses_below, signals_to_positions, report(), and compare_strategies()."""
    from polars_anta import compare_strategies, crosses_above, crosses_below

    df = pl.DataFrame(
        {
            "close": [100.0, 102.0, 105.0, 108.0, 104.0, 101.0, 106.0, 112.0],
            "fast": [99.0, 101.0, 106.0, 109.0, 103.0, 100.0, 107.0, 113.0],
            "slow": [100.0, 100.5, 102.0, 104.0, 105.0, 104.0, 105.0, 107.0],
            "rsi": [45.0, 28.0, 35.0, 75.0, 65.0, 25.0, 40.0, 55.0],
        }
    )

    # 1. Crossover expressions (column vs column and column vs scalar threshold)
    chk = df.select(
        [
            pl.col("fast").anta.crosses_above("slow").alias("golden_cross"),
            pl.col("fast").anta.crosses_below("slow").alias("death_cross"),
            crosses_below("rsi", 30.0).alias("rsi_oversold_cross"),
            crosses_above("rsi", 70.0).alias("rsi_overbought_cross"),
            pl.col("fast")
            .anta.crosses_above("slow")
            .anta.signals_to_positions(exits=pl.col("fast").anta.crosses_below("slow"))
            .alias("pos"),
        ]
    )
    assert chk["golden_cross"].to_list() == [False, True, False, False, False, False, True, False]
    assert chk["death_cross"].to_list() == [False, False, False, False, True, False, False, False]
    assert chk["rsi_oversold_cross"].to_list() == [
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert chk["rsi_overbought_cross"].to_list() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert chk["pos"].to_list() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    # 2. Entry / Exit backtest (Long/Short state machine)
    bt_trend = df.anta.backtest(
        entries=pl.col("fast").anta.crosses_above("slow"),
        exits=pl.col("fast").anta.crosses_below("slow"),
        short_entries=pl.col("fast").anta.crosses_below("slow"),
        short_exits=pl.col("fast").anta.crosses_above("slow"),
        fee_rate=0.0005,
        initial_capital=10_000.0,
    )
    bt_rsi = df.anta.backtest(
        entries=pl.col("rsi") < 30.0,
        exits=pl.col("rsi") > 70.0,
        fee_rate=0.0005,
        initial_capital=10_000.0,
    )

    rep_df = bt_trend.report()
    assert list(rep_df.columns) == ["metric", "value"]
    assert len(rep_df) >= 16
    assert "benchmark_return" in bt_trend.summary
    assert "excess_return" in bt_trend.summary
    assert "exposure" in bt_trend.summary
    assert "n_trades" in bt_trend.summary

    comp_df = compare_strategies({"Trend Crossover": bt_trend, "RSI MeanRev": bt_rsi})
    assert len(comp_df) == 2
    assert comp_df["strategy"].to_list() == ["Trend Crossover", "RSI MeanRev"]

    with pytest.raises(ValueError, match="strategies mapping must contain at least one"):
        compare_strategies({})


def test_multi_asset_weight_schemes():
    """Test dollar_neutral, long_only, and equal_weight portfolio weighting schemes."""
    df = pl.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"] * 3,
            "symbol": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "close": [100.0, 103.0, 106.0, 100.0, 101.0, 102.0, 100.0, 98.0, 96.0],
            "alpha": [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, -1.5, -1.5, -1.5],
        }
    )

    for scheme in ("dollar_neutral", "long_only", "equal_weight"):
        res = df.anta.backtest(
            signal="alpha",
            group_by="symbol",
            time_col="timestamp",
            weight_scheme=scheme,  # type: ignore[arg-type]
            signal_lag=1,
            initial_capital=100_000.0,
        )
        assert res["final_equity"] > 100_000.0
        assert res["total_return"] > 0.0
