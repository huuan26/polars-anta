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


def test_execution_realism_flat_mask_vol_target_and_buffer() -> None:
    """Test execution_price ('open', 'twap', 'vwap'), flat_mask, vol_target, and no_trade_buffer."""
    df = pl.DataFrame(
        {
            "timestamp": [f"2026-01-{i:02d}" for i in range(1, 11)],
            "open": [100.0, 101.0, 103.0, 102.0, 104.0, 106.0, 105.0, 108.0, 110.0, 112.0],
            "high": [102.0, 104.0, 105.0, 105.0, 107.0, 108.0, 109.0, 111.0, 113.0, 115.0],
            "low": [99.0, 100.0, 101.0, 100.0, 103.0, 104.0, 104.0, 107.0, 109.0, 111.0],
            "close": [101.0, 103.0, 102.0, 104.0, 106.0, 105.0, 108.0, 110.0, 112.0, 114.0],
            "signal": [1.0, 1.05, 1.08, 1.0, -1.0, -0.95, 0.0, 1.0, 1.0, 0.0],
            "is_eod": [False, False, False, True, False, False, False, False, True, False],
        }
    )

    # 1. Execution price modes ('open', 'twap', 'vwap')
    for exec_mode in ("open", "twap", "vwap"):
        res_exec = df.anta.backtest(
            signal="signal",
            execution_price=exec_mode,  # type: ignore[arg-type]
            signal_lag=1,
        )
        assert len(res_exec.data) == 10
        assert res_exec["n_bars"] == 10.0

    # 2. Intraday EOD flat_mask forces position = 0.0 on matching bars (indices 3 and 8)
    res_flat = df.anta.backtest(
        signal="signal",
        flat_mask=pl.col("is_eod"),
        signal_lag=1,
    )
    assert res_flat.data["position"][3] == pytest.approx(0.0)
    assert res_flat.data["position"][8] == pytest.approx(0.0)

    # 3. Dynamic Volatility Targeting with flat_mask
    res_vt = df.ta.backtest(
        signal="signal",
        vol_target=0.15,
        vol_lookback=3,
        max_leverage=2.0,
        flat_mask="is_eod",
        signal_lag=1,
    )
    assert res_vt.data["position"][3] == pytest.approx(0.0)
    assert res_vt.data["position"].abs().max() <= 2.0 + 1e-9

    # 4. No-trade buffer suppresses tiny changes (1.0 -> 1.05 -> 1.08 stays at 1.0 when buffer=0.1)
    res_buf = df.anta.backtest(
        signal="signal",
        no_trade_buffer=0.1,
        signal_lag=0,
    )
    assert res_buf.data["position"][0] == pytest.approx(1.0)
    assert res_buf.data["position"][1] == pytest.approx(1.0)
    assert res_buf.data["position"][2] == pytest.approx(1.0)
    # Full exit to 0.0 is always respected
    assert res_buf.data["position"][6] == pytest.approx(0.0)

    # Validation errors for new parameters
    with pytest.raises(ValueError, match="execution_price must be one of"):
        df.anta.backtest(execution_price="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vol_target must be a positive"):
        df.anta.backtest(vol_target=0.0)
    with pytest.raises(
        ValueError, match="vol_lookback must be an integer greater than or equal to 2"
    ):
        df.anta.backtest(vol_target=0.15, vol_lookback=1)
    with pytest.raises(ValueError, match="max_leverage must be a positive"):
        df.anta.backtest(vol_target=0.15, max_leverage=0.0)
    with pytest.raises(ValueError, match="no_trade_buffer must be non-negative"):
        df.anta.backtest(no_trade_buffer=-0.1)


def test_round_trip_trades_mae_mfe_and_parameter_sweep() -> None:
    """Test BacktestResult.trades() with MAE/MFE analytics and parameter_sweep()."""
    from polars_anta import extract_trades, parameter_sweep

    # Trade 1 (LONG on bars 1, 2, 3):
    # Price at bar 0 is 100.0 -> bar 1 drops to 98.0 (-2% intra-trade dip -> MAE = -0.02)
    # -> bar 2 rallies to 105.0 (+5% intra-trade peak -> MFE = +0.05)
    # -> bar 3 closes at 103.0 (+3% final gross return)
    # Trade 2 (SHORT on bars 5, 6):
    # Price at bar 4 is 100.0 -> bar 5 drops to 95.0 (+5% short gain) -> bar 6 drops to 90.25
    df = pl.DataFrame(
        {
            "timestamp": [f"2026-01-0{i}" for i in range(1, 9)],
            "close": [100.0, 98.0, 105.0, 103.0, 100.0, 95.0, 90.25, 90.25],
            "signal": [1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0],
        }
    )

    res = df.anta.backtest(
        signal="signal",
        close_col="close",
        time_col="timestamp",
        fee_rate=0.0005,
        signal_lag=1,
    )
    trades = res.trades()
    assert len(trades) == 2
    assert trades["trade_id"].to_list() == [1, 2]
    assert trades["side"].to_list() == ["LONG", "SHORT"]
    assert trades["holding_bars"].to_list() == [3, 2]
    assert "entry_time" in trades.columns and "exit_time" in trades.columns

    # Trade 1 MAE is -2% (from 100 -> 98) and MFE is +5% (from 100 -> 105), gross_return is +3% (100 -> 103)
    assert trades["mae"][0] == pytest.approx(-0.02, rel=1e-5)
    assert trades["mfe"][0] == pytest.approx(0.05, rel=1e-5)
    assert trades["gross_return"][0] == pytest.approx(0.03, rel=1e-5)
    assert trades["entry_price"][0] == pytest.approx(100.0)
    assert trades["exit_price"][0] == pytest.approx(103.0)

    # Summary includes round-trip trade metrics & PSR
    assert res["n_round_trips"] == 2.0
    assert res["trade_win_rate"] == pytest.approx(1.0)
    assert res["avg_holding_bars"] == pytest.approx(2.5)
    assert "psr" in res.summary

    # Direct extract_trades on flat signal (0 active trades) and without price column
    flat_bt = df.with_columns(pl.lit(0.0).alias("signal")).anta.backtest(signal="signal")
    assert len(flat_bt.trades()) == 0
    no_price_trades = extract_trades(res.data.drop("close"), price_col="close")
    assert len(no_price_trades) == 2
    assert no_price_trades["entry_price"].null_count() == 2

    # Parameter Sweep with Deflated Sharpe Ratio across multiple candidate signals
    sweep_df = df.lazy().ta.parameter_sweep(
        signals={
            "Long_Short_Base": pl.col("signal"),
            "Long_Only": pl.when(pl.col("signal") > 0).then(1.0).otherwise(0.0),
            "Inverted": -pl.col("signal"),
        },
        close_col="close",
        fee_rate=0.0005,
    )
    assert len(sweep_df) == 3
    assert "deflated_sharpe_ratio" in sweep_df.columns
    assert "sharpe_ratio" in sweep_df.columns
    # Sorted descending by sharpe_ratio
    assert sweep_df["sharpe_ratio"][0] >= sweep_df["sharpe_ratio"][-1]

    with pytest.raises(ValueError, match="signals mapping must contain at least one"):
        parameter_sweep(df, signals={})
