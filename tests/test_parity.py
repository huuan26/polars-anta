"""Mathematical parity and precision verification tests for polars-anta indicators.

Verifies that polars-anta indicators match textbook mathematical definitions
and golden reference implementations within machine epsilon / float precision (< 1e-6).
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_anta.alpha import fractal_dimension
from polars_anta.momentum import squeeze_momentum
from polars_anta.stats import linear_regression_slope, zscore
from polars_anta.trend import dema, ema, sma, ssl_channel, trend_intensity_index, wma
from polars_anta.volatility import bollinger_bands
from polars_anta.volume import vwap_bands


@pytest.fixture
def golden_price_series() -> list[float]:
    """Well-known golden closing price sequence (from Murphy's Technical Analysis)."""
    return [
        22.27,
        22.19,
        22.08,
        22.17,
        22.18,
        22.13,
        22.23,
        22.43,
        22.24,
        22.29,
        22.15,
        22.39,
        22.38,
        22.61,
        23.36,
        24.05,
        23.75,
        23.83,
        23.95,
        23.63,
        23.82,
        23.87,
        23.65,
        23.19,
        23.10,
        23.33,
        22.68,
        23.10,
        22.40,
        22.17,
    ]


def test_sma_mathematical_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(sma("close", period=period).alias("sma"))["sma"].to_list()

    # Manual textbook reference calculation
    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        expected_sma = sum(window) / float(period)
        assert pytest.approx(res[i], abs=1e-7) == expected_sma


def test_wma_mathematical_parity(golden_price_series: list[float]):
    period = 5
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(wma("close", period=period).alias("wma"))["wma"].to_list()

    weights = list(range(1, period + 1))
    weight_sum = sum(weights)

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        expected_wma = sum(w * x for w, x in zip(weights, window, strict=False)) / float(weight_sum)
        assert pytest.approx(res[i], abs=1e-7) == expected_wma


def test_ema_mathematical_parity(golden_price_series: list[float]):
    period = 10
    alpha = 2.0 / (period + 1.0)
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(ema("close", period=period).alias("ema"))["ema"].to_list()

    # Recurrence relation: EMA[t] = alpha * Price[t] + (1 - alpha) * EMA[t-1]
    curr = golden_price_series[0]
    assert pytest.approx(res[0], abs=1e-7) == curr

    for i in range(1, len(golden_price_series)):
        curr = alpha * golden_price_series[i] + (1.0 - alpha) * curr
        assert pytest.approx(res[i], abs=1e-6) == curr


def test_dema_mathematical_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    # DEMA = 2 * EMA - EMA(EMA)
    res_dema = df.select(dema("close", period=period).alias("dema"))["dema"].to_list()

    e1 = df.select(ema("close", period=period).alias("e1"))
    e2 = e1.select(ema("e1", period=period).alias("e2"))
    expected_dema = (2.0 * e1["e1"] - e2["e2"]).to_list()

    for r, e in zip(res_dema, expected_dema, strict=False):
        assert pytest.approx(r, abs=1e-6) == e


def test_bollinger_bands_parity(golden_price_series: list[float]):
    period = 10
    num_std = 2.0
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(bollinger_bands("close", period=period, num_std=num_std).alias("bb")).unnest(
        "bb"
    )

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        mean = sum(window) / float(period)
        # Sample standard deviation (ddof=1)
        variance = sum((x - mean) ** 2 for x in window) / float(period - 1)
        std = variance**0.5

        assert pytest.approx(res["middle"][i], abs=1e-7) == mean
        assert pytest.approx(res["upper"][i], abs=1e-6) == mean + num_std * std
        assert pytest.approx(res["lower"][i], abs=1e-6) == mean - num_std * std


def test_linear_regression_slope_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(linear_regression_slope("close", period=period).alias("slope"))[
        "slope"
    ].to_list()

    # OLS closed-form: sum((t - mean_t) * (y - mean_y)) / sum((t - mean_t)^2)
    t = list(range(period))
    mean_t = sum(t) / float(period)
    denom = sum((x - mean_t) ** 2 for x in t)

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        mean_y = sum(window) / float(period)
        nom = sum((k - mean_t) * (y - mean_y) for k, y in zip(t, window, strict=False))
        expected_slope = nom / denom
        assert pytest.approx(res[i], abs=1e-6) == expected_slope


def test_zscore_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(zscore("close", period=period).alias("z"))["z"].to_list()

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        mean = sum(window) / float(period)
        variance = sum((x - mean) ** 2 for x in window) / float(period - 1)
        std = variance**0.5
        expected_z = (golden_price_series[i] - mean) / std
        assert pytest.approx(res[i], abs=1e-6) == expected_z


def test_ssl_channel_mathematical_parity():
    # Synthetic test sequence with predictable state transitions
    highs = [12.0, 13.0, 14.0, 15.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0]
    lows = [8.0, 9.0, 10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0]
    closes = [10.0, 11.0, 15.5, 14.0, 13.0, 8.5, 9.0, 9.5, 6.0, 7.0]
    period = 3

    df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
    res = df.select(ssl_channel("high", "low", "close", period=period).alias("ssl")).unnest("ssl")

    ssl_up = res["ssl_up"].to_list()
    ssl_down = res["ssl_down"].to_list()
    hlv = res["hlv"].to_list()

    # Manual step-by-step verification
    state_hlv = 1
    for i in range(len(closes)):
        if i < period - 1:
            assert ssl_up[i] is None
            assert ssl_down[i] is None
            assert hlv[i] is None
            continue

        sma_h = sum(highs[i - period + 1 : i + 1]) / float(period)
        sma_l = sum(lows[i - period + 1 : i + 1]) / float(period)
        c = closes[i]

        if c > sma_h:
            state_hlv = 1
        elif c < sma_l:
            state_hlv = -1

        expected_up = sma_l if state_hlv < 0 else sma_h
        expected_down = sma_h if state_hlv < 0 else sma_l

        assert hlv[i] == state_hlv
        assert pytest.approx(ssl_up[i], abs=1e-7) == expected_up
        assert pytest.approx(ssl_down[i], abs=1e-7) == expected_down


def test_squeeze_momentum_mathematical_parity():
    # Synthetic test sequence
    highs = [10.0 + i * 0.2 for i in range(40)]
    lows = [8.0 + i * 0.2 for i in range(40)]
    closes = [9.0 + i * 0.2 for i in range(40)]

    df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
    res = df.select(
        squeeze_momentum(
            "high",
            "low",
            "close",
            bb_period=10,
            bb_mult=2.0,
            kc_period=10,
            kc_mult=1.5,
            mom_period=10,
        ).alias("sqz")
    ).unnest("sqz")

    sqz_on = res["squeeze_on"].to_list()
    sqz_off = res["squeeze_off"].to_list()
    momentum = res["momentum"].to_list()

    # Manual check at the final index
    idx = 39
    c_win = closes[idx - 9 : idx + 1]
    bb_mean = sum(c_win) / 10.0
    bb_std = (sum((x - bb_mean) ** 2 for x in c_win) / 9.0) ** 0.5
    up_bb = bb_mean + 2.0 * bb_std
    lo_bb = bb_mean - 2.0 * bb_std

    kc_mean = bb_mean
    tr_win = [
        max(highs[k] - lows[k], abs(highs[k] - closes[k - 1]), abs(lows[k] - closes[k - 1]))
        if k > 0
        else highs[k] - lows[k]
        for k in range(idx - 9, idx + 1)
    ]
    kc_range = (sum(tr_win) / 10.0) * 1.5
    up_kc = kc_mean + kc_range
    lo_kc = kc_mean - kc_range

    expected_on = lo_bb > lo_kc and up_bb < up_kc
    expected_off = lo_bb < lo_kc and up_bb > up_kc

    assert sqz_on[idx] == expected_on
    assert sqz_off[idx] == expected_off
    assert momentum[idx] is not None


def test_vwap_bands_mathematical_parity():
    highs = [10.5, 11.5, 12.0, 11.8, 12.5]
    lows = [9.5, 10.2, 11.0, 10.9, 11.8]
    closes = [10.0, 11.0, 11.5, 11.2, 12.0]
    volumes = [100.0, 150.0, 200.0, 120.0, 180.0]

    df = pl.DataFrame({"high": highs, "low": lows, "close": closes, "volume": volumes})
    res = df.select(vwap_bands("high", "low", "close", "volume").alias("vb")).unnest("vb")

    tps = [(h + lo_val + c) / 3.0 for h, lo_val, c in zip(highs, lows, closes, strict=False)]
    for i in range(len(closes)):
        cum_pv = sum(tp * v for tp, v in zip(tps[: i + 1], volumes[: i + 1], strict=False))
        cum_v = sum(volumes[: i + 1])
        expected_vwap = cum_pv / cum_v

        # Weighted variance
        cum_dev_sq = sum(
            v * ((tp - expected_vwap) ** 2)
            for tp, v in zip(tps[: i + 1], volumes[: i + 1], strict=False)
        )
        expected_std = (cum_dev_sq / cum_v) ** 0.5

        assert pytest.approx(res["vwap"][i], abs=1e-7) == expected_vwap
        assert pytest.approx(res["upper_1"][i], abs=1e-7) == expected_vwap + expected_std
        assert pytest.approx(res["lower_1"][i], abs=1e-7) == expected_vwap - expected_std
        assert pytest.approx(res["upper_2"][i], abs=1e-7) == expected_vwap + 2.0 * expected_std
        assert pytest.approx(res["lower_2"][i], abs=1e-7) == expected_vwap - 2.0 * expected_std


def test_trend_intensity_index_mathematical_parity():
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    period = 4
    signal_period = 2

    df = pl.DataFrame({"close": closes})
    res = df.select(
        trend_intensity_index("close", period=period, signal_period=signal_period).alias("tii")
    ).unnest("tii")

    # For strictly increasing sequence, close is always > SMA(close, 4)
    # dev is strictly positive, neg_dev is 0 -> TII must be 100.0 for all valid bars
    tii_vals = res["tii"].to_list()
    for i in range(period - 1, len(closes)):
        assert tii_vals[i] == 100.0


def test_fractal_dimension_mathematical_parity():
    # Linear trend should have low fractal dimension (~1.0)
    highs = [10.0 + i * 1.0 for i in range(30)]
    lows = [9.0 + i * 1.0 for i in range(30)]

    df = pl.DataFrame({"high": highs, "low": lows})
    res = df.select(fractal_dimension("high", "low", period=10).alias("fd"))

    vals = [x for x in res["fd"].to_list() if x is not None]
    assert len(vals) > 0
    # On straight linear diagonal line, fractal dimension is close to 1.0
    for v in vals:
        assert 1.0 <= v <= 1.2


def test_package_version_and_metadata():
    """Verify package version is 0.2.0 and matches project release."""
    import polars_anta

    assert polars_anta.__version__ == "0.2.0"
