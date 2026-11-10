"""Unit and statistical validation tests for polars_anta.inference."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

import polars_anta as pa
from polars_anta.evaluation import _newey_west_lrv, _nw_auto_lag


def test_ic_summary_nw_refactor_parity() -> None:
    """Verify exact numerical parity of ic_summary after extracting Newey-West helpers."""
    rng = np.random.default_rng(42)
    n_dates, n_assets = 80, 25
    dates = np.repeat(np.arange(n_dates), n_assets)
    assets = np.tile(np.arange(n_assets), n_dates)
    ret = rng.normal(0.0, 0.02, size=n_dates * n_assets)
    factor = 0.25 * ret + rng.normal(0.0, 0.02, size=n_dates * n_assets)
    df = pl.DataFrame({"date": dates, "asset": assets, "factor": factor, "fwd_ret": ret})

    expected = {
        None: 10.278339483747970,
        0: 10.937702275911013,
        5: 10.830998426443744,
    }
    for lags, exp_val in expected.items():
        res = pa.ic_summary(df, "factor", "fwd_ret", group_by="date", nw_lags=lags)
        assert math.isclose(res["nw_t_stat"], exp_val, rel_tol=1e-12)

    assert _nw_auto_lag(1) == 0
    u_2d = np.column_stack([ret[:50] - ret[:50].mean(), factor[:50] - factor[:50].mean()])
    lrv_2d = _newey_west_lrv(u_2d, 3)
    assert isinstance(lrv_2d, np.ndarray)
    assert lrv_2d.shape == (2,)
    assert math.isclose(float(lrv_2d[0]), float(_newey_west_lrv(u_2d[:, 0], 3)), rel_tol=1e-12)
