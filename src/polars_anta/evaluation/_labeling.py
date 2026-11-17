"""Quantitative Machine Learning labeling expressions (Triple-Barrier and Trend-Scanning)."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..core import IntoExpr, check_period, parse_expr


def triple_barrier_labels(
    close: IntoExpr,
    volatility: IntoExpr | float | int | None = None,
    horizon: int = 10,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    min_ret: float = 0.0,
    vol_period: int = 20,
    side: IntoExpr | None = None,
) -> pl.Expr:
    """Dynamic Triple-Barrier & Meta-Labeling Method (López de Prado, 2018).

    Evaluates the forward price trajectory over ``horizon`` bars against two dynamic
    horizontal barriers scaled by local volatility $\\sigma_t$ — an upper Profit-Taking (PT)
    barrier $+\\max(\\text{pt\\_mult} \\cdot \\sigma_t, \\text{min\\_ret})$ and a lower Stop-Loss (SL)
    barrier $-\\max(\\text{sl\\_mult} \\cdot \\sigma_t, \\text{min\\_ret})$ — plus a vertical expiration
    barrier at $t + \\text{horizon}$.

    Returns a ``pl.Struct`` with fields:
        - ``'label'``: ``+1.0`` if upper barrier touched first, ``-1.0`` if lower barrier
          touched first, or ``0.0`` if vertical horizon expired first.
        - ``'touch_bars'``: Number of bars elapsed until the first barrier touch ($1 \\dots \\text{horizon}$).
        - ``'realized_return'``: Realized return at the first touched barrier.
        - ``'meta_label'``: Binary meta-label in ``{0.0, 1.0}`` indicating whether the trade
          achieved its profit-taking target.

    :param close: Close price column or expression.
    :param volatility: Optional per-bar volatility column/expression or positive scalar target.
        When None, defaults to rolling standard deviation of returns over ``vol_period``.
    :param horizon: Vertical barrier maximum holding period in bars (default 10).
    :param pt_mult: Profit-taking volatility multiplier (default 2.0; set 0.0 to disable PT barrier).
    :param sl_mult: Stop-loss volatility multiplier (default 2.0; set 0.0 to disable SL barrier).
    :param min_ret: Minimum absolute return barrier threshold (default 0.0).
    :param vol_period: Rolling window used when ``volatility=None`` (default 20).
    :param side: Optional primary model directional signal ($\\in \\{-1, +1\\}$) for Meta-Labeling.
    :return: Polars Struct expression.
    """
    check_period(horizon, 1, "horizon")
    if pt_mult < 0.0 or sl_mult < 0.0:
        raise ValueError("pt_mult and sl_mult must be non-negative")
    if min_ret < 0.0:
        raise ValueError("min_ret must be non-negative")
    check_period(vol_period, 2, "vol_period")

    c = parse_expr(close).cast(pl.Float64)

    if volatility is None:
        vol_expr = c.pct_change().rolling_std(window_size=vol_period, min_samples=2)
    elif isinstance(volatility, bool):
        raise TypeError("volatility cannot be a boolean")
    elif isinstance(volatility, (int, float)):
        if volatility <= 0.0:
            raise ValueError("scalar volatility must be a positive number")
        vol_expr = pl.lit(float(volatility), dtype=pl.Float64)
    else:
        vol_expr = parse_expr(volatility).cast(pl.Float64)

    side_expr = (
        parse_expr(side).cast(pl.Float64).sign().fill_null(1.0)
        if side is not None
        else pl.lit(1.0, dtype=pl.Float64)
    )

    h = int(horizon)
    pt_m = float(pt_mult)
    sl_m = float(sl_mult)
    min_r = float(min_ret)

    return_dtype = pl.Struct(
        [
            pl.Field("label", pl.Float64),
            pl.Field("touch_bars", pl.Int64),
            pl.Field("realized_return", pl.Float64),
            pl.Field("meta_label", pl.Float64),
        ]
    )

    def _calc_tb(s_struct: pl.Series) -> pl.Series:
        df_chunk = s_struct.struct.unnest()
        c_vals = df_chunk["close"].to_numpy().astype(np.float64, copy=False)
        v_vals = df_chunk["vol"].to_numpy().astype(np.float64, copy=False)
        s_vals = df_chunk["side"].to_numpy().astype(np.float64, copy=False)
        n = len(c_vals)

        labels = np.full(n, np.nan, dtype=np.float64)
        touch_bars = np.full(n, -1, dtype=np.int64)
        realized_ret = np.full(n, np.nan, dtype=np.float64)
        meta_labels = np.full(n, np.nan, dtype=np.float64)

        if n > h:
            windows = np.lib.stride_tricks.sliding_window_view(c_vals, h + 1)
            m = len(windows)
            p0 = windows[:, :1]
            valid_mask = (
                (~np.isnan(windows).any(axis=1))
                & (p0[:, 0] > 0.0)
                & (~np.isnan(v_vals[:m]))
                & (s_vals[:m] != 0.0)
            )

            if valid_mask.any():
                valid_w = windows[valid_mask]
                valid_v = np.maximum(v_vals[:m][valid_mask], 1e-8)
                valid_s = np.sign(s_vals[:m][valid_mask])
                k = len(valid_w)

                raw_rets = (valid_w[:, 1:] - valid_w[:, :1]) / valid_w[:, :1]
                dir_rets = raw_rets * valid_s[:, None]

                if pt_m > 0.0:
                    u_thresh = np.maximum(pt_m * valid_v, min_r)[:, None]
                    hit_pt = dir_rets >= u_thresh
                    first_pt = np.where(hit_pt.any(axis=1), np.argmax(hit_pt, axis=1) + 1, h + 1)
                else:
                    first_pt = np.full(k, h + 1, dtype=np.int64)

                if sl_m > 0.0:
                    l_thresh = np.maximum(sl_m * valid_v, min_r)[:, None]
                    hit_sl = dir_rets <= -l_thresh
                    first_sl = np.where(hit_sl.any(axis=1), np.argmax(hit_sl, axis=1) + 1, h + 1)
                else:
                    first_sl = np.full(k, h + 1, dtype=np.int64)

                pt_wins = (first_pt < first_sl) & (first_pt <= h)
                sl_wins = (first_sl < first_pt) & (first_sl <= h)
                tie_hit = (first_pt == first_sl) & (first_pt <= h)

                lbl_trade = np.where(
                    pt_wins | (tie_hit & (dir_rets[:, 0] >= 0.0)),
                    1.0,
                    np.where(sl_wins | tie_hit, -1.0, 0.0),
                )
                lbl_valid = np.where(lbl_trade == 0.0, 0.0, lbl_trade * valid_s)
                t_bars_valid = np.minimum(np.minimum(first_pt, first_sl), h).astype(np.int64)
                row_idx = np.arange(k)
                r_ret_valid = raw_rets[row_idx, t_bars_valid - 1]
                meta_valid = np.where(lbl_trade > 0.0, 1.0, 0.0)

                labels_slice = labels[:m]
                t_bars_slice = touch_bars[:m]
                r_ret_slice = realized_ret[:m]
                meta_slice = meta_labels[:m]

                labels_slice[valid_mask] = lbl_valid
                t_bars_slice[valid_mask] = t_bars_valid
                r_ret_slice[valid_mask] = r_ret_valid
                meta_slice[valid_mask] = meta_valid

        tb_series = pl.Series("touch_bars", touch_bars, dtype=pl.Int64)
        tb_series = pl.select(
            pl.when(tb_series > 0).then(tb_series).otherwise(None).alias("touch_bars")
        )["touch_bars"]

        return pl.DataFrame(
            {
                "label": pl.Series(labels, dtype=pl.Float64).fill_nan(None),
                "touch_bars": tb_series,
                "realized_return": pl.Series(realized_ret, dtype=pl.Float64).fill_nan(None),
                "meta_label": pl.Series(meta_labels, dtype=pl.Float64).fill_nan(None),
            }
        ).to_struct("triple_barrier")

    combined = pl.struct(
        [
            c.alias("close"),
            vol_expr.alias("vol"),
            side_expr.alias("side"),
        ]
    )
    return combined.map_batches(_calc_tb, return_dtype=return_dtype)


def trend_scanning_labels(
    close: IntoExpr,
    min_window: int = 5,
    max_window: int = 20,
    step: int = 1,
) -> pl.Expr:
    """Trend-Scanning Multi-Horizon OLS Regression Labeling (López de Prado, 2020).

    Fits forward Ordinary Least Squares (OLS) linear regressions across all candidate
    lookahead windows $L \\in [\\text{min\\_window}, \\text{max\\_window}]$ (stepping by ``step``)
    using closed-form Polars expressions and selects the horizon $L^*$ that maximizes the
    absolute slope $t$-statistic $|\\hat{\\beta}_1(L) / \\text{SE}(\\hat{\\beta}_1(L))|$.

    Returns a ``pl.Struct`` with fields:
        - ``'label'``: Trend direction $\\text{sign}(t_{L^*}) \\in \\{-1.0, 0.0, 1.0\\}$
        - ``'t_value'``: Maximum-magnitude OLS slope $t$-statistic $t_{L^*}$
        - ``'optimal_window'``: Lookahead window length $L^*$ achieving $t_{L^*}$
        - ``'ret'``: Forward return $(P_{t + L^* - 1} / P_t) - 1$ over the optimal window

    :param close: Close price column or expression.
    :param min_window: Minimum forward regression window length $L \\ge 3$ (default 5).
    :param max_window: Maximum forward regression window length $L \\ge \\text{min\\_window}$ (default 20).
    :param step: Grid step size between candidate windows (default 1).
    :return: Polars Struct expression.
    """
    if min_window < 3:
        raise ValueError("min_window must be an integer greater than or equal to 3")
    if max_window < min_window:
        raise ValueError("max_window must be greater than or equal to min_window")
    if step <= 0:
        raise ValueError("step must be a positive integer")

    c = parse_expr(close).cast(pl.Float64)
    windows = list(range(min_window, max_window + 1, step))
    return_dtype = pl.Struct(
        [
            pl.Field("label", pl.Float64),
            pl.Field("t_value", pl.Float64),
            pl.Field("optimal_window", pl.Int64),
            pl.Field("ret", pl.Float64),
        ]
    )

    def _calc_ts(s: pl.Series) -> pl.Series:
        c_vals = s.to_numpy().astype(np.float64, copy=False)
        n = len(c_vals)
        best_t = np.full(n, np.nan, dtype=np.float64)
        best_abs = np.full(n, -1.0, dtype=np.float64)
        best_w = np.full(n, np.nan, dtype=np.float64)
        best_ret = np.full(n, np.nan, dtype=np.float64)
        valid_best = np.zeros(n, dtype=bool)
        is_fin = np.isfinite(c_vals)
        c_safe = np.where(is_fin, c_vals, 0.0)

        chunk = 16384
        for w in windows:
            if n < w:
                continue
            m = n - w + 1
            s_xx = float(w * (w - 1) * (w + 1)) / 12.0
            x_w = np.arange(w, dtype=np.float64) - 0.5 * float(w - 1)
            denom_base = float(w - 2) * s_xx

            views = np.lib.stride_tricks.sliding_window_view(c_safe, w)
            fviews = np.lib.stride_tricks.sliding_window_view(is_fin, w)
            for start in range(0, m, chunk):
                stop = min(start + chunk, m)
                v_blk = views[start:stop]
                ok = fviews[start:stop].all(axis=1)
                if not ok.any():
                    continue
                v_ok = v_blk[ok]
                y_c = v_ok - v_ok.mean(axis=1, keepdims=True)
                slope = (y_c @ x_w) / s_xx
                s_yy = np.sum(y_c * y_c, axis=1)
                rss = np.maximum(0.0, s_yy - (slope * slope) * s_xx)
                se = np.sqrt(rss / denom_base)

                t_ok = np.where(
                    se > 1e-12,
                    np.clip(slope / np.where(se > 1e-12, se, 1.0), -1e6, 1e6),
                    np.where(np.abs(slope) > 1e-12, np.sign(slope) * 1e6, 0.0),
                )
                p0 = v_ok[:, 0]
                pend = v_ok[:, -1]
                r_ok = np.where(p0 != 0.0, (pend / np.where(p0 != 0.0, p0, 1.0)) - 1.0, np.nan)
                ret_valid = np.isfinite(r_ok)

                idx_rel = np.flatnonzero(ok)[ret_valid] + start
                t_v = t_ok[ret_valid]
                a_v = np.abs(t_v)
                r_v = r_ok[ret_valid]

                upd = (~valid_best[idx_rel]) | (a_v > best_abs[idx_rel])
                u_idx = idx_rel[upd]
                best_t[u_idx] = t_v[upd]
                best_abs[u_idx] = a_v[upd]
                best_w[u_idx] = float(w)
                best_ret[u_idx] = r_v[upd]
                valid_best[u_idx] = True

        labels = np.where(valid_best, np.sign(best_t), np.nan)
        return pl.DataFrame(
            {
                "label": pl.Series(labels, dtype=pl.Float64).fill_nan(None),
                "t_value": pl.Series(best_t, dtype=pl.Float64).fill_nan(None),
                "optimal_window": pl.Series(best_w, dtype=pl.Float64).fill_nan(None).cast(pl.Int64),
                "ret": pl.Series(best_ret, dtype=pl.Float64).fill_nan(None),
            }
        ).to_struct("trend_scan")

    return c.map_batches(_calc_ts, return_dtype=return_dtype)
