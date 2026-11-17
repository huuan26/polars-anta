"""Cross-sectional alpha factor evaluation suite for Polars DataFrames and LazyFrames."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeVar, cast

import numpy as np
import polars as pl

from ._metrics import sharpe_ratio, win_rate

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)


def forward_returns(
    df: FrameType,
    close_col: str = "close",
    periods: Sequence[int] = (1, 5, 10),
    group_by: str | Sequence[str] | None = None,
    prefix: str = "fwd_ret_",
) -> FrameType:
    """Compute multi-horizon forward returns ``(Close[t + p] / Close[t]) - 1`` for alpha evaluation.

    :param df: Input Polars DataFrame or LazyFrame.
    :param close_col: Name of close price column (default 'close').
    :param periods: Sequence of forward lookahead horizons (default ``(1, 5, 10)``).
    :param group_by: Optional symbol/ticker column(s) to compute shifts within each asset.
    :param prefix: Column name prefix for generated forward return columns (default ``'fwd_ret_'``).
    :return: DataFrame or LazyFrame with appended forward return columns.
    """
    if not periods:
        raise ValueError("periods must contain at least one positive horizon")

    c = pl.col(close_col)
    exprs: list[pl.Expr] = []
    for p in periods:
        if p <= 0:
            raise ValueError("each forward return period must be a positive integer")
        fwd_expr = (c.shift(-p) / c) - 1.0
        if group_by is not None:
            fwd_expr = fwd_expr.over(group_by)
        exprs.append(fwd_expr.alias(f"{prefix}{p}"))

    return df.with_columns(exprs)


def information_coefficient(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str,
    group_by: str | Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pl.DataFrame:
    """Compute the Information Coefficient (IC) or Rank IC series between an alpha factor and forward returns.

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor signal.
    :param return_col: Column name of the target forward return.
    :param group_by: Optional cross-sectional grouping column(s) (typically ``'timestamp'`` or ``'date'``).
        When provided, computes cross-sectional IC per timestamp; when None, computes a single global IC.
    :param method: Correlation method: ``'spearman'`` (Rank IC, default) or ``'pearson'`` (Normal IC).
    :return: Eager Polars DataFrame with ``ic`` and ``n_obs`` columns.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be either 'spearman' or 'pearson'")

    f_col = pl.col(factor_col).cast(pl.Float64)
    r_col = pl.col(return_col).cast(pl.Float64)

    valid_filter = f_col.is_not_null() & r_col.is_not_null() & (~f_col.is_nan()) & (~r_col.is_nan())
    cleaned = df.filter(valid_filter)

    if method == "spearman":
        ic_expr = pl.corr(f_col.rank(), r_col.rank(), method="pearson").fill_nan(None).alias("ic")
    else:
        ic_expr = pl.corr(f_col, r_col, method="pearson").fill_nan(None).alias("ic")

    count_expr = pl.len().cast(pl.Int64).alias("n_obs")

    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        res = cleaned.group_by(gb_cols).agg([ic_expr, count_expr]).sort(gb_cols)
    else:
        res = cleaned.select([ic_expr, count_expr])

    return res.collect() if isinstance(res, pl.LazyFrame) else res


def _nw_auto_lag(n_obs: int) -> int:
    """Newey–West (1994) plug-in bandwidth: max(1, floor(4 * (T/100)^(2/9))), capped at T-1."""
    if n_obs <= 1:
        return 0
    return min(max(1, int(4.0 * ((float(n_obs) / 100.0) ** (2.0 / 9.0)))), n_obs - 1)


def _newey_west_lrv(u: np.ndarray, max_lag: int) -> np.ndarray | float:
    """Bartlett-kernel long-run variance of DEMEANED series.

    :param u: Demeaned array of shape ``(T,)`` or ``(T, K)``.
    :param max_lag: Maximum Bartlett kernel lag $L \\ge 0$.
    :return: ``float`` for 1-D input, or ``np.ndarray`` of shape ``(K,)`` for 2-D input, floored at ``1e-24``.
    """
    n_obs = u.shape[0]
    effective_lag = min(max(0, int(max_lag)), max(0, n_obs - 1))
    if u.ndim == 1:
        gamma_0 = float(np.dot(u, u)) / float(n_obs)
        lrv = gamma_0
        for lag_j in range(1, effective_lag + 1):
            weight = 1.0 - float(lag_j) / float(effective_lag + 1)
            gamma_j = float(np.dot(u[lag_j:], u[:-lag_j])) / float(n_obs)
            lrv += 2.0 * weight * gamma_j
        return max(lrv, 1e-24)

    gamma_0_vec = (u * u).sum(axis=0) / float(n_obs)
    lrv_vec = gamma_0_vec.copy()
    for lag_j in range(1, effective_lag + 1):
        weight = 1.0 - float(lag_j) / float(effective_lag + 1)
        gamma_j_vec = (u[lag_j:] * u[:-lag_j]).sum(axis=0) / float(n_obs)
        lrv_vec = lrv_vec + 2.0 * weight * gamma_j_vec
    return np.maximum(lrv_vec, 1e-24)


def ic_summary(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str,
    group_by: str | Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
    nw_lags: int | None = None,
) -> dict[str, float]:
    """Compute comprehensive Information Coefficient (IC) summary statistics with Newey-West HAC t-stat.

    Returns a dictionary containing:
    - ``ic_mean``: Mean IC across cross-sections.
    - ``ic_std``: Standard deviation of IC.
    - ``icir``: Information Ratio of IC (``ic_mean / ic_std``).
    - ``t_stat``: Independent-sample t-statistic testing ``IC != 0``.
    - ``nw_t_stat``: Newey-West (1987) HAC autocorrelation-consistent t-statistic.
    - ``hit_rate``: Proportion of periods where ``IC > 0``.
    - ``n_periods``: Total number of valid periods evaluated.

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor.
    :param return_col: Column name of the target forward return.
    :param group_by: Optional cross-sectional time column(s) (e.g., ``'timestamp'`` or ``'date'``).
    :param method: ``'spearman'`` (Rank IC) or ``'pearson'`` (Normal IC).
    :param nw_lags: Optional Bartlett kernel maximum lag $L \\ge 0$ for Newey-West HAC standard error.
        When None, uses the Newey-West (1994) automatic plug-in rule $\\max(1, \\lfloor 4(T/100)^{2/9} \\rfloor)$.
    :return: Dictionary of summary metrics.
    """
    if nw_lags is not None and nw_lags < 0:
        raise ValueError("nw_lags must be a non-negative integer")

    ic_df = information_coefficient(
        df,
        factor_col=factor_col,
        return_col=return_col,
        group_by=group_by,
        method=method,
    ).filter((pl.col("n_obs") > 0) & pl.col("ic").is_not_null() & (~pl.col("ic").is_nan()))

    n = len(ic_df)
    if n == 0:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "nw_t_stat": 0.0,
            "hit_rate": 0.0,
            "n_periods": 0.0,
        }

    ic_series = ic_df["ic"]
    mean_val = float(cast(float, ic_series.mean() or 0.0))
    std_val = float(cast(float, ic_series.std() or 0.0)) if n > 1 else 0.0
    icir_val = mean_val / std_val if std_val > 1e-12 else 0.0
    t_stat_val = (mean_val / (std_val / math.sqrt(float(n)))) if std_val > 1e-12 else 0.0
    hit_rate_val = float(cast(float, (ic_series > 0.0).mean() or 0.0))

    if n > 1 and std_val > 1e-12:
        u = ic_series.to_numpy().astype(np.float64, copy=False) - mean_val
        max_lag = min(int(nw_lags), n - 1) if nw_lags is not None else _nw_auto_lag(n)
        lrv = float(_newey_west_lrv(u, max_lag))
        nw_se = math.sqrt(lrv / float(n))
        nw_t_stat_val = mean_val / nw_se
    else:
        nw_t_stat_val = 0.0

    return {
        "ic_mean": mean_val,
        "ic_std": std_val,
        "icir": icir_val,
        "t_stat": t_stat_val,
        "nw_t_stat": nw_t_stat_val,
        "hit_rate": hit_rate_val,
        "n_periods": float(n),
    }


def ic_decay(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    close_col: str = "close",
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 20),
    asset_col: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pl.DataFrame:
    """Compute Multi-Horizon Information Coefficient (IC) Decay and Alpha Half-Life.

    Evaluates ``ic_summary`` across a grid of forward return horizons $h \\in \\text{horizons}$,
    automatically setting Newey-West HAC bandwidth ``nw_lags = max(1, h - 1)`` to correct for
    overlapping return autocorrelation, and estimates the exponential decay half-life (in bars).

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor signal.
    :param close_col: Column name of the close price (default 'close').
    :param horizons: Sequence of forward return horizons to evaluate (default ``(1, 2, 3, 5, 10, 20)``).
    :param asset_col: Optional asset/symbol column(s) for computing forward returns per instrument.
    :param group_by: Optional cross-sectional time column(s) (e.g., ``'timestamp'`` or ``'date'``).
    :param method: Correlation method: ``'spearman'`` (Rank IC, default) or ``'pearson'`` (Normal IC).
    :return: Eager Polars DataFrame with columns ``['horizon', 'ic_mean', 'ic_std', 'icir',
        't_stat', 'nw_t_stat', 'hit_rate', 'n_periods', 'half_life']``.
    """
    if not horizons:
        raise ValueError("horizons must contain at least one positive integer")
    for h in horizons:
        if h <= 0:
            raise ValueError("each horizon in horizons must be a positive integer")

    prefix = "_anta_decay_ret_"
    if isinstance(df, pl.LazyFrame):
        eager_df = forward_returns(
            df,
            close_col=close_col,
            periods=horizons,
            group_by=asset_col,
            prefix=prefix,
        ).collect()
    else:
        eager_df = forward_returns(
            df,
            close_col=close_col,
            periods=horizons,
            group_by=asset_col,
            prefix=prefix,
        )

    rows: list[dict[str, float | int]] = []
    abs_ics: list[float] = []
    h_vals: list[float] = []

    for h in horizons:
        ret_col = f"{prefix}{h}"
        stats = ic_summary(
            eager_df,
            factor_col=factor_col,
            return_col=ret_col,
            group_by=group_by,
            method=method,
            nw_lags=max(1, int(h) - 1),
        )
        rows.append(
            {
                "horizon": int(h),
                "ic_mean": stats["ic_mean"],
                "ic_std": stats["ic_std"],
                "icir": stats["icir"],
                "t_stat": stats["t_stat"],
                "nw_t_stat": stats["nw_t_stat"],
                "hit_rate": stats["hit_rate"],
                "n_periods": stats["n_periods"],
            }
        )
        abs_ics.append(max(abs(stats["ic_mean"]), 1e-8))
        h_vals.append(float(h))

    if len(h_vals) >= 2:
        x_arr = np.asarray(h_vals, dtype=np.float64)
        y_arr = np.log(np.asarray(abs_ics, dtype=np.float64))
        x_centered = x_arr - x_arr.mean()
        denom = float(np.dot(x_centered, x_centered))
        decay_rate = (
            -float(np.dot(x_centered, y_arr - y_arr.mean())) / denom if denom > 0.0 else 0.0
        )
        half_life = math.log(2.0) / decay_rate if decay_rate > 1e-8 else float("inf")
    else:
        half_life = float("inf")

    for row in rows:
        row["half_life"] = half_life

    return pl.DataFrame(rows)


def quantile_analysis(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str,
    quantiles: int = 5,
    group_by: str | Sequence[str] | None = None,
    annualization: float = 252.0,
) -> pl.DataFrame:
    """Group observations into N factor quantiles and compute return & risk metrics per bucket.

    Used to verify factor monotonicity (whether higher quantile buckets systematically
    earn higher forward returns than lower buckets).

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor.
    :param return_col: Column name of the target forward return.
    :param quantiles: Number of quantile buckets ``Q >= 2`` (default 5).
    :param group_by: Optional cross-sectional time column(s) (e.g., ``'date'``) to assign
        quantiles within each period.
    :param annualization: Periods per year for Sharpe computation (default 252.0).
    :return: Polars DataFrame sorted by ``quantile`` (``1`` to ``quantiles``).
    """
    if quantiles < 2:
        raise ValueError("quantiles must be an integer greater than or equal to 2")

    f_col = pl.col(factor_col).cast(pl.Float64)
    r_col = pl.col(return_col).cast(pl.Float64)

    valid_filter = f_col.is_not_null() & r_col.is_not_null() & (~f_col.is_nan()) & (~r_col.is_nan())
    cleaned = df.filter(valid_filter)

    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        pct_rank = (f_col.rank(method="ordinal") - 1.0).over(gb_cols) / pl.len().over(gb_cols)
    else:
        pct_rank = (f_col.rank(method="ordinal") - 1.0) / pl.len()

    bucket_expr = (
        (pct_rank * float(quantiles)).floor().clip(0.0, float(quantiles - 1)).cast(pl.Int32) + 1
    ).alias("quantile")

    with_q = cleaned.with_columns(bucket_expr)
    agg_df = (
        with_q.group_by("quantile")
        .agg(
            [
                pl.len().cast(pl.Int64).alias("count"),
                f_col.mean().alias("mean_factor"),
                r_col.mean().alias("mean_return"),
                r_col.std().fill_null(0.0).alias("std_return"),
                sharpe_ratio(r_col, annualization=annualization).alias("sharpe_ratio"),
                win_rate(r_col, include_zeros=True).alias("win_rate"),
                ((1.0 + r_col).product() - 1.0).alias("cumulative_return"),
            ]
        )
        .sort("quantile")
    )

    return agg_df.collect() if isinstance(agg_df, pl.LazyFrame) else agg_df


def factor_turnover(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    asset_col: str = "ticker",
    time_col: str = "timestamp",
    lag: int = 1,
) -> pl.DataFrame:
    """Compute factor rank autocorrelation and signal turnover across time.

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor.
    :param asset_col: Column name identifying each asset/ticker (default 'ticker').
    :param time_col: Column name identifying each timestamp/date (default 'timestamp').
    :param lag: Period lag for autocorrelation and turnover calculation (default 1).
    :return: Polars DataFrame with ``[time_col, 'rank_autocorr', 'turnover']``.
    """
    if lag <= 0:
        raise ValueError("lag must be a positive integer")

    f_col = pl.col(factor_col).cast(pl.Float64)
    working = (
        df.sort([asset_col, time_col])
        .with_columns(
            (f_col.rank().over(time_col) / f_col.count().over(time_col)).alias("_cs_rank"),
            (f_col / f_col.abs().sum().over(time_col)).fill_nan(0.0).alias("_weight"),
        )
        .with_columns(
            pl.col("_cs_rank").shift(lag).over(asset_col).alias("_prev_rank"),
            pl.col("_weight").shift(lag).over(asset_col).alias("_prev_weight"),
        )
        .filter(pl.col("_prev_rank").is_not_null())
    )

    res = (
        working.group_by(time_col)
        .agg(
            [
                pl.corr("_cs_rank", "_prev_rank", method="pearson")
                .fill_nan(0.0)
                .alias("rank_autocorr"),
                (pl.col("_weight") - pl.col("_prev_weight")).abs().sum().alias("turnover"),
            ]
        )
        .sort(time_col)
    )

    return res.collect() if isinstance(res, pl.LazyFrame) else res
