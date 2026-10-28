"""Factor evaluation (IC, Rank IC, Quantile Analysis) and strategy performance metrics for Polars."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeVar

import polars as pl

from .core import IntoExpr, parse_expr

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)


# =============================================================================
# Strategy Performance Expressions (pl.Expr)
# =============================================================================


def drawdown_series(
    returns_or_equity: IntoExpr,
    is_returns: bool = True,
) -> pl.Expr:
    """Compute the underwater drawdown series from period returns or an equity curve.

    Returns values in ``(-1.0, 0.0]`` representing the percentage drop from the
    running peak at each bar.

    :param returns_or_equity: Period returns column/expression (when ``is_returns=True``)
        or cumulative equity curve column/expression (when ``is_returns=False``).
    :param is_returns: If True, compounds ``(1 + r).cum_prod()`` first; if False,
        treats input directly as an equity/price series.
    :return: Polars expression of drawdown values ``<= 0.0``.
    """
    x = parse_expr(returns_or_equity)
    equity = (1.0 + x.fill_null(0.0)).cum_prod() if is_returns else x
    peak = equity.cum_max()
    return pl.when(peak.is_null() | (peak <= 0.0)).then(0.0).otherwise((equity - peak) / peak)


def max_drawdown(
    returns_or_equity: IntoExpr,
    is_returns: bool = True,
) -> pl.Expr:
    """Compute the Maximum Drawdown (MDD) scalar from period returns or an equity curve.

    Returns the worst peak-to-trough percentage decline (e.g., ``-0.18`` for an 18% drawdown).

    :param returns_or_equity: Period returns or equity curve column/expression.
    :param is_returns: If True, compounds ``(1 + r).cum_prod()`` first.
    :return: Scalar Polars expression.
    """
    return drawdown_series(returns_or_equity, is_returns=is_returns).min().fill_null(0.0)


def sharpe_ratio(
    returns: IntoExpr,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the annualized Sharpe Ratio scalar expression.

    :param returns: Period returns column or expression.
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression.
    """
    if annualization <= 0.0:
        raise ValueError("annualization must be a positive number greater than zero")

    r = parse_expr(returns)
    rf_per_bar = float(risk_free_rate) / float(annualization)
    excess = r - rf_per_bar
    mean_ex = excess.mean()
    std_ex = excess.std()
    return (
        pl.when(std_ex.is_null() | (std_ex == 0.0) | mean_ex.is_null())
        .then(0.0)
        .otherwise((mean_ex / std_ex) * math.sqrt(float(annualization)))
    )


def sortino_ratio(
    returns: IntoExpr,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the annualized Sortino Ratio scalar expression using downside deviation.

    :param returns: Period returns column or expression.
    :param risk_free_rate: Annualized risk-free rate (default 0.0).
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression.
    """
    if annualization <= 0.0:
        raise ValueError("annualization must be a positive number greater than zero")

    r = parse_expr(returns)
    rf_per_bar = float(risk_free_rate) / float(annualization)
    excess = r - rf_per_bar
    mean_ex = excess.mean()
    downside_sq = pl.when(excess < 0.0).then(excess**2).otherwise(0.0)
    downside_dev = downside_sq.mean().sqrt()
    return (
        pl.when(downside_dev.is_null() | (downside_dev == 0.0) | mean_ex.is_null())
        .then(0.0)
        .otherwise((mean_ex / downside_dev) * math.sqrt(float(annualization)))
    )


def calmar_ratio(
    returns: IntoExpr,
    annualization: float = 252.0,
) -> pl.Expr:
    """Compute the Calmar Ratio (Annualized Return / |Max Drawdown|) scalar expression.

    :param returns: Period returns column or expression.
    :param annualization: Number of trading periods per year (default 252.0).
    :return: Scalar Polars expression.
    """
    if annualization <= 0.0:
        raise ValueError("annualization must be a positive number greater than zero")

    r = parse_expr(returns)
    ann_return = r.mean() * float(annualization)
    mdd_abs = max_drawdown(r, is_returns=True).abs()
    return (
        pl.when(mdd_abs.is_null() | (mdd_abs == 0.0) | ann_return.is_null())
        .then(0.0)
        .otherwise(ann_return / mdd_abs)
    )


def profit_factor(
    returns: IntoExpr,
) -> pl.Expr:
    """Compute the Profit Factor (Gross Profit / |Gross Loss|) scalar expression.

    :param returns: Period returns or trade PnL column/expression.
    :return: Scalar Polars expression.
    """
    r = parse_expr(returns)
    gross_profit = pl.when(r > 0.0).then(r).otherwise(0.0).sum()
    gross_loss = pl.when(r < 0.0).then(r.abs()).otherwise(0.0).sum()
    return (
        pl.when(gross_loss == 0.0)
        .then(pl.when(gross_profit > 0.0).then(float("inf")).otherwise(0.0))
        .otherwise(gross_profit / gross_loss)
    )


def win_rate(
    returns: IntoExpr,
    include_zeros: bool = False,
) -> pl.Expr:
    """Compute the Win Rate (proportion of positive return periods) scalar expression.

    :param returns: Period returns or trade PnL column/expression.
    :param include_zeros: When False (default), computes win rate over active non-zero
        periods ``(r != 0)``; when True, includes zero-return periods in the denominator.
    :return: Scalar Polars expression in ``[0.0, 1.0]``.
    """
    r = parse_expr(returns)
    wins = (r > 0.0).cast(pl.Float64).sum()
    if include_zeros:
        total = r.is_not_null().cast(pl.Float64).sum()
    else:
        total = (r.is_not_null() & (r != 0.0)).cast(pl.Float64).sum()
    return pl.when(total == 0.0).then(0.0).otherwise(wins / total)


# =============================================================================
# DataFrame / LazyFrame Factor Evaluation Suite
# =============================================================================


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
        ic_expr = pl.corr(f_col.rank(), r_col.rank(), method="pearson").fill_nan(0.0).alias("ic")
    else:
        ic_expr = pl.corr(f_col, r_col, method="pearson").fill_nan(0.0).alias("ic")

    count_expr = pl.len().cast(pl.Int64).alias("n_obs")

    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        res = cleaned.group_by(gb_cols).agg([ic_expr, count_expr]).sort(gb_cols)
    else:
        res = cleaned.select([ic_expr, count_expr])

    return res.collect() if isinstance(res, pl.LazyFrame) else res


def ic_summary(
    df: pl.DataFrame | pl.LazyFrame,
    factor_col: str,
    return_col: str,
    group_by: str | Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> dict[str, float]:
    """Compute comprehensive Information Coefficient (IC) summary statistics.

    Returns a dictionary containing:
    - ``ic_mean``: Mean IC across cross-sections.
    - ``ic_std``: Standard deviation of IC.
    - ``icir``: Information Ratio of IC (``ic_mean / ic_std``).
    - ``t_stat``: t-statistic testing ``IC != 0``.
    - ``hit_rate``: Proportion of periods where ``IC > 0``.
    - ``n_periods``: Total number of valid periods evaluated.

    :param df: Input Polars DataFrame or LazyFrame.
    :param factor_col: Column name of the alpha factor.
    :param return_col: Column name of the target forward return.
    :param group_by: Optional cross-sectional time column(s) (e.g., ``'timestamp'`` or ``'date'``).
    :param method: ``'spearman'`` (Rank IC) or ``'pearson'`` (Normal IC).
    :return: Dictionary of summary metrics.
    """
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
            "hit_rate": 0.0,
            "n_periods": 0.0,
        }

    ic_series = ic_df["ic"]
    mean_val = float(ic_series.mean() or 0.0)
    std_val = float(ic_series.std() or 0.0) if n > 1 else 0.0
    icir_val = mean_val / std_val if std_val > 1e-12 else 0.0
    t_stat_val = (mean_val / (std_val / math.sqrt(float(n)))) if std_val > 1e-12 else 0.0
    hit_rate_val = float((ic_series > 0.0).mean() or 0.0)

    return {
        "ic_mean": mean_val,
        "ic_std": std_val,
        "icir": icir_val,
        "t_stat": t_stat_val,
        "hit_rate": hit_rate_val,
        "n_periods": float(n),
    }


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
