"""Panel factor falsification pipelines and parameter robustness sweeps."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from ..backtest import vectorized_backtest
from ..evaluation import _newey_west_lrv, information_coefficient
from ._bootstrap import adjust_pvalues, romano_wolf, spa_test
from ._overfitting import cusum_mean_test, empirical_bayes_shrinkage, pbo_cscv
from ._validation import _default_mean_block, _normalize_group_by, _two_sided_normal_p


def _join_factor_series(
    df: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    return_col: str,
    group_by: str | Sequence[str],
    compute_one: Callable[[pl.DataFrame, str, list[str]], pl.DataFrame],
) -> pl.DataFrame:
    f_cols = list(dict.fromkeys(str(c) for c in factor_cols))
    if not f_cols:
        raise ValueError("factor_cols must not be empty")
    gb_cols = _normalize_group_by(group_by)
    needed = [*gb_cols, return_col, *f_cols]
    base_df = df.select(needed).collect() if isinstance(df, pl.LazyFrame) else df.select(needed)
    wide_df: pl.DataFrame | None = None
    for f_col in f_cols:
        sub_df = compute_one(base_df, f_col, gb_cols)
        wide_df = (
            sub_df
            if wide_df is None
            else wide_df.join(sub_df, on=gb_cols, how="full", coalesce=True)
        )
    assert wide_df is not None
    return wide_df.sort(gb_cols)


def ic_matrix(
    df: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    return_col: str,
    group_by: str | Sequence[str],
    method: Literal["spearman", "pearson"] = "spearman",
) -> pl.DataFrame:
    """Compute a wide ``(T, K)`` DataFrame of per-period Information Coefficients across $K$ factors.

    Evaluates :func:`polars_anta.evaluation.information_coefficient` for each factor in
    ``factor_cols``, full-joins the resulting time series on ``group_by``, and sorts
    chronologically by ``group_by``.

    :param df: Input panel Polars DataFrame or LazyFrame.
    :param factor_cols: Non-empty sequence of factor column names.
    :param return_col: Target forward return column name.
    :param group_by: Cross-sectional timestamp/date column(s).
    :param method: ``'spearman'`` (Rank IC) or ``'pearson'`` (linear IC).
    :return: Eager Polars DataFrame with columns ``[*group_by, *factor_cols]``.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")
    return _join_factor_series(
        df,
        factor_cols,
        return_col,
        group_by,
        lambda base_df, f_col, gb_cols: (
            information_coefficient(
                base_df,
                factor_col=f_col,
                return_col=return_col,
                group_by=gb_cols,
                method=method,
            )
            .filter((pl.col("n_obs") > 0) & pl.col("ic").is_not_null() & (~pl.col("ic").is_nan()))
            .select([*gb_cols, pl.col("ic").alias(f_col)])
        ),
    )


def long_short_returns(
    df: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    return_col: str,
    group_by: str | Sequence[str],
    quantiles: int = 5,
    mode: Literal["long_short", "top_minus_mean"] = "long_short",
) -> pl.DataFrame:
    """Compute a wide ``(T, K)`` DataFrame of per-period quantile spread returns across $K$ factors.

    Within each ``group_by`` cross-section, observations with valid factor and return values are
    bucketed into $Q = \\text{quantiles}$ ordinal quantile portfolios ($1 \\dots Q$, matching
    :func:`polars_anta.evaluation.quantile_analysis`). For each period $t$:

    - ``mode='long_short'``: $\\bar{r}_{Q,t} - \\bar{r}_{1,t}$ (top quantile minus bottom quantile).
    - ``mode='top_minus_mean'``: $\\bar{r}_{Q,t} - \\bar{r}_{\\text{all},t}$ (top quantile minus equal-weight cross-sectional mean).

    :param df: Input panel Polars DataFrame or LazyFrame.
    :param factor_cols: Non-empty sequence of factor column names.
    :param return_col: Target forward return column name.
    :param group_by: Cross-sectional timestamp/date column(s).
    :param quantiles: Number of quantile buckets $Q \\ge 2$ (default ``5``).
    :param mode: ``'long_short'`` or ``'top_minus_mean'``.
    :return: Eager Polars DataFrame with columns ``[*group_by, *factor_cols]``.
    """
    q_num = int(quantiles)
    if q_num < 2:
        raise ValueError("quantiles must be an integer >= 2")
    if mode not in ("long_short", "top_minus_mean"):
        raise ValueError("mode must be 'long_short' or 'top_minus_mean'")
    r_expr = pl.col(return_col).cast(pl.Float64)

    def _compute_spread(base_df: pl.DataFrame, f_name: str, gb_cols: list[str]) -> pl.DataFrame:
        f_expr = pl.col(f_name).cast(pl.Float64)
        valid = (
            f_expr.is_not_null() & r_expr.is_not_null() & (~f_expr.is_nan()) & (~r_expr.is_nan())
        )
        pct_rank = (f_expr.rank(method="ordinal") - 1.0).over(gb_cols) / pl.len().over(gb_cols)
        q_expr = (
            (pct_rank * float(q_num)).floor().clip(0.0, float(q_num - 1)).cast(pl.Int32) + 1
        ).alias("_anta_q")
        top_mean = r_expr.filter(pl.col("_anta_q") == q_num).mean()
        bot_ref = (
            r_expr.filter(pl.col("_anta_q") == 1).mean() if mode == "long_short" else r_expr.mean()
        )
        return (
            base_df.filter(valid)
            .with_columns(q_expr)
            .group_by(gb_cols)
            .agg((top_mean - bot_ref).alias(f_name))
            .filter(pl.col(f_name).is_not_null() & (~pl.col(f_name).is_nan()))
        )

    return _join_factor_series(df, factor_cols, return_col, group_by, _compute_spread)


@dataclass
class FalsificationResult:
    """Container for multi-factor statistical falsification output.

    Attributes
    ----------
    table : pl.DataFrame
        Per-factor diagnostic table sorted by ``p_rw`` ascending with columns:
        ``factor``, ``ic_mean``, ``ic_std``, ``nw_t``, ``p_nw``, ``p_raw``, ``q_bh``,
        ``p_rw``, ``posterior_ic``, ``prob_positive``, ``p_cusum``, ``break_index``, ``verdict``.
    summary : dict[str, float]
        Aggregate funnel counts: ``n_tested``, ``n_periods_used``, ``n_raw_p05``,
        ``expected_false_raw``, ``n_bh``, ``n_rw``, ``n_robust``, ``n_decaying``, ``n_noise``.
    ic_series : pl.DataFrame
        Full wide IC time series returned by :func:`ic_matrix` prior to common-period intersection.
    """

    table: pl.DataFrame
    summary: dict[str, float]
    ic_series: pl.DataFrame

    def report(self) -> str:
        """Render a formatted human-readable falsification funnel and factor verdict report."""
        s = self.summary
        lines = [
            "==========================================================================",
            "                 POLARS-ANTA FACTOR FALSIFICATION REPORT                  ",
            "==========================================================================",
            f"Factors Tested        : {int(s['n_tested'])}",
            f"Common Periods Used   : {int(s['n_periods_used'])}",
            f"Raw p <= 0.05         : {int(s['n_raw_p05'])} (expected false under null: {s['expected_false_raw']:.2f})",
            f"BH FDR Survivors      : {int(s['n_bh'])}",
            f"Romano-Wolf Survivors : {int(s['n_rw'])}",
            "--------------------------------------------------------------------------",
            f"Verdict Breakdown     : robust={int(s['n_robust'])} | decaying={int(s['n_decaying'])} | noise={int(s['n_noise'])}",
            "--------------------------------------------------------------------------",
            f"{'Factor':<22} {'IC Mean':>9} {'NW t':>8} {'q_BH':>8} {'p_RW':>8} {'Post IC':>9} {'Verdict':>10}",
            "-" * 74,
        ]
        for row in self.table.iter_rows(named=True):
            lines.append(
                f"{str(row['factor'])[:22]:<22} "
                f"{float(row['ic_mean']):>9.4f} "
                f"{float(row['nw_t']):>8.2f} "
                f"{float(row['q_bh']):>8.4f} "
                f"{float(row['p_rw']):>8.4f} "
                f"{float(row['posterior_ic']):>9.4f} "
                f"{str(row['verdict']):>10}"
            )
        lines.append("==========================================================================")
        return "\n".join(lines)


def falsify_factors(
    df: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    return_col: str,
    group_by: str | Sequence[str],
    horizon: int = 1,
    method: Literal["spearman", "pearson"] = "spearman",
    n_boot: int = 1000,
    mean_block: float | None = None,
    fdr_q: float = 0.10,
    fwer_alpha: float = 0.05,
    prior: Literal["zero", "grand_mean"] = "zero",
    seed: int | None = None,
) -> FalsificationResult:
    """Run the end-to-end statistical falsification funnel across $K$ candidate alpha factors.

    Executes a 7-stage statistical sieve on the joint cross-sectional IC matrix:

    1. **Cross-sectional IC matrix** (:func:`ic_matrix`), retaining the intersection of periods
       where all $K$ factors have valid IC observations ($T \\ge 30$).
    2. **Newey–West HAC $t$-statistics** with lag $L = \\max(1, \\text{horizon} - 1)$.
    3. **Stationary bootstrap & Romano–Wolf stepdown max-$t$** (:func:`romano_wolf`) with default
       block length $\\max(\\text{horizon}, T^{1/3})$, yielding ``p_raw`` and FWER-adjusted ``p_rw``.
    4. **Benjamini–Hochberg FDR** (:func:`adjust_pvalues`) on ``p_raw``, yielding ``q_bh``.
    5. **Empirical-Bayes shrinkage** (:func:`empirical_bayes_shrinkage`) of ``ic_mean`` using
       Newey–West standard errors, yielding ``posterior_ic`` and ``prob_positive``.
    6. **OLS-CUSUM structural stability test** (:func:`cusum_mean_test`) on each IC series,
       yielding ``p_cusum`` and ``break_index``.
    7. **Pre-registered verdict classification** (priority ``decaying > robust > fdr_only > noise``):
       - ``'robust'``: ``p_rw <= fwer_alpha``
       - ``'fdr_only'``: ``q_bh <= fdr_q`` and not ``robust``
       - ``'decaying'``: (``robust`` or ``fdr_only``) and ``p_cusum <= 0.05`` and
         $|\\overline{\\text{IC}}_{\\text{second half}}| < |\\overline{\\text{IC}}_{\\text{first half}}|$
       - ``'noise'``: all remaining factors.

    :param df: Input panel Polars DataFrame or LazyFrame.
    :param factor_cols: Sequence of candidate factor column names.
    :param return_col: Target forward return column name.
    :param group_by: Cross-sectional timestamp/date column(s).
    :param horizon: Forward return horizon $h \\ge 1$ (controls NW lag and minimum block length).
    :param method: ``'spearman'`` (Rank IC) or ``'pearson'`` (Normal IC).
    :param n_boot: Number of stationary bootstrap replicates $B \\ge 1$.
    :param mean_block: Optional bootstrap mean block length (defaults to $\\max(h, T^{1/3})$).
    :param fdr_q: Benjamini–Hochberg FDR threshold $q \\in (0, 1)$, default ``0.10``.
    :param fwer_alpha: Romano–Wolf FWER threshold $\\alpha \\in (0, 1)$, default ``0.05``.
    :param prior: Empirical-Bayes prior center (``'zero'`` or ``'grand_mean'``).
    :param seed: Optional integer seed for deterministic reproducibility.
    :return: :class:`FalsificationResult` containing ``table``, ``summary``, and ``ic_series``.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be at least 1")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")
    if not (0.0 < float(fdr_q) < 1.0):
        raise ValueError("fdr_q must lie strictly in (0, 1)")
    if not (0.0 < float(fwer_alpha) < 1.0):
        raise ValueError("fwer_alpha must lie strictly in (0, 1)")

    f_cols = list(dict.fromkeys(str(c) for c in factor_cols))
    if not f_cols:
        raise ValueError("factor_cols must not be empty")
    gb_cols = _normalize_group_by(group_by)

    ic_wide = ic_matrix(
        df, factor_cols=f_cols, return_col=return_col, group_by=gb_cols, method=method
    )
    valid_row = pl.all_horizontal([pl.col(c).is_not_null() & (~pl.col(c).is_nan()) for c in f_cols])
    clean_ic = ic_wide.filter(valid_row)
    n_periods = len(clean_ic)
    if n_periods < 30:
        raise ValueError(
            f"falsify_factors requires at least 30 common valid IC periods across all factors, got {n_periods}"
        )

    ic_mat = clean_ic.select(f_cols).to_numpy().astype(np.float64, copy=False)
    k_factors = len(f_cols)

    ic_mean = ic_mat.mean(axis=0)
    ic_std = ic_mat.std(axis=0, ddof=1)
    nw_lag = max(1, h - 1)
    u_mat = ic_mat - ic_mean[None, :]
    lrv_vec = np.asarray(_newey_west_lrv(u_mat, nw_lag), dtype=np.float64)
    if lrv_vec.ndim == 0:
        lrv_vec = np.array([float(lrv_vec)], dtype=np.float64)
    ic_se_nw = np.sqrt(lrv_vec / float(n_periods))

    valid_std = ic_std > 1e-12
    nw_t = np.where(valid_std, ic_mean / ic_se_nw, 0.0)
    p_nw = np.array(
        [
            _two_sided_normal_p(float(t)) if is_v else 1.0
            for t, is_v in zip(nw_t, valid_std, strict=True)
        ],
        dtype=np.float64,
    )

    blk = (
        _default_mean_block(n_periods, mean_block)
        if mean_block is not None
        else max(float(h), float(n_periods) ** (1.0 / 3.0))
    )
    rw_df = romano_wolf(
        clean_ic,
        exclude=gb_cols,
        n_boot=n_boot,
        mean_block=blk,
        alternative="two-sided",
        seed=seed,
    )
    p_raw = rw_df["p_raw"].to_numpy()
    p_rw = rw_df["p_rw"].to_numpy()
    q_bh = adjust_pvalues(p_raw, method="bh")

    se_eb = np.where(valid_std, np.maximum(ic_se_nw, 1e-8), 1.0)
    eb_df = empirical_bayes_shrinkage(ic_mean, se_eb, names=f_cols, prior=prior)
    posterior_ic = eb_df["posterior_mean"].to_numpy()
    prob_positive = eb_df["prob_positive"].to_numpy()

    p_cusum = np.empty(k_factors, dtype=np.float64)
    break_index = np.empty(k_factors, dtype=np.int64)
    for idx_k in range(k_factors):
        cs_res = cusum_mean_test(ic_mat[:, idx_k], nw_lags=nw_lag)
        p_cusum[idx_k] = cs_res["p_value"]
        break_index[idx_k] = int(cs_res["break_index"])

    mid = n_periods // 2
    first_half_abs = np.abs(ic_mat[:mid, :].mean(axis=0))
    second_half_abs = np.abs(ic_mat[mid:, :].mean(axis=0))

    verdicts: list[str] = []
    for idx_k in range(k_factors):
        is_rob = bool(p_rw[idx_k] <= float(fwer_alpha))
        is_fdr = bool((q_bh[idx_k] <= float(fdr_q)) and not is_rob)
        is_dec = bool(
            (is_rob or is_fdr)
            and (p_cusum[idx_k] <= 0.05)
            and (second_half_abs[idx_k] < first_half_abs[idx_k])
        )
        if is_dec:
            verdicts.append("decaying")
        elif is_rob:
            verdicts.append("robust")
        elif is_fdr:
            verdicts.append("fdr_only")
        else:
            verdicts.append("noise")

    table = pl.DataFrame(
        {
            "factor": f_cols,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "nw_t": nw_t,
            "p_nw": p_nw,
            "p_raw": p_raw,
            "q_bh": q_bh,
            "p_rw": p_rw,
            "posterior_ic": posterior_ic,
            "prob_positive": prob_positive,
            "p_cusum": p_cusum,
            "break_index": break_index,
            "verdict": verdicts,
        }
    ).sort(["p_rw", "p_raw"])

    summary = {
        "n_tested": float(k_factors),
        "n_periods_used": float(n_periods),
        "n_raw_p05": float(np.sum(p_raw <= 0.05)),
        "expected_false_raw": 0.05 * float(k_factors),
        "n_bh": float(np.sum(q_bh <= float(fdr_q))),
        "n_rw": float(np.sum(p_rw <= float(fwer_alpha))),
        "n_robust": float(sum(1 for v in verdicts if v == "robust")),
        "n_fdr_only": float(sum(1 for v in verdicts if v == "fdr_only")),
        "n_decaying": float(sum(1 for v in verdicts if v == "decaying")),
        "n_noise": float(sum(1 for v in verdicts if v == "noise")),
    }
    return FalsificationResult(table=table, summary=summary, ic_series=ic_wide)


def sweep_robustness(
    df: pl.DataFrame | pl.LazyFrame,
    signals: Mapping[str, pl.Expr | str],
    time_col: str | None = None,
    n_splits: int = 16,
    n_boot: int = 1000,
    seed: int | None = None,
    **backtest_kwargs: Any,
) -> dict[str, object]:
    """Evaluate a parameter sweep of signals jointly via CSCV PBO, Hansen SPA, and Romano–Wolf.

    Executes :func:`polars_anta.backtest.vectorized_backtest` for each signal in ``signals``,
    constructs the aligned ``(T, K)`` ``net_return`` matrix, and runs :func:`pbo_cscv`,
    :func:`spa_test`, and :func:`romano_wolf`.

    :param df: Input OHLCV Polars DataFrame or LazyFrame.
    :param signals: Mapping of ``{strategy_name: signal_expr_or_col}`` ($K \\ge 2$).
    :param time_col: Optional timestamp column name to align return series across strategies.
    :param n_splits: Even number of CSCV blocks $S \\ge 4$ passed to :func:`pbo_cscv`.
    :param n_boot: Number of stationary bootstrap replicates passed to :func:`spa_test` and
        :func:`romano_wolf`.
    :param seed: Optional integer seed for deterministic reproducibility.
    :param backtest_kwargs: Additional keyword arguments forwarded to ``vectorized_backtest``.
    :return: Dictionary with keys ``returns_matrix``, ``pbo``, ``spa``, and ``romano_wolf``.
    """
    if len(signals) < 2:
        raise ValueError("sweep_robustness requires at least 2 signals")

    eager_df = df.collect() if isinstance(df, pl.LazyFrame) else df
    if time_col is not None and time_col not in eager_df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame")

    ret_wide: pl.DataFrame | None = None
    series_cols: list[pl.Series] = []

    for name, sig in signals.items():
        bt = vectorized_backtest(eager_df, signal=sig, **backtest_kwargs)
        net_ret = bt.data["net_return"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).alias(name)
        if time_col is not None:
            sub = pl.DataFrame({time_col: bt.data[time_col], name: net_ret})
            ret_wide = sub if ret_wide is None else ret_wide.join(sub, on=time_col, how="inner")
        else:
            series_cols.append(net_ret)

    if time_col is not None:
        assert ret_wide is not None
        returns_matrix = ret_wide.sort(time_col)
        excl: Sequence[str] = (time_col,)
    else:
        returns_matrix = pl.DataFrame(series_cols)
        excl = ()

    pbo_res = pbo_cscv(returns_matrix, exclude=excl, n_splits=n_splits, seed=seed)
    spa_res = spa_test(returns_matrix, exclude=excl, n_boot=n_boot, seed=seed)
    rw_res = romano_wolf(returns_matrix, exclude=excl, n_boot=n_boot, seed=seed)

    return {
        "returns_matrix": returns_matrix,
        "pbo": pbo_res,
        "spa": spa_res,
        "romano_wolf": rw_res,
    }
