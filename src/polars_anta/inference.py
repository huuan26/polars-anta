"""Statistical inference and factor falsification engine for quantitative research.

Looping convention (per AGENTS.md Invariant #2):
- Python loops over bars/timestamps are strictly forbidden; all time-axis reductions use
  vectorized Polars expressions, NumPy array operations, or BLAS matrix multiplications.
- Python loops over bootstrap replicate chunks, candidate factors/strategies, Newey–West
  lags, or CSCV combinations are permitted where inner time-series operations are vectorized.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import polars as pl

from polars_anta.backtest import BacktestResult, vectorized_backtest
from polars_anta.evaluation import _newey_west_lrv, _nw_auto_lag, information_coefficient

_STD_NORMAL = NormalDist(0.0, 1.0)


def _normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function $\\Phi(z)$."""
    if math.isnan(z):
        return float("nan")
    return float(_STD_NORMAL.cdf(z))


def _two_sided_normal_p(z: float) -> float:
    """Two-sided normal p-value $2(1 - \\Phi(|z|))$."""
    if math.isnan(z):
        return float("nan")
    return float(max(0.0, min(1.0, 2.0 * (1.0 - _STD_NORMAL.cdf(abs(z))))))


def _default_mean_block(n_obs: int, mean_block: float | None) -> float:
    """Validate or compute default stationary bootstrap block length $\\max(1, T^{1/3})$."""
    if mean_block is None:
        return max(1.0, float(n_obs) ** (1.0 / 3.0))
    val = float(mean_block)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError("mean_block must be a positive finite number")
    return val


def _to_1d_array(
    x: np.ndarray | pl.Series | Sequence[float],
    name: str = "x",
    min_len: int = 10,
) -> np.ndarray:
    """Convert a 1-D Series, sequence, or ndarray to a clean float64 NumPy vector."""
    if isinstance(x, pl.Series):
        if x.null_count() > 0:
            raise ValueError(f"{name} contains null values")
        arr = x.to_numpy().astype(np.float64, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {arr.ndim}D")
    if arr.size < min_len:
        raise ValueError(f"{name} must have at least {min_len} observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _to_matrix(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    exclude: Sequence[str] = (),
    min_rows: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """Convert a DataFrame, LazyFrame, or ndarray into a (T, K) float64 matrix and column names."""
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    if isinstance(data, pl.DataFrame):
        excl = set(exclude)
        cols = [c for c in data.columns if c not in excl]
        if not cols:
            raise ValueError("No columns remaining after applying exclude filter")
        sub = data.select(cols)
        null_counts = sub.null_count().row(0)
        if any(c > 0 for c in null_counts):
            raise ValueError("Input DataFrame contains null values in model/factor columns")
        mat = sub.to_numpy().astype(np.float64, copy=False)
        names = list(cols)
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            mat = arr[:, None]
        elif arr.ndim == 2:
            mat = arr
        else:
            raise ValueError(f"data array must be 1D or 2D, got {arr.ndim}D")
        names = [f"col_{i}" for i in range(mat.shape[1])]

    if mat.shape[0] < min_rows:
        raise ValueError(f"data must have at least {min_rows} rows, got {mat.shape[0]}")
    if mat.shape[1] < 1:
        raise ValueError("data must have at least 1 column")
    if not np.all(np.isfinite(mat)):
        raise ValueError("data contains NaN or infinite values")
    return mat, names


def stationary_bootstrap_indices(
    n_obs: int,
    n_boot: int = 1000,
    mean_block: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate circular stationary bootstrap index matrices without Python time-step loops.

    Implements the Stationary Bootstrap of Politis & Romano (1994), where resampled block
    lengths follow a geometric distribution with success probability $p = 1 / L$ (mean block
    length $L = \\text{mean\\_block}$) and indices wrap circularly modulo $T$:

    $$
    P(\\text{new block at } t) = p = \\frac{1}{L}, \\qquad I_{b,t} = \\left(U_{b, s(b,t)} + t - s(b,t)\\right) \\bmod T
    $$

    where $s(b,t) = \\max\\{\\tau \\le t : \\text{new}_{b,\\tau} = 1\\}$ is vectorized via
    ``np.maximum.accumulate``.

    References
    ----------
    Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303–1313.

    :param n_obs: Number of time-series observations $T \\ge 2$.
    :param n_boot: Number of bootstrap replicates $B \\ge 1$.
    :param mean_block: Expected block length $L > 0$. Defaults to $\\max(1, T^{1/3})$.
    :param seed: Optional integer seed for deterministic reproducibility.
    :return: Integer index array of shape ``(n_boot, n_obs)`` with dtype ``int64`` in ``[0, n_obs)``.
    """
    if int(n_obs) < 2:
        raise ValueError("n_obs must be at least 2")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")
    n = int(n_obs)
    b = int(n_boot)
    blk = _default_mean_block(n, mean_block)
    p = min(1.0, 1.0 / blk)

    rng = np.random.default_rng(seed)
    u = rng.integers(0, n, size=(b, n), dtype=np.int64)
    new = rng.random((b, n)) < p
    new[:, 0] = True
    t = np.arange(n, dtype=np.int64)
    bs = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    idx = (np.take_along_axis(u, bs, axis=1) + (t - bs)) % n
    return idx.astype(np.int64, copy=False)


def _bootstrap_means(
    x_mat: np.ndarray,
    idx: np.ndarray,
    chunk_bytes: int = 200_000_000,
) -> np.ndarray:
    """Compute bootstrap sample means ``(B, K)`` via chunked bincount weight matrices and BLAS ``@``.

    Avoids materializing the 3-D tensor ``(B, T, K)`` by counting index frequencies per replicate
    row $W_{b,t} = \\sum_{\\tau=0}^{T-1} \\mathbb{I}(I_{b,\\tau} = t)$ and multiplying
    $\\bar{X}^{\\ast} = \\frac{1}{T} W X$.
    """
    n_obs, n_cols = x_mat.shape
    n_boot = idx.shape[0]
    if idx.shape[1] != n_obs:
        raise ValueError(f"idx second dimension ({idx.shape[1]}) must match x_mat rows ({n_obs})")

    chunk_size = max(1, int(chunk_bytes // (8 * max(1, n_obs))))
    out = np.empty((n_boot, n_cols), dtype=np.float64)

    for start in range(0, n_boot, chunk_size):
        end = min(n_boot, start + chunk_size)
        b_chunk = end - start
        offsets = (np.arange(b_chunk, dtype=np.int64) * n_obs)[:, None]
        flat = (idx[start:end] + offsets).ravel()
        weights = np.bincount(flat, minlength=b_chunk * n_obs).reshape(b_chunk, n_obs)
        out[start:end] = (weights.astype(np.float64) @ x_mat) / float(n_obs)

    return out


def bootstrap_test(
    x: np.ndarray | pl.Series | Sequence[float],
    statistic: Literal["mean", "sharpe"] = "mean",
    n_boot: int = 2000,
    mean_block: float | None = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    ci_level: float = 0.95,
    annualization: float = 1.0,
    seed: int | None = None,
) -> dict[str, float]:
    """Stationary-bootstrap hypothesis test and confidence interval for a mean or Sharpe ratio.

    Tests $H_0: \\theta = 0$ using the null-centered stationary bootstrap distribution
    $\\tilde{\\theta}^{\\ast}_b = \\hat{\\theta}^{\\ast}_b - \\hat{\\theta}$ with finite-sample
    p-value correction:

    $$
    p = \\frac{1 + \\sum_{b=1}^{B} \\mathbb{I}\\left(|\\hat{\\theta}^{\\ast}_b - \\hat{\\theta}| \\ge |\\hat{\\theta}|\\right)}{B + 1}
    $$

    References
    ----------
    Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303–1313.

    :param x: 1-D input series of returns or factor values ($T \\ge 10$).
    :param statistic: ``'mean'`` or ``'sharpe'`` ($\\sqrt{A} \\cdot \\bar{x} / s_x$ with ``ddof=1``).
    :param n_boot: Number of stationary bootstrap replicates $B \\ge 1$.
    :param mean_block: Expected geometric block length. Defaults to $\\max(1, T^{1/3})$.
    :param alternative: ``'two-sided'``, ``'greater'``, or ``'less'``.
    :param ci_level: Confidence level in ``(0, 1)``, default ``0.95``.
    :param annualization: Annualization factor $A > 0$ applied when ``statistic='sharpe'``.
    :param seed: Optional integer seed for deterministic reproducibility.
    :return: Dictionary with keys ``estimate``, ``std_error``, ``ci_lower``, ``ci_upper``,
        ``p_value``, ``n_obs``, and ``mean_block``.
    """
    if statistic not in ("mean", "sharpe"):
        raise ValueError("statistic must be 'mean' or 'sharpe'")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if not (0.0 < float(ci_level) < 1.0):
        raise ValueError("ci_level must be strictly between 0 and 1")
    if float(annualization) <= 0.0:
        raise ValueError("annualization must be positive")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")

    arr = _to_1d_array(x, name="x", min_len=10)
    n_obs = arr.size
    blk = _default_mean_block(n_obs, mean_block)
    idx = stationary_bootstrap_indices(n_obs, n_boot=n_boot, mean_block=blk, seed=seed)

    if statistic == "mean":
        theta_hat = float(arr.mean())
        theta_star = _bootstrap_means(arr[:, None], idx)[:, 0]
    else:
        scale = math.sqrt(float(annualization))
        std_hat = float(arr.std(ddof=1))
        theta_hat = (float(arr.mean()) / std_hat * scale) if std_hat > 1e-15 else 0.0
        moments = _bootstrap_means(np.column_stack([arr, arr * arr]), idx)
        m1 = moments[:, 0]
        m2 = moments[:, 1]
        var_star = np.maximum(0.0, (m2 - m1 * m1) * (float(n_obs) / float(n_obs - 1)))
        std_star = np.sqrt(var_star)
        theta_star = np.where(std_star > 1e-15, (m1 / std_star) * scale, 0.0)

    centered = theta_star - theta_hat
    if alternative == "two-sided":
        exceed = int(np.sum(np.abs(centered) >= abs(theta_hat)))
    elif alternative == "greater":
        exceed = int(np.sum(centered >= theta_hat))
    else:
        exceed = int(np.sum(centered <= theta_hat))

    p_value = float((1 + exceed) / (int(n_boot) + 1))
    alpha = 1.0 - float(ci_level)
    ci_low, ci_high = np.quantile(theta_star, [alpha / 2.0, 1.0 - alpha / 2.0])
    std_err = float(np.std(theta_star, ddof=1)) if int(n_boot) > 1 else 0.0

    return {
        "estimate": theta_hat,
        "std_error": std_err,
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "p_value": p_value,
        "n_obs": float(n_obs),
        "mean_block": float(blk),
    }


def adjust_pvalues(
    p_values: np.ndarray | pl.Series | Sequence[float],
    method: Literal["bonferroni", "holm", "bh", "by"] = "bh",
) -> np.ndarray:
    """Adjust a vector of p-values for multiple hypothesis testing (FWER or FDR control).

    Supports four classical procedures over $m$ non-NaN hypotheses sorted as
    $p_{(1)} \\le p_{(2)} \\le \\dots \\le p_{(m)}$:

    - ``'bonferroni'`` (FWER): $\\tilde{p}_i = \\min(1, m \\cdot p_i)$.
    - ``'holm'`` (Holm 1979 step-down FWER):
      $\\tilde{p}_{(i)} = \\min\\left(1, \\max_{j \\le i} (m - j + 1) p_{(j)}\\right)$.
    - ``'bh'`` (Benjamini & Hochberg 1995 step-up FDR):
      $\\tilde{p}_{(i)} = \\min\\left(1, \\min_{j \\ge i} \\frac{m}{j} p_{(j)}\\right)$.
    - ``'by'`` (Benjamini & Yekutieli 2001 arbitrary-dependence FDR):
      $\\tilde{p}_{(i)} = \\min\\left(1, c(m) \\min_{j \\ge i} \\frac{m}{j} p_{(j)}\\right)$ with
      $c(m) = \\sum_{k=1}^{m} \\frac{1}{k}$.

    References
    ----------
    - Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
      *Scandinavian Journal of Statistics*, 6(2), 65–70.
    - Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate.
      *JRSS-B*, 57(1), 289–300.
    - Benjamini, Y., & Yekutieli, D. (2001). The Control of the False Discovery Rate in Multiple
      Testing under Dependency. *Annals of Statistics*, 29(4), 1165–1188.

    :param p_values: 1-D sequence, Series, or array of raw p-values in ``[0, 1]`` (``NaN`` preserved).
    :param method: ``'bonferroni'``, ``'holm'``, ``'bh'``, or ``'by'``.
    :return: ``float64`` NumPy array of adjusted p-values in original input order.
    """
    if method not in ("bonferroni", "holm", "bh", "by"):
        raise ValueError("method must be 'bonferroni', 'holm', 'bh', or 'by'")

    if isinstance(p_values, pl.Series):
        raw = p_values.cast(pl.Float64).fill_null(float("nan")).to_numpy()
    else:
        raw = np.asarray(p_values, dtype=np.float64)

    if raw.ndim != 1:
        raise ValueError(f"p_values must be 1-dimensional, got {raw.ndim}D")

    out = np.full(raw.shape, np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(raw)
    valid_p = raw[valid_mask]
    m = valid_p.size
    if m == 0:
        return out
    if np.any((valid_p < 0.0) | (valid_p > 1.0)):
        raise ValueError("All non-NaN p_values must lie in [0.0, 1.0]")

    if method == "bonferroni":
        adj = np.clip(float(m) * valid_p, 0.0, 1.0)
    elif method == "holm":
        order = np.argsort(valid_p, kind="mergesort")
        sorted_p = valid_p[order]
        multipliers = float(m) - np.arange(m, dtype=np.float64)
        stepdown = np.maximum.accumulate(sorted_p * multipliers)
        adj = np.empty(m, dtype=np.float64)
        adj[order] = np.clip(stepdown, 0.0, 1.0)
    else:
        order = np.argsort(valid_p, kind="mergesort")
        sorted_p = valid_p[order]
        ranks = np.arange(1, m + 1, dtype=np.float64)
        factor = float(m) / ranks
        if method == "by":
            c_m = float(np.sum(1.0 / ranks))
            factor = factor * c_m
        stepup = np.minimum.accumulate((sorted_p * factor)[::-1])[::-1]
        adj = np.empty(m, dtype=np.float64)
        adj[order] = np.clip(stepup, 0.0, 1.0)

    out[valid_mask] = adj
    return out


def romano_wolf(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    exclude: Sequence[str] = (),
    n_boot: int = 2000,
    mean_block: float | None = None,
    alternative: Literal["two-sided", "greater"] = "two-sided",
    seed: int | None = None,
) -> pl.DataFrame:
    """Romano–Wolf (2005, 2016) stepdown max-$t$ multiple hypothesis test via stationary bootstrap.

    Controls the Family-Wise Error Rate (FWER) across $K$ jointly tested factors or strategies
    while accounting for cross-sectional correlation and serial dependence:

    $$
    t_k = \\frac{\\sqrt{T}\\,\\bar{X}_k}{\\hat{\\omega}_k}, \\qquad
    t^{\\ast}_{b,k} = \\frac{\\sqrt{T}\\left(\\bar{X}^{\\ast}_{b,k} - \\bar{X}_k\\right)}{\\hat{\\omega}_k}
    $$

    Ordering hypotheses as $t_{(1)} \\ge t_{(2)} \\ge \\dots \\ge t_{(K)}$, stepdown adjusted
    p-values are computed in a single bootstrap pass via backward cumulative maxima
    $M^{\\ast}_{b,j} = \\max_{l \\ge j} t^{\\ast}_{b,(l)}$ with monotonicity enforcement:

    $$
    \\hat{p}^{\\text{step}}_{(j)} = \\frac{1 + \\sum_{b=1}^{B} \\mathbb{I}\\left(M^{\\ast}_{b,j} \\ge t_{(j)}\\right)}{B + 1},
    \\qquad \\hat{p}^{\\text{RW}}_{(j)} = \\max_{i \\le j} \\hat{p}^{\\text{step}}_{(i)}
    $$

    References
    ----------
    - Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping.
      *Econometrica*, 73(4), 1237–1282.
    - Romano, J. P., & Wolf, M. (2016). Efficient Computation of Adjusted p-Values for
      Resampling-Based Stepdown Multiple Testing. *Statistics & Probability Letters*, 113, 38–40.

    :param data: ``(T, K)`` NumPy array, Polars DataFrame, or LazyFrame of returns/IC series.
    :param exclude: Column names to exclude when ``data`` is a DataFrame/LazyFrame (e.g., ``("date",)``).
    :param n_boot: Number of stationary bootstrap replicates $B \\ge 1$.
    :param mean_block: Expected geometric block length. Defaults to $\\max(1, T^{1/3})$.
    :param alternative: ``'two-sided'`` (uses $|t_k|$ and $|t^{\\ast}_{b,k}|$) or ``'greater'``.
    :param seed: Optional integer seed for deterministic reproducibility.
    :return: Polars DataFrame with columns ``["name", "mean", "t_stat", "p_raw", "p_rw"]`` in
        the original column order.
    """
    if alternative not in ("two-sided", "greater"):
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")

    x_mat, names = _to_matrix(data, exclude=exclude, min_rows=10)
    t_obs, k_cols = x_mat.shape
    b_reps = int(n_boot)
    sqrt_t = math.sqrt(float(t_obs))

    means = x_mat.mean(axis=0)
    idx = stationary_bootstrap_indices(t_obs, n_boot=b_reps, mean_block=mean_block, seed=seed)
    m_star = _bootstrap_means(x_mat, idx)

    if b_reps > 1:
        omega = sqrt_t * np.std(m_star, axis=0, ddof=1)
    else:
        omega = np.std(x_mat, axis=0, ddof=1)

    valid = omega > 1e-15
    omega_safe = np.where(valid, omega, 1.0)
    t_signed = np.where(valid, sqrt_t * means / omega_safe, 0.0)
    t_star_signed = np.where(
        valid[None, :],
        sqrt_t * (m_star - means[None, :]) / omega_safe[None, :],
        0.0,
    )

    if alternative == "two-sided":
        t_eval = np.abs(t_signed)
        t_star_eval = np.abs(t_star_signed)
    else:
        t_eval = t_signed.copy()
        t_star_eval = t_star_signed.copy()

    # Raw bootstrap p-values per column
    raw_exceed = np.sum(t_star_eval >= t_eval[None, :], axis=0)
    p_raw = (1.0 + raw_exceed.astype(np.float64)) / float(b_reps + 1)
    p_raw = np.where(valid, p_raw, 1.0)

    # Sort hypotheses strongest-first; place zero-variance columns last
    sort_key = np.where(valid, t_eval, -np.inf)
    order = np.argsort(-sort_key, kind="mergesort")
    s_star = t_star_eval[:, order]
    max_rest = np.maximum.accumulate(s_star[:, ::-1], axis=1)[:, ::-1]
    t_sorted = t_eval[order]
    step_exceed = np.sum(max_rest >= t_sorted[None, :], axis=0)
    p_step = (1.0 + step_exceed.astype(np.float64)) / float(b_reps + 1)
    p_rw_sorted = np.maximum.accumulate(p_step)

    p_rw = np.empty(k_cols, dtype=np.float64)
    p_rw[order] = np.clip(p_rw_sorted, 0.0, 1.0)
    p_rw = np.where(valid, p_rw, 1.0)

    return pl.DataFrame(
        {
            "name": names,
            "mean": means,
            "t_stat": t_signed,
            "p_raw": p_raw,
            "p_rw": p_rw,
        }
    )


def spa_test(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    benchmark: np.ndarray | pl.Series | Sequence[float] | str | float | None = None,
    exclude: Sequence[str] = (),
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int | None = None,
) -> dict[str, float | str]:
    """Hansen (2005) Superior Predictive Ability (SPA) & White (2000) Reality Check test.

    Tests the composite null hypothesis $H_0: \\max_{k=1,\\dots,K} \\mathbb{E}[d_{k,t}] \\le 0$ that no
    candidate strategy outperforms the benchmark, using studentized test statistic:

    $$
    T_{\\text{SPA}} = \\max\\left(0, \\max_{1 \\le k \\le K} \\frac{\\sqrt{T}\\,\\bar{d}_k}{\\hat{\\omega}_k}\\right)
    $$

    and three null recentering estimators $g_k^{(i)}$ ($i \\in \\{\\text{lower}, \\text{consistent}, \\text{upper}\\}$):

    $$
    g_k^{(l)} = \\max(\\bar{d}_k, 0), \\qquad
    g_k^{(c)} = \\bar{d}_k \\cdot \\mathbb{I}\\left(\\frac{\\sqrt{T}\\,\\bar{d}_k}{\\hat{\\omega}_k} \\ge -\\sqrt{2 \\ln \\ln T}\\right), \\qquad
    g_k^{(u)} = \\bar{d}_k
    $$

    where ``p_upper`` corresponds to the studentized White (2000) Reality Check and satisfies
    $p_{\\text{lower}} \\le p_{\\text{consistent}} \\le p_{\\text{upper}}$.

    References
    ----------
    - Hansen, P. R. (2005). A Test for Superior Predictive Ability.
      *Journal of Business & Economic Statistics*, 23(4), 365–380.
    - White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097–1126.

    :param data: ``(T, K)`` NumPy array, Polars DataFrame, or LazyFrame of strategy returns.
    :param benchmark: Benchmark return series: ``None`` (zero benchmark), scalar ``float``,
        1-D array/Series of length $T$, or column name ``str`` in ``data``.
    :param exclude: Column names to exclude when ``data`` is a DataFrame/LazyFrame.
    :param n_boot: Number of stationary bootstrap replicates $B \\ge 1$.
    :param mean_block: Expected geometric block length. Defaults to $\\max(1, T^{1/3})$.
    :param seed: Optional integer seed for deterministic reproducibility.
    :return: Dictionary with keys ``statistic``, ``p_lower``, ``p_consistent``, ``p_upper``,
        ``best_name``, ``n_models``, and ``n_obs``.
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")

    excl_list = list(exclude)
    bench_vec: np.ndarray | float
    if isinstance(benchmark, str):
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        if not isinstance(data, pl.DataFrame) or benchmark not in data.columns:
            raise ValueError(f"Benchmark column '{benchmark}' not found in DataFrame")
        bench_vec = _to_1d_array(data[benchmark], name="benchmark", min_len=10)
        if benchmark not in excl_list:
            excl_list.append(benchmark)
    elif benchmark is None:
        bench_vec = 0.0
    elif isinstance(benchmark, (int, float)):
        bench_vec = float(benchmark)
    else:
        bench_vec = _to_1d_array(benchmark, name="benchmark", min_len=10)

    x_mat, names = _to_matrix(data, exclude=excl_list, min_rows=10)
    t_obs, k_models = x_mat.shape
    if isinstance(bench_vec, np.ndarray):
        if bench_vec.size != t_obs:
            raise ValueError(f"benchmark length ({bench_vec.size}) must match data rows ({t_obs})")
        d_mat = x_mat - bench_vec[:, None]
    else:
        d_mat = x_mat - float(bench_vec)

    b_reps = int(n_boot)
    sqrt_t = math.sqrt(float(t_obs))
    d_bar = d_mat.mean(axis=0)
    idx = stationary_bootstrap_indices(t_obs, n_boot=b_reps, mean_block=mean_block, seed=seed)
    d_star = _bootstrap_means(d_mat, idx)

    if b_reps > 1:
        omega = sqrt_t * np.std(d_star, axis=0, ddof=1)
    else:
        omega = np.std(d_mat, axis=0, ddof=1)

    valid = omega > 1e-15
    omega_safe = np.where(valid, omega, 1.0)
    t_vec = np.where(valid, sqrt_t * d_bar / omega_safe, 0.0)
    t_obs_stat = float(max(0.0, float(np.max(t_vec))))
    best_idx = int(np.argmax(t_vec))

    thresh = -math.sqrt(max(0.0, 2.0 * math.log(max(1.0000001, math.log(float(t_obs))))))
    g_lower = np.maximum(d_bar, 0.0)
    g_consistent = np.where(t_vec >= thresh, d_bar, 0.0)
    g_upper = d_bar

    p_vals: list[float] = []
    for g_vec in (g_lower, g_consistent, g_upper):
        z_star = np.where(
            valid[None, :],
            sqrt_t * (d_star - g_vec[None, :]) / omega_safe[None, :],
            0.0,
        )
        t_boot = np.maximum(0.0, np.max(z_star, axis=1))
        exceed = int(np.sum(t_boot >= t_obs_stat))
        p_vals.append(float((1 + exceed) / (b_reps + 1)))

    return {
        "statistic": t_obs_stat,
        "p_lower": p_vals[0],
        "p_consistent": p_vals[1],
        "p_upper": p_vals[2],
        "best_name": names[best_idx],
        "n_models": float(k_models),
        "n_obs": float(t_obs),
    }


def _sharpe_from_moments(sum_x: np.ndarray, sum_x2: np.ndarray, n_obs: float) -> np.ndarray:
    """Compute sample Sharpe ratios ``mean / std(ddof=1)`` from sufficient statistics."""
    mean = sum_x / n_obs
    var = np.maximum(0.0, (sum_x2 - (sum_x * sum_x) / n_obs) / max(1.0, n_obs - 1.0))
    std = np.sqrt(var)
    return np.where(std > 1e-15, mean / std, 0.0)


def pbo_cscv(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    exclude: Sequence[str] = (),
    n_splits: int = 16,
    max_combinations: int = 20_000,
    seed: int | None = None,
) -> dict[str, object]:
    """Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation.

    Implements Bailey, Borwein, López de Prado & Zhu (2017). Partitions the $(T, N)$ return matrix
    into $S$ equal-length contiguous sub-matrices ($S$ even) and evaluates $\\binom{S}{S/2}$
    in-sample / out-of-sample split combinations via block sufficient statistics:

    $$
    n^{\\ast}_c = \\arg\\max_{1 \\le n \\le N} \\widehat{\\text{SR}}^{\\text{IS}}_{c,n}, \\qquad
    \\bar{\\omega}_c = \\frac{\\text{rank}_{\\text{OOS}}(n^{\\ast}_c)}{N + 1}, \\qquad
    \\lambda_c = \\ln\\frac{\\bar{\\omega}_c}{1 - \\bar{\\omega}_c}
    $$

    $$
    \\text{PBO} = \\frac{1}{|\\mathcal{C}|} \\sum_{c \\in \\mathcal{C}} \\mathbb{I}(\\lambda_c \\le 0)
    $$

    References
    ----------
    Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The Probability of
    Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39–69.

    :param data: ``(T, N)`` NumPy array, Polars DataFrame, or LazyFrame of strategy returns ($N \\ge 2$).
    :param exclude: Column names to exclude when ``data`` is a DataFrame/LazyFrame.
    :param n_splits: Even number of contiguous time blocks $S \\ge 4$ (default ``16``).
    :param max_combinations: Maximum number of $\\binom{S}{S/2}$ combinations evaluated before
        deterministic random subsampling via ``seed``.
    :param seed: Optional integer seed for combination subsampling.
    :return: Dictionary with keys ``pbo``, ``logits``, ``is_sharpe_best``, ``oos_sharpe_best``,
        ``degradation_slope``, ``prob_oos_loss``, ``n_combinations``, and ``n_strategies``.
    """
    s_splits = int(n_splits)
    if s_splits < 4 or s_splits % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 4")
    max_comb = int(max_combinations)
    if max_comb < 1:
        raise ValueError("max_combinations must be at least 1")

    x_mat, _ = _to_matrix(data, exclude=exclude, min_rows=max(10, 2 * s_splits))
    t_raw, n_strat = x_mat.shape
    if n_strat < 2:
        raise ValueError("pbo_cscv requires at least 2 strategies (columns)")

    rem = t_raw % s_splits
    if rem > 0:
        x_mat = x_mat[rem:]
    t_used = x_mat.shape[0]
    block_len = t_used // s_splits
    if block_len < 2:
        raise ValueError("Each CSCV split block must contain at least 2 observations")

    blocks = x_mat.reshape(s_splits, block_len, n_strat)
    sum_s = blocks.sum(axis=1)
    sumsq_s = (blocks * blocks).sum(axis=1)

    half = s_splits // 2
    total_comb = math.comb(s_splits, half)
    if total_comb <= max_comb:
        combos = list(itertools.combinations(range(s_splits), half))
    else:
        rng = np.random.default_rng(seed)
        if total_comb <= 250_000:
            all_combos = list(itertools.combinations(range(s_splits), half))
            chosen = rng.choice(total_comb, size=max_comb, replace=False)
            combos = [all_combos[int(i)] for i in chosen]
        else:
            seen: set[tuple[int, ...]] = set()
            combos = []
            while len(combos) < max_comb:
                comb = tuple(sorted(int(i) for i in rng.choice(s_splits, size=half, replace=False)))
                if comb not in seen:
                    seen.add(comb)
                    combos.append(comb)

    n_comb = len(combos)
    c_mat = np.zeros((n_comb, s_splits), dtype=np.float64)
    for row_i, comb in enumerate(combos):
        c_mat[row_i, list(comb)] = 1.0

    n_half = float(half * block_len)
    is_sum = c_mat @ sum_s
    is_sq = c_mat @ sumsq_s
    oos_sum = sum_s.sum(axis=0)[None, :] - is_sum
    oos_sq = sumsq_s.sum(axis=0)[None, :] - is_sq

    is_sharpe = _sharpe_from_moments(is_sum, is_sq, n_half)
    oos_sharpe = _sharpe_from_moments(oos_sum, oos_sq, n_half)

    n_star = np.argmax(is_sharpe, axis=1)
    rows = np.arange(n_comb)
    is_best = is_sharpe[rows, n_star]
    oos_best = oos_sharpe[rows, n_star]

    less = np.sum(oos_sharpe < oos_best[:, None], axis=1)
    equal = np.sum(oos_sharpe == oos_best[:, None], axis=1)
    rank_oos = 1.0 + less.astype(np.float64) + 0.5 * (equal.astype(np.float64) - 1.0)
    omega_bar = np.clip(rank_oos / float(n_strat + 1), 1e-12, 1.0 - 1e-12)
    logits = np.log(omega_bar / (1.0 - omega_bar))

    pbo_val = float(np.mean(logits <= 0.0))
    prob_loss = float(np.mean(oos_best < 0.0))
    var_is = float(np.var(is_best))
    deg_slope = float(np.cov(is_best, oos_best, ddof=0)[0, 1] / var_is) if var_is > 1e-15 else 0.0

    return {
        "pbo": pbo_val,
        "logits": logits,
        "is_sharpe_best": is_best,
        "oos_sharpe_best": oos_best,
        "degradation_slope": deg_slope,
        "prob_oos_loss": prob_loss,
        "n_combinations": int(n_comb),
        "n_strategies": int(n_strat),
    }


def empirical_bayes_shrinkage(
    estimates: np.ndarray | pl.Series | Sequence[float],
    std_errors: np.ndarray | pl.Series | Sequence[float],
    names: Sequence[str] | None = None,
    prior: Literal["zero", "grand_mean"] = "zero",
    groups: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Empirical-Bayes normal–normal shrinkage of factor estimates (James–Stein / Efron–Morris).

    Models observed estimates $\\hat{\\theta}_k \\mid \\theta_k \\sim \\mathcal{N}(\\theta_k, s_k^2)$ with
    hierarchical prior $\\theta_k \\sim \\mathcal{N}(\\mu_0, \\tau^2)$ (within each group when ``groups``
    is provided):

    $$
    \\hat{\\tau}^2 = \\begin{cases}
    \\max\\!\\left(0,\\; \\frac{1}{K}\\sum_{k=1}^{K}(\\hat{\\theta}_k - \\mu_0)^2 - \\frac{1}{K}\\sum_{k=1}^{K} s_k^2\\right) & \\text{if prior = 'zero'} \\\\[6pt]
    \\max\\!\\left(0,\\; \\text{Var}_{\\text{ddof}=1}(\\hat{\\theta}) - \\frac{1}{K}\\sum_{k=1}^{K} s_k^2\\right) & \\text{if prior = 'grand\\_mean'}
    \\end{cases}
    $$

    $$
    B_k = \\frac{s_k^2}{s_k^2 + \\hat{\\tau}^2}, \\qquad
    \\hat{\\theta}^{\\text{EB}}_k = \\mu_0 + (1 - B_k)(\\hat{\\theta}_k - \\mu_0), \\qquad
    \\sigma^{\\text{post}}_k = \\sqrt{\\frac{\\hat{\\tau}^2 s_k^2}{\\hat{\\tau}^2 + s_k^2}}
    $$

    When $\\hat{\\tau}^2 = 0$, ``shrinkage = 1.0``, ``posterior_mean = prior_mean``,
    ``posterior_sd = 0.0``, and ``prob_positive = 0.5``.

    References
    ----------
    - Efron, B., & Morris, C. (1975). Data Analysis Using Stein's Estimator and Its Generalizations.
      *JASA*, 70(350), 311–319.
    - Jensen, T. I., Kelly, B., & Pedersen, L. H. (2023). Is There a Replication Crisis in Finance?
      *Journal of Finance*, 78(5), 2465–2518.

    :param estimates: 1-D sequence, Series, or array of point estimates $\\hat{\\theta}_k$.
    :param std_errors: 1-D sequence, Series, or array of standard errors $s_k > 0$.
    :param names: Optional sequence of factor/strategy names.
    :param prior: ``'zero'`` ($\\mu_0 = 0$) or ``'grand_mean'`` ($\\mu_0 = \\bar{\\theta}$).
    :param groups: Optional sequence of cluster/family labels for group-specific priors.
    :return: Polars DataFrame with columns ``["name", "group", "estimate", "std_error",
        "prior_mean", "tau2", "shrinkage", "posterior_mean", "posterior_sd", "prob_positive"]``.
    """
    if prior not in ("zero", "grand_mean"):
        raise ValueError("prior must be 'zero' or 'grand_mean'")

    est = _to_1d_array(estimates, name="estimates", min_len=1)
    se = _to_1d_array(std_errors, name="std_errors", min_len=1)
    k = est.size
    if se.size != k:
        raise ValueError(f"estimates ({k}) and std_errors ({se.size}) must have the same length")
    if np.any(se <= 0.0):
        raise ValueError("std_errors must be strictly positive")

    if names is None:
        name_list = [f"factor_{i}" for i in range(k)]
    else:
        name_list = [str(n) for n in names]
        if len(name_list) != k:
            raise ValueError(f"names length ({len(name_list)}) must match estimates ({k})")

    if groups is None:
        group_list = ["all"] * k
    else:
        group_list = [str(g) for g in groups]
        if len(group_list) != k:
            raise ValueError(f"groups length ({len(group_list)}) must match estimates ({k})")

    prior_mean_arr = np.empty(k, dtype=np.float64)
    tau2_arr = np.empty(k, dtype=np.float64)
    shrink_arr = np.empty(k, dtype=np.float64)
    post_mean_arr = np.empty(k, dtype=np.float64)
    post_sd_arr = np.empty(k, dtype=np.float64)
    prob_pos_arr = np.empty(k, dtype=np.float64)

    unique_groups = list(dict.fromkeys(group_list))
    group_np = np.array(group_list, dtype=object)

    for grp in unique_groups:
        mask = group_np == grp
        est_g = est[mask]
        se_g = se[mask]
        se2_g = se_g * se_g

        if prior == "zero":
            mu0 = 0.0
            tau2 = max(0.0, float(np.mean(est_g * est_g) - np.mean(se2_g)))
        else:
            mu0 = float(np.mean(est_g))
            var_est = float(np.var(est_g, ddof=1)) if est_g.size > 1 else 0.0
            tau2 = max(0.0, var_est - float(np.mean(se2_g)))

        prior_mean_arr[mask] = mu0
        tau2_arr[mask] = tau2

        if tau2 <= 0.0:
            shrink_arr[mask] = 1.0
            post_mean_arr[mask] = mu0
            post_sd_arr[mask] = 0.0
            prob_pos_arr[mask] = 0.5
        else:
            b_k = se2_g / (se2_g + tau2)
            post_m = mu0 + (1.0 - b_k) * (est_g - mu0)
            post_s = np.sqrt((tau2 * se2_g) / (tau2 + se2_g))
            prob_p = np.array(
                [
                    _normal_cdf(float(m / s)) if s > 0.0 else 0.5
                    for m, s in zip(post_m, post_s, strict=True)
                ],
                dtype=np.float64,
            )
            shrink_arr[mask] = b_k
            post_mean_arr[mask] = post_m
            post_sd_arr[mask] = post_s
            prob_pos_arr[mask] = prob_p

    return pl.DataFrame(
        {
            "name": name_list,
            "group": group_list,
            "estimate": est,
            "std_error": se,
            "prior_mean": prior_mean_arr,
            "tau2": tau2_arr,
            "shrinkage": shrink_arr,
            "posterior_mean": post_mean_arr,
            "posterior_sd": post_sd_arr,
            "prob_positive": prob_pos_arr,
        }
    )


def oos_r2(
    y_true: np.ndarray | pl.Series | Sequence[float],
    y_pred: np.ndarray | pl.Series | Sequence[float],
    min_train: int = 20,
) -> dict[str, float]:
    """Campbell & Thompson (2008) out-of-sample $R^2_{\\text{OOS}}$ against the expanding historical mean.

    Evaluates forecasts $\\hat{y}_t$ over $t = m, \\dots, T-1$ ($m = \\text{min\\_train}$) against the
    strictly causal expanding sample mean $\\bar{y}_{t-1} = \\frac{1}{t}\\sum_{i=0}^{t-1} y_i$:

    $$
    R^2_{\\text{OOS}} = 1 - \\frac{\\sum_{t=m}^{T-1} (y_t - \\hat{y}_t)^2}{\\sum_{t=m}^{T-1} (y_t - \\bar{y}_{t-1})^2}
    $$

    References
    ----------
    Campbell, J. Y., & Thompson, S. B. (2008). Predicting Excess Stock Returns Out of Sample.
    *Review of Financial Studies*, 21(4), 1509–1531.

    :param y_true: 1-D series of realized target returns $y_t$.
    :param y_pred: 1-D series of out-of-sample return forecasts $\\hat{y}_t$.
    :param min_train: Initial warm-up observations $m \\ge 1$ used for the first historical mean.
    :return: Dictionary with keys ``r2_oos`` and ``n_obs``.
    """
    m = int(min_train)
    if m < 1:
        raise ValueError("min_train must be at least 1")
    y = _to_1d_array(y_true, name="y_true", min_len=max(10, m + 1))
    yp = _to_1d_array(y_pred, name="y_pred", min_len=max(10, m + 1))
    if y.size != yp.size:
        raise ValueError(f"y_true ({y.size}) and y_pred ({yp.size}) must have the same length")

    cum_sum = np.cumsum(y[:-1])
    counts = np.arange(1, y.size, dtype=np.float64)
    hist_mean = (cum_sum / counts)[m - 1 :]

    y_eval = y[m:]
    yp_eval = yp[m:]
    sse_model = float(np.sum((y_eval - yp_eval) ** 2))
    sse_bench = float(np.sum((y_eval - hist_mean) ** 2))
    r2 = (1.0 - sse_model / sse_bench) if sse_bench > 1e-24 else 0.0
    return {"r2_oos": r2, "n_obs": float(y_eval.size)}


def clark_west_test(
    y_true: np.ndarray | pl.Series | Sequence[float],
    pred_restricted: np.ndarray | pl.Series | Sequence[float],
    pred_unrestricted: np.ndarray | pl.Series | Sequence[float],
    nw_lags: int | None = None,
) -> dict[str, float]:
    """Clark & West (2007) MSPE-adjusted test for equal predictive accuracy in nested models.

    Corrects the downward bias in out-of-sample MSPE of the unrestricted model under the null by
    testing $H_0: \\mathbb{E}[f_t] \\le 0$ vs. $H_1: \\mathbb{E}[f_t] > 0$ (one-sided upper tail):

    $$
    f_t = (y_t - \\hat{y}_{r,t})^2 - \\left[(y_t - \\hat{y}_{u,t})^2 - (\\hat{y}_{r,t} - \\hat{y}_{u,t})^2\\right],
    \\qquad t_{\\text{CW}} = \\frac{\\bar{f}}{\\sqrt{\\hat{\\Omega}_{\\text{NW}}(f) / T}}
    $$

    References
    ----------
    Clark, T. E., & West, K. D. (2007). Approximately Normal Tests for Equal Predictive Accuracy in
    Nested Models. *Journal of Econometrics*, 138(1), 291–311.

    :param y_true: 1-D series of realized target values $y_t$ ($T \\ge 10$).
    :param pred_restricted: 1-D series of restricted benchmark forecasts $\\hat{y}_{r,t}$.
    :param pred_unrestricted: 1-D series of unrestricted model forecasts $\\hat{y}_{u,t}$.
    :param nw_lags: Optional Newey–West Bartlett lag $L \\ge 0$ (defaults to plug-in rule).
    :return: Dictionary with keys ``statistic`` and ``p_value`` (one-sided $1 - \\Phi(t_{\\text{CW}})$).
    """
    if nw_lags is not None and int(nw_lags) < 0:
        raise ValueError("nw_lags must be a non-negative integer")
    y = _to_1d_array(y_true, name="y_true", min_len=10)
    pr = _to_1d_array(pred_restricted, name="pred_restricted", min_len=10)
    pu = _to_1d_array(pred_unrestricted, name="pred_unrestricted", min_len=10)
    if not (y.size == pr.size == pu.size):
        raise ValueError("y_true, pred_restricted, and pred_unrestricted must have the same length")

    n_obs = y.size
    f_t = (y - pr) ** 2 - ((y - pu) ** 2 - (pr - pu) ** 2)
    f_bar = float(f_t.mean())
    if float(np.std(f_t)) <= 1e-15:
        return {"statistic": 0.0, "p_value": 0.5}

    u = f_t - f_bar
    lags = _nw_auto_lag(n_obs) if nw_lags is None else int(nw_lags)
    lrv = float(_newey_west_lrv(u, lags))
    t_stat = f_bar / math.sqrt(lrv / float(n_obs))
    p_val = float(max(0.0, min(1.0, 1.0 - _normal_cdf(t_stat))))
    return {"statistic": t_stat, "p_value": p_val}


def diebold_mariano_test(
    y_true: np.ndarray | pl.Series | Sequence[float],
    pred_1: np.ndarray | pl.Series | Sequence[float],
    pred_2: np.ndarray | pl.Series | Sequence[float],
    loss: Literal["squared", "absolute"] = "squared",
    horizon: int = 1,
) -> dict[str, float]:
    """Diebold & Mariano (1995) test for equal predictive accuracy between two forecasts.

    Tests $H_0: \\mathbb{E}[d_t] = 0$ where $d_t = L(e_{1,t}) - L(e_{2,t})$ ($d_t < 0$ indicates
    ``pred_1`` has lower loss than ``pred_2``) with Newey–West lag $L = h - 1$:

    $$
    \\text{DM} = \\frac{\\bar{d}}{\\sqrt{\\hat{\\Omega}_{\\text{NW}, h-1}(d) / T}}, \\qquad
    p = 2\\left(1 - \\Phi(|\\text{DM}|)\\right)
    $$

    References
    ----------
    Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy.
    *Journal of Business & Economic Statistics*, 13(3), 253–263.

    :param y_true: 1-D series of realized values $y_t$ ($T \\ge 10$).
    :param pred_1: 1-D series of first model forecasts $\\hat{y}_{1,t}$.
    :param pred_2: 1-D series of second model forecasts $\\hat{y}_{2,t}$.
    :param loss: ``'squared'`` ($e^2$) or ``'absolute'`` ($|e|$).
    :param horizon: Forecast horizon $h \\ge 1$ (sets Newey–West lag to $h - 1$).
    :return: Dictionary with keys ``statistic`` and ``p_value`` (two-sided).
    """
    if loss not in ("squared", "absolute"):
        raise ValueError("loss must be 'squared' or 'absolute'")
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be at least 1")

    y = _to_1d_array(y_true, name="y_true", min_len=10)
    p1 = _to_1d_array(pred_1, name="pred_1", min_len=10)
    p2 = _to_1d_array(pred_2, name="pred_2", min_len=10)
    if not (y.size == p1.size == p2.size):
        raise ValueError("y_true, pred_1, and pred_2 must have the same length")

    e1 = y - p1
    e2 = y - p2
    d_t = (e1 * e1 - e2 * e2) if loss == "squared" else (np.abs(e1) - np.abs(e2))
    d_bar = float(d_t.mean())
    if float(np.std(d_t)) <= 1e-15:
        return {"statistic": 0.0, "p_value": 1.0}

    n_obs = y.size
    u = d_t - d_bar
    lrv = float(_newey_west_lrv(u, max(0, h - 1)))
    stat = d_bar / math.sqrt(lrv / float(n_obs))
    return {"statistic": stat, "p_value": _two_sided_normal_p(stat)}


def cusum_mean_test(
    x: np.ndarray | pl.Series | Sequence[float],
    nw_lags: int | None = None,
) -> dict[str, float]:
    """Ploberger & Krämer (1992) OLS-CUSUM structural break test for stability of the mean.

    Detects regime shifts or alpha decay in a return or IC time series via the scaled partial sums
    of demeaned residuals $e_t = x_t - \\bar{x}$:

    $$
    S_k = \\frac{1}{\\hat{\\sigma}_{\\text{NW}}\\sqrt{T}} \\sum_{t=1}^{k} (x_t - \\bar{x}), \\qquad
    Q_{\\text{CUSUM}} = \\max_{1 \\le k \\le T} |S_k|
    $$

    Asymptotic p-values are evaluated from the Kolmogorov distribution
    $P(\\sup_{0 \\le u \\le 1} |B^0(u)| \\ge x) = 2 \\sum_{j=1}^{\\infty} (-1)^{j-1} e^{-2 j^2 x^2}$.

    References
    ----------
    Ploberger, W., & Krämer, W. (1992). The CUSUM Test with OLS Residuals.
    *Econometrica*, 60(2), 271–285.

    :param x: 1-D input series ($T \\ge 10$).
    :param nw_lags: Optional Newey–West Bartlett lag $L \\ge 0$ (defaults to plug-in rule).
    :return: Dictionary with keys ``statistic``, ``p_value``, ``break_index``, ``mean_before``,
        ``mean_after``, and ``n_obs``.
    """
    if nw_lags is not None and int(nw_lags) < 0:
        raise ValueError("nw_lags must be a non-negative integer")
    arr = _to_1d_array(x, name="x", min_len=10)
    n_obs = arr.size
    e = arr - float(arr.mean())
    lags = _nw_auto_lag(n_obs) if nw_lags is None else int(nw_lags)
    sigma_hat = math.sqrt(float(_newey_west_lrv(e, lags)))

    if float(np.std(arr)) <= 1e-15 or sigma_hat <= 1e-12:
        brk = 0
        stat = 0.0
        p_val = 1.0
    else:
        s_k = np.abs(np.cumsum(e)) / (sigma_hat * math.sqrt(float(n_obs)))
        brk = int(np.argmax(s_k))
        stat = float(s_k[brk])
        if stat < 0.3:
            p_val = 1.0
        else:
            j = np.arange(1, 101, dtype=np.float64)
            signs = np.where((j.astype(np.int64) % 2) == 1, 1.0, -1.0)
            p_raw = 2.0 * float(np.sum(signs * np.exp(-2.0 * (j * j) * (stat * stat))))
            p_val = float(max(0.0, min(1.0, p_raw)))

    split = max(1, min(n_obs - 1, brk + 1))
    return {
        "statistic": stat,
        "p_value": p_val,
        "break_index": float(brk),
        "mean_before": float(arr[:split].mean()),
        "mean_after": float(arr[split:].mean()),
        "n_obs": float(n_obs),
    }


def breakeven_cost(
    gross_returns: np.ndarray | pl.Series | Sequence[float],
    turnover: np.ndarray | pl.Series | Sequence[float],
) -> float:
    """Compute the linear one-way transaction cost rate $c^{\\ast}$ at which net return equals zero.

    Matches the proportional cost model $r^{\\text{net}}_t = r^{\\text{gross}}_t - c \\cdot \\text{turnover}_t$
    of :func:`polars_anta.backtest.vectorized_backtest`:

    $$
    c^{\\ast} = \\frac{\\sum_{t=1}^{T} r^{\\text{gross}}_t}{\\sum_{t=1}^{T} \\text{turnover}_t}
    $$

    :param gross_returns: 1-D series of gross strategy returns.
    :param turnover: 1-D series of non-negative period turnover $|\\Delta w_t|$.
    :return: Breakeven proportional cost per unit of turnover (``inf`` if $\\sum \\text{turnover} = 0$
        and $\\sum r^{\\text{gross}} > 0$, else ``0.0`` when $\\sum \\text{turnover} = 0$).
    """
    g = _to_1d_array(gross_returns, name="gross_returns", min_len=1)
    to = _to_1d_array(turnover, name="turnover", min_len=1)
    if g.size != to.size:
        raise ValueError(f"gross_returns ({g.size}) and turnover ({to.size}) must match in length")
    if np.any(to < 0.0):
        raise ValueError("turnover must be non-negative")

    sum_gross = float(np.sum(g))
    sum_to = float(np.sum(to))
    if sum_to <= 1e-15:
        return float("inf") if sum_gross > 0.0 else 0.0
    return sum_gross / sum_to


def breakeven_cost_from_backtest(result: BacktestResult) -> float:
    """Compute breakeven one-way transaction cost rate directly from a :class:`BacktestResult`."""
    if not isinstance(result, BacktestResult):
        raise ValueError("result must be a BacktestResult instance")
    return breakeven_cost(result.data["gross_return"], result.data["turnover"])


def _normalize_group_by(group_by: str | Sequence[str]) -> list[str]:
    """Normalize ``group_by`` into a non-empty list of column names."""
    if isinstance(group_by, str):
        cols = [group_by]
    else:
        cols = [str(c) for c in group_by]
    if not cols or any(not c for c in cols):
        raise ValueError("group_by must specify at least one column name")
    return cols


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
    f_cols = list(dict.fromkeys(str(c) for c in factor_cols))
    if not f_cols:
        raise ValueError("factor_cols must not be empty")
    gb_cols = _normalize_group_by(group_by)

    needed = [*gb_cols, return_col, *f_cols]
    base_df = df.select(needed).collect() if isinstance(df, pl.LazyFrame) else df.select(needed)

    wide_df: pl.DataFrame | None = None
    for f_col in f_cols:
        ic_df = (
            information_coefficient(
                base_df,
                factor_col=f_col,
                return_col=return_col,
                group_by=gb_cols,
                method=method,
            )
            .filter((pl.col("n_obs") > 0) & pl.col("ic").is_not_null() & (~pl.col("ic").is_nan()))
            .select([*gb_cols, pl.col("ic").alias(f_col)])
        )
        if wide_df is None:
            wide_df = ic_df
        else:
            wide_df = wide_df.join(ic_df, on=gb_cols, how="full", coalesce=True)

    assert wide_df is not None
    return wide_df.sort(gb_cols)


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
    f_cols = list(dict.fromkeys(str(c) for c in factor_cols))
    if not f_cols:
        raise ValueError("factor_cols must not be empty")
    gb_cols = _normalize_group_by(group_by)

    needed = [*gb_cols, return_col, *f_cols]
    base_df = df.select(needed).collect() if isinstance(df, pl.LazyFrame) else df.select(needed)
    r_expr = pl.col(return_col).cast(pl.Float64)

    wide_df: pl.DataFrame | None = None
    for f_name in f_cols:
        f_expr = pl.col(f_name).cast(pl.Float64)
        valid = (
            f_expr.is_not_null() & r_expr.is_not_null() & (~f_expr.is_nan()) & (~r_expr.is_nan())
        )
        pct_rank = (f_expr.rank(method="ordinal") - 1.0).over(gb_cols) / pl.len().over(gb_cols)
        q_expr = (
            (pct_rank * float(q_num)).floor().clip(0.0, float(q_num - 1)).cast(pl.Int32) + 1
        ).alias("_anta_q")

        top_mean = r_expr.filter(pl.col("_anta_q") == q_num).mean()
        if mode == "long_short":
            bot_mean = r_expr.filter(pl.col("_anta_q") == 1).mean()
            spread_expr = (top_mean - bot_mean).alias(f_name)
        else:
            spread_expr = (top_mean - r_expr.mean()).alias(f_name)

        sub_df = (
            base_df.filter(valid)
            .with_columns(q_expr)
            .group_by(gb_cols)
            .agg(spread_expr)
            .filter(pl.col(f_name).is_not_null() & (~pl.col(f_name).is_nan()))
        )
        if wide_df is None:
            wide_df = sub_df
        else:
            wide_df = wide_df.join(sub_df, on=gb_cols, how="full", coalesce=True)

    assert wide_df is not None
    return wide_df.sort(gb_cols)


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
