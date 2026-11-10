"""Statistical inference and factor falsification engine for quantitative research.

Looping convention (per AGENTS.md Invariant #2):
- Python loops over bars/timestamps are strictly forbidden; all time-axis reductions use
  vectorized Polars expressions, NumPy array operations, or BLAS matrix multiplications.
- Python loops over bootstrap replicate chunks, candidate factors/strategies, Newey–West
  lags, or CSCV combinations are permitted where inner time-series operations are vectorized.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Literal

import numpy as np
import polars as pl

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
