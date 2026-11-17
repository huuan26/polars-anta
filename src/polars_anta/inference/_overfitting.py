"""Overfitting diagnostics, forecast evaluation, shrinkage, and breakeven cost."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from ..backtest import BacktestResult
from ..evaluation import _newey_west_lrv, _nw_auto_lag
from ._validation import _normal_cdf, _to_1d_array, _to_matrix, _two_sided_normal_p


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
