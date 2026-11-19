"""Hierarchical Risk Parity (HRP) and Equal Risk Parity (ERC / IVP) allocation.

Implements two non-inversion / risk-budgeting portfolio allocators in pure NumPy + Polars:

1. **Hierarchical Risk Parity (HRP)** (Lopez de Prado, 2016):
   - Tree clustering on correlation distance ``d_ij = sqrt((1 - rho_ij) / 2)`` via
     vectorized Lance-Williams linkage (Ward or single).
   - Quasi-diagonalization via dendrogram leaf ordering.
   - Recursive bisection weighted inversely by inverse-variance cluster variance.
2. **Equal Risk Parity (ERC)** (Maillard, Roncalli & Teiletche, 2010; Spinu, 2013):
   - Solves exact equal risk contribution ``w_i (Sigma w)_i = (1 / N) w^T Sigma w``
     via strictly convex Cyclical Coordinate Descent (``method="erc"``), or falls back
     to diagonal inverse-volatility weighting (``method="ivp"``).

Requires zero SciPy or C dependencies and composes directly with ``polars_anta.covariance``
estimators (Ledoit-Wolf, Marchenko-Pastur RMT denoising, exponential, and sample).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from .covariance import (
    exponential_covariance,
    ledoit_wolf_shrinkage,
    marchenko_pastur_denoise,
    nearest_spd,
    sample_covariance,
)

__all__ = ["hierarchical_risk_parity", "equal_risk_parity"]

_Shrinkage = Literal["ledoit_wolf", "marchenko_pastur", "exponential", "sample"]
_Linkage = Literal["ward", "single"]
_ErpMethod = Literal["erc", "ivp"]


def _as_return_matrix(
    returns: pl.DataFrame | pl.LazyFrame | np.ndarray,
    columns: Sequence[str] | None,
    time_col: str | None,
    min_rows: int,
) -> tuple[np.ndarray, list[str]]:
    """Return a complete-observation return matrix and the matching asset names."""
    if isinstance(returns, (pl.DataFrame, pl.LazyFrame)):
        df = returns.collect() if isinstance(returns, pl.LazyFrame) else returns
        if columns is None:
            names = [c for c in df.columns if c != time_col and df.schema[c].is_numeric()]
        else:
            names = list(columns)
            missing = [c for c in names if c not in df.columns]
            if missing:
                raise ValueError(f"columns not found in frame: {missing}")
        x = df.select([pl.col(c).cast(pl.Float64) for c in names]).to_numpy()
    else:
        x = np.asarray(returns, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("returns must be a 2-D array (observations x assets)")
        names = [f"asset_{i}" for i in range(x.shape[1])]

    if x.shape[1] < 2:
        raise ValueError("need at least two return columns to allocate")
    x = np.where(np.isfinite(x), x, np.nan)
    x = x[~np.isnan(x).any(axis=1)]
    if x.shape[0] < min_rows:
        raise ValueError(f"need at least {min_rows} complete observations, got {x.shape[0]}")
    if np.any(np.var(x, axis=0) <= 0.0):
        raise ValueError("constant return series cannot be allocated")
    return x, names


def _covariance_matrix(x: np.ndarray, shrinkage: _Shrinkage | None) -> np.ndarray:
    """Estimate covariance via ``polars_anta.covariance`` estimators."""
    if shrinkage == "ledoit_wolf":
        cov, _ = ledoit_wolf_shrinkage(x)
        return np.asarray(cov, dtype=np.float64)
    if shrinkage == "marchenko_pastur":
        return np.asarray(marchenko_pastur_denoise(x), dtype=np.float64)
    if shrinkage == "exponential":
        return np.asarray(exponential_covariance(x), dtype=np.float64)
    if shrinkage in (None, "sample"):
        return np.asarray(sample_covariance(x), dtype=np.float64)
    raise ValueError(
        "shrinkage must be 'ledoit_wolf', 'marchenko_pastur', 'exponential', 'sample', or None"
    )


def _correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix with unit diagonal."""
    std = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _linkage_leaf_order(corr: np.ndarray, method: _Linkage) -> list[int]:
    """Vectorized Lance-Williams agglomeration returning dendrogram leaf order."""
    k = corr.shape[0]
    total_nodes = 2 * k - 1
    dist = np.full((total_nodes, total_nodes), np.inf, dtype=np.float64)
    dist[:k, :k] = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist[:k, :k], 0.0)
    dist_sq = dist * dist

    # Active upper-triangular search buffer with diagonal and inactive entries masked to inf.
    work = np.full((total_nodes, total_nodes), np.inf, dtype=np.float64)
    triu_i, triu_j = np.triu_indices(k, k=1)
    work[triu_i, triu_j] = dist[triu_i, triu_j]

    sizes = np.zeros(total_nodes, dtype=np.int64)
    sizes[:k] = 1
    is_active = np.zeros(total_nodes, dtype=bool)
    is_active[:k] = True
    children: dict[int, tuple[int, int]] = {}

    for next_id in range(k, total_nodes):
        flat_idx = int(np.argmin(work))
        s, t = divmod(flat_idx, total_nodes)
        if s > t:
            s, t = t, s
        children[next_id] = (s, t)

        is_active[s] = False
        is_active[t] = False
        work[s, :] = np.inf
        work[:, s] = np.inf
        work[t, :] = np.inf
        work[:, t] = np.inf

        active_idx = np.flatnonzero(is_active)
        if active_idx.size > 0:
            if method == "single":
                merged = np.minimum(dist[s, active_idx], dist[t, active_idx])
                merged_sq = merged * merged
            else:
                ns = float(sizes[s])
                nt = float(sizes[t])
                nu = sizes[active_idx].astype(np.float64)
                merged_sq = np.maximum(
                    (
                        (nu + ns) * dist_sq[s, active_idx]
                        + (nu + nt) * dist_sq[t, active_idx]
                        - nu * dist_sq[s, t]
                    )
                    / (ns + nt + nu),
                    0.0,
                )
                merged = np.sqrt(merged_sq)

            dist[active_idx, next_id] = merged
            dist[next_id, active_idx] = merged
            dist_sq[active_idx, next_id] = merged_sq
            dist_sq[next_id, active_idx] = merged_sq
            # active_idx entries are strictly < next_id, so (active_idx, next_id) stays upper-tri.
            work[active_idx, next_id] = merged

        sizes[next_id] = sizes[s] + sizes[t]
        is_active[next_id] = True

    root = total_nodes - 1
    order: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node < k:
            order.append(node)
        else:
            left, right = children[node]
            stack.append(right)
            stack.append(left)
    return order


def _cluster_variance(cov: np.ndarray, idx: list[int]) -> float:
    """Inverse-variance cluster variance ``w^T Sigma w`` clamped non-negative."""
    diag = np.diag(cov)[idx]
    if np.any(diag <= 0.0):
        return 0.0
    inv = 1.0 / diag
    weights = inv / inv.sum()
    sub = cov[np.ix_(idx, idx)]
    return float(max(weights @ sub @ weights, 0.0))


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Halve the ordered list repeatedly, splitting risk by inverse cluster variance."""
    weights = np.ones(cov.shape[0], dtype=np.float64)
    stack = [list(order)]
    while stack:
        idx = stack.pop()
        if len(idx) < 2:
            continue
        mid = len(idx) // 2
        left, right = idx[:mid], idx[mid:]
        v_left = _cluster_variance(cov, left)
        v_right = _cluster_variance(cov, right)
        total = v_left + v_right
        alpha = 0.5 if total <= 0.0 else float(np.clip(1.0 - v_left / total, 0.0, 1.0))
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        stack.append(left)
        stack.append(right)
    return weights


def _solve_erc_ccd(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 500) -> np.ndarray:
    """Solve Equal Risk Contribution via Spinu / Griveau-Billion Cyclical Coordinate Descent."""
    k = cov.shape[0]
    sigma = 0.5 * (cov + cov.T)
    if float(np.linalg.eigvalsh(sigma)[0]) <= 1e-12:
        sigma = nearest_spd(sigma, eps=1e-12)

    diag = np.diag(sigma)
    inv_vol = 1.0 / np.sqrt(np.maximum(diag, 1e-300))
    x = inv_vol / inv_vol.sum()
    sx = sigma @ x
    budget = 1.0 / float(k)

    for _ in range(max_iter):
        for i in range(k):
            a_i = float(diag[i])
            c_i = float(sx[i] - a_i * x[i])
            disc = max(c_i * c_i + 4.0 * a_i * budget, 0.0)
            x_new = (-c_i + float(np.sqrt(disc))) / (2.0 * a_i)
            delta = x_new - float(x[i])
            x[i] = x_new
            sx += delta * sigma[:, i]

        w = x / x.sum()
        rc = w * (sigma @ w)
        total_var = float(rc.sum())
        if total_var > 0.0 and float(np.max(np.abs(rc / total_var - budget))) <= tol:
            break

    return x / x.sum()


def _weights_frame(
    names: Sequence[str],
    weights: np.ndarray,
    asset_col: str,
    weight_col: str,
) -> pl.DataFrame:
    """Assemble the long-form weight frame, normalizing to sum to one."""
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("allocation produced non-positive total weight")
    return pl.DataFrame(
        {
            asset_col: list(names),
            weight_col: np.asarray(weights, dtype=np.float64) / total,
        }
    )


def hierarchical_risk_parity(
    returns: pl.DataFrame | pl.LazyFrame | np.ndarray,
    *,
    columns: Sequence[str] | None = None,
    time_col: str | None = None,
    method: _Linkage = "ward",
    shrinkage: _Shrinkage | None = "ledoit_wolf",
    min_rows: int = 10,
    asset_col: str = "asset",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Hierarchical Risk Parity weights (Lopez de Prado, 2016).

    Parameters
    ----------
    returns:
        Wide return frame (one column per asset, optional ``time_col``) or a 2-D
        array of shape ``(observations, assets)``. Rows with any missing value are
        dropped.
    columns:
        Asset columns to allocate. Defaults to every numeric column except
        ``time_col``.
    time_col:
        Column to ignore when auto-selecting assets.
    method:
        Linkage used for tree clustering: ``"ward"`` (default) or ``"single"``.
    shrinkage:
        Covariance estimator: ``"ledoit_wolf"`` (default), ``"marchenko_pastur"``,
        ``"exponential"``, ``"sample"``, or ``None``.
    min_rows:
        Minimum number of complete observations required.
    asset_col, weight_col:
        Output column names.

    Returns
    -------
    polars.DataFrame
        Long-form frame with one row per asset and weights summing to 1.0.
    """
    if method not in ("ward", "single"):
        raise ValueError("method must be 'ward' or 'single'")
    x, names = _as_return_matrix(returns, columns, time_col, min_rows)
    cov = _covariance_matrix(x, shrinkage)
    corr = _correlation_from_cov(cov)
    order = _linkage_leaf_order(corr, method)
    weights = _recursive_bisection(cov, order)
    return _weights_frame(names, weights, asset_col, weight_col)


def equal_risk_parity(
    returns: pl.DataFrame | pl.LazyFrame | np.ndarray,
    *,
    columns: Sequence[str] | None = None,
    time_col: str | None = None,
    method: _ErpMethod = "erc",
    shrinkage: _Shrinkage | None = "ledoit_wolf",
    min_rows: int = 10,
    tol: float = 1e-12,
    max_iter: int = 500,
    asset_col: str = "asset",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Equal Risk Parity (ERC) or Inverse-Volatility (IVP) portfolio weights.

    When ``method="erc"`` (default), solves the exact Equal Risk Contribution
    problem ``w_i (Sigma w)_i = (1 / N) w^T Sigma w`` using the full shrunk or
    denoised covariance matrix via Cyclical Coordinate Descent. When assets are
    uncorrelated, this reduces identically to inverse-volatility weighting.
    Pass ``method="ivp"`` for the diagonal ``1 / sigma_i`` baseline.

    Returns
    -------
    polars.DataFrame
        Long-form frame with one row per asset and weights summing to 1.0.
    """
    if method not in ("erc", "ivp"):
        raise ValueError("method must be 'erc' or 'ivp'")
    x, names = _as_return_matrix(returns, columns, time_col, min_rows)
    cov = _covariance_matrix(x, shrinkage)
    if method == "ivp":
        vol = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        return _weights_frame(names, 1.0 / vol, asset_col, weight_col)
    weights = _solve_erc_ccd(cov, tol=tol, max_iter=max_iter)
    return _weights_frame(names, weights, asset_col, weight_col)
