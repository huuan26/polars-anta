"""Hierarchical Risk Parity and inverse-volatility portfolio allocation.

This module implements the allocation side of Lopez de Prado (2016),
*Building Diversified Portfolios that Outperform Out of Sample*:

1. **Tree clustering** on correlation distance d_ij = sqrt((1 - rho_ij) / 2).
2. **Quasi-diagonalization**: reorder assets so similar assets sit next to
   each other.
3. **Recursive bisection**: split the ordered list in halves and allocate
   inversely to each half's inverse-variance cluster variance.

Only NumPy is used for the linear algebra; the linkage step is a direct
Lance-Williams implementation, so the package keeps its two-dependency
footprint (polars + numpy) and needs no SciPy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from .covariance import ledoit_wolf_shrinkage, sample_covariance

__all__ = ["hierarchical_risk_parity", "equal_risk_parity"]

_Shrinkage = Literal["ledoit_wolf", "sample"]
_Linkage = Literal["ward", "single"]


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
    """Sample or Ledoit-Wolf shrunk covariance of the return matrix."""
    if shrinkage == "ledoit_wolf":
        cov, _ = ledoit_wolf_shrinkage(x)
        return np.asarray(cov, dtype=np.float64)
    if shrinkage in (None, "sample"):
        return np.asarray(sample_covariance(x), dtype=np.float64)
    raise ValueError("shrinkage must be 'ledoit_wolf', 'sample' or None")


def _correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix with unit diagonal."""
    std = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _linkage_leaf_order(corr: np.ndarray, method: _Linkage) -> list[int]:
    """Agglomerate assets and return the dendrogram leaf order (quasi-diagonalization)."""
    k = corr.shape[0]
    # Agglomeration creates k - 1 new nodes, so the working matrices must hold 2k - 1 rows.
    total_nodes = 2 * k - 1
    dist = np.full((total_nodes, total_nodes), np.inf)
    dist[:k, :k] = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    dist_sq = dist * dist

    sizes = [1] * k
    children: dict[int, tuple[int, int]] = {}
    active = list(range(k))
    next_id = k

    while len(active) > 1:
        best_v = np.inf
        s = t = -1
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                v = float(dist[active[i], active[j]])
                if v < best_v:
                    best_v, s, t = v, active[i], active[j]
        children[next_id] = (s, t)

        for u in active:
            if u == s or u == t:
                continue
            if method == "single":
                merged = min(dist[s, u], dist[t, u])
                dist[u, next_id] = dist[next_id, u] = merged
                dist_sq[u, next_id] = dist_sq[next_id, u] = merged * merged
            else:
                ns, nt, nu = sizes[s], sizes[t], sizes[u]
                merged_sq = (
                    (nu + ns) * dist_sq[s, u] + (nu + nt) * dist_sq[t, u] - nu * dist_sq[s, t]
                ) / (ns + nt + nu)
                merged_sq = max(float(merged_sq), 0.0)
                dist_sq[u, next_id] = dist_sq[next_id, u] = merged_sq
                dist[u, next_id] = dist[next_id, u] = np.sqrt(merged_sq)

        sizes.append(sizes[s] + sizes[t])
        active = [u for u in active if u != s and u != t]
        active.append(next_id)
        next_id += 1

    root = active[0]
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
    """Inverse-variance cluster variance w'Sw, the recursive-bisection spread."""
    diag = np.diag(cov)[idx]
    if np.any(diag <= 0.0):
        return 0.0
    inv = 1.0 / diag
    weights = inv / inv.sum()
    sub = cov[np.ix_(idx, idx)]
    return float(weights @ sub @ weights)


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
        alpha = 1.0 if total <= 0.0 else 1.0 - v_left / total
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        stack.append(left)
        stack.append(right)
    return weights


def _weights_frame(
    names: Sequence[str],
    weights: np.ndarray,
    asset_col: str,
    weight_col: str,
) -> pl.DataFrame:
    """Assemble the long-form weight frame, normalising to sum to one."""
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
        Wide return frame (one column per asset, optional time_col) or a 2-D
        array of shape (observations, assets). Rows with any missing value are
        dropped.
    columns:
        Asset columns to allocate. Defaults to every numeric column except
        time_col.
    time_col:
        Column to ignore when auto-selecting assets.
    method:
        Linkage used for tree clustering: ward (default) or single.
    shrinkage:
        Covariance estimator: ledoit_wolf (default), sample, or None.
    min_rows:
        Minimum number of complete observations required.
    asset_col, weight_col:
        Output column names.

    Returns
    -------
    polars.DataFrame
        Long-form frame with one row per asset and weights summing to 1.0.

    Notes
    -----
    HRP never inverts the covariance matrix: assets are clustered on
    correlation distance, quasi-diagonalised, then split recursively with
    weights inversely proportional to each half's inverse-variance cluster
    variance. Each cluster gets a comparable risk budget regardless of how many
    assets it holds, which is what keeps the result stable when the covariance
    matrix is close to singular.
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
    shrinkage: _Shrinkage | None = "ledoit_wolf",
    min_rows: int = 10,
    asset_col: str = "asset",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Inverse-volatility weights (equal risk contribution if uncorrelated).

    Each asset receives weight proportional to 1 / sigma_i using the covariance
    diagonal, so every asset contributes the same variance under the
    deliberately naive assumption of uncorrelated returns. This is the Equal
    Risk Parity baseline HRP is usually compared against.

    Returns
    -------
    polars.DataFrame
        Long-form frame with one row per asset and weights summing to 1.0.
    """
    x, names = _as_return_matrix(returns, columns, time_col, min_rows)
    cov = _covariance_matrix(x, shrinkage)
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    return _weights_frame(names, 1.0 / vol, asset_col, weight_col)
