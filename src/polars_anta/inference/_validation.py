"""Internal normal-distribution, array-coercion, and grouping helpers for inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

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


def _normalize_group_by(group_by: str | Sequence[str]) -> list[str]:
    """Normalize ``group_by`` into a non-empty list of column names."""
    if isinstance(group_by, str):
        cols = [group_by]
    else:
        cols = [str(c) for c in group_by]
    if not cols or any(not c for c in cols):
        raise ValueError("group_by must specify at least one column name")
    return cols
