"""Pure-NumPy covariance estimation, Ledoit-Wolf shrinkage, and Marchenko-Pastur RMT denoising."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

__all__ = [
    "nearest_spd",
    "sample_covariance",
    "exponential_covariance",
    "ledoit_wolf_shrinkage",
    "marchenko_pastur_denoise",
]


def _extract_2d_obs(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    exclude: Sequence[str] = (),
    min_rows: int = 2,
) -> np.ndarray:
    """Convert array or Polars frame into a finite 2D ``(T, K)`` float64 matrix."""
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        eager = data.collect() if isinstance(data, pl.LazyFrame) else data
        excl = set(exclude)
        cols = [c for c in eager.columns if c not in excl]
        if not cols:
            raise ValueError("No numeric columns remain after applying exclude")
        mat = eager.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    else:
        mat = np.asarray(data, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[:, None]
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D observation matrix, got {mat.ndim}D")

    valid_rows = np.isfinite(mat).all(axis=1)
    clean = np.asarray(mat[valid_rows], dtype=np.float64)
    if clean.shape[0] < min_rows:
        raise ValueError(f"At least {min_rows} finite observations required, got {clean.shape[0]}")
    if clean.shape[1] < 1:
        raise ValueError("Observation matrix must have at least 1 column")
    return clean


def nearest_spd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a square matrix onto the cone of Symmetric Positive Definite matrices."""
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive")
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix must be 2D square")
    if not np.all(np.isfinite(a)):
        raise ValueError("Input matrix contains NaN or Inf values")

    sym = 0.5 * (a + a.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, float(eps))
    out = (eigvecs * clipped[None, :]) @ eigvecs.T
    return 0.5 * (out + out.T)


def sample_covariance(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    exclude: Sequence[str] = (),
    ddof: int = 1,
) -> np.ndarray:
    """Compute finite-row sample covariance matrix of shape ``(K, K)``."""
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    x = _extract_2d_obs(data, exclude=exclude, min_rows=ddof + 1)
    x_c = x - np.mean(x, axis=0, keepdims=True)
    cov = (x_c.T @ x_c) / float(x.shape[0] - ddof)
    return 0.5 * (cov + cov.T)


def exponential_covariance(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    half_life: float = 60.0,
    exclude: Sequence[str] = (),
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute unbiased exponentially weighted covariance matrix with ``half_life`` decay."""
    if half_life <= 0.0 or not np.isfinite(half_life):
        raise ValueError("half_life must be a positive finite number")
    x = _extract_2d_obs(data, exclude=exclude, min_rows=2)
    t_obs = x.shape[0]

    decay = np.exp(-np.log(2.0) / float(half_life))
    exponents = np.arange(t_obs - 1, -1, -1, dtype=np.float64)
    weights = np.power(decay, exponents)
    weights /= np.sum(weights)

    mean_w = np.sum(weights[:, None] * x, axis=0, keepdims=True)
    x_c = x - mean_w
    cov = (x_c.T * weights[None, :]) @ x_c
    denom = 1.0 - float(np.sum(weights * weights))
    if denom > 1e-12:
        cov = cov / denom
    return nearest_spd(cov, eps=eps)


def ledoit_wolf_shrinkage(
    data: np.ndarray | pl.DataFrame | pl.LazyFrame,
    target: Literal["constant_correlation", "identity"] = "constant_correlation",
    exclude: Sequence[str] = (),
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Analytical Ledoit-Wolf (2004) covariance shrinkage estimator using loop-free BLAS."""
    if target not in ("constant_correlation", "identity"):
        raise ValueError("target must be 'constant_correlation' or 'identity'")

    x = _extract_2d_obs(data, exclude=exclude, min_rows=4)
    t_obs, k_cols = x.shape
    t_f = float(t_obs)

    x_c = x - np.mean(x, axis=0, keepdims=True)
    s_mle = (x_c.T @ x_c) / t_f
    var = np.diag(s_mle)
    std = np.sqrt(np.maximum(var, 1e-15))

    if k_cols == 1:
        return nearest_spd((t_f / (t_f - 1.0)) * s_mle, eps=eps), 0.0

    # Vectorized fourth-moment matrix: pi_ij = (1/T) sum_t (x_{ti} x_{tj} - S_{ij})^2
    x2 = x_c * x_c
    pi_mat = (x2.T @ x2) / t_f - s_mle * s_mle
    pi_hat = float(np.sum(pi_mat))

    if target == "identity":
        mu = float(np.trace(s_mle)) / float(k_cols)
        target_mat = mu * np.eye(k_cols, dtype=np.float64)
        d2 = float(np.sum((s_mle - target_mat) ** 2))
        delta = min(1.0, max(0.0, (pi_hat / t_f) / d2)) if d2 > 1e-15 else 0.0
    else:
        denom_sd = np.outer(std, std)
        corr = np.clip(s_mle / denom_sd, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        r_bar = float((np.sum(corr) - float(k_cols)) / float(k_cols * (k_cols - 1)))
        target_mat = r_bar * denom_sd
        np.fill_diagonal(target_mat, var)

        d2 = float(np.sum((s_mle - target_mat) ** 2))
        if d2 <= 1e-15:
            return nearest_spd((t_f / (t_f - 1.0)) * s_mle, eps=eps), 0.0

        # Vectorized asymptotic cross-covariance: theta_ij = (1/T) sum_t x_{ti}^3 x_{tj} - S_{ii} S_{ij}
        theta_mat = ((x_c * x2).T @ x_c) / t_f - var[:, None] * s_mle
        np.fill_diagonal(theta_mat, 0.0)
        rho_diag = float(np.sum(np.diag(pi_mat)))
        rho_off = r_bar * float(np.sum((std[None, :] / std[:, None]) * theta_mat))
        kappa = (pi_hat - (rho_diag + rho_off)) / d2
        delta = min(1.0, max(0.0, kappa / t_f))

    shrunk = (delta * target_mat + (1.0 - delta) * s_mle) * (t_f / (t_f - 1.0))
    return nearest_spd(shrunk, eps=eps), float(delta)


def marchenko_pastur_denoise(
    data_or_cov: np.ndarray | pl.DataFrame | pl.LazyFrame,
    n_obs: int | None = None,
    method: Literal["constant_residual", "targeted_shrink"] = "constant_residual",
    shrink_alpha: float = 0.5,
    is_cov: bool = False,
    exclude: Sequence[str] = (),
    eps: float = 1e-8,
) -> np.ndarray:
    """Random Matrix Theory (Marchenko-Pastur) spectral covariance/correlation denoiser.

    Identifies eigenvalues below the Marchenko-Pastur bulk edge
    $\\lambda_+ = \\hat{\\sigma}^2 (1 + \\sqrt{K / T})^2$ (estimated iteratively on the bulk to avoid
    inflation from dominant market/style modes) and replaces bulk noise via trace-preserving
    constant-residual averaging or targeted diagonal shrinkage.
    """
    if method not in ("constant_residual", "targeted_shrink"):
        raise ValueError("method must be 'constant_residual' or 'targeted_shrink'")
    if not (0.0 <= shrink_alpha <= 1.0):
        raise ValueError("shrink_alpha must lie in [0.0, 1.0]")

    if is_cov:
        if n_obs is None or n_obs < 2:
            raise ValueError("n_obs >= 2 is required when is_cov=True")
        cov = np.asarray(data_or_cov, dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance matrix must be 2D square when is_cov=True")
        t_obs = int(n_obs)
    else:
        x = _extract_2d_obs(data_or_cov, exclude=exclude, min_rows=2)
        t_obs = x.shape[0] if n_obs is None else int(n_obs)
        if t_obs < 2:
            raise ValueError("n_obs must be at least 2")
        x_c = x - np.mean(x, axis=0, keepdims=True)
        cov = (x_c.T @ x_c) / float(x.shape[0] - 1)

    cov = 0.5 * (cov + cov.T)
    k_cols = cov.shape[0]
    if k_cols <= 1:
        return nearest_spd(cov, eps=eps)

    var = np.maximum(np.diag(cov), float(eps))
    std = np.sqrt(var)
    denom_sd = np.outer(std, std)
    corr = np.clip(cov / denom_sd, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    corr = 0.5 * (corr + corr.T)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.0)
    q_ratio = float(k_cols) / float(t_obs)

    # Fixed-point bulk variance estimation so signal spikes do not inflate lambda_plus
    sigma2 = 1.0
    edge_factor = (1.0 + np.sqrt(q_ratio)) ** 2
    for _ in range(5):
        lambda_plus = sigma2 * edge_factor
        bulk = eigvals[eigvals <= lambda_plus]
        if len(bulk) == 0 or len(bulk) == k_cols:
            break
        sigma2 = float(np.mean(bulk))
    lambda_plus = sigma2 * edge_factor
    noise_mask = eigvals <= lambda_plus

    if np.any(noise_mask) and not np.all(noise_mask):
        if method == "constant_residual":
            eig_clean = eigvals.copy()
            eig_clean[noise_mask] = float(np.mean(eigvals[noise_mask]))
            c_raw = (eigvecs * eig_clean[None, :]) @ eigvecs.T
        else:
            v_n = eigvecs[:, noise_mask]
            c_noise = (v_n * eigvals[None, noise_mask]) @ v_n.T
            v_s = eigvecs[:, ~noise_mask]
            c_sig = (v_s * eigvals[None, ~noise_mask]) @ v_s.T
            c_raw = (
                c_sig
                + float(shrink_alpha) * c_noise
                + (1.0 - float(shrink_alpha)) * np.diag(np.diag(c_noise))
            )
    elif np.all(noise_mask):
        c_raw = np.eye(k_cols, dtype=np.float64)
    else:
        c_raw = corr

    c_raw = 0.5 * (c_raw + c_raw.T)
    diag_c = np.sqrt(np.maximum(np.diag(c_raw), 1e-15))
    c_denoised = c_raw / np.outer(diag_c, diag_c)
    np.fill_diagonal(c_denoised, 1.0)
    cov_denoised = c_denoised * denom_sd
    return nearest_spd(cov_denoised, eps=eps)
