"""Cross-sectional alpha conditioning, FWL neutralization, and symmetric orthogonalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar

import numpy as np
import polars as pl

from .core import IntoExpr, parse_expr

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)

__all__ = [
    "cs_gaussianize",
    "lowdin_matrix",
    "neutralize_panel",
    "orthogonalize_panel",
    "collinearity_summary",
]

# Peter J. Acklam's inverse normal CDF rational approximation coefficients (max error < 1.15e-9)
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def _horner(x: pl.Expr, coeffs: tuple[float, ...]) -> pl.Expr:
    acc: pl.Expr = pl.lit(coeffs[0])
    for c in coeffs[1:]:
        acc = acc * x + c
    return acc


def _inv_norm_cdf_expr(p: pl.Expr) -> pl.Expr:
    """Evaluate standard normal quantile Phi^{-1}(p) for p in (0, 1) as a native Polars Expr."""
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = _horner(r_mid, _A) * q_mid / (_horner(r_mid, _B) * r_mid + 1.0)

    p_tail = pl.when(p < 0.5).then(p).otherwise(1.0 - p).clip(lower_bound=1e-15)
    q_tail = (-2.0 * p_tail.log()).sqrt()
    z_tail_abs = _horner(q_tail, _C) / (_horner(q_tail, _D) * q_tail + 1.0)
    z_tail = pl.when(p < 0.5).then(z_tail_abs).otherwise(-z_tail_abs)
    return pl.when((p >= _P_LOW) & (p <= 1.0 - _P_LOW)).then(z_mid).otherwise(z_tail)


def cs_gaussianize(
    expr: IntoExpr,
    over: IntoExpr | Sequence[str] | None = None,
    clip_bound: float | None = 4.0,
) -> pl.Expr:
    """Cross-sectional rank-to-normal (Van der Waerden / Blom) transformation."""
    if clip_bound is not None and clip_bound <= 0.0:
        raise ValueError("clip_bound must be strictly positive when provided")

    clean = parse_expr(expr).cast(pl.Float64).fill_nan(None)
    over_cols = (
        None if over is None else ([over] if isinstance(over, (str, pl.Expr)) else list(over))
    )
    ranked = clean.rank(method="average")
    count = clean.count()
    if over_cols is not None:
        ranked = ranked.over(over_cols)
        count = count.over(over_cols)

    u = ((ranked - 0.5) / count.cast(pl.Float64)).clip(1e-12, 1.0 - 1e-12)
    valid = (clean.is_not_null() & (count >= 2)).fill_null(False)
    # Keep finite column-rooted `u` first inside `.then(...)` so Polars preserves the input column name
    z = pl.when(valid).then((u * 0.0) + _inv_norm_cdf_expr(u)).otherwise(None)
    return z.clip(-float(clip_bound), float(clip_bound)) if clip_bound is not None else z


def lowdin_matrix(
    mat: np.ndarray,
    rcond: float = 1e-10,
    standardize: bool = True,
) -> np.ndarray:
    """Compute Lowdin symmetric orthogonalization via thin SVD polar decomposition (2D or 3D)."""
    if rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    x = np.asarray(mat, dtype=np.float64)
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {x.ndim}D")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input matrix contains NaN or Inf values")

    n_obs, k_cols = x.shape[-2], x.shape[-1]
    if n_obs < 2 or n_obs < k_cols:
        raise ValueError(f"Number of rows ({n_obs}) must be >= max(2, K={k_cols})")

    work = x - np.mean(x, axis=-2, keepdims=True) if standardize else x.copy()
    u_mat, s_vals, vt_mat = np.linalg.svd(work, full_matrices=False)
    s_max = np.max(s_vals, axis=-1, keepdims=True)
    valid = (s_max > 1e-15) & (s_vals > float(rcond) * s_max)
    ortho = np.where(valid[..., None, :], u_mat, 0.0) @ vt_mat
    return ortho * np.sqrt(float(n_obs - 1)) if standardize else ortho


def neutralize_panel(
    df: FrameType,
    factor_cols: Sequence[str],
    time_col: str = "date",
    risk_cols: Sequence[str] | None = None,
    group_col: str | Sequence[str] | None = None,
    standardize: bool = True,
    ridge: float = 0.0,
    suffix: str = "_neut",
) -> FrameType:
    """Simultaneously neutralize factors against categorical groups and continuous risks (FWL)."""
    factors = list(factor_cols)
    if not factors:
        raise ValueError("factor_cols must contain at least one column name")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    risks = list(risk_cols) if risk_cols is not None else []
    groups = (
        []
        if group_col is None
        else ([group_col] if isinstance(group_col, str) else list(group_col))
    )
    clean_cols = list(dict.fromkeys([*factors, *risks]))
    out = df.with_columns(
        [pl.col(c).cast(pl.Float64).fill_nan(None).alias(f"_anta_clean_{c}") for c in clean_cols]
    )

    risk_valid = (
        pl.all_horizontal([pl.col(f"_anta_clean_{r}").is_not_null() for r in risks])
        if risks
        else pl.lit(True)
    )
    demean_over = [time_col, *groups] if groups else [time_col]
    out_exprs: list[pl.Expr] = []

    for f in factors:
        valid = pl.col(f"_anta_clean_{f}").is_not_null() & risk_valid
        f_masked = pl.when(valid).then(pl.col(f"_anta_clean_{f}")).otherwise(None)
        y_res = f_masked - f_masked.mean().over(demean_over)

        if risks:
            u_list = [
                pl.when(valid).then(pl.col(f"_anta_clean_{r}")).otherwise(None)
                - pl.when(valid)
                .then(pl.col(f"_anta_clean_{r}"))
                .otherwise(None)
                .mean()
                .over(demean_over)
                for r in risks
            ]
            for j, q_j in enumerate(u_list):
                var_q = (q_j * q_j).mean().over(time_col)
                denom = var_q + float(ridge)
                beta_y = (
                    pl.when(var_q > 1e-15)
                    .then((y_res * q_j).mean().over(time_col) / denom)
                    .otherwise(0.0)
                )
                y_res = y_res - beta_y * q_j
                for m in range(j + 1, len(u_list)):
                    beta_m = (
                        pl.when(var_q > 1e-15)
                        .then((u_list[m] * q_j).mean().over(time_col) / denom)
                        .otherwise(0.0)
                    )
                    u_list[m] = u_list[m] - beta_m * q_j

        if standardize:
            std_y = y_res.std().over(time_col)
            y_res = (
                pl.when(valid & std_y.is_not_null() & (std_y > 1e-15))
                .then(y_res / std_y)
                .otherwise(pl.when(valid).then(0.0).otherwise(None))
            )
        out_exprs.append(y_res.alias(f"{f}{suffix}"))

    return out.with_columns(out_exprs).drop([f"_anta_clean_{c}" for c in clean_cols])


def orthogonalize_panel(
    df: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    time_col: str = "date",
    method: Literal["lowdin", "lowdin_corr", "gram_schmidt"] = "lowdin",
    rcond: float = 1e-10,
    standardize: bool = True,
    suffix: str = "_ortho",
) -> pl.DataFrame:
    """Orthogonalize multiple alpha columns within each ``time_col`` slice without bar loops.

    Subspace invariance note: because $\\widetilde{\\mathbf{F}}_t = \\mathbf{Z}_t \\mathbf{M}_t$ forms
    cross-factor linear combinations, prior sector/risk neutrality ($\\mathbf{X}_t^T \\mathbf{F}_t = \\mathbf{0}$)
    is preserved if and only if **all** columns in ``factor_cols`` have already been neutralized
    against $\\mathbf{X}_t$ via :func:`neutralize_panel` before calling ``orthogonalize_panel``.
    """
    factors = list(factor_cols)
    k = len(factors)
    if k < 1:
        raise ValueError("factor_cols must contain at least one column name")
    if method not in ("lowdin", "lowdin_corr", "gram_schmidt"):
        raise ValueError("method must be 'lowdin', 'lowdin_corr', or 'gram_schmidt'")
    if rcond < 0.0:
        raise ValueError("rcond must be non-negative")

    eager = df.collect() if isinstance(df, pl.LazyFrame) else df
    clean_names = [f"_anta_f_{i}" for i in range(k)]
    valid_expr = pl.all_horizontal(
        [pl.col(c).cast(pl.Float64).fill_nan(None).is_not_null() for c in factors]
    )
    work = eager.with_columns(
        [
            pl.when(valid_expr)
            .then(pl.col(factors[i]).cast(pl.Float64).fill_nan(None))
            .otherwise(None)
            .alias(clean_names[i])
            for i in range(k)
        ]
    )

    z_names = [f"_anta_z_{i}" for i in range(k)]
    z_exprs: list[pl.Expr] = []
    for i in range(k):
        col_e = pl.col(clean_names[i])
        centered = col_e - col_e.mean().over(time_col)
        if method == "lowdin_corr":
            sd = col_e.std().over(time_col)
            centered = pl.when(sd.is_not_null() & (sd > 1e-15)).then(centered / sd).otherwise(0.0)
        z_exprs.append(pl.when(valid_expr).then(centered).otherwise(None).alias(z_names[i]))
    work = work.with_columns(z_exprs)

    if method == "gram_schmidt":
        q_exprs: list[pl.Expr] = []
        for i in range(k):
            u_i = pl.col(z_names[i])
            for j in range(i):
                q_j = q_exprs[j]
                vq = (q_j * q_j).mean().over(time_col)
                beta = (
                    pl.when(vq > float(rcond))
                    .then((u_i * q_j).mean().over(time_col) / vq)
                    .otherwise(0.0)
                )
                u_i = u_i - beta * q_j
            q_exprs.append(u_i)

        out_cols: list[pl.Expr] = []
        for i, q_i in enumerate(q_exprs):
            if standardize:
                sd_q = q_i.std().over(time_col)
                q_out = (
                    pl.when(valid_expr & sd_q.is_not_null() & (sd_q > 1e-15))
                    .then(q_i / sd_q)
                    .otherwise(pl.when(valid_expr).then(0.0).otherwise(None))
                )
            else:
                q_out = pl.when(valid_expr).then(q_i).otherwise(None)
            out_cols.append(q_out.alias(f"{factors[i]}{suffix}"))
        return work.with_columns(out_cols).drop([*clean_names, *z_names])

    gram_aggs: list[pl.Expr] = [pl.col(z_names[0]).count().alias("_anta_n")]
    for a in range(k):
        for b in range(a, k):
            gram_aggs.append((pl.col(z_names[a]) * pl.col(z_names[b])).sum().alias(f"_g_{a}_{b}"))

    gram_df = work.group_by(time_col, maintain_order=True).agg(gram_aggs)
    n_vec = gram_df["_anta_n"].to_numpy().astype(np.float64)
    g_3d = np.zeros((gram_df.height, k, k), dtype=np.float64)
    for a in range(k):
        for b in range(a, k):
            vals = gram_df[f"_g_{a}_{b}"].fill_null(0.0).to_numpy().astype(np.float64)
            g_3d[:, a, b] = vals
            g_3d[:, b, a] = vals

    eigvals, eigvecs = np.linalg.eigh(g_3d)
    max_eig = np.max(eigvals, axis=-1, keepdims=True)
    enough_obs = (n_vec >= max(2, k))[:, None]
    eig_ok = enough_obs & (max_eig > 1e-15) & (eigvals > float(rcond) * max_eig)
    inv_sqrt = np.where(eig_ok, 1.0 / np.sqrt(np.maximum(eigvals, 1e-30)), 0.0)
    if standardize:
        inv_sqrt = inv_sqrt * np.sqrt(np.maximum(n_vec - 1.0, 0.0))[:, None]

    m_3d = (eigvecs * inv_sqrt[:, None, :]) @ np.swapaxes(eigvecs, -1, -2)
    m_cols: dict[str, pl.Series | np.ndarray] = {time_col: gram_df[time_col]}
    for a in range(k):
        for b in range(k):
            m_cols[f"_m_{a}_{b}"] = m_3d[:, a, b]
    m_df = pl.DataFrame(m_cols).with_columns(pl.lit(enough_obs.ravel()).alias("_anta_ok"))

    joined = work.join(m_df, on=time_col, how="left")
    ortho_exprs = [
        pl.when(valid_expr & pl.col("_anta_ok"))
        .then(pl.sum_horizontal([pl.col(z_names[a]) * pl.col(f"_m_{a}_{b}") for a in range(k)]))
        .otherwise(None)
        .alias(f"{factors[b]}{suffix}")
        for b in range(k)
    ]
    drop_cols = [
        *clean_names,
        *z_names,
        "_anta_ok",
        *(f"_m_{a}_{b}" for a in range(k) for b in range(k)),
    ]
    return joined.with_columns(ortho_exprs).drop(drop_cols)


def collinearity_summary(
    df: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    time_col: str | None = None,
    rcond: float = 1e-10,
) -> pl.DataFrame:
    """Compute VIF, participation-ratio effective rank ($N_{\\text{eff}}$), and condition number."""
    factors = list(factor_cols)
    k = len(factors)
    if k < 1:
        raise ValueError("factor_cols must contain at least one column name")

    eager = df.collect() if isinstance(df, pl.LazyFrame) else df
    valid = pl.all_horizontal(
        [pl.col(c).cast(pl.Float64).fill_nan(None).is_not_null() for c in factors]
    )
    clean = eager.filter(valid)
    if clean.height < 2:
        raise ValueError("At least 2 complete observations are required for collinearity summary")

    if time_col is not None and time_col in eager.columns:
        z_df = clean.select(
            [time_col, *(pl.col(c).cast(pl.Float64) for c in factors)]
        ).with_columns(
            [
                ((pl.col(c) - pl.col(c).mean().over(time_col)) / pl.col(c).std().over(time_col))
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias(c)
                for c in factors
            ]
        )
        mat = z_df.select(factors).to_numpy().astype(np.float64)
    else:
        mat = (
            clean.select([pl.col(c).cast(pl.Float64) for c in factors])
            .to_numpy()
            .astype(np.float64)
        )

    stds = np.std(mat, axis=0, ddof=1)
    means = np.mean(mat, axis=0, keepdims=True)
    z = np.where(stds[None, :] > 1e-15, (mat - means) / np.maximum(stds[None, :], 1e-15), 0.0)
    corr = 0.5 * ((z.T @ z) / float(max(len(z) - 1, 1)))
    corr = corr + corr.T

    eigvals = np.maximum(np.linalg.eigvalsh(corr), 0.0)
    sum_eig = float(np.sum(eigvals))
    sum_sq = float(np.sum(eigvals * eigvals))
    n_eff = (sum_eig * sum_eig / sum_sq) if sum_sq > 1e-15 else 0.0
    max_eig = float(np.max(eigvals)) if len(eigvals) else 0.0
    pos_eigs = eigvals[eigvals > float(rcond) * max(max_eig, 1e-15)]
    cond_num = float(np.sqrt(max_eig / np.min(pos_eigs))) if len(pos_eigs) > 0 else float("inf")
    vifs = np.maximum(np.diag(np.linalg.pinv(corr, rcond=float(rcond))), 1.0)

    return pl.DataFrame(
        {
            "factor": factors,
            "vif": vifs.tolist(),
            "effective_rank": [n_eff] * k,
            "condition_number": [cond_num] * k,
        }
    )
