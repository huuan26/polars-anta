"""Cost-aware multi-factor FISTA optimization and strictly causal walk-forward allocation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from .alpha import cs_zscore
from .covariance import (
    exponential_covariance,
    ledoit_wolf_shrinkage,
    marchenko_pastur_denoise,
    sample_covariance,
)
from .evaluation import ic_summary
from .inference import empirical_bayes_shrinkage, ic_matrix

__all__ = [
    "OptimizationResult",
    "WalkForwardAllocationResult",
    "project_simplex",
    "project_box_simplex",
    "prox_l1_shifted",
    "solve_fista",
    "combine_alphas_walkforward",
]


@dataclass(frozen=True)
class OptimizationResult:
    """Structured output for a single-period convex alpha allocation solve."""

    weights: np.ndarray
    converged: bool
    iterations: int
    objective: float
    turnover: float
    kkt_residual: float

    def summary(self) -> dict[str, Any]:
        """Return scalar convergence and optimality diagnostics."""
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "objective": round(self.objective, 8),
            "turnover": round(self.turnover, 8),
            "kkt_residual": self.kkt_residual,
        }


@dataclass(frozen=True)
class WalkForwardAllocationResult:
    """Container for strictly causal walk-forward multi-factor allocation."""

    data: pl.DataFrame
    weights_history: pl.DataFrame
    ic_matrix: pl.DataFrame
    composite_ic_summary: dict[str, float]
    equal_weight_ic_summary: dict[str, float]
    mean_weight_turnover: float
    embargo_bars: int

    def summary(self) -> dict[str, Any]:
        """Compare walk-forward composite alpha against equal-weight baseline."""
        return {
            "fista_ic_mean": round(self.composite_ic_summary["ic_mean"], 6),
            "fista_icir": round(self.composite_ic_summary["icir"], 6),
            "fista_nw_t_stat": round(self.composite_ic_summary["nw_t_stat"], 4),
            "ew_ic_mean": round(self.equal_weight_ic_summary["ic_mean"], 6),
            "ew_icir": round(self.equal_weight_ic_summary["icir"], 6),
            "ew_nw_t_stat": round(self.equal_weight_ic_summary["nw_t_stat"], 4),
            "mean_weight_turnover": round(self.mean_weight_turnover, 6),
            "embargo_bars": self.embargo_bars,
            "n_oos_periods": int(self.composite_ic_summary["n_periods"]),
        }


def project_simplex(v: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0 : sum(w) == budget}`` (Duchi et al., 2008)."""
    if budget <= 0.0:
        raise ValueError("budget must be strictly positive")
    arr = np.asarray(v, dtype=np.float64).ravel()
    k = len(arr)
    if k == 0:
        raise ValueError("Input vector must be non-empty")

    u = np.sort(arr)[::-1]
    cssv = np.cumsum(u)
    idx = np.arange(1, k + 1, dtype=np.float64)
    cond = u + (float(budget) - cssv) / idx > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    theta = (float(budget) - cssv[rho]) / float(rho + 1)
    return np.maximum(arr + theta, 0.0)


def prox_l1_shifted(v: np.ndarray, center: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal operator for ``threshold * ||w - center||_1``."""
    arr = np.asarray(v, dtype=np.float64).ravel()
    c = np.asarray(center, dtype=np.float64).ravel()
    if threshold <= 0.0:
        return arr.copy()
    diff = arr - c
    return c + np.sign(diff) * np.maximum(np.abs(diff) - float(threshold), 0.0)


def _joint_prox_box_simplex_l1(
    z: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: float,
    w_prev: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Exact joint proximal operator of ``tau * ||w - w_prev||_1 + I_{lb <= w <= ub, sum(w) == B}``.

    By Lagrangian duality on the budget hyperplane ``sum_k w_k = B``, each coordinate satisfies
    ``w_k(nu) = clip(prox_{tau |.-w_prev_k|}(z_k - nu), lb_k, ub_k)``, which is continuous and
    monotonically non-increasing in ``nu``. Exact 1D bisection achieves float64 precision.
    """
    if np.any(lb > ub):
        raise ValueError("lower_bounds must not exceed upper_bounds")
    sum_lb = float(np.sum(lb))
    sum_ub = float(np.sum(ub))
    if budget < sum_lb - 1e-10 or budget > sum_ub + 1e-10:
        raise ValueError(
            f"Infeasible budget={budget:.6f} for bounds sum [{sum_lb:.6f}, {sum_ub:.6f}]"
        )
    if abs(budget - sum_lb) <= 1e-12:
        return lb.copy()
    if abs(budget - sum_ub) <= 1e-12:
        return ub.copy()

    nu_low = float(np.min(z - tau - ub)) - 1.0
    nu_high = float(np.max(z + tau - lb)) + 1.0

    for _ in range(85):
        nu_mid = 0.5 * (nu_low + nu_high)
        y_mid = prox_l1_shifted(z - nu_mid, w_prev, tau) if tau > 0.0 else (z - nu_mid)
        w_mid = np.clip(y_mid, lb, ub)
        diff = float(np.sum(w_mid)) - budget
        if abs(diff) <= 1e-13:
            return w_mid
        if diff > 0.0:
            nu_low = nu_mid
        else:
            nu_high = nu_mid

    nu_star = 0.5 * (nu_low + nu_high)
    y_star = prox_l1_shifted(z - nu_star, w_prev, tau) if tau > 0.0 else (z - nu_star)
    return np.clip(y_star, lb, ub)


def project_box_simplex(
    v: np.ndarray,
    lower_bounds: np.ndarray | float = 0.0,
    upper_bounds: np.ndarray | float = 1.0,
    budget: float = 1.0,
) -> np.ndarray:
    """Euclidean projection onto ``{lb <= w <= ub : sum(w) == budget}``."""
    arr = np.asarray(v, dtype=np.float64).ravel()
    k = len(arr)
    if k == 0:
        raise ValueError("Input vector must be non-empty")
    lb = np.broadcast_to(np.asarray(lower_bounds, dtype=np.float64), (k,)).copy()
    ub = np.broadcast_to(np.asarray(upper_bounds, dtype=np.float64), (k,)).copy()
    return _joint_prox_box_simplex_l1(arr, lb, ub, float(budget), np.zeros(k), 0.0)


def solve_fista(
    expected_ic: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.0,
    w_prev: np.ndarray | None = None,
    l2_reg: float = 1e-5,
    constraint: Literal["box_simplex", "box", "unconstrained"] = "box_simplex",
    lower_bounds: np.ndarray | float = 0.0,
    upper_bounds: np.ndarray | float = 1.0,
    budget: float = 1.0,
    max_iter: int = 800,
    tol: float = 1e-9,
) -> OptimizationResult:
    """Solve convex turnover-penalized alpha allocation via Monotone FISTA with exact joint prox."""
    if risk_aversion < 0.0 or turnover_penalty < 0.0 or l2_reg < 0.0:
        raise ValueError("risk_aversion, turnover_penalty, and l2_reg must be non-negative")
    if constraint not in ("box_simplex", "box", "unconstrained"):
        raise ValueError("constraint must be 'box_simplex', 'box', or 'unconstrained'")

    mu = np.asarray(expected_ic, dtype=np.float64).ravel()
    k = len(mu)
    if k < 1:
        raise ValueError("expected_ic must be non-empty")
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (k, k):
        raise ValueError(f"covariance shape {cov.shape} does not match K={k}")

    cov_sym = 0.5 * (cov + cov.T)
    lb = np.broadcast_to(np.asarray(lower_bounds, dtype=np.float64), (k,)).copy()
    ub = np.broadcast_to(np.asarray(upper_bounds, dtype=np.float64), (k,)).copy()
    if constraint != "unconstrained" and np.any(lb > ub):
        raise ValueError("lower_bounds must not exceed upper_bounds")

    if w_prev is None:
        w_ref = np.full(k, float(budget) / k if constraint == "box_simplex" else 0.0)
    else:
        w_ref = np.asarray(w_prev, dtype=np.float64).ravel()
        if len(w_ref) != k:
            raise ValueError("w_prev length must match expected_ic")

    hess = float(risk_aversion) * cov_sym + float(l2_reg) * np.eye(k, dtype=np.float64)
    eigs = np.linalg.eigvalsh(hess)
    lipschitz = max(float(np.max(eigs)), 1e-8)
    step = 1.0 / lipschitz
    tau = step * float(turnover_penalty)

    def _prox_step(z_vec: np.ndarray) -> np.ndarray:
        if constraint == "box_simplex":
            return _joint_prox_box_simplex_l1(z_vec, lb, ub, float(budget), w_ref, tau)
        p = prox_l1_shifted(z_vec, w_ref, tau)
        return np.clip(p, lb, ub) if constraint == "box" else p

    def _obj(w_vec: np.ndarray) -> float:
        quad = 0.5 * float(w_vec @ (hess @ w_vec)) - float(mu @ w_vec)
        pen = float(turnover_penalty) * float(np.sum(np.abs(w_vec - w_ref)))
        return quad + pen

    w = _prox_step(w_ref)
    y = w.copy()
    t_val = 1.0
    f_cur = _obj(w)
    converged = False
    n_iters = 0

    for step_idx in range(1, int(max_iter) + 1):
        n_iters = step_idx
        grad_y = hess @ y - mu
        w_cand = _prox_step(y - step * grad_y)
        f_cand = _obj(w_cand)

        # Monotone FISTA + O'Donoghue-Candes gradient restart
        if f_cand > f_cur + 1e-14:
            grad_w = hess @ w - mu
            w_next = _prox_step(w - step * grad_w)
            f_next = _obj(w_next)
            t_next = 1.0
            y_next = w_next.copy()
        else:
            w_next = w_cand
            f_next = f_cand
            if float((y - w_next) @ (w_next - w)) > 0.0:
                t_next = 1.0
                y_next = w_next.copy()
            else:
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_val * t_val))
                y_next = w_next + ((t_val - 1.0) / t_next) * (w_next - w)

        res_step = float(np.linalg.norm(w_next - w))
        w, y, t_val, f_cur = w_next, y_next, t_next, f_next
        if res_step <= float(tol) * max(1.0, float(np.linalg.norm(w))):
            # Verify true proximal-gradient stationarity before stopping
            kkt_check = float(np.linalg.norm(w - _prox_step(w - step * (hess @ w - mu)))) / step
            if kkt_check <= max(float(tol) * lipschitz * 10.0, 1e-7):
                converged = True
                break

    kkt_res = float(np.linalg.norm(w - _prox_step(w - step * (hess @ w - mu)))) / step
    turnover = float(np.sum(np.abs(w - w_ref)))
    return OptimizationResult(
        weights=w,
        converged=converged,
        iterations=n_iters,
        objective=f_cur,
        turnover=turnover,
        kkt_residual=kkt_res,
    )


def _factor_return_and_turnover_matrices(
    df: pl.DataFrame,
    factors: list[str],
    return_col: str,
    time_col: str,
    asset_col: str | None,
    horizon: int,
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame]:
    """Compute per-period dollar-neutral factor returns, asset-level turnovers, and return dispersion."""
    work = df.sort([asset_col, time_col] if asset_col is not None else [time_col])
    r_clean = pl.col(return_col).cast(pl.Float64).fill_nan(None)

    w_exprs: list[pl.Expr] = []
    w_ret_exprs: list[pl.Expr] = []
    for i, c in enumerate(factors):
        f_c = pl.col(c).cast(pl.Float64).fill_nan(None)
        # Signal-only dollar-neutral weights (causal at t even when return_col is null)
        dm_all = f_c - f_c.mean().over(time_col)
        den_all = dm_all.abs().sum().over(time_col)
        w_sig = (
            pl.when(f_c.is_not_null() & (den_all > 1e-15))
            .then(dm_all / den_all)
            .otherwise(0.0)
            .alias(f"_ws_{i}")
        )
        w_exprs.append(w_sig)

        # Joint-support dollar-neutral weights for realized factor return r_{k,t}
        valid_jr = f_c.is_not_null() & r_clean.is_not_null()
        f_jr = pl.when(valid_jr).then(f_c).otherwise(None)
        dm_jr = f_jr - f_jr.mean().over(time_col)
        den_jr = dm_jr.abs().sum().over(time_col)
        w_jr = pl.when(valid_jr & (den_jr > 1e-15)).then(dm_jr / den_jr).otherwise(None)
        w_ret_exprs.append((w_jr * r_clean).alias(f"_fr_{i}"))

    work = work.with_columns([*w_exprs, *w_ret_exprs, r_clean.alias("_anta_r_clean")])

    ret_aggs = [
        pl.when(pl.col(f"_fr_{i}").count() > 0)
        .then(pl.col(f"_fr_{i}").sum())
        .otherwise(None)
        .alias(factors[i])
        for i in range(len(factors))
    ]
    fac_ret_df = work.group_by(time_col).agg(ret_aggs).sort(time_col)
    disp_df = (
        work.group_by(time_col).agg(pl.col("_anta_r_clean").std().alias("_cs_disp")).sort(time_col)
    )

    if asset_col is None:
        return fac_ret_df, None, disp_df

    turn_exprs = [
        (
            pl.col(f"_ws_{i}")
            - pl.col(f"_ws_{i}").shift(int(horizon), fill_value=0.0).over(asset_col)
        )
        .abs()
        .alias(f"_ft_{i}")
        for i in range(len(factors))
    ]
    turn_aggs = [pl.col(f"_ft_{i}").sum().alias(factors[i]) for i in range(len(factors))]
    fac_turn_df = work.with_columns(turn_exprs).group_by(time_col).agg(turn_aggs).sort(time_col)
    return fac_ret_df, fac_turn_df, disp_df


def combine_alphas_walkforward(
    df: pl.DataFrame | pl.LazyFrame,
    factor_cols: Sequence[str],
    return_col: str,
    time_col: str = "date",
    asset_col: str | None = None,
    lookback: int = 60,
    horizon: int = 1,
    signal_lag: int = 1,
    embargo: int | None = None,
    objective_mode: Literal["ic", "factor_return"] = "ic",
    one_way_cost_bps: float = 0.0,
    signal_ema_alpha: float | None = None,
    risk_aversion: float = 2.0,
    turnover_penalty: float = 0.05,
    l2_reg: float = 1e-4,
    constraint: Literal["box_simplex", "box", "unconstrained"] = "box_simplex",
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    budget: float = 1.0,
    cov_estimator: Literal[
        "ledoit_wolf", "marchenko_pastur", "exponential", "sample"
    ] = "ledoit_wolf",
    ic_method: Literal["spearman", "pearson"] = "spearman",
    ic_shrinkage: Literal["none", "empirical_bayes"] = "none",
    ic_ema_halflife: float | None = None,
    no_trade_buffer: float = 0.0,
    normalize_composite: bool = True,
    min_weight_norm: float = 1e-8,
    composite_col: str = "composite_alpha",
    equal_weight_col: str = "equal_weight_alpha",
) -> WalkForwardAllocationResult:
    """Strictly causal walk-forward multi-factor allocator with embargo and stock-turnover cost control.

    At decision index ``idx`` (`time_col[idx]`), uses only realized observations up to
    ``end_excl = idx - embargo`` where ``embargo = max(0, horizon + signal_lag - 1)``. Supports
    optimizing either cross-sectional Information Coefficients (``objective_mode='ic'``) or
    realized dollar-neutral factor portfolio returns (``objective_mode='factor_return'``), with
    per-factor stock-level transaction cost deduction (``one_way_cost_bps``) and optional causal
    exponential smoothing (``signal_ema_alpha``) applied symmetrically across factor estimation,
    ``composite_alpha``, and ``equal_weight_alpha``.
    """
    factors = list(factor_cols)
    k = len(factors)
    if k < 2:
        raise ValueError("At least 2 factor columns are required for combination")
    if lookback < 4:
        raise ValueError("lookback must be at least 4 periods")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if signal_lag < 0:
        raise ValueError("signal_lag must be non-negative")
    if no_trade_buffer < 0.0 or one_way_cost_bps < 0.0 or min_weight_norm < 0.0:
        raise ValueError(
            "no_trade_buffer, one_way_cost_bps, and min_weight_norm must be non-negative"
        )
    if objective_mode not in ("ic", "factor_return"):
        raise ValueError("objective_mode must be 'ic' or 'factor_return'")
    if one_way_cost_bps > 0.0 and asset_col is None:
        raise ValueError("asset_col is required when one_way_cost_bps > 0")
    if signal_ema_alpha is not None:
        if not (0.0 < float(signal_ema_alpha) <= 1.0):
            raise ValueError("signal_ema_alpha must lie in (0.0, 1.0]")
        if asset_col is None:
            raise ValueError("asset_col is required when signal_ema_alpha is set")

    emb = max(0, int(horizon) + int(signal_lag) - 1) if embargo is None else max(0, int(embargo))
    eager = df.collect() if isinstance(df, pl.LazyFrame) else df
    use_ema = signal_ema_alpha is not None and asset_col is not None
    orig_backup_cols = [f"_anta_orig_{i}" for i in range(k)]
    if use_ema:
        assert asset_col is not None and signal_ema_alpha is not None
        sorted_eager = eager.sort([asset_col, time_col])
        eval_df = sorted_eager.with_columns(
            [pl.col(c).alias(orig_c) for c, orig_c in zip(factors, orig_backup_cols, strict=True)]
        ).with_columns(
            [
                pl.col(c)
                .cast(pl.Float64)
                .fill_nan(None)
                .ewm_mean(alpha=float(signal_ema_alpha), adjust=False)
                .over(asset_col)
                .alias(c)
                for c in factors
            ]
        )
    else:
        eval_df = eager

    # Master timeline from factor observations (includes live dates where return_col is null)
    timeline_df = eval_df.select(time_col).unique().sort(time_col)
    realized_ic = ic_matrix(
        eval_df,
        factor_cols=factors,
        return_col=return_col,
        group_by=time_col,
        method=ic_method,
    )
    ic_aligned = timeline_df.join(realized_ic, on=time_col, how="left").sort(time_col)
    if ic_aligned.height <= lookback + emb:
        raise ValueError(
            f"Panel timestamps ({ic_aligned.height}) must exceed lookback + embargo ({lookback + emb})"
        )

    need_fac_stats = (objective_mode == "factor_return") or (one_way_cost_bps > 0.0)
    turn_arr: np.ndarray | None = None
    disp_arr: np.ndarray | None = None
    if need_fac_stats:
        fac_ret_df, fac_turn_df, disp_df = _factor_return_and_turnover_matrices(
            eval_df, factors, return_col, time_col, asset_col, horizon
        )
        ret_aligned = timeline_df.join(fac_ret_df, on=time_col, how="left").sort(time_col)
        disp_aligned = timeline_df.join(disp_df, on=time_col, how="left").sort(time_col)
        disp_arr = disp_aligned["_cs_disp"].to_numpy().astype(np.float64)
        if fac_turn_df is not None:
            turn_aligned = timeline_df.join(fac_turn_df, on=time_col, how="left").sort(time_col)
            turn_arr = turn_aligned.select(factors).to_numpy().astype(np.float64)
        obs_arr = (
            ret_aligned.select(factors).to_numpy().astype(np.float64)
            if objective_mode == "factor_return"
            else ic_aligned.select(factors).to_numpy().astype(np.float64)
        )
    else:
        obs_arr = ic_aligned.select(factors).to_numpy().astype(np.float64)

    times = ic_aligned[time_col].to_list()
    cost_rate = float(one_way_cost_bps) * 1e-4

    w_prev = np.full(
        k,
        float(budget) / k if constraint == "box_simplex" else 0.0,
        dtype=np.float64,
    )
    weight_rows: list[dict[str, Any]] = []
    turnovers: list[float] = []

    for idx in range(lookback + emb, len(times)):
        end_excl = idx - emb
        win_raw = obs_arr[end_excl - lookback : end_excl]
        finite_rows = np.isfinite(win_raw).all(axis=1)
        win = win_raw[finite_rows]
        if len(win) < 4:
            continue

        if objective_mode == "factor_return":
            d_win = (
                disp_arr[end_excl - lookback : end_excl][finite_rows]
                if disp_arr is not None
                else np.ones(1, dtype=np.float64)
            )
            cs_scale = max(0.8 * float(np.nanmean(d_win)), 1e-4)
            win_eff = win / cs_scale
        else:
            d_win = (
                disp_arr[end_excl - lookback : end_excl][finite_rows]
                if disp_arr is not None
                else np.ones(1, dtype=np.float64)
            )
            cs_scale = max(0.8 * float(np.nanmean(d_win)), 1e-4)
            win_eff = win

        if ic_ema_halflife is not None and ic_ema_halflife > 0.0:
            dec = np.exp(-np.log(2.0) / float(ic_ema_halflife))
            ew = np.power(dec, np.arange(len(win_eff) - 1, -1, -1, dtype=np.float64))
            ew /= np.sum(ew)
            mu_hat = np.sum(win_eff * ew[:, None], axis=0)
        else:
            mu_hat = np.mean(win_eff, axis=0)

        # Deduct asset-level transaction cost drag per factor before Empirical Bayes & FISTA
        if cost_rate > 0.0 and turn_arr is not None:
            t_win = turn_arr[end_excl - lookback : end_excl][finite_rows]
            mean_tau = np.nanmean(t_win, axis=0)
            mean_tau = np.where(np.isfinite(mean_tau), mean_tau, 0.0)
            drag_vec = (cost_rate * mean_tau) / cs_scale
            if constraint != "unconstrained" and float(lower_bound) >= 0.0:
                mu_hat = mu_hat - drag_vec
            else:
                mu_hat = np.sign(mu_hat) * np.maximum(np.abs(mu_hat) - drag_vec, 0.0)

        if ic_shrinkage == "empirical_bayes":
            se = np.maximum(np.std(win_eff, axis=0, ddof=1) / np.sqrt(float(len(win_eff))), 1e-8)
            eb_df = empirical_bayes_shrinkage(mu_hat, se, prior="zero")
            mu_hat = eb_df["posterior_mean"].to_numpy().astype(np.float64)

        if cov_estimator == "ledoit_wolf":
            cov_hat, _ = ledoit_wolf_shrinkage(win_eff)
        elif cov_estimator == "marchenko_pastur":
            cov_hat = marchenko_pastur_denoise(win_eff)
        elif cov_estimator == "exponential":
            cov_hat = exponential_covariance(win_eff, half_life=max(4.0, lookback / 2.0))
        else:
            cov_hat = sample_covariance(win_eff)

        opt = solve_fista(
            expected_ic=mu_hat,
            covariance=cov_hat,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            w_prev=w_prev,
            l2_reg=l2_reg,
            constraint=constraint,
            lower_bounds=lower_bound,
            upper_bounds=upper_bound,
            budget=budget,
        )
        w_new = opt.weights
        if no_trade_buffer > 0.0 and float(np.sum(np.abs(w_new - w_prev))) < float(no_trade_buffer):
            w_new = w_prev.copy()

        step_turn = float(np.sum(np.abs(w_new - w_prev)))
        turnovers.append(step_turn)
        w_prev = w_new

        row: dict[str, Any] = {
            time_col: times[idx],
            "weight_turnover": step_turn,
            "kkt_residual": opt.kkt_residual,
            "converged": opt.converged,
        }
        for j, f_name in enumerate(factors):
            row[f"w_{f_name}"] = float(w_new[j])
        weight_rows.append(row)

    if not weight_rows:
        raise ValueError("No out-of-sample periods had at least 4 finite realized observations")

    w_df = pl.DataFrame(weight_rows)
    w_cols = [f"w_{c}" for c in factors]
    joined = eval_df.join(w_df.select([time_col, *w_cols]), on=time_col, how="inner")

    has_any_sig = pl.any_horizontal(
        [pl.col(c).cast(pl.Float64).fill_nan(None).is_not_null() for c in factors]
    )
    w_abs_sum = pl.sum_horizontal([pl.col(f"w_{c}").abs() for c in factors])
    comp_raw = (
        pl.when(has_any_sig)
        .then(
            pl.sum_horizontal(
                [
                    pl.col(c).cast(pl.Float64).fill_nan(0.0).fill_null(0.0) * pl.col(f"w_{c}")
                    for c in factors
                ]
            )
        )
        .otherwise(None)
    )
    ew_raw = pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_nan(None) for c in factors])

    comp_norm = cs_zscore(comp_raw, over=time_col) if normalize_composite else comp_raw
    ew_norm = cs_zscore(ew_raw, over=time_col) if normalize_composite else ew_raw
    comp_final = (
        pl.when(~has_any_sig)
        .then(None)
        .when(w_abs_sum > float(min_weight_norm))
        .then(comp_norm)
        .otherwise(0.0)
        .alias(composite_col)
    )
    out_panel = joined.with_columns(comp_final, ew_norm.alias(equal_weight_col))
    if use_ema:
        out_panel = out_panel.with_columns(
            [pl.col(orig_c).alias(c) for c, orig_c in zip(factors, orig_backup_cols, strict=True)]
        ).drop(orig_backup_cols)
    if asset_col is not None:
        out_panel = out_panel.sort([time_col, asset_col])

    nw_lags = max(1, int(horizon) - 1)
    comp_sum = ic_summary(
        out_panel,
        factor_col=composite_col,
        return_col=return_col,
        group_by=time_col,
        method=ic_method,
        nw_lags=nw_lags,
    )
    ew_sum = ic_summary(
        out_panel,
        factor_col=equal_weight_col,
        return_col=return_col,
        group_by=time_col,
        method=ic_method,
        nw_lags=nw_lags,
    )

    return WalkForwardAllocationResult(
        data=out_panel,
        weights_history=w_df,
        ic_matrix=ic_aligned,
        composite_ic_summary=comp_sum,
        equal_weight_ic_summary=ew_sum,
        mean_weight_turnover=float(np.mean(turnovers)),
        embargo_bars=emb,
    )
