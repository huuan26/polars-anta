"""Analytical nonlinear shrinkage (Ledoit-Wolf QIS) estimator tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_anta.covariance import (
    _epanechnikov_hilbert,
    ledoit_wolf_shrinkage,
    nearest_spd,
    nonlinear_shrinkage_qis,
    sample_covariance,
)


def _numeric_hilbert(x: float, n: int = 400001) -> float:
    """Principal-value integral (1/pi) * int k(t)/(x - t) dt on |t| <= 1."""
    t = np.linspace(-1.0, 1.0, n)
    k = 0.75 * (1.0 - t * t)
    return float(np.trapezoid(k / (x - t), t) / np.pi)


@pytest.mark.parametrize("x", [-2.5, -1.4, 1.3, 2.5])
def test_epanechnikov_hilbert_matches_quadrature(x: float) -> None:
    formula = float(_epanechnikov_hilbert(np.array([x]))[0])
    assert formula == pytest.approx(_numeric_hilbert(x), abs=5e-4)


def _spiked_sample(seed: int, k_cols: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spikes = np.array([20.0, 10.0, 5.0, 3.0, 2.0])
    spectrum = np.ones(k_cols)
    spectrum[: spikes.size] = spikes
    basis, _ = np.linalg.qr(rng.normal(size=(k_cols, k_cols)))
    truth = basis @ np.diag(spectrum) @ basis.T
    t_obs = int(round(k_cols / ratio))
    chol = np.linalg.cholesky(truth + 1e-10 * np.eye(k_cols))
    x = rng.normal(size=(t_obs, k_cols)) @ chol.T
    return truth, x


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8])
def test_qis_strictly_beats_baselines_on_spiked_spectrum(ratio: float) -> None:
    errors = {"sample": [], "lw": [], "qis": []}
    conds = {"sample": [], "lw": [], "qis": []}
    for seed in range(5):
        truth, x = _spiked_sample(seed, 60, ratio)
        est_sample = sample_covariance(x)
        est_lw, _ = ledoit_wolf_shrinkage(x)
        est_qis = nonlinear_shrinkage_qis(x)
        for name, est in (("sample", est_sample), ("lw", est_lw), ("qis", est_qis)):
            errors[name].append(np.linalg.norm(est - truth) / np.linalg.norm(truth))
            conds[name].append(np.linalg.cond(est))
    mean_err = {k: float(np.mean(v)) for k, v in errors.items()}
    med_cond = {k: float(np.median(v)) for k, v in conds.items()}
    assert mean_err["qis"] < mean_err["sample"]
    assert mean_err["qis"] < mean_err["lw"]
    assert med_cond["qis"] < med_cond["sample"]


@pytest.mark.parametrize("ratio", [0.2, 0.8, 1.2])
def test_qis_output_is_spd_for_all_regimes(ratio: float) -> None:
    _, x = _spiked_sample(3, 40, ratio)
    est = nonlinear_shrinkage_qis(x)
    assert est.shape == (40, 40)
    assert np.all(np.isfinite(est))
    assert np.allclose(est, est.T, atol=1e-10)
    assert float(np.min(np.linalg.eigvalsh(est))) > 0.0
    assert np.allclose(est, nearest_spd(est), atol=1e-8)


def test_qis_preserves_trace() -> None:
    _, x = _spiked_sample(5, 30, 0.5)
    est = nonlinear_shrinkage_qis(x)
    x_c = x - np.mean(x, axis=0, keepdims=True)
    mle_trace = float(np.trace((x_c.T @ x_c) / float(x.shape[0])))
    assert float(np.trace(est)) == pytest.approx(mle_trace, rel=1e-6)


def test_qis_rank_deficient_regime_beats_sample_frobenius() -> None:
    truth, x = _spiked_sample(9, 50, 1.25)
    est_qis = nonlinear_shrinkage_qis(x)
    est_sample = sample_covariance(x)
    err_qis = np.linalg.norm(est_qis - truth) / np.linalg.norm(truth)
    err_sample = np.linalg.norm(est_sample - truth) / np.linalg.norm(truth)
    assert err_qis < err_sample
    assert float(np.min(np.linalg.eigvalsh(est_qis))) > 0.0


def test_qis_accepts_polars_frame_and_exclude() -> None:
    _, x = _spiked_sample(1, 8, 0.5)
    frame = pl.DataFrame({f"a{i}": x[:, i] for i in range(8)})
    frame = frame.with_columns(pl.Series("label", ["x"] * frame.height))
    est = nonlinear_shrinkage_qis(frame, exclude=("label",))
    assert est.shape == (8, 8)
    assert np.all(np.linalg.eigvalsh(est) > 0.0)


def test_qis_handles_nonfinite_rows_and_single_column() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 3))
    x[0, 1] = np.nan
    est = nonlinear_shrinkage_qis(x)
    assert est.shape == (3, 3)
    assert np.all(np.isfinite(est))

    col = rng.normal(size=(50, 1))
    est1 = nonlinear_shrinkage_qis(col)
    assert est1.shape == (1, 1)
    assert est1[0, 0] == pytest.approx(float(np.var(col[:, 0], ddof=1)), rel=1e-6)


def test_qis_rejects_bad_bandwidth() -> None:
    x = np.random.default_rng(0).normal(size=(30, 4))
    with pytest.raises(ValueError):
        nonlinear_shrinkage_qis(x, bandwidth=0.0)
    with pytest.raises(ValueError):
        nonlinear_shrinkage_qis(x, bandwidth=np.nan)


def test_qis_is_deterministic() -> None:
    _, x = _spiked_sample(2, 12, 0.6)
    first = nonlinear_shrinkage_qis(x)
    second = nonlinear_shrinkage_qis(x)
    assert np.array_equal(first, second)
