"""Unit and mathematical invariant tests for polars_anta.conditioning."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import polars as pl
import pytest

from polars_anta.conditioning import (
    collinearity_summary,
    cs_gaussianize,
    lowdin_matrix,
    neutralize_panel,
    orthogonalize_panel,
)


def test_cs_gaussianize_matches_exact_normal_quantile() -> None:
    """Acklam rational approximation must match exact inverse normal CDF across tails and center."""
    n = 500
    vals = np.exp(np.linspace(-5.0, 8.0, n))
    df = pl.DataFrame({"date": ["2026-01-02"] * n, "x": vals})
    out = df.with_columns(cs_gaussianize("x", over=["date"], clip_bound=None).alias("g"))[
        "g"
    ].to_numpy()

    nd = NormalDist()
    expected = np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)], dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-7)


def test_cs_gaussianize_handles_nan_null_and_clip() -> None:
    df = pl.DataFrame(
        {
            "date": ["d1"] * 6 + ["d2"],
            "x": [1.0, 2.0, 3.0, 1000.0, float("nan"), None, 5.0],
        }
    )
    res = df.with_columns(cs_gaussianize("x", over="date", clip_bound=1.5).alias("g"))
    g = res["g"].to_list()
    assert g[4] is None
    assert g[5] is None
    assert g[6] is None
    valid_d1 = [v for v in g[:4] if v is not None]
    assert len(valid_d1) == 4
    assert np.isclose(sum(valid_d1), 0.0, atol=1e-12)
    assert all(-1.5 <= v <= 1.5 for v in valid_d1)


def test_lowdin_matrix_orthogonality_and_permutation_symmetry() -> None:
    rng = np.random.default_rng(42)
    base = rng.standard_normal((120, 1))
    mat = 0.75 * base + 0.25 * rng.standard_normal((120, 4))

    ortho = lowdin_matrix(mat, standardize=True)
    gram = (ortho.T @ ortho) / (len(mat) - 1)
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-11)

    perm = [2, 0, 3, 1]
    ortho_perm = lowdin_matrix(mat[:, perm], standardize=True)
    np.testing.assert_allclose(ortho_perm, ortho[:, perm], atol=1e-12)

    batch = np.stack([mat, mat[:, perm]], axis=0)
    batch_out = lowdin_matrix(batch, standardize=True)
    np.testing.assert_allclose(batch_out[0], ortho, atol=1e-12)
    np.testing.assert_allclose(batch_out[1], ortho_perm, atol=1e-12)


def test_neutralize_panel_simultaneous_sector_and_risk_fwl_invariant() -> None:
    """Verify exact FWL orthogonality to BOTH categorical sectors and continuous risk exposures."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    sectors = ["Fin", "Tech", "Ind", "Cons"]
    for d in ("2026-01-02", "2026-01-05", "2026-01-06"):
        n_t = 80
        beta = rng.normal(1.0, 0.4, size=n_t)
        size = rng.normal(10.0, 2.0, size=n_t)
        sec_idx = rng.integers(0, len(sectors), size=n_t)
        sec_effect = np.array([0.8, -0.5, 0.3, -0.6])[sec_idx]
        pure_a = rng.standard_normal(n_t)
        f1 = 2.0 * sec_effect + 1.5 * beta - 0.9 * size + 0.3 * pure_a
        f2 = -1.2 * sec_effect + 0.7 * beta + 1.1 * size + rng.standard_normal(n_t)
        for i in range(n_t):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{i:03d}",
                    "sector": sectors[int(sec_idx[i])],
                    "beta": float(beta[i]),
                    "log_mcap": float(size[i]),
                    "f1": float(f1[i]),
                    "f2": float(f2[i]),
                }
            )
    panel = pl.DataFrame(rows)
    panel = (
        panel.with_row_index("_idx")
        .with_columns(
            pl.when(pl.col("_idx") == 3).then(float("nan")).otherwise(pl.col("f1")).alias("f1"),
            pl.when(pl.col("_idx") == 10).then(None).otherwise(pl.col("beta")).alias("beta"),
        )
        .drop("_idx")
    )

    out_eager = neutralize_panel(
        panel,
        factor_cols=["f1", "f2"],
        time_col="date",
        risk_cols=["beta", "log_mcap"],
        group_col="sector",
        standardize=True,
    )
    out_lazy = neutralize_panel(
        panel.lazy(),
        factor_cols=["f1", "f2"],
        time_col="date",
        risk_cols=["beta", "log_mcap"],
        group_col="sector",
        standardize=True,
    ).collect()
    assert out_eager.columns == out_lazy.columns
    for col_name in ("f1_neut", "f2_neut"):
        np.testing.assert_allclose(
            out_eager[col_name].fill_null(np.nan).to_numpy(),
            out_lazy[col_name].fill_null(np.nan).to_numpy(),
            atol=1e-12,
            equal_nan=True,
        )

    for d in ("2026-01-02", "2026-01-05", "2026-01-06"):
        for col_n in ("f1_neut", "f2_neut"):
            sub = out_eager.filter((pl.col("date") == d) & pl.col(col_n).is_not_null())
            vec = sub[col_n].to_numpy()
            for s_name in sectors:
                mask = (sub["sector"] == s_name).to_numpy().astype(np.float64)
                assert abs(float(vec @ mask)) < 1e-10
            assert abs(float(vec @ sub["beta"].to_numpy())) < 1e-10
            assert abs(float(vec @ sub["log_mcap"].to_numpy())) < 1e-10
            assert np.isclose(np.std(vec, ddof=1), 1.0, atol=1e-10)


def test_orthogonalize_panel_unbalanced_permutation_and_duplicate_guard() -> None:
    """Verify batched Lowdin orthogonality, order symmetry, and resilience to duplicate columns."""
    rng = np.random.default_rng(123)
    frames: list[pl.DataFrame] = []
    for d, n_t in [("d1", 60), ("d2", 45), ("d3", 2)]:
        common = rng.standard_normal(n_t)
        f1 = common + 0.3 * rng.standard_normal(n_t)
        f2 = 0.8 * common + 0.5 * rng.standard_normal(n_t)
        f3 = -0.6 * common + 0.7 * rng.standard_normal(n_t)
        frames.append(pl.DataFrame({"date": [d] * n_t, "f1": f1, "f2": f2, "f3": f3}))
    panel = pl.concat(frames)

    out_abc = orthogonalize_panel(panel, ["f1", "f2", "f3"], time_col="date", method="lowdin")
    out_cab = orthogonalize_panel(panel, ["f3", "f1", "f2"], time_col="date", method="lowdin")

    for d, n_t in [("d1", 60), ("d2", 45)]:
        sub_abc = (
            out_abc.filter(pl.col("date") == d)
            .select(["f1_ortho", "f2_ortho", "f3_ortho"])
            .to_numpy()
        )
        sub_cab = (
            out_cab.filter(pl.col("date") == d)
            .select(["f1_ortho", "f2_ortho", "f3_ortho"])
            .to_numpy()
        )
        gram = (sub_abc.T @ sub_abc) / (n_t - 1)
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(sub_abc, sub_cab, atol=1e-11)

    assert out_abc.filter(pl.col("date") == "d3")["f1_ortho"].null_count() == 2

    dup_panel = panel.filter(pl.col("date") == "d1").with_columns(
        pl.col("f1").alias("f1_dup"),
        pl.lit(0.0).alias("f_zero"),
    )
    out_dup = orthogonalize_panel(
        dup_panel,
        ["f1", "f1_dup", "f2", "f_zero"],
        time_col="date",
        method="lowdin_corr",
    )
    assert out_dup["f2_ortho"].null_count() == 0
    assert np.std(out_dup["f2_ortho"].to_numpy(), ddof=1) > 0.5

    out_gs = orthogonalize_panel(
        panel, ["f1", "f2", "f3"], time_col="date", method="gram_schmidt", standardize=False
    )
    sub_gs = (
        out_gs.filter(pl.col("date") == "d1")
        .select(["f1_ortho", "f2_ortho", "f3_ortho"])
        .to_numpy()
    )
    off_diag = (sub_gs.T @ sub_gs) - np.diag(np.diag(sub_gs.T @ sub_gs))
    np.testing.assert_allclose(off_diag, np.zeros((3, 3)), atol=1e-10)


def test_collinearity_summary_detects_redundancy_and_orthogonality() -> None:
    rng = np.random.default_rng(99)
    z1 = rng.standard_normal(300)
    z2 = rng.standard_normal(300)
    z3 = rng.standard_normal(300)
    df = pl.DataFrame(
        {
            "date": ["d1"] * 150 + ["d2"] * 150,
            "a": z1,
            "b": z1 + 0.05 * z3,  # High collinearity with a (VIF > 100), yet full rank 3
            "c": z2,
        }
    )
    raw_sum = collinearity_summary(df, ["a", "b", "c"], time_col="date")
    assert raw_sum["effective_rank"][0] < 2.3
    assert raw_sum.filter(pl.col("factor") == "a")["vif"][0] > 100.0

    ortho_df = orthogonalize_panel(df, ["a", "b", "c"], time_col="date", method="lowdin")
    ortho_sum = collinearity_summary(ortho_df, ["a_ortho", "b_ortho", "c_ortho"], time_col="date")
    np.testing.assert_allclose(ortho_sum["vif"].to_numpy(), np.ones(3), atol=1e-6)
    assert np.isclose(ortho_sum["effective_rank"][0], 3.0, atol=1e-6)

    # Also verify pooled (time_col=None) summary path
    pooled = collinearity_summary(ortho_df, ["a_ortho", "b_ortho", "c_ortho"], time_col=None)
    assert np.isclose(pooled["effective_rank"][0], 3.0, atol=1e-2)


def test_subspace_invariance_neutralize_all_before_orthogonalize() -> None:
    """Lowdin preserves exact sector/risk neutrality iff all input columns were FWL-neutralized."""
    rng = np.random.default_rng(515)
    n_t = 60
    sectors = [f"S{i % 3}" for i in range(n_t)]
    beta = rng.normal(1.0, 0.35, size=n_t)
    sec_num = np.array([1.0 if s == "S0" else (-0.8 if s == "S1" else 0.2) for s in sectors])
    df = pl.DataFrame(
        {
            "date": ["2026-03-02"] * n_t,
            "sector": sectors,
            "beta": beta,
            "f1": 1.5 * sec_num + 0.9 * beta + rng.standard_normal(n_t),
            "f2": -1.1 * sec_num + 1.2 * beta + rng.standard_normal(n_t),
        }
    )
    neut = neutralize_panel(
        df, ["f1", "f2"], time_col="date", risk_cols=["beta"], group_col="sector"
    )
    clean_ortho = orthogonalize_panel(
        neut, ["f1_neut", "f2_neut"], time_col="date", method="lowdin"
    )
    for c in ("f1_neut_ortho", "f2_neut_ortho"):
        v = clean_ortho[c].to_numpy()
        assert abs(float(v @ beta)) < 1e-10
        for s_lbl in ("S0", "S1", "S2"):
            assert abs(float(v @ (np.array(sectors) == s_lbl).astype(np.float64))) < 1e-10

    # Mixing an un-neutralized column into Lowdin contaminates the neutralized column's subspace
    mixed_ortho = orthogonalize_panel(neut, ["f1_neut", "f2"], time_col="date", method="lowdin")
    assert abs(float(mixed_ortho["f1_neut_ortho"].to_numpy() @ beta)) > 0.5


def test_conditioning_validation_errors() -> None:
    df = pl.DataFrame({"date": ["d1", "d1"], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="clip_bound"):
        cs_gaussianize("x", clip_bound=0.0)
    with pytest.raises(ValueError, match="factor_cols"):
        neutralize_panel(df, [])
    with pytest.raises(ValueError, match="ridge"):
        neutralize_panel(df, ["x"], ridge=-0.1)
    with pytest.raises(ValueError, match="factor_cols"):
        orthogonalize_panel(df, [])
    with pytest.raises(ValueError, match="method"):
        orthogonalize_panel(df, ["x"], method="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rcond"):
        orthogonalize_panel(df, ["x"], rcond=-1.0)
    with pytest.raises(ValueError, match="rcond"):
        lowdin_matrix(np.eye(3), rcond=-1.0)
    with pytest.raises(ValueError, match="2D or 3D"):
        lowdin_matrix(np.ones(3))
    with pytest.raises(ValueError, match="NaN or Inf"):
        lowdin_matrix(np.array([[1.0, float("nan")], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="Number of rows"):
        lowdin_matrix(np.ones((2, 4)))
    with pytest.raises(ValueError, match="factor_cols"):
        collinearity_summary(df, [])
    with pytest.raises(ValueError, match="At least 2"):
        collinearity_summary(pl.DataFrame({"x": [1.0]}), ["x"])
