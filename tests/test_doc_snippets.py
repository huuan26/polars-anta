"""Syntax validation for all ```python code blocks in README.md and docs/**/*.md."""

from __future__ import annotations

import ast
import json
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_anta  # noqa: F401 - triggers namespace registration

REPO_ROOT = Path(__file__).resolve().parents[1]


def _collect_python_blocks() -> list[tuple[str, str]]:
    """Return a list of (file:start_line, code_str) for every ```python block."""
    md_files = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    blocks: list[tuple[str, str]] = []

    for md_path in md_files:
        if not md_path.exists():
            continue
        rel_path = md_path.relative_to(REPO_ROOT).as_posix()
        lines = md_path.read_text(encoding="utf-8").splitlines()

        in_python_block = False
        block_start_line = 0
        block_lines: list[str] = []
        fence_marker = ""

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not in_python_block:
                if stripped.startswith("```python") or stripped.startswith("````python"):
                    in_python_block = True
                    block_start_line = idx
                    block_lines = []
                    fence_marker = "````" if stripped.startswith("````") else "```"
            else:
                if stripped == fence_marker:
                    in_python_block = False
                    blocks.append(
                        (f"{rel_path}:{block_start_line}", textwrap.dedent("\n".join(block_lines)))
                    )
                else:
                    block_lines.append(line)

    return blocks


_PYTHON_BLOCKS = _collect_python_blocks()


@pytest.mark.parametrize(
    ("location", "code"),
    _PYTHON_BLOCKS,
    ids=[loc for loc, _ in _PYTHON_BLOCKS],
)
def test_markdown_python_snippet_syntax(location: str, code: str) -> None:
    """Every ```python fenced block in README.md and docs/ must be valid Python syntax."""
    try:
        ast.parse(code, filename=location)
    except SyntaxError as exc:
        pytest.fail(
            f"SyntaxError in markdown snippet at {location} "
            f"(snippet line {exc.lineno}, col {exc.offset}): {exc.msg}\n{exc.text}"
        )


def test_readme_and_alpha_engineering_snippets_execute() -> None:
    """Every Python snippet in README.md and alpha-engineering.md must execute without error."""
    readme_blocks = [(loc, code) for loc, code in _PYTHON_BLOCKS if loc.startswith("README.md:")]
    alpha_blocks = [
        (loc, code)
        for loc, code in _PYTHON_BLOCKS
        if loc.startswith("docs/indicators/alpha-engineering.md:")
    ]
    assert len(readme_blocks) == 5
    assert len(alpha_blocks) == 1

    # 1. Standalone Quick Start & docs snippets execute in an empty namespace
    for loc, code in (readme_blocks[0], alpha_blocks[0]):
        exec(compile(code, loc, "exec"), {})

    # 2. Core Workflows 1-4 execute with canonical schema fixtures
    rng = np.random.default_rng(42)
    n_dates, n_assets = 90, 20
    dates = np.repeat(np.arange(n_dates), n_assets)
    tickers = np.tile([f"T{i:02d}" for i in range(n_assets)], n_dates)
    closes = 100.0 + rng.normal(0.0, 2.0, n_dates * n_assets)
    a1 = rng.standard_normal(n_dates * n_assets)
    a2 = rng.standard_normal(n_dates * n_assets)
    a3 = rng.standard_normal(n_dates * n_assets)
    panel_alpha = pl.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "sector": np.tile([f"S{i % 4}" for i in range(n_assets)], n_dates),
            "beta": rng.uniform(0.7, 1.3, n_dates * n_assets),
            "close": closes,
            "volume": rng.uniform(1e5, 1e6, n_dates * n_assets),
            "alpha_1": a1,
            "alpha_2": a2,
            "alpha_3": a3,
            "fwd_1": 0.004 * a1 + 0.003 * a3 + rng.normal(0.0, 0.015, n_dates * n_assets),
        }
    )
    ns_alpha: dict[str, Any] = {"pl": pl, "pa": polars_anta, "panel": panel_alpha}
    exec(compile(readme_blocks[1][1], readme_blocks[1][0], "exec"), ns_alpha)

    n_bar = 60
    t0 = datetime(2026, 1, 2, 9, 0)
    ts = [t0 + timedelta(minutes=i) for i in range(n_bar)]
    raw_contracts = pl.DataFrame(
        {
            "timestamp": ts,
            "trading_date": [t.date() for t in ts],
            "contract": ["F1"] * 30 + ["F2"] * 30,
            "open": 1200.0 + np.cumsum(rng.normal(0.0, 1.0, n_bar)),
            "high": 1202.0 + np.cumsum(rng.normal(0.0, 1.0, n_bar)),
            "low": 1198.0 + np.cumsum(rng.normal(0.0, 1.0, n_bar)),
            "close": 1200.0 + np.cumsum(rng.normal(0.0, 1.0, n_bar)),
            "volume": rng.uniform(500.0, 1500.0, n_bar),
            "bid_price": 1199.9 + rng.normal(0.0, 0.2, n_bar),
            "bid_size": rng.uniform(10.0, 50.0, n_bar),
            "ask_price": 1200.1 + rng.normal(0.0, 0.2, n_bar),
            "ask_size": rng.uniform(10.0, 50.0, n_bar),
            "spot_close": 1195.0 + np.cumsum(rng.normal(0.0, 1.0, n_bar)),
            "days_to_expiry": np.linspace(30.0, 15.0, n_bar),
            "open_interest": 10_000.0 + np.cumsum(rng.normal(0.0, 20.0, n_bar)),
        }
    )
    df_bt = pl.DataFrame(
        {
            "ticker": ["F1M"] * n_bar,
            "open": 100.0 + np.cumsum(rng.normal(0.0, 1.0, n_bar)),
            "close": 100.0 + np.cumsum(rng.normal(0.0, 1.0, n_bar)),
            "primary_signal": np.sign(rng.normal(0.0, 1.0, n_bar)),
            "is_atc_bar": [False] * n_bar,
        }
    )
    ns_wf: dict[str, Any] = {
        "pl": pl,
        "pa": polars_anta,
        "raw_contracts": raw_contracts,
        "df": df_bt,
    }
    for loc, code in readme_blocks[2:]:
        exec(compile(code, loc, "exec"), ns_wf)
        if loc == readme_blocks[3][0]:
            assert ns_wf["ml_dataset"].height == n_bar


def _notebook_code_cells(path: Path) -> list[str]:
    """Executable source of every code cell in a notebook (stdlib json, no nbformat needed)."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells: list[str] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        lines = [
            line
            for line in "".join(cell["source"]).splitlines()
            if not line.strip().startswith(("!pip", "%"))
        ]
        code = "\n".join(lines).strip()
        if code:
            cells.append(code)
    return cells


def test_example_08_vpin_order_flow_notebook_executes() -> None:
    """Issue #59: the VPIN / OFI tutorial must run end to end on its synthetic tape."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    notebook = REPO_ROOT / "examples" / "08_crypto_vpin_order_flow_toxicity.ipynb"
    cells = _notebook_code_cells(notebook)
    assert len(cells) >= 8

    namespace: dict[str, Any] = {}
    for index, code in enumerate(cells):
        exec(compile(code, f"{notebook.name}:cell{index}", "exec"), namespace)

    bars = namespace["bars"]
    assert bars.height > 150
    assert {"vpin", "vpin_threshold", "toxic"} <= set(bars.columns)
    assert int(bars["toxic"].sum()) > 0

    vpin = bars["vpin"].drop_nulls()
    assert vpin.len() > 0
    assert vpin.min() >= 0.0
    assert vpin.max() <= 1.0

    ofi = namespace["ofi"]
    assert {"ofi", "ofi_norm", "fwd_ret_60s"} <= set(ofi.columns)
    corr = namespace["corr_60s"]
    assert corr is not None and -1.0 <= corr <= 1.0
    assert namespace["quintiles"].height == 5

    import matplotlib.pyplot as plt

    plt.close("all")
