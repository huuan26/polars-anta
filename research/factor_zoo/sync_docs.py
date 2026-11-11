"""Deterministically generate docs/research/factor-zoo.md from research/factor_zoo/REPORT.md (Plan E §5 E-C7).

Transforms relative repository links and figure paths for MkDocs strict builds, and copies
all figures from research/factor_zoo/results/figures/ into docs/assets/research/.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from research.factor_zoo.config import REPO_ROOT, RESULTS_DIR

REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"
DOCS_TARGET_PATH = REPO_ROOT / "docs" / "research" / "factor-zoo.md"
DOCS_FIGURES_DIR = REPO_ROOT / "docs" / "assets" / "research"
GITHUB_BLOB_BASE = "https://github.com/huuan26/polars-anta/blob/master/research/factor_zoo"


def sync_docs(
    report_path: Path = REPORT_PATH,
    target_path: Path = DOCS_TARGET_PATH,
    figures_dir: Path = RESULTS_DIR / "figures",
    docs_figures_dir: Path = DOCS_FIGURES_DIR,
) -> dict[str, object]:
    """Synchronize REPORT.md and rendered figures to docs/research/factor-zoo.md."""
    text = report_path.read_text(encoding="utf-8")

    # 1. Rewrite figure image paths: (results/figures/X.png) -> (../assets/research/X.png)
    text = re.sub(
        r"\]\(results/figures/([^)]+\.png)\)",
        r"](../assets/research/\1)",
        text,
    )

    # 2. Rewrite links to ../../docs/... -> ../... (relative to docs/research/)
    text = re.sub(
        r"\]\(\.\./\.\./docs/([^)]+)\)",
        r"](../\1)",
        text,
    )

    # 3. Rewrite sibling files and results/ artifacts in research/factor_zoo/ to GitHub URLs
    text = re.sub(
        r"\]\(((?:PREREGISTRATION\.md|config\.py|registry\.py|pipeline\.py|data\.py|exploratory\.py|render\.py|run\.py|results/[^)]+))\)",
        rf"]({GITHUB_BLOB_BASE}/\1)",
        text,
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8")

    docs_figures_dir.mkdir(parents=True, exist_ok=True)
    copied_figures: list[str] = []
    if figures_dir.exists():
        for png_path in sorted(figures_dir.glob("*.png")):
            dest = docs_figures_dir / png_path.name
            shutil.copy2(png_path, dest)
            copied_figures.append(png_path.name)

    return {
        "source": str(report_path),
        "target": str(target_path),
        "copied_figures": copied_figures,
    }


if __name__ == "__main__":
    import json

    out = sync_docs()
    print(json.dumps(out, indent=2))
