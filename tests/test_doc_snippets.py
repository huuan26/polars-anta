"""Syntax validation for all ```python code blocks in README.md and docs/**/*.md."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

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
