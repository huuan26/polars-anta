"""Command-line entry point for the Vietnamese Equity Factor Zoo falsification study.

Usage:
    python -m research.factor_zoo.run --stage discovery
    python -m research.factor_zoo.run --stage holdout --unlock-holdout
    python -m research.factor_zoo.run --stage exploratory
"""

from __future__ import annotations

import argparse
import json
import sys

from research.factor_zoo.exploratory import run_exploratory
from research.factor_zoo.pipeline import run_discovery, run_holdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Vietnamese Equity Factor Zoo falsification pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=("discovery", "holdout", "exploratory"),
        required=True,
        help="Pipeline stage to execute ('discovery', 'holdout', or 'exploratory').",
    )
    parser.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="Explicitly unlock the 2022+ holdout period (requires committed PREREGISTRATION.md).",
    )
    args = parser.parse_args(argv)

    if args.stage == "discovery":
        summary = run_discovery()
        print(json.dumps(summary["funnel"], indent=2))
        return 0

    if args.stage == "exploratory":
        summary = run_exploratory()
        print(json.dumps(summary, indent=2))
        return 0

    try:
        summary = run_holdout(unlock_holdout=args.unlock_holdout)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary["funnel_final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
