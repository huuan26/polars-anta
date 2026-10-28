# Contributing to polars-anta

Thank you for your interest in contributing to **polars-anta**! We are building the fastest, most ergonomic technical analysis and factor library for the Polars ecosystem.

---

## Project Governance & Roles

To ensure architectural consistency, numerical rigor, and high performance standards, `polars-anta` defines clear project roles:

* **👑 Creator & Lead Maintainer (Owner)**: **Phan Bùi Hữu Ân** ([@huuan26](https://github.com/huuan26))
  * Sole authority over project vision, architectural design, API conventions, versioning, and PyPI releases.
  * Designated as the global **Code Owner** (see [`.github/CODEOWNERS`](.github/CODEOWNERS)); all pull requests require review and final approval from `@huuan26` prior to merging into `master`.
* **🤝 Community Contributors**:
  * Developers, quants, and traders who contribute new indicators, performance optimizations, bug fixes, test parity suites, and tutorial notebooks.
  * All merged contributions are highlighted in official GitHub release notes and community discussions.

---

## How Can You Contribute?

1. **Implement New Indicators**: Add indicators from financial quantitative literature.
2. **Performance Optimizations**: Benchmark and refine Polars expressions for maximum SIMD efficiency.
3. **Bug Reports & Feature Requests**: Open detailed issues with minimal reproducible examples.
4. **Documentation & Examples**: Improve tutorials, use cases, and notebooks.

---

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/polars-anta.git
   cd polars-anta
   ```

2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev,benchmark,docs]"
   ```

3. Run the test suite:
   ```bash
   pytest
   ```

4. Format and lint your code:
   ```bash
   ruff check --fix .
   ruff format .
   ```

---

## Design Principles for New Indicators

When implementing indicators:
- **Zero Python loops**: All computations must be expressed as pure `pl.Expr` expressions whenever possible.
- **Support Grouping (`.over()`)**: Ensure expressions work cleanly within Polars window contexts across symbols.
- **Standard Formulas**: Match standard financial literature (Wilder, Kaufman, Bollinger, Murphy, WorldQuant).
- **Comprehensive Unit Tests**: Add tests in `tests/` checking correctness, edge cases, and namespace integration.

---

## Pull Request Review Workflow

1. Push your branch to your fork: `git push origin feat/your-feature`.
2. Open a Pull Request targeting `master` on `huuan26/polars-anta`.
3. Through GitHub's [`.github/CODEOWNERS`](.github/CODEOWNERS) system, `@huuan26` is automatically designated as the lead reviewer.
4. Once all CI checks (multi-platform tests across Python 3.10-3.13 and Ruff linting) pass and the PR is approved by `@huuan26`, your contribution will be merged into the project!
