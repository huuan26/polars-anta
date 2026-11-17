# Contributing to `polars-anta`

We welcome contributions from quantitative researchers, Python developers, and the open-source community!

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/huuan26/polars-anta.git
   cd polars-anta
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e .[dev,benchmark,docs]
   ```

---

## Guidelines for New Indicators

When adding a new indicator:

1. **Pure Expression First**: Implement using `pl.Expr` wherever possible. Avoid Python loops over rows.
2. **Support String or Expr**: Use `parse_expr` from `polars_anta.core` so users can pass either column names (`"close"`) or existing expressions (`pl.col("close")`).
3. **Register on Dual Namespaces**: Indicators added to `TechnicalAnalysisExpr` (`polars_anta/namespace/_expr_ns.py`) and `_FRAME_EXPR_SPECS` (`polars_anta/namespace/_frame_specs.py`) are automatically available under both `.anta` and `.ta`.
4. **Multi-Asset Compatibility**: Ensure the indicator executes without errors inside `.over("ticker")`.
5. **Structured Outputs**: If the indicator produces multiple values (like bands or oscillators with signals), return a typed `pl.Struct`.
6. **Comprehensive Tests**: Add test cases to the appropriate file in `tests/`. Verify test coverage remains at or above 95%:
   ```bash
   pytest -v
   ```
7. **Code Quality & Static Typing**: Ensure linter, formatter, and static type checker pass without errors:
   ```bash
   ruff check .
   ruff format --check .
   mypy src/polars_anta
   ```

---

## Building Documentation Locally

Preview the documentation website with hot reload:

```bash
mkdocs serve
```

Build the static HTML site:

```bash
mkdocs build
```
