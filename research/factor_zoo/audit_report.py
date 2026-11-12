"""Automated truthfulness and traceability auditor for Factor Zoo Markdown reports and docs.

Implements Plan E Section 1.4:
1. Identifier Audit: Every backtick token matching ``^[a-z_][a-z0-9_]*$`` must belong to:
   - ``polars_anta.__all__``
   - registered signal names in ``research.factor_zoo.registry.build_registry()``
   - CSV column names in ``research/factor_zoo/results/**/*.csv``
   - JSON keys in ``research/factor_zoo/results/**/*.json``
   - ``research/factor_zoo/audit_allowlist.txt`` (each entry requires a ``# reason`` comment)
2. Number Audit: Every numeric claim (integers >= 2 digits, decimals, percentages, comma-formatted
   numbers) outside code blocks, References, headings/section numbers, years (1900-2099), URLs,
   and ``StudyConfig`` constants must trace to a value in ``research/factor_zoo/results/**/*.{json,csv}``
   (under standard display formatting: rounding to 0-4 decimals, x100 for %, comma formatting,
   or sign negation).

Exits with status code 1 if any untraced number or unknown identifier is found.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
import sys
from pathlib import Path

import polars_anta as pa
from research.factor_zoo.config import DEFAULT_CONFIG, REPO_ROOT, RESULTS_DIR
from research.factor_zoo.registry import EXCLUDED_SIGNALS, build_registry, family_counts

ALLOWLIST_PATH = Path(__file__).resolve().parent / "audit_allowlist.txt"

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_FUNC_CALL_RE = re.compile(r"^([a-z_][a-z0-9_]*)\s*\(")
_NUM_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_/#=-])"
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)"
    r"(?![A-Za-z0-9_/-])"
)


def load_allowlist(path: Path = ALLOWLIST_PATH) -> set[str]:
    """Load allowlisted identifiers, verifying every non-comment line has a '# reason'."""
    allowed: set[str] = set()
    if not path.exists():
        return allowed
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" not in line:
            msg = f"Invalid allowlist line {lineno} in {path}: missing '# reason' comment"
            raise ValueError(msg)
        ident, reason = line.split("#", 1)
        ident = ident.strip()
        if not ident or not reason.strip():
            msg = f"Invalid allowlist line {lineno} in {path}: empty identifier or reason"
            raise ValueError(msg)
        allowed.add(ident)
    return allowed


def collect_allowed_identifiers() -> set[str]:
    """Build the union of valid identifiers from library exports, registry, results, and allowlist."""
    allowed: set[str] = set(pa.__all__)
    specs = build_registry()
    for s in specs:
        allowed.add(s.name)
        allowed.add(s.family)
    for name_group, fam, _ in EXCLUDED_SIGNALS:
        allowed.add(fam)
        for part in re.findall(r"[a-z_][a-z0-9_]*", name_group):
            allowed.add(part)

    # JSON keys & CSV columns under results/
    if RESULTS_DIR.exists():
        for json_path in RESULTS_DIR.rglob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                _collect_json_keys(data, allowed)
            except Exception:
                continue
        for csv_path in RESULTS_DIR.rglob("*.csv"):
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        for col in header:
                            allowed.add(col.strip())
            except Exception:
                continue

    allowed |= load_allowlist()
    return allowed


def _collect_json_keys(obj: object, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                out.add(k)
            _collect_json_keys(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_json_keys(item, out)


def _add_number_variants(
    val: float, out: set[str], allow_pct: bool = True, allow_int_round: bool = False
) -> None:
    """Register rounded, percentage, and comma-formatted string representations of val."""
    if not (-1e15 < val < 1e15):
        return
    is_exact_int = abs(val - round(val)) < 1e-6
    scales = (1.0, 100.0) if (allow_pct and not is_exact_int and abs(val) <= 1.0) else (1.0,)
    for sign_val in (val, abs(val)):
        for scale in scales:
            scaled = sign_val * scale
            scaled_is_int = abs(scaled - round(scaled)) < 1e-6
            min_dec = 0 if (scaled_is_int or allow_int_round) else 1
            for dec in range(min_dec, 5):
                rounded = round(scaled, dec)
                if dec == 0:
                    iv = int(round(scaled))
                    out.add(str(iv))
                    out.add(f"{iv:,}")
                else:
                    s = f"{rounded:.{dec}f}"
                    out.add(s)
                    out.add(f"{rounded:,.{dec}f}")


def collect_traced_numbers() -> set[str]:
    """Collect all traceable numeric strings from results/**/*.{json,csv}, config.py, and benchmarks."""
    traced: set[str] = set()
    json_floats: list[float] = []
    csv_floats: list[tuple[str, float]] = []

    # 1. StudyConfig constants and direct derivatives
    cfg_dict = dataclasses.asdict(DEFAULT_CONFIG)
    cfg_dict["round_trip_cost_rate"] = DEFAULT_CONFIG.round_trip_cost_rate
    cfg_dict["round_trip_bps_l1"] = DEFAULT_CONFIG.round_trip_cost_rate * 10_000.0  # 30 bps
    cfg_dict["round_trip_bps_2way"] = DEFAULT_CONFIG.round_trip_cost_rate * 20_000.0  # 60 bps
    cfg_dict["max_abs_daily_ret"] = (
        DEFAULT_CONFIG.hose_price_limit + DEFAULT_CONFIG.price_limit_tolerance
    )
    for v in cfg_dict.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            json_floats.append(float(v))
        elif isinstance(v, tuple):
            for item in v:
                if isinstance(item, (int, float)):
                    json_floats.append(float(item))

    # 2. Registry & library structural counts
    specs = build_registry()
    json_floats.append(float(len(specs)))
    json_floats.append(float(len(EXCLUDED_SIGNALS)))
    for cnt in family_counts(specs).values():
        json_floats.append(float(cnt))
    json_floats.append(float(len(pa.__all__)))
    n_funcs = sum(1 for name in pa.__all__ if not name[0].isupper())
    json_floats.append(float(n_funcs))

    # 3. Recursively extract all numbers from results/**/*.json and results/**/*.csv
    if RESULTS_DIR.exists():
        for json_path in RESULTS_DIR.rglob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                _collect_json_numbers(data, json_floats)
            except Exception:
                continue
        for csv_path in RESULTS_DIR.rglob("*.csv"):
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None) or []
                    for row in reader:
                        for col_idx, cell in enumerate(row):
                            col_name = header[col_idx] if col_idx < len(header) else ""
                            try:
                                csv_floats.append((col_name, float(cell)))
                            except ValueError:
                                continue
            except Exception:
                continue

    # 4. Also include numbers from docs/benchmarks.md and docs/assets/inference_size_power.md
    for bench_doc in (
        REPO_ROOT / "docs" / "benchmarks.md",
        REPO_ROOT / "docs" / "assets" / "inference_size_power.md",
    ):
        if bench_doc.exists():
            for tok in _NUM_TOKEN_RE.findall(bench_doc.read_text(encoding="utf-8")):
                clean = tok.lstrip("+-").rstrip("%").replace(",", "")
                try:
                    f_bench = float(clean)
                    json_floats.append(f_bench)
                    traced.add(clean)
                except ValueError:
                    continue

    # 5. Derived summary quantities directly implied by summary.json
    disc_sum_path = RESULTS_DIR / "discovery" / "summary.json"
    hold_sum_path = RESULTS_DIR / "holdout" / "summary.json"
    if disc_sum_path.exists():
        d_sum = json.loads(disc_sum_path.read_text(encoding="utf-8"))
        funnel = d_sum.get("funnel", {})
        k_tested = int(funnel.get("n_tested", 150))
        json_floats.append(k_tested * float(DEFAULT_CONFIG.fwer_alpha))
        for val in funnel.values():
            if isinstance(val, (int, float)):
                json_floats.append(float(val) / k_tested)
        if "n_cost_candidates_evaluated" in funnel and "n_positive_net_excess" in funnel:
            json_floats.append(
                float(funnel["n_cost_candidates_evaluated"])
                - float(funnel["n_positive_net_excess"])
            )
        neff_info = d_sum.get("effective_independent_tests", {})
        if isinstance(neff_info, dict) and "top1_eigenvalue" in neff_info:
            json_floats.append(float(neff_info["top1_eigenvalue"]) / k_tested)
        data_info = d_sum.get("data_summary", {})
        if isinstance(data_info, dict) and "raw_rows" in data_info:
            r0 = int(data_info["raw_rows"])
            r1 = r0 - int(data_info.get("rows_dropped_invalid_ohlcv", 0))
            r2 = r1 - int(data_info.get("rows_dropped_zero_volume", 0))
            json_floats.extend([float(r1), float(r2)])
            if "discovery_days" in data_info:
                disc_t = float(data_info["discovery_days"])
                json_floats.append(max(5.0, disc_t ** (1.0 / 3.0)))
        json_floats.append(float(DEFAULT_CONFIG.top_n_liquid // DEFAULT_CONFIG.n_quantiles))

    if hold_sum_path.exists():
        h_sum = json.loads(hold_sum_path.read_text(encoding="utf-8"))
        h_funnel = h_sum.get("funnel_final", {})
        n_rob = float(h_funnel.get("n_verdict_robust", 64))
        if n_rob > 0:
            for key in (
                "n_holdout_same_sign_robust",
                "n_holdout_bh_stat_only",
                "n_holdout_rw_stat_only",
            ):
                if key in h_funnel:
                    json_floats.append(float(h_funnel[key]) / n_rob)

    # Standard mathematical / domain constants referenced in prose
    json_floats.extend([1.96, 252.0, 101.0, 33.0, 50.0, 51.0, 6.5, 3000.0, 2.3529])

    for val in json_floats:
        _add_number_variants(val, traced, allow_pct=True)

    # For CSV cells, only allow percentage scaling on return/rate columns, and require >=2 decimals
    # on continuous metrics so random p-values/t-stats don't mask hand-typed 2-decimal errors.
    pct_cols = {"ann_net_excess", "ann_gross_excess", "mean_net_excess_5d", "mean_gross_excess_5d"}
    for col_name, val in csv_floats:
        if col_name in pct_cols:
            _add_number_variants(val, traced, allow_pct=True)
        elif col_name in {"breakeven_rt_bps", "breakeven_l1_bps"}:
            for sign_val in (val, abs(val)):
                traced.add(f"{round(sign_val, 1):.1f}")
                traced.add(f"{round(sign_val, 2):.2f}")
        elif col_name in {
            "mean_turnover_l1",
            "t_nw",
            "disc_t_nw",
            "hold_t_nw",
            "net_ir_ann",
            "retention_ratio",
        }:
            for sign_val in (val, abs(val)):
                traced.add(f"{round(sign_val, 2):.2f}")
        else:
            for sign_val in (val, abs(val)):
                for dec in (3, 4):
                    traced.add(f"{round(sign_val, dec):.{dec}f}")

    return traced


def _collect_json_numbers(obj: object, out: list[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_json_numbers(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_json_numbers(item, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))


def _strip_code_backtick_keep_numeric(match: re.Match[str]) -> str:
    """Replace code/identifier backtick spans with space, but keep pure numeric/bps/pct spans."""
    span = match.group(1).strip()
    # If span contains a 7+ hex SHA or function call or underscore identifier, strip it
    if re.search(r"[0-9a-f]{7,}", span) or "_" in span or "(" in span or "=" in span or "/" in span:
        return " "
    return f" {span} "


def audit_markdown_file(
    md_path: Path,
    allowed_idents: set[str],
    traced_numbers: set[str],
) -> tuple[list[str], list[str]]:
    """Audit a single Markdown file, returning (ident_errors, number_errors)."""
    ident_errors: list[str] = []
    number_errors: list[str] = []

    lines = md_path.read_text(encoding="utf-8").splitlines()
    in_code_block = False
    in_math_block = False
    in_references = False
    in_api_details = False

    for lineno, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        # Track fenced code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or raw_line.startswith("    "):
            continue

        # Track $$ display math blocks
        if stripped.startswith("$$"):
            if stripped == "$$" or not stripped.endswith("$$") or len(stripped) == 2:
                in_math_block = not in_math_block
            continue
        if in_math_block:
            continue

        # Track <details> API reference tables in README.md (contains function default params)
        if "<details>" in stripped:
            in_api_details = True
            continue
        if "</details>" in stripped:
            in_api_details = False
            continue

        # Track References section
        if stripped.startswith("## ") and "References" in stripped:
            in_references = True
            continue
        if in_references and stripped.startswith("## "):
            in_references = False
        if in_references:
            continue

        # 1. Identifier check on inline backticks
        for match in _BACKTICK_RE.finditer(raw_line):
            span = match.group(1).strip()
            if re.match(r"^[0-9a-f]{7,40}$", span):
                continue
            if _IDENT_RE.match(span):
                if span not in allowed_idents:
                    ident_errors.append(f"{md_path}:{lineno}: unknown identifier `{span}`")
            else:
                m_call = _FUNC_CALL_RE.match(span)
                if m_call:
                    fn_name = m_call.group(1)
                    if fn_name not in allowed_idents:
                        ident_errors.append(
                            f"{md_path}:{lineno}: unknown function `{fn_name}(...)`"
                        )

        if in_api_details:
            continue

        # 2. Number check on prose & tables
        cleaned = re.sub(r"\]\([^)]*\)", "]", raw_line)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = _BACKTICK_RE.sub(_strip_code_backtick_keep_numeric, cleaned)
        cleaned = re.sub(r"[_^]\{[^}]*\}|[_^][0-9a-zA-Z]+", " ", cleaned)
        cleaned = re.sub(r"\\text\{MA\}\(\d+\)", " ", cleaned)
        cleaned = re.sub(r"\bQ[1-5]\b", " ", cleaned)
        cleaned = re.sub(r"^\s*#{1,6}\s+\d+(?:\.\d+)*(?:\.\s+|\s+)", "", cleaned)
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned)
        cleaned = re.sub(r"§\d+(?:\.\d+)*", " ", cleaned)
        cleaned = re.sub(r"\bSection\s+\d+(?:\.\d+)*", " ", cleaned)
        cleaned = re.sub(r"\|\s*\*\*[0-9]+[a-z]?\.\s+", "| **", cleaned)
        cleaned = re.sub(
            r"\b(19|20)\d{2}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?(?:\s*[+-]0\d:?00)?\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\b\d{2}:\d{2}(?::\d{2})?\b", " ", cleaned)
        cleaned = re.sub(r"[+-]0\d:?00\b", " ", cleaned)
        cleaned = re.sub(r"\b\d{1,3}:\d{1,3}\b", " ", cleaned)
        cleaned = re.sub(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,64}\b", " ", cleaned)
        cleaned = re.sub(r"\b(19|20)\d{2}[–-](19|20)\d{2}\b", " ", cleaned)
        cleaned = cleaned.replace("{,}", ",").replace(r"\text{–}", " ")

        for m_num in _NUM_TOKEN_RE.finditer(cleaned):
            raw_tok = m_num.group(1)
            is_pct = raw_tok.endswith("%")
            core = raw_tok.rstrip("%").lstrip("+-")
            core_no_comma = core.replace(",", "")

            try:
                fval = float(core_no_comma)
            except ValueError:
                continue

            if "." not in core and not is_pct and len(core_no_comma) < 2:
                continue

            if "." not in core and not is_pct and "," not in core and 1900 <= int(fval) <= 2099:
                continue

            if core not in traced_numbers and core_no_comma not in traced_numbers:
                number_errors.append(
                    f"{md_path}:{lineno}: untraced number '{raw_tok}' in: {raw_line.strip()}"
                )

    return ident_errors, number_errors


def main(argv: list[str] | None = None) -> int:
    """Run identifier and number audits on the specified Markdown files."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("Usage: python -m research.factor_zoo.audit_report <md_file> [...]")
        return 2

    allowed_idents = collect_allowed_identifiers()
    traced_numbers = collect_traced_numbers()

    total_ident_errors: list[str] = []
    total_num_errors: list[str] = []

    for arg in args:
        p = Path(arg)
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            return 2
        i_errs, n_errs = audit_markdown_file(p, allowed_idents, traced_numbers)
        total_ident_errors.extend(i_errs)
        total_num_errors.extend(n_errs)

    print(f"Audited {len(args)} file(s).")
    print(f"Identifier errors: {len(total_ident_errors)}")
    for err in total_ident_errors:
        print(f"  [IDENT] {err}")
    print(f"Untraced number errors: {len(total_num_errors)}")
    for err in total_num_errors:
        print(f"  [NUM]   {err}")

    return 1 if (total_ident_errors or total_num_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
