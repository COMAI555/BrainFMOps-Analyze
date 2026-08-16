"""Pre-publication audit for the BrainFMOps-Analyze release folder.

Run from the repository root:
    python audit_repository.py

The script is read-only except for repository_audit_report.txt.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path.cwd().resolve()
REPORT = ROOT / "repository_audit_report.txt"
SELF = Path(__file__).resolve()
TEXT_EXTENSIONS = {".py", ".ipynb", ".json", ".csv", ".md", ".yaml", ".yml", ".toml"}
BLOCKED_EXTENSIONS = {".nii", ".gz", ".hdr", ".img", ".dcm", ".pt", ".pth", ".h5"}
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}

PATTERNS = {
    "Windows absolute path": re.compile(r"[A-Za-z]:[\\/]+(?:Users|Brain|Data|OASIS|[^\\/\s]+[\\/])", re.I),
    "Windows user directory": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.I),
    "Unix home directory": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "Possible secret assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\b\s*[:=]\s*[\"'][^\"']+[\"']"
    ),
}


def candidates():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == SELF or path == REPORT:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def safe_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def python_imports(text: str) -> set[str]:
    found: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def notebook_code_and_outputs(text: str) -> tuple[str, int]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text, 0
    code: list[str] = []
    output_count = 0
    for cell in obj.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            code.append("".join(source) if isinstance(source, list) else str(source))
            output_count += len(cell.get("outputs", []))
    return "\n".join(code), output_count


def main() -> int:
    findings: list[str] = []
    warnings: list[str] = []
    imports: Counter[str] = Counter()
    scanned = 0

    for path in candidates():
        rel = path.relative_to(ROOT)
        suffix = path.suffix.lower()
        if suffix in BLOCKED_EXTENSIONS or path.name.lower().endswith(".nii.gz"):
            findings.append(f"BLOCKED FILE: {rel}")
            continue
        if suffix not in TEXT_EXTENSIONS:
            continue
        text = safe_text(path)
        if text is None:
            warnings.append(f"Could not read as UTF-8: {rel}")
            continue
        scanned += 1
        analysis_text = text
        if suffix == ".ipynb":
            analysis_text, output_count = notebook_code_and_outputs(text)
            if output_count:
                warnings.append(f"Notebook contains {output_count} stored output(s): {rel}")
        if suffix == ".py":
            imports.update(python_imports(analysis_text))
        elif suffix == ".ipynb":
            imports.update(python_imports(analysis_text))
        for label, pattern in PATTERNS.items():
            for line_number, line in enumerate(analysis_text.splitlines(), 1):
                if pattern.search(line):
                    findings.append(f"{label}: {rel}:{line_number}")
        if suffix in {".csv", ".json"} and "public_data" in rel.parts:
            subject_pattern = re.compile(r"\bOAS1_\d{4}(?:_MR\d+)?\b", re.I)
            for line_number, line in enumerate(analysis_text.splitlines(), 1):
                if subject_pattern.search(line):
                    findings.append(f"OASIS case identifier in public data: {rel}:{line_number}")

    report: list[str] = [
        "BrainFMOps-Analyze pre-publication repository audit",
        f"Repository: {ROOT}",
        f"Text files scanned: {scanned}",
        "",
        "BLOCKING FINDINGS",
    ]
    report.extend(findings or ["None"])
    report.extend(["", "WARNINGS"])
    report.extend(warnings or ["None"])
    report.extend(["", "DETECTED TOP-LEVEL IMPORTS"])
    report.extend(sorted(imports) or ["None"])
    report.extend(["", f"Audit result: {'FAIL' if findings else 'PASS'}"])
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nReport written to: {REPORT.name}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
