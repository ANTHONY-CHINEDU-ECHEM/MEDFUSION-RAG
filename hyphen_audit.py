"""Repository wide audit that guarantees no hyphen or dash characters exist.

The project follows a strict typographic policy: no hyphen, minus sign, en dash
or em dash may appear in any source file, dataset, configuration, document or
generated text artefact. The forbidden characters are declared by code point so
that this file itself remains compliant.

Usage:
    python tools/hyphen_audit.py            audit the repository root
    python tools/hyphen_audit.py some/dir   audit a specific directory
"""
from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN_CODE_POINTS = {
    45: "HYPHEN MINUS",
    173: "SOFT HYPHEN",
    8208: "HYPHEN",
    8209: "NON BREAKING HYPHEN",
    8210: "FIGURE DASH",
    8211: "EN DASH",
    8212: "EM DASH",
    8213: "HORIZONTAL BAR",
    8722: "MINUS SIGN",
    11834: "TWO EM DASH",
    11835: "THREE EM DASH",
    65112: "SMALL EM DASH",
    65123: "SMALL HYPHEN MINUS",
    65293: "FULLWIDTH HYPHEN MINUS",
}
FORBIDDEN = {chr(code): name for code, name in FORBIDDEN_CODE_POINTS.items()}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".cfg", ".toml", ".ini", ".yaml", ".yml", ".html"}
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "cache"}


def audit_text(text: str):
    """Return (line, column, character name) for every forbidden character."""
    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for column, char in enumerate(line, start=1):
            if char in FORBIDDEN:
                findings.append((line_number, column, FORBIDDEN[char]))
    return findings


def audit_tree(root: Path, max_report: int = 50):
    """Scan every text artefact and file name below root."""
    violations = []
    files_scanned = 0
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if any(char in FORBIDDEN for char in relative):
            violations.append((relative, 0, 0, "FILE NAME"))
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files_scanned += 1
        text = path.read_text(encoding="utf8", errors="replace")
        for line_number, column, name in audit_text(text):
            violations.append((relative, line_number, column, name))
    return {"files_scanned": files_scanned, "violations": violations[:max_report], "violation_count": len(violations)}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    report = audit_tree(root)
    print(f"Scanned {report['files_scanned']} text files under {root}")
    if report["violation_count"]:
        print(f"FAILED: {report['violation_count']} forbidden characters found")
        for relative, line, column, name in report["violations"]:
            print(f"  {relative}:{line}:{column} {name}")
        return 1
    print("PASSED: no hyphens, minus signs, en dashes or em dashes found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
