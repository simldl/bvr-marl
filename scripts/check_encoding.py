#!/usr/bin/env python3
"""CI script: verify that all Python source files are valid UTF-8 without BOM.

Run with: python scripts/check_encoding.py
Exit code 0 = clean. Exit code 1 = violations found (CI will fail).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"


def check_file(path: Path) -> list[str]:
    violations = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        violations.append(f"{path}: cannot read — {exc}")
        return violations

    if raw.startswith(b"\xef\xbb\xbf"):
        violations.append(f"{path}: UTF-8 BOM detected — strip the BOM")
        return violations

    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        violations.append(f"{path}: not valid UTF-8 — {exc}")

    return violations


def main() -> int:
    all_violations: list[str] = []
    checked = 0
    for py_file in sorted(ROOT.rglob("*.py")):
        all_violations.extend(check_file(py_file))
        checked += 1

    if all_violations:
        print("Source encoding violations found:")
        for v in all_violations:
            print(f"  {v}")
        print(
            "\nAll Python source files must be UTF-8 encoded without BOM. "
            "Fix the listed files and re-run."
        )
        return 1

    print(f"OK — all {checked} source files are valid UTF-8 without BOM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
