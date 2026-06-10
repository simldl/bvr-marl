#!/usr/bin/env python3
"""CI script: verify that bvr_marl_core never imports from other bvr_marl_* packages.

The simulation core must remain self-contained. Importing any other package
in the bvr_marl_* namespace would create an undeclared circular dependency.

Run with: python scripts/check_import_boundaries.py
Exit code 0 = clean. Exit code 1 = violations found (CI will fail).
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "bvr_marl_core"
OWN_PACKAGE = "bvr_marl_core"
NAMESPACE_PREFIX = "bvr_marl_"


def _is_forbidden(module: str) -> bool:
    return module.startswith(NAMESPACE_PREFIX) and not (
        module == OWN_PACKAGE or module.startswith(OWN_PACKAGE + ".")
    )


def check_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: cannot read — {exc}"]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error — {exc}"]

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    violations.append(f"{path}:{node.lineno}: forbidden import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden(module):
                violations.append(f"{path}:{node.lineno}: forbidden 'from {module} import ...'")

    return violations


def main() -> int:
    all_violations: list[str] = []
    checked = 0
    for py_file in sorted(ROOT.rglob("*.py")):
        all_violations.extend(check_file(py_file))
        checked += 1

    if all_violations:
        print(
            f"Import boundary violations found ({OWN_PACKAGE} must not import other bvr_marl_* packages):"
        )
        for v in all_violations:
            print(f"  {v}")
        print(
            f"\n{OWN_PACKAGE} must remain self-contained. "
            "Remove or relocate the listed imports and re-run."
        )
        return 1

    print(f"OK — all {checked} source files respect import boundaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
