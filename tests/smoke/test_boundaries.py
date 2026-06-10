"""Regression tests: verify that bvr_marl_core never imports from bvr_marl_behavior.

The dependency arrow is strictly one-directional: bvr_marl_behavior may import
from bvr_marl_core, never the reverse.  These tests guard that boundary so
that any accidental import breaks CI immediately.
"""

from pathlib import Path

import pytest


@pytest.mark.smoke
def test_core_does_not_import_behavior():
    """bvr_marl_core must never import from bvr_marl_behavior at the module level."""
    import ast

    from bvr_marl_core import __file__ as pkg_init

    ROOT = Path(pkg_init).resolve().parent

    violations = []
    for py_file in sorted(ROOT.rglob("*.py")):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            for mod in mods:
                if mod == "bvr_marl_behavior" or mod.startswith("bvr_marl_behavior."):
                    violations.append(f"{py_file.relative_to(ROOT.parent)}:{node.lineno} — '{mod}'")

    assert violations == [], (
        "Forbidden bvr_marl_behavior imports found in bvr_marl_core:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
