"""Architecture checks for the authoritative operational track contract."""

import ast
from pathlib import Path

_ROOT = Path(__file__).parents[2] / "src" / "bvr_marl_core"
_OPERATIONAL_MODULES = (
    _ROOT / "aircraft",
    _ROOT / "missiles",
    _ROOT / "radar",
    _ROOT / "rl" / "environment",
)


def test_operational_code_does_not_decode_positional_track_tuples():
    violations = []
    for root in _OPERATIONAL_MODULES:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                    if node.value.id in {"track", "track_data"}:
                        violations.append(f"{path.relative_to(_ROOT)}:{node.lineno}")

    assert violations == [], "positional track access remains at " + ", ".join(violations)
