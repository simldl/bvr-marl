"""
Root conftest — shared pytest fixtures and marker registration.

Markers
-------
smoke       End-to-end entry-point checks (fast, ~4 s total).
slow        Tests that run a full physics simulation (>0.5 s each).
            Deselect for a fast feedback loop: ``pytest -m "not slow"``
integration Tests that exercise multiple subsystems together.
unit        Pure unit tests — no real simulator, no RL deps.

Directory → default marker mapping (applied by pytest_collection_modifyitems):
    tests/aircrafts/   → unit
    tests/missiles/    → unit
    tests/physics/     → unit
    tests/radar/       → unit
    tests/rl/          → unit
    tests/simulator/   → integration
"""

import matplotlib
import pytest

matplotlib.use("Agg")  # force non-interactive backend before any test imports matplotlib

# Maps directory path fragments to their default marker.
# Tests that already carry an explicit pytestmark keep it in addition.
_DIR_MARKERS: dict[str, str] = {
    "tests/aircrafts": "unit",
    "tests/missiles": "unit",
    "tests/physics": "unit",
    "tests/radar": "unit",
    "tests/rl": "unit",
    "tests/simulator": "integration",
}


def pytest_collection_modifyitems(items: list) -> None:
    """Apply directory-level default markers to unmarked test items."""
    for item in items:
        item_path = str(item.fspath).replace("\\", "/")
        for path_fragment, marker_name in _DIR_MARKERS.items():
            if path_fragment in item_path:
                item.add_marker(getattr(pytest.mark, marker_name))
                break
