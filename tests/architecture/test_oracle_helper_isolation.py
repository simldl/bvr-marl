"""Structural isolation of oracle/truth-resolving helpers from the sensor-limited path.

The information firewall relies on the fact that the only helpers able to resolve an
anonymous track/contact identity back to a live world-truth entity --
``resolve_truth_unit`` (observation) and ``TargetSorter.select_target`` (action) -- are
(a) called from exactly one place each, and (b) reachable only behind an explicit
sensor-limited guard. These structural tests fail loudly if a refactor spreads a truth
helper into a new site or drops its guard, complementing the runtime trip-wire in
``truth_access_guard`` and the invariance tests.

Own-unit access (``active_units[agent_id]`` for an agent reading its OWN state) is
legitimate and deliberately not treated as a truth helper here.
"""

from pathlib import Path

import bvr_marl_core.rl as rl_pkg

_RL_ROOT = Path(rl_pkg.__file__).parent


def _py_files():
    return [p for p in _RL_ROOT.rglob("*.py") if "__pycache__" not in str(p)]


def _read(*parts):
    return (_RL_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def test_truth_helpers_are_confined_to_their_single_module():
    # Each truth-resolving helper is *called* in exactly one module. select_target's
    # definition lives in target_sorting.py (``def select_target``) and is excluded by
    # requiring the method-call form ``.select_target(``.
    allowed = {
        "resolve_truth_unit(": {"enemy_info_builder.py"},
        ".select_target(": {"action_processor.py"},
    }
    for helper, modules in allowed.items():
        callers = {p.name for p in _py_files() if helper in p.read_text(encoding="utf-8")}
        assert callers <= modules, f"{helper} appears outside {modules}: {callers - modules}"
        assert callers == modules, f"{helper} expected in {modules}, found {callers}"


def test_resolve_truth_unit_call_is_behind_a_sensor_limited_guard():
    normalized = " ".join(
        _read("environment", "spaces", "observation", "enemy_info_builder.py").split()
    )
    sites = 0
    idx = 0
    needle = "resolve_truth_unit("
    while True:
        found = normalized.find(needle, idx)
        if found == -1:
            break
        window = normalized[max(0, found - 60) : found]
        if "import" in window:  # the `from ... import resolve_truth_unit` line, not a call
            idx = found + 1
            continue
        sites += 1
        assert "self.sensor_limited" in window, window
        idx = found + 1
    assert sites >= 1, "expected a guarded resolve_truth_unit call site"


def test_select_target_truth_path_is_the_non_sensor_limited_branch():
    src = _read("environment", "spaces", "action_space", "action_processor.py")
    i_sl = src.index("InformationMode.SENSOR_LIMITED")
    i_contact = src.index("select_contact", i_sl)  # sensor-limited branch: truth-free
    i_else = src.index("else:", i_sl)
    i_target = src.index(".select_target(", i_sl)  # truth path
    # Ordering: sensor-limited -> select_contact -> else -> select_target.
    assert i_contact < i_else < i_target
