"""Tests for the GUI config-builder helpers that back the opt-in scenario-variety
options (asymmetric counts, heterogeneous per-slot types, spawn variation), and a
round-trip of a GUI-shaped config through the real environment config path.
"""

import pytest

pytest.importorskip("streamlit", reason="config_builder imports streamlit")

from bvr_marl_core.gui.components.config_builder import (
    collapse_type_selection,
    expand_type_selection,
    spawn_variation_enabled,
    write_spawn_variation,
)
from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig
from bvr_marl_core.rl.utils.type_maps import AIRCRAFT_TYPE_MAP, resolve_aircraft_config


class TestCollapseTypeSelection:
    def test_single_type_collapses_to_string(self):
        assert collapse_type_selection(["F22", "F22", "F22"]) == "F22"

    def test_mixed_types_stay_a_list(self):
        assert collapse_type_selection(["F22", "F35", "F35"]) == ["F22", "F35", "F35"]

    def test_empty_is_none(self):
        assert collapse_type_selection([]) is None
        assert collapse_type_selection([None, ""]) is None


class TestExpandTypeSelection:
    def test_string_repeats_to_count(self):
        assert expand_type_selection("Su57", 3) == ["Su57", "Su57", "Su57"]

    def test_list_cycles_to_count(self):
        assert expand_type_selection(["F22", "F35"], 3) == ["F22", "F35", "F22"]

    def test_none_falls_back_to_default(self):
        out = expand_type_selection(None, 2)
        assert len(out) == 2 and all(isinstance(t, str) for t in out)

    def test_zero_count_is_empty(self):
        assert expand_type_selection("F22", 0) == []


class TestSpawnVariation:
    def test_disabled_by_default(self):
        assert spawn_variation_enabled({}) is False

    def test_enable_writes_all_keys(self):
        env = {}
        write_spawn_variation(env, True, 200, 300, 6000, 11000)
        assert spawn_variation_enabled(env) is True
        assert env["spawn_speed_min"] == 200.0 and env["spawn_speed_max"] == 300.0
        assert env["spawn_alt_min"] == 6000.0 and env["spawn_alt_max"] == 11000.0

    def test_disable_clears_keys(self):
        env = {
            "spawn_speed_min": 200.0,
            "spawn_speed_max": 300.0,
            "spawn_alt_min": 6000.0,
            "spawn_alt_max": 11000.0,
        }
        write_spawn_variation(env, False, 0, 0, 0, 0)
        assert spawn_variation_enabled(env) is False
        assert not any(k.startswith("spawn_") for k in env)


def test_gui_shaped_config_round_trips_through_env_config():
    """A config with the GUI's new options resolves through the real env path."""
    env = {
        "num_agents_per_side": 4,
        "num_opponents": 3,
        "map_size": 500,
        "aircraft_config": {
            "agent_type": ["F22", "F35", "F35", "F35"],  # heterogeneous list of strings
            "opponent_type": "Su57",  # homogeneous string
        },
        "spawn_speed_min": 200.0,
        "spawn_speed_max": 300.0,
        "spawn_alt_min": 6000.0,
        "spawn_alt_max": 11000.0,
    }
    env.update(resolve_aircraft_config(env))
    cfg = BVREnvConfig.from_dict(env)

    assert cfg.agent_ids == ["A0", "A1", "A2", "A3"]
    assert cfg.opponent_ids == ["B0", "B1", "B2"]
    assert cfg.map_size_km == 500.0
    assert cfg.aircraft_type_map["A0"] is AIRCRAFT_TYPE_MAP["F22"]
    assert cfg.aircraft_type_map["A1"] is AIRCRAFT_TYPE_MAP["F35"]
    assert cfg.aircraft_type_map["B0"] is AIRCRAFT_TYPE_MAP["Su57"]
