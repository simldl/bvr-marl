"""Tests for the opt-in scenario-variety capabilities (workshop "general/map"):

- asymmetric team counts (``num_opponents`` != agent count),
- heterogeneous aircraft types (a list of types cycled per slot),
- opt-in per-spawn start speed/altitude variation.

These exercise config parsing and the spawn helpers directly; existing scenarios
that set none of these keys keep their previous fixed behaviour.
"""

from types import SimpleNamespace

import numpy as np

from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig
from bvr_marl_core.rl.environment.gym.spawn_utils import _apply_spawn_variation
from bvr_marl_core.rl.utils.type_maps import AIRCRAFT_TYPE_MAP, resolve_aircraft_config


class TestAsymmetricCounts:
    def test_defaults_to_symmetric(self):
        """Without num_opponents the opponent count mirrors the agent count."""
        cfg = BVREnvConfig.from_dict({"num_agents_per_side": 3, "map_size": 200})
        assert cfg.agent_ids == ["A0", "A1", "A2"]
        assert cfg.opponent_ids == ["B0", "B1", "B2"]
        assert cfg.all_agent_ids == ["A0", "A1", "A2", "B0", "B1", "B2"]

    def test_asymmetric_counts(self):
        """num_opponents overrides the opponent count independently (e.g. 4 v 2)."""
        cfg = BVREnvConfig.from_dict(
            {"num_agents_per_side": 4, "num_opponents": 2, "map_size": 500}
        )
        assert cfg.agent_ids == ["A0", "A1", "A2", "A3"]
        assert cfg.opponent_ids == ["B0", "B1"]
        assert cfg.num_agents_per_team == 4


class TestHeterogeneousTypes:
    def test_single_type_fills_whole_team(self):
        cfg = BVREnvConfig.from_dict(
            {
                "num_agents_per_side": 2,
                "map_size": 200,
                "aircraft_types": {
                    "agent": AIRCRAFT_TYPE_MAP["Eurofighter"],
                    "opponent": AIRCRAFT_TYPE_MAP["Su57"],
                },
            }
        )
        assert cfg.aircraft_type_map["A0"] is AIRCRAFT_TYPE_MAP["Eurofighter"]
        assert cfg.aircraft_type_map["A1"] is AIRCRAFT_TYPE_MAP["Eurofighter"]
        assert cfg.aircraft_type_map["B0"] is AIRCRAFT_TYPE_MAP["Su57"]

    def test_list_type_assigned_per_slot_and_cycled(self):
        """A list of types is assigned per slot and cycled (mixed formation)."""
        cfg = BVREnvConfig.from_dict(
            {
                "num_agents_per_side": 3,
                "map_size": 500,
                "aircraft_types": {
                    "agent": [AIRCRAFT_TYPE_MAP["F22"], AIRCRAFT_TYPE_MAP["F35"]],
                    "opponent": AIRCRAFT_TYPE_MAP["Su57"],
                },
            }
        )
        assert cfg.aircraft_type_map["A0"] is AIRCRAFT_TYPE_MAP["F22"]
        assert cfg.aircraft_type_map["A1"] is AIRCRAFT_TYPE_MAP["F35"]
        # Cycles back to the first type for the third slot.
        assert cfg.aircraft_type_map["A2"] is AIRCRAFT_TYPE_MAP["F22"]

    def test_resolve_aircraft_config_accepts_list_of_strings(self):
        """The YAML-facing resolver turns a list of type strings into classes."""
        resolved = resolve_aircraft_config(
            {"aircraft_config": {"agent_type": ["F22", "F35"], "opponent_type": "Su57"}}
        )
        types = resolved["aircraft_types"]
        assert types["agent"] == [AIRCRAFT_TYPE_MAP["F22"], AIRCRAFT_TYPE_MAP["F35"]]
        assert types["opponent"] is AIRCRAFT_TYPE_MAP["Su57"]


class TestSpawnVariation:
    def _pos(self, alt=8000.0):
        return SimpleNamespace(alt=alt)

    def test_disabled_by_default(self):
        """No spawn-range keys -> speed and altitude are returned unchanged."""
        env = SimpleNamespace(config={})
        pos = self._pos(8000.0)
        speed, out = _apply_spawn_variation(env, 250.0, pos)
        assert speed == 250.0
        assert out.alt == 8000.0

    def test_opt_in_randomises_within_range(self):
        env = SimpleNamespace(
            config={
                "spawn_speed_min": 200.0,
                "spawn_speed_max": 300.0,
                "spawn_alt_min": 6000.0,
                "spawn_alt_max": 11000.0,
            }
        )
        np.random.seed(0)
        for _ in range(20):
            pos = self._pos(8000.0)
            speed, out = _apply_spawn_variation(env, 250.0, pos)
            assert 200.0 <= speed <= 300.0
            assert 6000.0 <= out.alt <= 11000.0

    def test_partial_config_only_varies_provided_axis(self):
        """Only speed range set -> altitude untouched."""
        env = SimpleNamespace(config={"spawn_speed_min": 200.0, "spawn_speed_max": 300.0})
        pos = self._pos(8000.0)
        speed, out = _apply_spawn_variation(env, 250.0, pos)
        assert 200.0 <= speed <= 300.0
        assert out.alt == 8000.0
