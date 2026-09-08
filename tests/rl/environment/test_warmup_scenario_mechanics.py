"""Unit tests for warmup scenario mechanics added for the boundary-first curriculum.

Covers:
* the ``random_map`` full-map spawn regime,
* the ``stationary_hold`` / ``opponent_hold_fire`` opponent scripting override,
* the launch-envelope (R2/R3 in, R1/R4 out) tagging helper.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.rl.environment.gym.gym_components.config import ScenarioConfigData
from bvr_marl_core.rl.environment.gym.gym_components.step_processor import StepProcessor
from bvr_marl_core.rl.environment.scenarios.geometry_sampler import (
    RANDOM_MAP_REGIME,
    GeometrySampler,
)
from bvr_marl_core.rl.environment.spaces.action_space.weapon_firing import WeaponFiringHandler


def test_random_map_regime_is_registered():
    assert RANDOM_MAP_REGIME == "random_map"
    assert "random_map" in GeometrySampler.list_regimes()


def test_random_map_positions_stay_in_inner_box_and_separated():
    sampler = GeometrySampler(map_size_km=160)
    rng = np.random.default_rng(0)
    half_inner_deg = (160 / 2 * (1 - 0.2)) / 111.0
    for _ in range(25):
        data = sampler.compute_random_map_positions(
            num_agents=1, num_opponents=1, margin_frac=0.2, min_separation_m=40_000.0, rng=rng
        )
        agent = data["agent_positions"][0]
        opp = data["opponent_positions"][0]
        assert abs(agent.lat) <= half_inner_deg + 1e-6
        assert abs(agent.lon) <= half_inner_deg + 1e-6
        sep_km = math.hypot((agent.lat - opp.lat) * 111.0, (agent.lon - opp.lon) * 111.0)
        assert sep_km >= 39.5  # best-effort min separation
        assert 0.0 <= data["agent_headings"][0] < 360.0


def test_scenario_stationary_hold_implies_hold_fire_via_parser():
    from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig

    cfg = BVREnvConfig.from_dict(
        {
            "num_agents_per_side": 1,
            "scenario_config": {"opponent_behavior": "stationary_hold"},
        }
    )
    assert cfg.scenario_config.opponent_behavior == "stationary_hold"
    assert cfg.scenario_config.opponent_hold_fire is True


def test_scenario_defaults_leave_opponent_on_policy():
    sc = ScenarioConfigData()
    assert sc.opponent_behavior == "policy"
    assert sc.opponent_hold_fire is False
    assert sc.opponent_non_engageable is False


def test_scenario_opponent_non_engageable_parses():
    from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig

    cfg = BVREnvConfig.from_dict(
        {
            "num_agents_per_side": 1,
            "scenario_config": {"opponent_non_engageable": True},
        }
    )
    assert cfg.scenario_config.opponent_non_engageable is True


def test_scripted_opponent_flags_set_non_engageable_for_opponent_only():
    from types import SimpleNamespace

    from bvr_marl_core.rl.environment.gym.spawn_utils import _apply_scripted_opponent_flags

    cfg = {"scenario_config": {"opponent_non_engageable": True}}

    opp = SimpleNamespace(is_non_engageable=False)
    _apply_scripted_opponent_flags(opp, "opponent", cfg)
    assert opp.is_non_engageable is True

    # The opponent flag never makes the trained agent non-engageable.
    agent = SimpleNamespace(is_non_engageable=False)
    _apply_scripted_opponent_flags(agent, "agent", cfg)
    assert agent.is_non_engageable is False

    # Without the flag the opponent stays engageable (default behavior).
    plain = SimpleNamespace(is_non_engageable=False)
    _apply_scripted_opponent_flags(plain, "opponent", {"scenario_config": {}})
    assert plain.is_non_engageable is False


def test_scripted_opponent_action_holds_fire_and_neutralizes_flight():
    act = np.array([0.9, 0.1, 0.8, 0.3, 1.0, 1.0, 0.7, 0.2, 0.0, 0.5], np.float32)

    stationary = StepProcessor._scripted_opponent_action(act, behavior="stationary_hold")
    assert np.allclose(stationary[0:3], 0.5)  # neutral straight-and-level
    assert stationary[3] == pytest.approx(0.0)  # missile held
    assert stationary[4] == pytest.approx(0.5)  # neutral target selection
    assert np.allclose(stationary[5:10], 0.0)  # no gun / countermeasures

    hold_fire_only = StepProcessor._scripted_opponent_action(act, behavior="policy")
    assert np.allclose(hold_fire_only[0:3], act[0:3])  # flight untouched
    assert hold_fire_only[3] == pytest.approx(0.0)  # missile silenced
    assert hold_fire_only[4] == pytest.approx(act[4])  # target selection untouched
    assert np.allclose(hold_fire_only[5:10], 0.0)  # gun/countermeasures silenced


def test_scripted_opponent_action_anchored_orbits_and_holds_fire():
    act = np.array([0.9, 0.1, 0.8, 0.3, 1.0, 1.0, 0.7, 0.2, 0.0, 0.5], np.float32)

    anchored = StepProcessor._scripted_opponent_action(act, behavior="anchored_hold")
    assert anchored[0] == pytest.approx(0.5)  # hold energy
    # Constant banked level turn: bank off-center, load factor slightly above 1g.
    assert anchored[2] == pytest.approx(StepProcessor._ANCHOR_BANK_ACTION)
    assert anchored[2] != pytest.approx(0.5)
    assert anchored[1] > 0.5
    assert anchored[3] == pytest.approx(0.0)
    assert anchored[4] == pytest.approx(0.5)
    assert np.allclose(anchored[5:10], 0.0)  # never fires gun/countermeasures


def test_scenario_anchored_hold_implies_hold_fire_via_parser():
    from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig

    cfg = BVREnvConfig.from_dict(
        {
            "num_agents_per_side": 1,
            "scenario_config": {"opponent_behavior": "anchored_hold"},
        }
    )
    assert cfg.scenario_config.opponent_behavior == "anchored_hold"
    assert cfg.scenario_config.opponent_hold_fire is True


class _FakeWez:
    def __init__(self, zone):
        self._zone = zone

    def compute_dlz(self, _t):
        return object()

    def _slant_range_m(self, _o, _t):
        return 0.0

    def zone_for_range(self, _r, _d):
        return self._zone


@pytest.mark.parametrize(
    "zone,expected",
    [("R2", True), ("R3", True), ("R1", False), ("R4", False)],
)
def test_shot_in_envelope_classifies_dlz_zones(zone, expected):
    from types import SimpleNamespace

    unit = SimpleNamespace(wez=_FakeWez(zone))
    assert WeaponFiringHandler._shot_in_envelope(unit, SimpleNamespace()) is expected


def test_shot_in_envelope_returns_none_without_wez():
    from types import SimpleNamespace

    assert WeaponFiringHandler._shot_in_envelope(SimpleNamespace(), SimpleNamespace()) is None


def test_contact_shot_envelope_uses_estimated_track_only():
    from types import SimpleNamespace

    class EstimatedWez:
        def compute_dlz_from_track(self, state, covariance):
            assert state[0] == 40_000.0
            assert np.asarray(covariance).shape == (6, 6)
            return SimpleNamespace(nominal=object())

        def compute_dlz(self, _target):
            raise AssertionError("truth DLZ path used")

        def zone_for_range(self, range_m, _dlz):
            assert range_m == pytest.approx(40_000.0)
            return "R3"

    contact = TacticalContact(
        track_id=7,
        state=(40_000.0, 0.0, 0.0, -200.0, 0.0, 0.0),
        covariance=tuple(tuple(row) for row in np.eye(6)),
        confidence=0.8,
        classification="aircraft",
    )

    assert WeaponFiringHandler._shot_in_envelope(SimpleNamespace(wez=EstimatedWez()), contact)
