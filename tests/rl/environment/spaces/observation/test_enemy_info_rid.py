"""Radar-identification (RID) features in the enemy-fighter observation block."""

from types import SimpleNamespace

import numpy as np

from bvr_marl_core.rl.environment.spaces.observation.constants import (
    EF_IDX_RID_AIRFRAME,
    EF_IDX_RID_KNOWN,
    EF_IDX_RID_MISSILES_LEFT,
    EF_IDX_RID_WEAPON,
)
from bvr_marl_core.rl.environment.spaces.observation.enemy_info_builder import EnemyInfoBuilder
from bvr_marl_core.rl.environment.spaces.observation.rid import airframe_id_norm, weapon_id_norm
from tests.helpers.track_snapshot import track_snapshot


class _Config:
    ef_slots = 1
    em_slots = 1
    information_mode = "oracle"


def _make_env(*, confidence, locked, target):
    track = track_snapshot(
        "bandit-1",
        state=(0.0, 70_000.0, 0.0, 0.0, -250.0, 0.0),
        confidence=confidence,
    )
    own = SimpleNamespace(
        id="A0",
        group="BLUE",
        yaw_deg=0.0,
        sensor=SimpleNamespace(
            sensor_tracks=[track],
            get_locked_targets=lambda: {"bandit-1"} if locked else set(),
            get_nez_features=lambda _sim, _sel: {},
        ),
        wez=SimpleNamespace(
            zone_for_range=lambda *_a, **_k: "R4",
            sqi=lambda *_a, **_k: 0.0,
            compute_dlz=lambda *_a, **_k: None,
        ),
        target=target,
    )
    sim = SimpleNamespace(active_units={"A0": own, "bandit-1": target})
    return sim


def _mk_eurofighter():
    from types import SimpleNamespace as NS

    from bvr_marl_core.registry import get_aircraft_class
    from bvr_marl_core.simulator.core.helpers import Position

    ML = NS(left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000)
    ef = get_aircraft_class("Eurofighter")(
        Position(lat=0.6, lon=0.0, alt=9000.0), 180.0, 250.0, "RED", ML, 0.0, 20000.0
    )
    ef.id = "bandit-1"
    return ef


def test_rid_hidden_when_not_locked_and_low_confidence():
    target = SimpleNamespace(id="bandit-1", group="RED", is_missile=False)
    sim = _make_env(confidence=0.3, locked=False, target=target)
    _em_arr, ef_arr = EnemyInfoBuilder(sim, _Config()).build("A0")
    slot = ef_arr[0]
    assert slot[EF_IDX_RID_KNOWN] == np.float32(0.0)
    assert slot[EF_IDX_RID_AIRFRAME] == np.float32(0.0)
    assert slot[EF_IDX_RID_WEAPON] == np.float32(0.0)
    assert slot[EF_IDX_RID_MISSILES_LEFT] == np.float32(0.0)


def test_rid_revealed_when_locked():
    target = _mk_eurofighter()
    sim = _make_env(confidence=0.2, locked=True, target=target)
    _em_arr, ef_arr = EnemyInfoBuilder(sim, _Config()).build("A0")
    slot = ef_arr[0]
    assert slot[EF_IDX_RID_KNOWN] == np.float32(1.0)
    # Exact airframe + primary-weapon identity revealed (matches the helper).
    assert slot[EF_IDX_RID_AIRFRAME] == np.float32(airframe_id_norm(target))
    assert slot[EF_IDX_RID_WEAPON] == np.float32(weapon_id_norm(target))
    assert slot[EF_IDX_RID_AIRFRAME] > 0.0
    # Full missile load at spawn -> normalized remaining is 1.0.
    assert slot[EF_IDX_RID_MISSILES_LEFT] == np.float32(1.0)


def test_rid_revealed_on_high_confidence_without_lock():
    target = _mk_eurofighter()
    sim = _make_env(confidence=0.9, locked=False, target=target)
    _em_arr, ef_arr = EnemyInfoBuilder(sim, _Config()).build("A0")
    assert ef_arr[0][EF_IDX_RID_KNOWN] == np.float32(1.0)
