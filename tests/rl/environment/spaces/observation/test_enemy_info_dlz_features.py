from types import SimpleNamespace

import numpy as np

from bvr_marl_core.rl.environment.spaces.observation.constants import (
    EF_IDX_DLZ_R_MIN,
    EF_IDX_DLZ_R_TR,
    EF_IDX_DLZ_ZONE,
    EF_IDX_MASK,
    EF_IDX_NEZ,
    EF_IDX_SQI,
    d_EF,
)
from bvr_marl_core.rl.environment.spaces.observation.enemy_info_builder import EnemyInfoBuilder
from tests.helpers.track_snapshot import track_snapshot


class _Config:
    ef_slots = 1
    em_slots = 1
    information_mode = "oracle"


def test_enemy_fighter_obs_contains_trimmed_dlz_features():
    dlz = SimpleNamespace(
        r_min_m=5_000.0,
        r_tr_m=50_000.0,
        r_pi_m=90_000.0,
        r_aero_m=120_000.0,
        r_nez_in_m=5_000.0,
        r_nez_out_m=50_000.0,
    )
    target = SimpleNamespace(id="bandit-1", group="RED", is_missile=False)
    own = SimpleNamespace(
        id="A0",
        group="BLUE",
        yaw_deg=0.0,
        sensor=SimpleNamespace(
            sensor_tracks=[
                track_snapshot(
                    "bandit-1",
                    state=(0.0, 70_000.0, 0.0, 0.0, -250.0, 0.0),
                )
            ],
            get_nez_features=lambda _sim, _selected: {
                "active_nez_by_target": {"bandit-1": 50_000.0},
                "active_dlz_by_target": {"bandit-1": dlz},
                "passive_nez": {"bandit-1": 45_000.0},
            },
        ),
        wez=SimpleNamespace(
            zone_for_range=lambda _range, _dlz: "R3",
            sqi=lambda *_args, **_kwargs: 0.42,
        ),
        target=target,
    )
    sim = SimpleNamespace(active_units={"A0": own, "bandit-1": target})

    _em_arr, ef_arr = EnemyInfoBuilder(sim, _Config()).build("A0")

    assert ef_arr.shape == (1, d_EF)
    slot = ef_arr[0]
    assert slot[EF_IDX_MASK] == np.float32(1.0)
    # NEZ is the merged max(active, passive) = 50 km, normalized by the 150 km scale.
    assert slot[EF_IDX_NEZ] == np.float32(50_000.0 / 150_000.0)
    assert slot[EF_IDX_DLZ_R_MIN] == np.float32(5_000.0 / 150_000.0)
    assert slot[EF_IDX_DLZ_R_TR] == np.float32(50_000.0 / 150_000.0)
    assert slot[EF_IDX_SQI] == np.float32(0.42)
    assert slot[EF_IDX_DLZ_ZONE] == np.float32(0.75)
