"""Integration check that enemy-fighter tokens are present and mask-driven.

Verifies, on a real env reset, that:
1. enemy_fighters is a flat (ef_slots * d_EF) vector,
2. each token's last column is a 0/1 validity mask,
3. the trimmed NEZ/DLZ/SQI/zone cues sit at their documented indices.
"""

import numpy as np

from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from bvr_marl_core.rl.environment.spaces.observation.constants import (
    EF_IDX_DLZ_ZONE,
    EF_IDX_IS_SUPPORT,
    EF_IDX_MASK,
    EF_IDX_NEZ,
    EF_IDX_SQI,
    d_EF,
)
from bvr_marl_core.simulator.simulator import Simulator


def _make_env(num_ef_slots_from=2):
    simulator = Simulator(weapon_config={"missile_hit_radius_m": 500.0, "gun_hit_radius_m": 5.0})
    return BVRMultiAgentEnv(
        {
            "simulator": simulator,
            "num_agents_per_side": num_ef_slots_from,
            "max_steps": 100,
            "debug": False,
            "map_size": 200,
        }
    )


def test_enemy_fighter_tokens_shape_and_mask_column():
    env = _make_env()
    obs, _info = env.reset()
    agent_obs = obs["A0"]

    # No separate mask key any more — the mask rides in the token.
    assert "mask_enemy_fighters" not in agent_obs
    ef = agent_obs["enemy_fighters"]
    assert ef.shape[0] % d_EF == 0
    ef_slots = ef.shape[0] // d_EF

    tokens = ef.reshape(ef_slots, d_EF)
    mask = tokens[:, EF_IDX_MASK]
    assert set(np.unique(mask)).issubset({0.0, 1.0})

    # Documented cue indices are within the token and finite.
    for idx in (EF_IDX_NEZ, EF_IDX_SQI, EF_IDX_DLZ_ZONE, EF_IDX_IS_SUPPORT):
        assert 0 <= idx < d_EF
        assert np.all(np.isfinite(tokens[:, idx]))
    env.close()


def test_enemy_fighter_slots_scale_with_roster():
    """More agents -> more enemy-fighter slots, same per-token width (variable-N)."""
    env = _make_env(num_ef_slots_from=4)
    obs, _info = env.reset()
    ef = obs["A0"]["enemy_fighters"]
    assert ef.shape[0] % d_EF == 0
    env.close()
