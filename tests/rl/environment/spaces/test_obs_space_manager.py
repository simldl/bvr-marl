import numpy as np

from bvr_marl_core.rl.environment.spaces.obs_space_manager import (
    EnvConfig,
    ObservationSpaceManager,
)
from bvr_marl_core.rl.environment.spaces.observation.constants import (
    d_EF,
    d_EM,
    d_FF,
    d_FM,
    d_MWS,
    d_OWN,
    d_PR,
)


def test_observation_space_manager():
    cfg = EnvConfig(
        own_dim=d_OWN,
        fm_slots=2,
        ff_slots=1,
        em_slots=2,
        ef_slots=1,
        pr_slots=1,
        warn_sectors=4,
    )
    mgr = ObservationSpaceManager(["A1", "B1"], cfg)
    all_spaces = mgr.all()
    print("All available agent IDs in space manager:", list(all_spaces.keys()))
    assert "A1" in all_spaces
    space = mgr.get("A1")
    assert space is not None, "mgr.get('A1') returned None"
    keys = list(space.keys())
    print("Space keys:", keys)

    # Mask-driven token structure: entity blocks only, mask folded into tokens.
    assert set(keys) == {
        "own_state",
        "friendly_missiles",
        "friendly_fighters",
        "enemy_missiles",
        "enemy_fighters",
        "missile_warnings",
        "passive_radar",
    }
    # No separate mask/index keys remain.
    assert not any(k.startswith("mask_") for k in keys)
    assert "fm_target_indices" not in keys and "ff_lock_indices" not in keys

    # Check shapes (slots * token_dim, mask column included in token_dim).
    assert space["own_state"].shape == (d_OWN,)
    assert space["friendly_missiles"].shape == (2 * d_FM,)
    assert space["friendly_fighters"].shape == (1 * d_FF,)
    assert space["enemy_missiles"].shape == (2 * d_EM,)
    assert space["enemy_fighters"].shape == (1 * d_EF,)
    assert space["missile_warnings"].shape == (2 * d_MWS,)
    assert space["passive_radar"].shape == (1 * d_PR,)

    all_spaces = mgr.all()
    assert set(all_spaces.keys()) == {"A1", "B1"}


def test_env_own_state_dim_tracks_d_OWN_and_total_matches():
    """The runtime env's own_state_dim must equal d_OWN, and the built space's
    total must equal get_total_obs_dim() (the two obs systems must not drift)."""
    from gymnasium.spaces.utils import flatdim

    from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
    from bvr_marl_core.rl.environment.spaces.observation.constants import get_total_obs_dim
    from bvr_marl_core.simulator.simulator import Simulator

    env = BVRMultiAgentEnv(
        {"simulator": Simulator(weapon_config={}), "num_agents_per_side": 2, "map_size": 200}
    )
    obs, _ = env.reset()
    assert env.config.own_state_dim == d_OWN

    space = env.observation_space["A0"]
    total = sum(flatdim(s) for s in space.spaces.values())
    assert total == get_total_obs_dim()
    # Actual observation arrays match the declared space exactly (no coercion pad).
    for key, subspace in space.spaces.items():
        assert obs["A0"][key].shape[0] == flatdim(subspace)
    env.close()
