import numpy as np

from bvr_marl_core.rl.environment.spaces.obs_space_manager import (
    EnvConfig,
    ObservationSpaceManager,
)


def test_observation_space_manager():
    cfg = EnvConfig(
        own_dim=21,
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

    # Check individual keys (updated for new structure)
    assert "own_state" in keys
    assert "friendly_missiles" in keys
    assert "mask_friendly_missiles" in keys
    assert "friendly_fighters" in keys
    assert "mask_friendly_fighters" in keys
    assert "fm_target_indices" in keys
    assert "mask_fm_targets" in keys
    assert "ff_lock_indices" in keys
    assert "mask_ff_locks" in keys
    assert "enemy_missiles" in keys
    assert "mask_enemy_missiles" in keys
    assert "enemy_fighters" in keys
    assert "mask_enemy_fighters" in keys
    # REMOVED: NEZ is now embedded in enemy_fighters, not separate arrays
    # assert "enemy_nez_active" in keys
    # assert "enemy_nez_passive" in keys
    assert "missile_warning_flag" in keys
    assert "missile_warning_dirs" in keys
    assert "mask_warning_dirs" in keys
    assert "passive_radar" in keys
    assert "mask_passive_radar" in keys

    # Check shapes (updated dimensions)
    assert space["own_state"].shape == (21,)
    assert space["friendly_missiles"].shape == (2 * 8,)  # Updated from 6 to 8 dims/slot
    assert space["friendly_fighters"].shape == (1 * 6,)  # Unchanged
    assert space["fm_target_indices"].shape == (2 * 1,)
    assert space["ff_lock_indices"].shape == (1 * 1,)
    assert space["enemy_missiles"].shape == (2 * 7,)  # Updated from 6 to 7 dims/slot
    assert space["enemy_fighters"].shape == (
        1 * 10,
    )  # Updated from 9 to 10 dims/slot (includes is_support_asset)
    # REMOVED: NEZ arrays no longer exist
    # assert space["enemy_nez_active"].shape == (1,)
    # assert space["enemy_nez_passive"].shape == (1,)
    assert space["missile_warning_flag"].shape == (1,)
    assert space["missile_warning_dirs"].shape == (2 * (1 + 4),)  # em_slots * warn_dim
    assert space["passive_radar"].shape == (1 * 3,)

    all_spaces = mgr.all()
    assert set(all_spaces.keys()) == {"A1", "B1"}
