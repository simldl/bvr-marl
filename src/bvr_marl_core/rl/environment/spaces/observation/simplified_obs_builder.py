"""
Simplified observation builder for the lightweight training environment.

Reads truth data directly from bvr_marl_core.simulator.active_units — no radar pipeline required.

Observation keys per agent:
  own_state              (7,)  — lat, lon, alt, yaw_deg, speed, pitch_deg, missiles_remaining
  friendly_fighters      (ff_slots * 4,)  — [lat, lon, alt, speed] per teammate
  mask_friendly_fighters (ff_slots,)
  enemy_fighters         (ef_slots * 4,)  — [lat, lon, alt, speed] per enemy
  mask_enemy_fighters    (ef_slots,)
  missile_warning_flag   (1,)
  missile_warning_dirs   (em_slots * (1 + warn_sectors),)
  mask_warning_dirs      (em_slots,)
"""

from __future__ import annotations

import numpy as np

from bvr_marl_core.rl.environment.spaces.obs_space_manager import SimplifiedEnvConfig
from bvr_marl_core.rl.environment.spaces.observation.missile_warning_builder import (
    MissileWarningBuilder,
)


def _pad(rows: list, n_slots: int, dim: int):
    """Pad or truncate a list of 1-D arrays to (n_slots, dim). Returns (arr, mask)."""
    arr = np.zeros((n_slots, dim), dtype=np.float32)
    mask = np.zeros(n_slots, dtype=np.float32)
    for i, row in enumerate(rows[:n_slots]):
        arr[i] = row
        mask[i] = 1.0
    return arr, mask


class SimplifiedObservationBuilder:
    """
    Builds observations for all agents using truth positions / speeds from the
    simulator without touching the radar tracking pipeline.
    """

    # Dimension of the entity vector for each other aircraft
    ENTITY_DIM = 4  # [lat, lon, alt, speed]
    OWN_DIM = 7  # [lat, lon, alt, yaw_deg, speed, pitch_deg, missiles_remaining]

    def __init__(self, simulator, agent_ids: list[str], config: SimplifiedEnvConfig):
        self.simulator = simulator
        self.agent_ids = agent_ids  # friendly agent IDs (trainable side)
        self.all_agent_ids = list(agent_ids)
        self.config = config

        self._warning_builder = MissileWarningBuilder(simulator, config)

    # Public interface

    def build(self, unit_id: str) -> dict[str, np.ndarray]:
        """Build complete observation for the unit with the given ID."""
        unit = self.simulator.active_units[unit_id]
        obs: dict[str, np.ndarray] = {}

        # Own state
        if hasattr(unit, "remaining_missiles"):
            remaining_missiles = float(unit.remaining_missiles)
        elif hasattr(unit, "weapons") and hasattr(unit.weapons, "remaining_missiles"):
            remaining_missiles = float(unit.weapons.remaining_missiles)
        elif hasattr(unit, "max_missiles") and hasattr(unit, "missiles"):
            remaining_missiles = float(unit.max_missiles - len(unit.missiles))
        else:
            remaining_missiles = 0.0
        obs["own_state"] = np.array(
            [
                unit.position.lat,
                unit.position.lon,
                unit.position.alt,
                unit.yaw_deg,
                unit.speed,
                unit.pitch_deg,
                remaining_missiles,
            ],
            dtype=np.float32,
        )

        # Categorise all alive non-missile units (excluding self)
        friendlies = []
        enemies = []
        for uid, u in self.simulator.active_units.items():
            if uid == unit_id:
                continue
            if getattr(u, "is_missile", False):
                continue
            row = np.array(
                [u.position.lat, u.position.lon, u.position.alt, u.speed], dtype=np.float32
            )
            if u.group == unit.group:
                friendlies.append(row)
            else:
                enemies.append(row)

        # Friendly fighters
        ff_arr, ff_mask = _pad(friendlies, self.config.ff_slots, self.ENTITY_DIM)
        obs["friendly_fighters"] = ff_arr.flatten()
        obs["mask_friendly_fighters"] = ff_mask

        # Enemy fighters
        ef_arr, ef_mask = _pad(enemies, self.config.ef_slots, self.ENTITY_DIM)
        obs["enemy_fighters"] = ef_arr.flatten()
        obs["mask_enemy_fighters"] = ef_mask

        # Missile warnings (reuse existing builder — independent of radar)
        flag, dirs, dirs_mask = self._warning_builder.build(unit)
        obs["missile_warning_flag"] = np.array([flag], dtype=np.float32)
        obs["missile_warning_dirs"] = dirs
        obs["mask_warning_dirs"] = dirs_mask

        return obs
