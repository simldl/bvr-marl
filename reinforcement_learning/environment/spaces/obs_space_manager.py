from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from gymnasium.spaces import Box, Dict as SpaceDict

def _fbox(shape, low=-np.inf, high=np.inf, dtype=np.float32):
    # Ensure low/high are float32 arrays to avoid precision warnings.
    low_arr  = np.full(shape, low,  dtype=np.float32)
    high_arr = np.full(shape, high, dtype=np.float32)
    return Box(low=low_arr, high=high_arr, shape=shape, dtype=dtype)

def _mbox(shape):  # mask boxes in [0,1], float32
    return _fbox(shape, low=0.0, high=1.0, dtype=np.float32)

@dataclass(frozen=True)
class EnvConfig:
    own_dim: int                 # Dimension of own_state
    fm_slots: int                # Number of friendly missile slots
    ff_slots: int                # Number of friendly fighter slots
    em_slots: int                # Number of enemy missile slots
    ef_slots: int                # Number of enemy fighter slots
    pr_slots: int                # Number of passive radar detection slots
    warn_sectors: int            # Number of warning sectors
    all_agent_ids: tuple = ()    # All agent IDs (optional, set by ObservationBuilder)

class ObservationSpaceManager:
    """
    Creates fixed Gym observation spaces per agent based on EnvConfig.

    Updated for new observation structure:
      - own_state: 22 dims (reduced from 26, removed DLZ range redundancy)
      - friendly_missiles: 8 dims per slot (added phase + seeker)
      - friendly_fighters: 6 dims per slot (unchanged)
      - enemy_missiles: 7 dims per slot (added TTI)
      - enemy_fighters: 9 dims per slot (added per-fighter NEZ: active + passive)
      - passive_radar: unchanged
      - missile_warning: unchanged
    """
    def __init__(self, agent_ids: List[str], config: EnvConfig):
        self.agent_ids = agent_ids
        self.config = config
        self.spaces = {aid: self._build_space(config) for aid in agent_ids}

    def _build_space(self, config: EnvConfig) -> SpaceDict:
        # Entity dimensions (updated)
        fm_ent_dim = 8  # [dx, dy, dz, dvx, dvy, dvz, phase, seeker_lock]
        ff_ent_dim = 6  # [dx, dy, dz, dvx, dvy, dvz]
        em_ent_dim = 7  # [dx, dy, dz, dvx, dvy, dvz, tti]
        ef_ent_dim = 9  # [dx, dy, dz, dvx, dvy, dvz, confidence, nez_active, nez_passive]

        warn_dim = 1 + config.warn_sectors

        obs_space = {
            # Own state (reduced from 26 to 22)
            "own_state": _fbox((config.own_dim,)),

            # Friendlies
            "friendly_missiles":      _fbox((config.fm_slots * fm_ent_dim,)),
            "mask_friendly_missiles": _mbox((config.fm_slots,)),

            "friendly_fighters":      _fbox((config.ff_slots * ff_ent_dim,)),
            "mask_friendly_fighters": _mbox((config.ff_slots,)),

            # Targets/locks (builder provides these)
            "fm_target_indices":      _fbox((config.fm_slots * 1,), low=-1.0, high=float(np.inf)),
            "mask_fm_targets":        _mbox((config.fm_slots,)),
            "ff_lock_indices":        _fbox((config.ff_slots * 1,), low=-1.0, high=float(np.inf)),
            "mask_ff_locks":          _mbox((config.ff_slots,)),

            # Enemies
            "enemy_missiles":         _fbox((config.em_slots * em_ent_dim,)),
            "mask_enemy_missiles":    _mbox((config.em_slots,)),

            "enemy_fighters":         _fbox((config.ef_slots * ef_ent_dim,)),
            "mask_enemy_fighters":    _mbox((config.ef_slots,)),

            # Missile warning (builder provides these)
            "missile_warning_flag":   _mbox((1,)),
            "missile_warning_dirs":   _mbox((config.em_slots * warn_dim,)),
            "mask_warning_dirs":      _mbox((config.em_slots,)),

            # Passive radar
            "passive_radar":          _fbox((config.pr_slots * 3,)),
            "mask_passive_radar":     _mbox((config.pr_slots,)),
        }
        return SpaceDict(obs_space)


    def get(self, agent_id: str) -> SpaceDict:
        return self.spaces[agent_id]

    def all(self) -> Dict[str, SpaceDict]:
        return self.spaces
