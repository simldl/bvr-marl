from dataclasses import dataclass

import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as SpaceDict


def _fbox(shape, low=-np.inf, high=np.inf, dtype=np.float32):
    # Ensure low/high are float32 arrays to avoid precision warnings.
    low_arr = np.full(shape, low, dtype=np.float32)
    high_arr = np.full(shape, high, dtype=np.float32)
    return Box(low=low_arr, high=high_arr, shape=shape, dtype=dtype)


def _mbox(shape):  # mask boxes in [0,1], float32
    return _fbox(shape, low=0.0, high=1.0, dtype=np.float32)


@dataclass(frozen=True)
class EnvConfig:
    own_dim: int  # Dimension of own_state
    fm_slots: int  # Number of friendly missile slots
    ff_slots: int  # Number of friendly fighter slots
    em_slots: int  # Number of enemy missile slots
    ef_slots: int  # Number of enemy fighter slots
    pr_slots: int  # Number of passive radar detection slots
    warn_sectors: int  # Number of warning sectors
    all_agent_ids: tuple = ()  # All agent IDs (optional, set by ObservationBuilder)


class ObservationSpaceManager:
    """
    Creates fixed Gym observation spaces per agent based on EnvConfig.

    Updated for new observation structure:
      - own_state: 21 dims (SQI removed from policy observations)
      - friendly_missiles: 8 dims per slot (added phase + seeker)
      - friendly_fighters: 6 dims per slot (unchanged)
      - enemy_missiles: 7 dims per slot (added TTI)
      - enemy_fighters: 10 dims per slot (added per-fighter NEZ: active + passive, is_support_asset)
      - passive_radar: unchanged
      - missile_warning: unchanged
    """

    def __init__(self, agent_ids: list[str], config: EnvConfig):
        self.agent_ids = agent_ids
        self.config = config
        self.spaces = {aid: self._build_space(config) for aid in agent_ids}

    def _build_space(self, config: EnvConfig) -> SpaceDict:
        # Entity dimensions (updated)
        fm_ent_dim = 8  # [dx, dy, dz, dvx, dvy, dvz, phase, seeker_lock]
        ff_ent_dim = 6  # [dx, dy, dz, dvx, dvy, dvz]
        em_ent_dim = 7  # [dx, dy, dz, dvx, dvy, dvz, tti]
        ef_ent_dim = (
            10  # [dx, dy, dz, dvx, dvy, dvz, confidence, nez_active, nez_passive, is_support_asset]
        )

        warn_dim = 1 + config.warn_sectors

        obs_space = {
            # Own state (SQI removed; available only through metrics/evaluation)
            "own_state": _fbox((config.own_dim,)),
            # Friendlies
            "friendly_missiles": _fbox((config.fm_slots * fm_ent_dim,)),
            "mask_friendly_missiles": _mbox((config.fm_slots,)),
            "friendly_fighters": _fbox((config.ff_slots * ff_ent_dim,)),
            "mask_friendly_fighters": _mbox((config.ff_slots,)),
            # Targets/locks (builder provides these)
            "fm_target_indices": _fbox((config.fm_slots * 1,), low=-1.0, high=float(np.inf)),
            "mask_fm_targets": _mbox((config.fm_slots,)),
            "ff_lock_indices": _fbox((config.ff_slots * 1,), low=-1.0, high=float(np.inf)),
            "mask_ff_locks": _mbox((config.ff_slots,)),
            # Enemies
            "enemy_missiles": _fbox((config.em_slots * em_ent_dim,)),
            "mask_enemy_missiles": _mbox((config.em_slots,)),
            "enemy_fighters": _fbox((config.ef_slots * ef_ent_dim,)),
            "mask_enemy_fighters": _mbox((config.ef_slots,)),
            # Missile warning (builder provides these)
            "missile_warning_flag": _mbox((1,)),
            "missile_warning_dirs": _mbox((config.em_slots * warn_dim,)),
            "mask_warning_dirs": _mbox((config.em_slots,)),
            # Passive radar
            "passive_radar": _fbox((config.pr_slots * 3,)),
            "mask_passive_radar": _mbox((config.pr_slots,)),
        }
        return SpaceDict(obs_space)

    def get(self, agent_id: str) -> SpaceDict:
        return self.spaces[agent_id]

    def all(self) -> dict[str, SpaceDict]:
        return self.spaces


@dataclass(frozen=True)
class SimplifiedEnvConfig:
    """Configuration for SimplifiedObservationSpaceManager."""

    ff_slots: int  # Friendly fighter slots (excluding self)
    ef_slots: int  # Enemy fighter slots
    em_slots: int  # Warning slots (re-used for missile warner)
    warn_sectors: int  # Number of warning sectors per missile warning
    all_agent_ids: tuple = ()


class SimplifiedObservationSpaceManager:
    """
    Gym observation space for the lightweight training environment.

    Keys per agent:
      own_state              (7,)               lat, lon, alt, yaw, speed, pitch, missiles_remaining
      friendly_fighters      (ff_slots * 4,)    [lat, lon, alt, speed] per friendly
      mask_friendly_fighters (ff_slots,)
      enemy_fighters         (ef_slots * 4,)    [lat, lon, alt, speed] per enemy
      mask_enemy_fighters    (ef_slots,)
      missile_warning_flag   (1,)
      missile_warning_dirs   (em_slots * (1 + warn_sectors),)
      mask_warning_dirs      (em_slots,)
    """

    # Dimensions of truth-position entity entries
    ENTITY_DIM = 4  # [lat, lon, alt, speed]
    OWN_DIM = 7  # [lat, lon, alt, yaw, speed, pitch, missiles_remaining]

    def __init__(self, agent_ids: list[str], config: SimplifiedEnvConfig):
        self.agent_ids = agent_ids
        self.config = config
        self.spaces = {aid: self._build_space(config) for aid in agent_ids}

    def _build_space(self, config: SimplifiedEnvConfig) -> SpaceDict:
        warn_dim = 1 + config.warn_sectors
        obs_space = {
            "own_state": _fbox((self.OWN_DIM,)),
            "friendly_fighters": _fbox((config.ff_slots * self.ENTITY_DIM,)),
            "mask_friendly_fighters": _mbox((config.ff_slots,)),
            "enemy_fighters": _fbox((config.ef_slots * self.ENTITY_DIM,)),
            "mask_enemy_fighters": _mbox((config.ef_slots,)),
            "missile_warning_flag": _mbox((1,)),
            "missile_warning_dirs": _mbox((config.em_slots * warn_dim,)),
            "mask_warning_dirs": _mbox((config.em_slots,)),
        }
        return SpaceDict(obs_space)

    def get(self, agent_id: str) -> SpaceDict:
        return self.spaces[agent_id]

    def all(self) -> dict[str, SpaceDict]:
        return self.spaces
