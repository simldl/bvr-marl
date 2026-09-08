from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as SpaceDict

from bvr_marl_core.rl.environment.spaces.observation.constants import (
    WARN_FEATURE_DIM,
    d_EM,
    d_FF,
    d_FM,
    d_MWS,
    d_PR,
    ef_token_dim,
)


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
    information_mode: str = "sensor_limited"
    oracle_use_reason: str | None = None
    ef_extra_dim: int = 0  # extra enemy-fighter token columns supplied by an extension
    emcon_action_enabled: bool = False  # widen ownship vector with own radar-emission state
    # Opaque passthrough for an extension package's own builder options. Core never
    # reads it; it exists so a subclass builder can be configured without core
    # growing a field per extension.
    extension_options: Mapping[str, Any] = field(default_factory=dict)


class ObservationSpaceManager:
    """
    Creates fixed Gym observation spaces per agent based on EnvConfig.

    Observation structure (mask-driven tokens): every entity block is a flat
    ``slots * token_dim`` vector where each token's last column is a validity
    mask. Relations (missile/fighter -> enemy slot) and warnings are folded into
    their tokens, so there are no separate mask/index keys. The key order here
    matches the encoder's branch order.
      - own_state:         (own_dim,)
      - friendly_missiles: (fm_slots * d_FM)
      - friendly_fighters: (ff_slots * d_FF)
      - enemy_missiles:    (em_slots * d_EM)
      - enemy_fighters:    (ef_slots * d_EF)
      - missile_warnings:  (em_slots * d_MWS)
      - passive_radar:     (pr_slots * d_PR)
    """

    def __init__(self, agent_ids: list[str], config: EnvConfig):
        self.agent_ids = agent_ids
        self.config = config
        self.spaces = {aid: self._build_space(config) for aid in agent_ids}

    def _build_space(self, config: EnvConfig) -> SpaceDict:
        d_ef = ef_token_dim(config.ef_extra_dim)
        obs_space = {
            "own_state": _fbox((config.own_dim,)),
            "friendly_missiles": _fbox((config.fm_slots * d_FM,)),
            "friendly_fighters": _fbox((config.ff_slots * d_FF,)),
            "enemy_missiles": _fbox((config.em_slots * d_EM,)),
            "enemy_fighters": _fbox((config.ef_slots * d_ef,)),
            "missile_warnings": _fbox((config.em_slots * d_MWS,)),
            "passive_radar": _fbox((config.pr_slots * d_PR,)),
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
        warn_dim = WARN_FEATURE_DIM
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
