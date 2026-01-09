"""
Episode reset and initialization logic.
"""

from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from gymnasium import spaces

from reinforcement_learning.environment.gym.spawn_utils import spawn_unit
from reinforcement_learning.environment.gym.env_helpers import (
    to_f32_obs,
    coerce_obs_to_space,
    spaces_from_sample_observation,
)

if TYPE_CHECKING:
    from simulator.core.simulator import Simulator
    from reinforcement_learning.environment.spaces.obs_space_builder import ObservationBuilder


class EpisodeManager:
    """Manages episode reset and initialization."""

    def __init__(self, simulator: Simulator, obs_builder, bvr_config, original_config_dict, obs_space_mgr=None):
        self.simulator = simulator
        self.obs_builder = obs_builder
        self.bvr_config = bvr_config
        self.obs_space_mgr = obs_space_mgr
        
        # Keep original config dict as .config for spawn_unit compatibility
        self.config = original_config_dict

        # Agent management
        self.agent_to_unit_id: Dict[str, int] = {}
        self.agents: list[str] = []  # Currently alive agents

        # Observation spaces
        self.observation_space: Dict[str, spaces.Space] = {}
        self._need_space_update = True
        self._printed_space_keys = False

        # Radar lock fix tracking
        self.radar_locks_fixed = False
        
        # Expose attributes needed by spawn_unit
        self.map_limits = bvr_config.map_limits
        self.map_size_km = bvr_config.map_size_km
        self.aircraft_type_map = bvr_config.aircraft_type_map

    def reset(self, seed: Optional[int] = None) -> tuple[Dict, Dict]:
        """Reset the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self.simulator, "seed"):
                try:
                    self.simulator.seed(seed)
                except Exception:
                    pass

        self.simulator.reset_sim(units={})
        self.agent_to_unit_id.clear()

        # Reset radar lock fix flag
        self.radar_locks_fixed = False

        # Spawn and store numeric ids
        for aid in self.bvr_config.all_agent_ids:
            group = "agent" if aid in self.bvr_config.agent_ids else "opponent"
            uid = spawn_unit(self, aid, group)
            self.agent_to_unit_id[aid] = uid

        # Update obs builder with unit IDs
        try:
            self.obs_builder.all_agent_ids = list(self.agent_to_unit_id.values())
        except Exception:
            pass

        # Build raw observations
        raw_obs: Dict[str, Dict[str, np.ndarray]] = {}
        for aid in self.bvr_config.all_agent_ids:
            ob = self.obs_builder.build(self.agent_to_unit_id[aid])
            raw_obs[aid] = to_f32_obs(ob)

        # Update observation spaces with real spaces after reset
        if self._need_space_update or not self.observation_space:
            try:
                if self.obs_space_mgr:
                    mgr = self.obs_space_mgr.all()
                    if all(isinstance(v, spaces.Space) for v in mgr.values()):
                        self.observation_space = dict(mgr)
                        self._need_space_update = False
                    else:
                        raise TypeError
                else:
                    raise TypeError
            except Exception:
                # Build from actual observations to avoid shape mismatches
                sample_any = next(iter(raw_obs.values()))
                sample_space = spaces_from_sample_observation(sample_any)
                self.observation_space = {aid: sample_space for aid in self.bvr_config.all_agent_ids}
                self._need_space_update = False

        # Coerce obs strictly to spaces
        obs: Dict[str, Dict[str, np.ndarray]] = {}
        infos: Dict[str, dict] = {}
        for aid in self.bvr_config.all_agent_ids:
            obs[aid] = coerce_obs_to_space(self.observation_space[aid], raw_obs[aid])
            infos[aid] = {}

        self.agents = list(self.bvr_config.all_agent_ids)

        if not self._printed_space_keys:
            import logging
            logger = logging.getLogger(__name__)
            first = next(iter(self.observation_space))
            logger.debug("OBS SPACE KEYS: %s", list(self.observation_space[first].spaces.keys()))
            self._printed_space_keys = True

        return obs, infos

    def get_agent_position(self, agent_id: str) -> Optional[tuple]:
        """Get current position of an agent."""
        uid = self.agent_to_unit_id.get(agent_id)
        unit = self.simulator.active_units.get(uid) if uid is not None else None
        if unit and hasattr(unit, "position"):
            return (unit.position.lon, unit.position.lat, getattr(unit.position, 'alt', 0.0))
        return None

    def create_dummy_observation_space(self) -> spaces.Space:
        """Create a dummy observation space that matches the expected structure."""
        obs_dict = {}
        obs_dict["own_state"] = spaces.Box(
            low=np.full((self.bvr_config.own_state_dim,), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.own_state_dim,), np.inf, dtype=np.float32),
            shape=(self.bvr_config.own_state_dim,), dtype=np.float32)
        obs_dict["friendly_fighters"] = spaces.Box(
            low=np.full((self.bvr_config.num_ff, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_ff, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_ff, 7), dtype=np.float32)
        obs_dict["friendly_missiles"] = spaces.Box(
            low=np.full((self.bvr_config.num_fm, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_fm, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_fm, 7), dtype=np.float32)
        obs_dict["enemy_fighters"] = spaces.Box(
            low=np.full((self.bvr_config.num_ef, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_ef, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_ef, 7), dtype=np.float32)
        obs_dict["enemy_missiles"] = spaces.Box(
            low=np.full((self.bvr_config.num_em, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_em, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_em, 7), dtype=np.float32)

        # Add mask spaces and other required spaces...
        return spaces.Dict(obs_dict)
