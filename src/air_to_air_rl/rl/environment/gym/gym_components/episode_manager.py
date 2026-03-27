"""
Episode reset and initialization logic.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
from gymnasium import spaces

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from air_to_air_rl.rl.environment.gym.env_helpers import (
    coerce_obs_to_space,
    spaces_from_sample_observation,
    to_f32_obs,
)
from air_to_air_rl.rl.environment.gym.spawn_utils import (
    compute_awacs_orbit_center,
    spawn_awacs,
    spawn_unit,
    spawn_unit_with_geometry,
    spawn_with_geometry_config,
)
from air_to_air_rl.rl.environment.scenarios.geometry_sampler import GeometrySampler

if TYPE_CHECKING:
    from air_to_air_rl.rl.environment.spaces.obs_space_builder import ObservationBuilder
    from air_to_air_rl.simulator.core.simulator import Simulator

logger = logging.getLogger(__name__)


class EpisodeManager:
    """Manages episode reset and initialization."""

    def __init__(
        self,
        simulator: Simulator,
        obs_builder,
        bvr_config,
        original_config_dict,
        obs_space_mgr=None,
    ):
        self.simulator = simulator
        self.obs_builder = obs_builder
        self.bvr_config = bvr_config
        self.obs_space_mgr = obs_space_mgr

        # Keep original config dict as .config for spawn_unit compatibility
        self.config = original_config_dict

        # Agent management
        self.agent_to_unit_id: dict[str, int] = {}
        self.agents: list[str] = []  # Currently alive agents

        # AWACS unit IDs (not part of agent_to_unit_id since they're not RL agents)
        self.awacs_unit_ids: dict[str, int] = {}  # "agent" -> uid, "opponent" -> uid

        # Observation spaces
        self.observation_space: dict[str, spaces.Space] = {}
        self._need_space_update = True
        self._printed_space_keys = False

        # Radar lock fix tracking
        self.radar_locks_fixed = False

        # Expose attributes needed by spawn_unit / spawn_awacs
        self.map_limits = bvr_config.map_limits
        self.full_map_limits = bvr_config.full_map_limits or bvr_config.map_limits
        self.map_size_km = bvr_config.map_size_km
        self.aircraft_type_map = bvr_config.aircraft_type_map

        # Geometry sampler for controlled spawning
        self.geometry_sampler = GeometrySampler(map_size_km=bvr_config.map_size_km)

        # Track current episode's regime for logging/debugging
        self.current_regime: str | None = None

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Reset the environment for a new episode."""
        if seed is not None:
            # Seed all RNG sources for reproducibility
            random.seed(seed)
            np.random.seed(seed)

            # Seed PyTorch if available
            if HAS_TORCH:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Seed simulator if it supports it
            if hasattr(self.simulator, "seed"):
                try:
                    self.simulator.seed(seed)
                except Exception:
                    pass

        self.simulator.reset_sim(units={})
        self.agent_to_unit_id.clear()
        self.awacs_unit_ids.clear()

        # Reset radar lock fix flag
        self.radar_locks_fixed = False

        # Spawn fighters using scenario-based or random spawning
        geometry_data = self._spawn_fighters()

        # Spawn AWACS if configured
        self._spawn_awacs_units(geometry_data)

        # Update obs builder with unit IDs (fighters only, not AWACS)
        try:
            self.obs_builder.all_agent_ids = list(self.agent_to_unit_id.values())
        except Exception:
            pass

        # Build raw observations
        raw_obs: dict[str, dict[str, np.ndarray]] = {}
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
                self.observation_space = {
                    aid: sample_space for aid in self.bvr_config.all_agent_ids
                }
                self._need_space_update = False

        # Coerce obs strictly to spaces
        obs: dict[str, dict[str, np.ndarray]] = {}
        infos: dict[str, dict] = {}
        for aid in self.bvr_config.all_agent_ids:
            obs[aid] = coerce_obs_to_space(self.observation_space[aid], raw_obs[aid])
            infos[aid] = {}

        self.agents = list(self.bvr_config.all_agent_ids)

        if not self._printed_space_keys:
            first = next(iter(self.observation_space))
            logger.debug("OBS SPACE KEYS: %s", list(self.observation_space[first].spaces.keys()))
            self._printed_space_keys = True

        return obs, infos

    def _get_allowed_regimes(self) -> list[str]:
        """Get the list of allowed geometry regimes for spawning."""
        # Check scenario_config first
        scenario_config = self.bvr_config.scenario_config
        if scenario_config.allowed_regimes:
            return scenario_config.allowed_regimes

        # Fall back to geometry_config regime as single-element list
        geometry_config = self.bvr_config.geometry_config
        if geometry_config.use_controlled_geometry:
            return [geometry_config.regime]

        # No controlled geometry - return empty (will use random spawning)
        return []

    def _select_regime(self) -> str | None:
        """Select a geometry regime for this episode."""
        allowed_regimes = self._get_allowed_regimes()

        if not allowed_regimes:
            return None

        scenario_config = self.bvr_config.scenario_config
        if scenario_config.randomize_regime and len(allowed_regimes) > 1:
            # Randomly select from allowed regimes
            return random.choice(allowed_regimes)
        else:
            # Use first/only regime
            return allowed_regimes[0]

    def _spawn_fighters(self) -> dict | None:
        """
        Spawn fighter aircraft using either controlled geometry or random spawning.

        Returns:
            Geometry data dict if using controlled geometry, None otherwise.
        """
        geometry_config = self.bvr_config.geometry_config
        scenario_config = self.bvr_config.scenario_config

        # Determine if we should use controlled geometry
        use_geometry = geometry_config.use_controlled_geometry or bool(
            scenario_config.allowed_regimes
        )

        if use_geometry:
            # Select regime for this episode
            regime = self._select_regime()
            self.current_regime = regime

            if regime:
                logger.debug(f"Using controlled geometry regime: {regime}")

                # Sample geometry with optional noise
                if (
                    geometry_config.position_noise_m > 0
                    or geometry_config.altitude_noise_m > 0
                    or geometry_config.heading_noise_deg > 0
                ):
                    geometry = self.geometry_sampler.sample_with_noise(
                        regime,
                        position_noise_m=geometry_config.position_noise_m,
                        altitude_noise_m=geometry_config.altitude_noise_m,
                        heading_noise_deg=geometry_config.heading_noise_deg,
                    )
                else:
                    geometry = self.geometry_sampler.sample(regime)

                # Compute positions
                geometry_data = self.geometry_sampler.compute_positions(
                    geometry,
                    num_agents=len(self.bvr_config.agent_ids),
                    num_opponents=len(self.bvr_config.opponent_ids),
                    formation_spread_m=geometry_config.formation_spread_m,
                )

                # Spawn using geometry
                self.agent_to_unit_id = spawn_with_geometry_config(self, geometry_data)
                return geometry_data

        # Fall back to random spawning
        self.current_regime = None
        logger.debug("Using random spawning")

        for aid in self.bvr_config.all_agent_ids:
            group = "agent" if aid in self.bvr_config.agent_ids else "opponent"
            uid = spawn_unit(self, aid, group)
            self.agent_to_unit_id[aid] = uid

        return None

    def _spawn_awacs_units(self, geometry_data: dict | None) -> None:
        """
        Spawn AWACS units if configured in scenario_config.

        Args:
            geometry_data: Geometry data from fighter spawning (for positioning AWACS).
        """
        scenario_config = self.bvr_config.scenario_config
        awacs_config = scenario_config.awacs_config

        # Spawn agent team AWACS (blue / west side)
        if awacs_config.agent_awacs:
            try:
                uid = spawn_awacs(self, "agent", awacs_config)
                if uid is not None:
                    self.awacs_unit_ids["agent"] = uid
                    logger.debug(f"Spawned agent AWACS with unit ID {uid}")
            except Exception as e:
                logger.warning(f"Failed to spawn agent AWACS: {e}")

        # Spawn opponent team AWACS (red / east side)
        if awacs_config.opponent_awacs:
            try:
                uid = spawn_awacs(self, "opponent", awacs_config)
                if uid is not None:
                    self.awacs_unit_ids["opponent"] = uid
                    logger.debug(f"Spawned opponent AWACS with unit ID {uid}")
            except Exception as e:
                logger.warning(f"Failed to spawn opponent AWACS: {e}")

    def get_agent_position(self, agent_id: str) -> tuple | None:
        """Get current position of an agent."""
        uid = self.agent_to_unit_id.get(agent_id)
        unit = self.simulator.active_units.get(uid) if uid is not None else None
        if unit and hasattr(unit, "position"):
            return (unit.position.lon, unit.position.lat, getattr(unit.position, "alt", 0.0))
        return None

    def create_dummy_observation_space(self) -> spaces.Space:
        """Create a dummy observation space that matches the expected structure."""
        obs_dict = {}
        obs_dict["own_state"] = spaces.Box(
            low=np.full((self.bvr_config.own_state_dim,), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.own_state_dim,), np.inf, dtype=np.float32),
            shape=(self.bvr_config.own_state_dim,),
            dtype=np.float32,
        )
        obs_dict["friendly_fighters"] = spaces.Box(
            low=np.full((self.bvr_config.num_ff, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_ff, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_ff, 7),
            dtype=np.float32,
        )
        obs_dict["friendly_missiles"] = spaces.Box(
            low=np.full((self.bvr_config.num_fm, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_fm, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_fm, 7),
            dtype=np.float32,
        )
        obs_dict["enemy_fighters"] = spaces.Box(
            low=np.full((self.bvr_config.num_ef, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_ef, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_ef, 7),
            dtype=np.float32,
        )
        obs_dict["enemy_missiles"] = spaces.Box(
            low=np.full((self.bvr_config.num_em, 7), -np.inf, dtype=np.float32),
            high=np.full((self.bvr_config.num_em, 7), np.inf, dtype=np.float32),
            shape=(self.bvr_config.num_em, 7),
            dtype=np.float32,
        )

        # Add mask spaces and other required spaces...
        return spaces.dict(obs_dict)
