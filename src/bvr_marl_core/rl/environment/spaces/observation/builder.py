"""
Main Observation Builder - Orchestrates all sub-builders.

This is the main entry point for building observations. It delegates to
specialized builders for different observation components:
- OwnStateBuilder: Own state vector
- FriendlyInfoBuilder: Friendly units info
- EnemyInfoBuilder: Enemy units from sensor tracks
- MissileWarningBuilder: Missile warnings with sector encoding
- PassiveRadarBuilder: Passive radar detections
"""

import numpy as np

from bvr_marl_core.domain.information_mode import InformationMode, resolve_information_mode
from bvr_marl_core.domain.truth_access_guard import forbidden_truth_access
from bvr_marl_core.rl.environment.spaces.observation.enemy_info_builder import EnemyInfoBuilder
from bvr_marl_core.rl.environment.spaces.observation.friendly_info_builder import (
    FriendlyInfoBuilder,
)
from bvr_marl_core.rl.environment.spaces.observation.missile_warning_builder import (
    MissileWarningBuilder,
)
from bvr_marl_core.rl.environment.spaces.observation.own_state_builder import OwnStateBuilder
from bvr_marl_core.rl.environment.spaces.observation.passive_radar_builder import (
    PassiveRadarBuilder,
)


class ObservationBuilder:
    """
    Main observation builder that orchestrates all sub-builders.

    This class maintains the same interface as the original monolithic builder
    but delegates to specialized sub-builders for each observation component.

    ``enemy_builder_class`` is the extension point for widening the enemy-fighter
    token: a subclass may point it at an ``EnemyInfoBuilder`` subclass that appends
    extra columns. The env config must declare the matching ``ef_extra_dim``.
    """

    enemy_builder_class = EnemyInfoBuilder

    def __init__(self, simulator, agent_ids: list[str], config):
        """
        Initialize the observation builder with all sub-builders.

        Args:
            simulator: The simulation environment
            agent_ids: list of agent IDs to build observations for
            config: Configuration object with observation space parameters
        """
        self.simulator = simulator
        self.agent_ids = agent_ids
        self.all_agent_ids = agent_ids
        self.config = config

        # Store config values as attributes (for backward compatibility)
        self.num_fm = config.fm_slots
        self.num_ff = config.ff_slots
        self.num_em = config.em_slots
        self.num_ef = config.ef_slots
        self.num_pr = config.pr_slots
        self.num_warn = config.warn_sectors

        # Ensure config has all_agent_ids for builders
        # Use object.__setattr__ for frozen dataclasses
        if not hasattr(config, "all_agent_ids") or not config.all_agent_ids:
            object.__setattr__(config, "all_agent_ids", tuple(agent_ids))

        # Initialize sub-builders
        self._sensor_limited = (
            resolve_information_mode(
                getattr(config, "information_mode", None),
                default=InformationMode.SENSOR_LIMITED,
            )
            is InformationMode.SENSOR_LIMITED
        )
        self.own_state_builder = OwnStateBuilder(simulator, config)
        self.friendly_builder = FriendlyInfoBuilder(simulator, config)
        self.enemy_builder = self.enemy_builder_class(simulator, config)
        self.missile_warning_builder = MissileWarningBuilder(simulator, config)
        self.passive_radar_builder = PassiveRadarBuilder(simulator, config)

    def build(self, agent_id: str) -> dict[str, np.ndarray]:
        """
        Build complete observation for an agent.

        Delegates to sub-builders and assembles the complete observation dict.

        Structure (all entity blocks are self-contained tokens whose last column
        is a validity mask; relations/warnings are folded into their tokens):
        - own_state:         own-state vector (d_OWN,)
        - friendly_missiles: [fm_slots * d_FM]  (incl. target relation + mask)
        - friendly_fighters: [ff_slots * d_FF]  (incl. lock relation + mask)
        - enemy_missiles:    [em_slots * d_EM]  (incl. TTI + mask)
        - enemy_fighters:    [ef_slots * d_EF]  (incl. NEZ/DLZ/RID/BDA + mask)
        - missile_warnings:  [em_slots * d_MWS] (incl. mask)
        - passive_radar:     [pr_slots * d_PR]  (incl. mask)

        Args:
            agent_id: The agent ID to build observations for

        Returns:
            Dictionary of flattened observation arrays.
        """
        if self._sensor_limited:
            # Runtime firewall trip-wire: any track-identity -> truth-entity
            # resolution inside a sensor-limited observation build raises
            # (see domain.truth_access_guard). Own-state access is unaffected.
            with forbidden_truth_access("sensor_limited_observation"):
                return self._build(agent_id)
        return self._build(agent_id)

    def _build(self, agent_id: str) -> dict[str, np.ndarray]:
        unit = self.simulator.active_units[agent_id]
        obs: dict[str, np.ndarray] = {}

        # Own state
        obs["own_state"] = self.own_state_builder.build(unit)

        # Friendly tokens (target/lock relation + mask folded in)
        fm_tokens, ff_tokens = self.friendly_builder.build(agent_id)
        obs["friendly_missiles"] = fm_tokens.ravel()
        obs["friendly_fighters"] = ff_tokens.ravel()

        # Enemy tokens (mask folded in)
        em_tokens, ef_tokens = self.enemy_builder.build(agent_id)
        obs["enemy_missiles"] = em_tokens.ravel()
        obs["enemy_fighters"] = ef_tokens.ravel()

        # Threat/ESM tokens (mask folded in)
        obs["missile_warnings"] = self.missile_warning_builder.build(unit).ravel()
        obs["passive_radar"] = self.passive_radar_builder.build(unit).ravel()

        return obs
