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
from typing import List, Dict
import numpy as np

from .own_state_builder import OwnStateBuilder
from .friendly_info_builder import FriendlyInfoBuilder
from .enemy_info_builder import EnemyInfoBuilder
from .missile_warning_builder import MissileWarningBuilder
from .passive_radar_builder import PassiveRadarBuilder


class ObservationBuilder:
    """
    Main observation builder that orchestrates all sub-builders.

    This class maintains the same interface as the original monolithic builder
    but delegates to specialized sub-builders for each observation component.
    """

    def __init__(self, simulator, agent_ids: List[str], config):
        """
        Initialize the observation builder with all sub-builders.

        Args:
            simulator: The simulation environment
            agent_ids: List of agent IDs to build observations for
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
        if not hasattr(config, 'all_agent_ids') or not config.all_agent_ids:
            object.__setattr__(config, 'all_agent_ids', tuple(agent_ids))

        # Initialize sub-builders
        self.own_state_builder = OwnStateBuilder(simulator, config)
        self.friendly_builder = FriendlyInfoBuilder(simulator, config)
        self.enemy_builder = EnemyInfoBuilder(simulator, config)
        self.missile_warning_builder = MissileWarningBuilder(simulator, config)
        self.passive_radar_builder = PassiveRadarBuilder(simulator, config)

    def build(self, agent_id: str) -> Dict[str, np.ndarray]:
        """
        Build complete observation for an agent.

        Delegates to sub-builders and assembles the complete observation dict.

        Updated structure:
        - NEZ features are now embedded in enemy_fighters (no separate arrays)
        - Friendly missiles include phase and seeker lock
        - Enemy missiles include TTI

        Args:
            agent_id: The agent ID to build observations for

        Returns:
            Dictionary of observation arrays with keys:
            - own_state: Own state vector (22 dims)
            - friendly_missiles, mask_friendly_missiles: Friendly missile states (8 dims/slot)
            - friendly_fighters, mask_friendly_fighters: Friendly fighter states (6 dims/slot)
            - fm_target_indices, mask_fm_targets: Missile target indices
            - ff_lock_indices, mask_ff_locks: Fighter lock indices
            - enemy_missiles, mask_enemy_missiles: Enemy missile states (7 dims/slot)
            - enemy_fighters, mask_enemy_fighters: Enemy fighter states (9 dims/slot, includes NEZ)
            - missile_warning_flag: Warning count normalized
            - missile_warning_dirs, mask_warning_dirs: Warning directions
            - passive_radar, mask_passive_radar: Passive radar detections
        """
        unit = self.simulator.active_units[agent_id]
        obs: Dict[str, np.ndarray] = {}

        # Build own state
        obs["own_state"] = self.own_state_builder.build(unit)

        # Build friendly info
        fm_s, fm_m, ff_s, ff_m, fm_t, fm_t_m, ff_l, ff_l_m = self.friendly_builder.build(agent_id)
        obs.update({
            "friendly_missiles":       fm_s.flatten(),
            "mask_friendly_missiles":  fm_m,
            "friendly_fighters":       ff_s.flatten(),
            "mask_friendly_fighters":  ff_m,
            "fm_target_indices":       fm_t,
            "mask_fm_targets":         fm_t_m,
            "ff_lock_indices":         ff_l,
            "mask_ff_locks":           ff_l_m,
        })

        # Build enemy info (NEZ now embedded in ef_s, no separate arrays)
        em_s, em_m, ef_s, ef_m = self.enemy_builder.build(agent_id)
        obs.update({
            "enemy_missiles":        em_s.flatten(),
            "mask_enemy_missiles":   em_m,
            "enemy_fighters":        ef_s.flatten(),
            "mask_enemy_fighters":   ef_m,
        })

        # Build missile warnings
        flag, dirs, dirs_m = self.missile_warning_builder.build(unit)
        obs.update({
            "missile_warning_flag":   np.array([flag], np.float32),
            "missile_warning_dirs":   dirs,
            "mask_warning_dirs":      dirs_m,
        })

        # Build passive radar
        pr, pr_m = self.passive_radar_builder.build(unit)
        obs.update({
            "passive_radar":          pr.flatten(),
            "mask_passive_radar":     pr_m,
        })

        return obs
