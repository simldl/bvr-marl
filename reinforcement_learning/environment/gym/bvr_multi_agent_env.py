from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import logging

from ray.rllib.policy import Policy

logger = logging.getLogger(__name__)

from reinforcement_learning.environment.gym.base_env import BaseMultiAgentEnv
from reinforcement_learning.environment.spaces.obs_space_manager import (
    EnvConfig,
    ObservationSpaceManager,
)
from reinforcement_learning.environment.spaces.obs_space_builder import ObservationBuilder
from reinforcement_learning.environment.spaces.action_space import (
    ActionSpaceManager,
    EnergyLiftVectorActionProcessor,
)
from tacview.logger import TacviewLogger

from reinforcement_learning.environment.gym.gym_components import (
    BVREnvConfig,
    StateTracker,
    EpisodeManager,
    StepProcessor,
    TerminationChecker,
    AgentHelpers,
)
from reinforcement_learning.environment.gym.gym_components.reward_config import create_reward_calculator


class BVRMultiAgentEnv(BaseMultiAgentEnv):
    """
    Multi-agent env:
    - All spaces are gymnasium Spaces (float32).
    - Observations are strictly coerced to their subspaces.
    - Actions are normalized Box([0,1]^10) and applied to numeric unit-ids.
    - 10-dim action space: [Ps, n, φ, target, missile_fire, gun_fire, 4×CM]
      where Ps = specific energy rate, n = load factor, φ = bank angle
    """

    def __init__(self, config: dict, model: Optional[Policy] = None):
        super().__init__(config)
        self.debug = bool(config.get("debug", False))
        self.model = model

        # Parse configuration
        self.config = BVREnvConfig.from_dict(config)

        # Episode tracking
        self.current_step = 0

        # Expose config attributes for backward compatibility
        self.max_steps = self.config.max_steps
        self.tick_secs = self.config.tick_secs

        # Tacview logging (optional)
        self.tacview_logger = TacviewLogger(self.config.tacview_logfile) if self.config.tacview_logfile else None

        # PettingZoo-like attributes
        self.possible_agents = self.config.possible_agents
        self.agents: list[str] = []

        # Initialize reward calculator
        self.reward_calculator = create_reward_calculator(config)

        # Builders/Managers
        env_conf = EnvConfig(
            own_dim=self.config.own_state_dim,
            fm_slots=self.config.num_fm, ff_slots=self.config.num_ff,
            em_slots=self.config.num_em, ef_slots=self.config.num_ef,
            pr_slots=self.config.num_pr, warn_sectors=self.config.num_warn,
        )
        self.obs_space_mgr = ObservationSpaceManager(self.config.all_agent_ids, env_conf)
        self.obs_builder = ObservationBuilder(self.simulator, self.config.all_agent_ids, env_conf)
        self.action_processor = EnergyLiftVectorActionProcessor(self.simulator)

        # Configure automated missile firing
        self.action_processor.configure_automation(
            enable_missile_automation=self.config.enable_missile_automation,
            missile_auto_sqi_threshold=self.config.missile_auto_sqi_threshold,
            missile_auto_max_per_target=self.config.missile_auto_max_per_target,
            missile_auto_long_cooldown_s=self.config.missile_auto_long_cooldown_s
        )

        # Link action processor to simulator
        self.simulator.action_processor = self.action_processor

        # Initialize action spaces
        self.action_space_mgr = ActionSpaceManager(self.config.all_agent_ids, shape=10)
        self.action_space = self.action_space_mgr.all()

        # Initialize modular components
        self.state_tracker = StateTracker()
        self.episode_manager = EpisodeManager(self.simulator, self.obs_builder, self.config, config, self.obs_space_mgr)
        self.helpers = AgentHelpers(
            self.simulator, self.config.agent_ids, self.config.opponent_ids,
            self.episode_manager.agent_to_unit_id, self.config.force_locks_all_enemies
        )
        self.step_processor = StepProcessor(
            self.simulator, self.obs_builder, self.action_processor,
            self.reward_calculator, self.config
        )
        self.termination_checker = TerminationChecker(self.simulator, self.config)

        # Set observation space from episode manager
        self.observation_space = self.episode_manager.observation_space

        # Override base class attributes with config values
        self.agent_ids = self.config.agent_ids
        self.opponent_ids = self.config.opponent_ids
        self.all_agent_ids = self.config.all_agent_ids
        self.agent_to_unit_id = self.episode_manager.agent_to_unit_id

        # Expose map_limits for visualization
        self.map_limits = self.episode_manager.map_limits

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for new episode."""
        self.current_step = 0
        self.state_tracker.reset()
        self.termination_checker.reset_timing(self.config.tick_secs)

        # Reset episode
        obs, infos = self.episode_manager.reset(seed=seed)

        # Update observation space reference
        self.observation_space = self.episode_manager.observation_space

        # Initialize state tracker for all agents
        for aid in self.config.all_agent_ids:
            position = self.episode_manager.get_agent_position(aid)
            self.state_tracker.initialize_agent(aid, position)

        self.agents = self.episode_manager.agents

        return obs, infos

    def step(self, actions: Optional[Dict[str, np.ndarray]] = None):
        """Execute one environment step."""
        self.current_step += 1
        actions = actions or {}

        # Get currently alive agents
        current_agents = list(self.agents) if self.agents else list(self.config.all_agent_ids)

        # Fix radar locks if needed
        if self.config.fix_radar_lock_after_first_obs and not self.episode_manager.radar_locks_fixed and self.current_step == 1:
            self.helpers.fix_all_radar_locks(self.config.all_agent_ids)
            self.episode_manager.radar_locks_fixed = True

        # Process step
        obs, rewards, terminateds, truncateds, infos, _ = self.step_processor.process_step(
            actions, current_agents, self.episode_manager.agent_to_unit_id,
            self.episode_manager.observation_space, self.state_tracker,
            self.helpers, self.config.tick_secs
        )

        # Update simulation time
        self.termination_checker.update_simulation_time(self.config.tick_secs)

        # Check termination
        terminated, truncated, end_reasons, _ = self.termination_checker.check_termination(
            self.current_step, self.episode_manager.agent_to_unit_id,
            self.config.agent_ids, self.config.opponent_ids
        )

        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        # Ensure per-agent flags are consistent
        if terminated:
            for aid in current_agents:
                terminateds[aid] = True
                truncateds[aid] = False
        elif truncated:
            for aid in current_agents:
                if not terminateds.get(aid, False):
                    truncateds[aid] = True

        # Add episode-end info
        if terminated or truncated:
            agents_alive = any(
                self.episode_manager.agent_to_unit_id.get(aid) in self.simulator.active_units
                for aid in self.config.agent_ids
                if self.episode_manager.agent_to_unit_id.get(aid) is not None
            )
            opponents_alive = any(
                self.episode_manager.agent_to_unit_id.get(aid) in self.simulator.active_units
                for aid in self.config.opponent_ids
                if self.episode_manager.agent_to_unit_id.get(aid) is not None
            )

            episode_info = self.termination_checker.compute_episode_info(
                end_reasons, agents_alive, opponents_alive,
                self.current_step, self.state_tracker
            )

            for aid in list(infos.keys()):
                agent_info = {
                    "missiles_fired": self.state_tracker.episode_missiles_fired.get(aid, 0),
                    "kills": self.state_tracker.episode_kills.get(aid, 0),
                    "died": self.state_tracker.episode_deaths.get(aid, 0),
                }
                infos[aid].update(episode_info)
                infos[aid].update(agent_info)

        # Update alive agents for next step
        if terminated or truncated:
            self.agents = []
        else:
            self.agents = [
                aid for aid in current_agents
                if not terminateds.get(aid, False) and not truncateds.get(aid, False)
            ]

        if self.tacview_logger is not None:
            self.tacview_logger.log_tick(self.simulator)

        return obs, rewards, terminateds, truncateds, infos
