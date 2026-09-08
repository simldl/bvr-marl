from __future__ import annotations

import logging
import math

import numpy as np
from ray.rllib.policy import Policy

from bvr_marl_core.rl.environment.gym.base_env import BaseMultiAgentEnv
from bvr_marl_core.rl.environment.gym.gym_components import (
    AgentHelpers,
    BVREnvConfig,
    EpisodeManager,
    StateTracker,
    StepProcessor,
    TerminationChecker,
)
from bvr_marl_core.rl.environment.gym.gym_components.reward_config import (
    create_reward_calculator,
)
from bvr_marl_core.rl.environment.spaces.action_space import (
    FULL_ACTION_DIM,
    ActionSpaceManager,
    EnergyLiftVectorActionProcessor,
    emcon_action_dim,
)
from bvr_marl_core.rl.environment.spaces.obs_space_builder import ObservationBuilder
from bvr_marl_core.rl.environment.spaces.obs_space_manager import (
    EnvConfig,
    ObservationSpaceManager,
)
from bvr_marl_core.rl.environment.spaces.observation.layout import observation_layout
from bvr_marl_core.simulator.core.experiment_metadata import (
    build_experiment_metadata,
    prototype_warnings,
)
from bvr_marl_core.tacview.logger import TacviewLogger

logger = logging.getLogger(__name__)


class BVRMultiAgentEnv(BaseMultiAgentEnv):
    """
    Multi-agent env:
    - All spaces are gymnasium Spaces (float32).
    - Observations are strictly coerced to their subspaces.
    - Actions are normalized Box([0,1]^10) and applied to numeric unit-ids.
    - 10-dim action space: [Ps, n, φ, missile_fire, target, gun_fire, 4×CM]
      where Ps = specific energy rate, n = load factor, φ = bank angle

    ``observation_builder_class`` is the extension point for a widened observation:
    a subclass may point it at an ``ObservationBuilder`` subclass. The extra
    enemy-fighter width must also be declared as ``ef_extra_dim`` on the env config
    so the observation space matches what the builder emits.
    """

    observation_builder_class = ObservationBuilder

    def __init__(self, config: dict, model: Policy | None = None):
        super().__init__(config)
        self.debug = bool(config.get("debug", False))
        self.model = model

        # Parse configuration
        self.config = BVREnvConfig.from_dict(config)
        self.simulator.information_mode = self.config.information_mode
        self.simulator.replay_metadata.update(build_experiment_metadata(config))
        self.simulator.replay_metadata.update(
            {
                "information_mode": self.config.information_mode,
                "oracle_use_reason": self.config.oracle_use_reason,
            }
        )

        # Datalink 0/1 dropout rate (0.0 = always up) — read by DataLink each tick.
        self.simulator.datalink_drop_prob = self.config.datalink_drop_prob

        # Trace recording is only needed for visualization/tacview. Keep it on when
        # a tacview log is requested; otherwise honor the config (default True).
        self.simulator.record_traces = bool(
            self.config.record_traces or self.config.tacview_logfile
        )

        # Episode tracking
        self.current_step = 0

        # Expose config attributes for backward compatibility
        self.max_steps = self.config.max_steps
        self.tick_secs = self.config.tick_secs
        # Episode duration in sim-seconds, read by the own-state builder for the
        # time-remaining observation (the builder only sees the simulator).
        self.simulator.episode_max_s = self.max_steps * self.tick_secs
        self.map_size_km = self.config.map_size_km
        self.map_width_km = self.config.map_width_km
        self.map_height_km = self.config.map_height_km

        # Tacview logging (optional)
        self.tacview_logger = (
            TacviewLogger(self.config.tacview_logfile) if self.config.tacview_logfile else None
        )

        # PettingZoo-like attributes
        self.possible_agents = self.config.possible_agents
        self.agents: list[str] = []

        # Initialize reward calculator
        self.reward_calculator = create_reward_calculator(config)
        self.simulator.replay_metadata.update(self.reward_calculator.metadata())
        governance_warnings = prototype_warnings(self.simulator.replay_metadata)
        self.simulator.replay_metadata["governance_warnings"] = list(governance_warnings)
        for warning in governance_warnings:
            logger.warning("Experiment governance: %s", warning)

        # Resolved observation layout, recorded so a checkpoint/config disagreement can
        # be reported as "expected num_fm=6, got 4" instead of surfacing as a matrix
        # shape error deep inside a forward pass. See observation/layout.py for the
        # incident this exists to prevent.
        self.observation_layout = observation_layout(config)

        # Builders/Managers
        env_conf = EnvConfig(
            own_dim=self.config.own_state_dim,
            fm_slots=self.config.num_fm,
            ff_slots=self.config.num_ff,
            em_slots=self.config.num_em,
            ef_slots=self.config.num_ef,
            pr_slots=self.config.num_pr,
            warn_sectors=self.config.num_warn,
            information_mode=self.config.information_mode,
            oracle_use_reason=self.config.oracle_use_reason,
            ef_extra_dim=self.config.ef_extra_dim,
            extension_options=self.config.extension_options,
            emcon_action_enabled=self.config.emcon_action_enabled,
        )
        self.obs_space_mgr = ObservationSpaceManager(self.config.all_agent_ids, env_conf)
        self.obs_builder = self.observation_builder_class(
            self.simulator, self.config.all_agent_ids, env_conf
        )
        self.action_processor = EnergyLiftVectorActionProcessor(
            self.simulator, information_mode=self.config.information_mode
        )
        # Scripted active-sensing baseline (no-op for the default "learned" policy).
        from bvr_marl_core.aircraft.systems.emcon_controller import EmconController

        self.action_processor.emcon_controller = EmconController(
            self.config.emcon_policy,
            period_steps=self.config.emcon_period_steps,
            duty=self.config.emcon_duty,
        )

        subsystem_modes = {
            self.action_processor.information_mode.value,
            self.obs_builder.own_state_builder.information_mode.value,
            self.obs_builder.friendly_builder.information_mode.value,
            self.obs_builder.enemy_builder.information_mode.value,
            self.obs_builder.missile_warning_builder.information_mode.value,
        }
        if subsystem_modes != {self.config.information_mode}:
            raise ValueError(f"Inconsistent subsystem information modes: {sorted(subsystem_modes)}")

        # Configure automated missile firing and weapon toggles
        self.action_processor.configure_automation(
            enable_missile_automation=self.config.enable_missile_automation,
            missile_auto_sqi_threshold=self.config.missile_auto_sqi_threshold,
            missile_auto_max_per_target=self.config.missile_auto_max_per_target,
            missile_auto_long_cooldown_s=self.config.missile_auto_long_cooldown_s,
            missile_fire_threshold=self.config.missile_fire_threshold,
            enable_gun=self.config.enable_gun,
        )

        # Link action processor to simulator
        self.simulator.action_processor = self.action_processor

        # Initialize action spaces. Base is 9 (flight + weapons + countermeasures);
        # the EMCON radar toggle adds a 10th action when env.emcon_action_enabled is
        # set (config-driven, falling back to the EMCON_ACTION_ENABLED default).
        action_dim = emcon_action_dim(True) if self.config.emcon_action_enabled else FULL_ACTION_DIM
        self.action_space_mgr = ActionSpaceManager(self.config.all_agent_ids, shape=action_dim)
        self.action_space = self.action_space_mgr.all()

        # Initialize modular components
        self.state_tracker = StateTracker()
        self.episode_manager = EpisodeManager(
            self.simulator, self.obs_builder, self.config, config, self.obs_space_mgr
        )
        self.helpers = AgentHelpers(
            self.simulator,
            self.config.agent_ids,
            self.config.opponent_ids,
            self.episode_manager.agent_to_unit_id,
            self.config.force_locks_all_enemies,
        )
        self.step_processor = StepProcessor(
            self.simulator,
            self.obs_builder,
            self.action_processor,
            self.reward_calculator,
            self.config,
        )
        self.termination_checker = TerminationChecker(self.simulator, self.config)

        # Set observation space from episode manager
        self.observation_space = self.episode_manager.observation_space

        # Override base class attributes with config values
        self.agent_ids = self.config.agent_ids
        self.opponent_ids = self.config.opponent_ids
        self.all_agent_ids = self.config.all_agent_ids
        self.agent_to_unit_id = self.episode_manager.agent_to_unit_id

        # Expose map limits for visualization
        self.map_limits = self.episode_manager.map_limits
        self.full_map_limits = (
            self.episode_manager.full_map_limits
        )  # extended area incl. AWACS zones

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset environment for new episode."""
        self.current_step = 0
        self.state_tracker.reset()
        self.termination_checker.reset_timing(self.config.tick_secs)
        # Per-agent action state does not survive an episode: see
        # ActionProcessorBase.reset. Omitting this let the contact-slot registry
        # accumulate ghost contacts across episodes and silently corrupt target
        # selection.
        self.action_processor.reset()

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

    def step(self, actions: dict[str, np.ndarray] | None = None):
        """Execute one environment step."""
        self.current_step += 1
        actions = actions or {}

        # Get currently alive agents
        current_agents = list(self.agents) if self.agents else list(self.config.all_agent_ids)

        # Fix radar locks if needed
        lock_fix_steps = 1
        if self.config.lock_fix_duration_s > 0.0:
            lock_fix_steps = max(
                1,
                int(math.ceil(self.config.lock_fix_duration_s / max(self.config.tick_secs, 1e-6))),
            )
        if self.config.fix_radar_lock_after_first_obs and self.current_step <= lock_fix_steps:
            self.helpers.fix_all_radar_locks(self.config.all_agent_ids)
            self.episode_manager.radar_locks_fixed = self.current_step >= lock_fix_steps

        # Process step
        obs, rewards, terminateds, truncateds, infos, _ = self.step_processor.process_step(
            actions,
            current_agents,
            self.episode_manager.agent_to_unit_id,
            self.episode_manager.observation_space,
            self.state_tracker,
            self.helpers,
            self.config.tick_secs,
            current_step=self.current_step,
            sim_time_s=self.current_step * self.config.tick_secs,
        )

        self.termination_checker.update_simulation_time(self.config.tick_secs)

        # Check termination
        terminated, truncated, end_reasons, _ = self.termination_checker.check_termination(
            self.current_step,
            self.episode_manager.agent_to_unit_id,
            self.config.agent_ids,
            self.config.opponent_ids,
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
                end_reasons,
                agents_alive,
                opponents_alive,
                self.current_step,
                self.state_tracker,
                self.episode_manager.agent_to_unit_id,
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
                aid
                for aid in current_agents
                if not terminateds.get(aid, False) and not truncateds.get(aid, False)
            ]

        if self.tacview_logger is not None:
            self.tacview_logger.log_tick(self.simulator)

        return obs, rewards, terminateds, truncateds, infos
