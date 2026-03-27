"""
SimplifiedMultiAgentEnv — lightweight training environment.

Differences from BVRMultiAgentEnv:
  - No AWACS units.
  - Observations use truth positions/speeds directly (no radar pipeline).
  - Action space is 4-dimensional: [energy, lift_phi, lift_N, missile_trigger].
  - Target selection is automatic (closest alive enemy).
  - Missiles fire via fire_missile_direct() (no radar lock required).
  - No gun, no countermeasures.

All other components (physics, reward, termination, spawning) are shared with
the full environment to keep the simplified env consistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from air_to_air_rl.rl.environment.gym.base_env import BaseMultiAgentEnv
from air_to_air_rl.rl.environment.gym.gym_components import (
    AgentHelpers,
    BVREnvConfig,
    EpisodeManager,
    StateTracker,
    StepProcessor,
    TerminationChecker,
)
from air_to_air_rl.rl.environment.gym.gym_components.reward_config import create_reward_calculator
from air_to_air_rl.rl.environment.spaces.action_space import ActionSpaceManager
from air_to_air_rl.rl.environment.spaces.action_space.simplified_processor import (
    SIMPLIFIED_ACTION_DIM,
    SimplifiedActionProcessor,
)
from air_to_air_rl.rl.environment.spaces.obs_space_manager import (
    SimplifiedEnvConfig,
    SimplifiedObservationSpaceManager,
)
from air_to_air_rl.rl.environment.spaces.observation.simplified_obs_builder import (
    SimplifiedObservationBuilder,
)

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedConfig:
    """
    Minimal configuration for the SimplifiedMultiAgentEnv.

    Only exposes the settings that are relevant to this stripped-down env.
    All other BVREnvConfig fields default to sensible values internally.
    """

    # Team sizes
    num_agents_per_team: int = 2

    # Episode length
    max_steps: int = 600
    tick_secs: float = 1.0

    # Map
    map_size_km: float = 600.0
    max_alt: float = 25000.0

    # Aircraft types (class name strings, e.g. "F22", "Su57")
    agent_aircraft_type: str | None = None
    opponent_aircraft_type: str | None = None

    # Observation slots
    num_ff: int = 2  # friendly fighter slots (excluding self)
    num_ef: int = 2  # enemy fighter slots
    num_warn: int = 4  # missile warning slots / sectors

    # Missile cooldown (seconds)
    missile_cooldown_s: float = 3.0

    # Termination
    early_term_all_enemies_dead: bool = True
    early_term_mission_complete: bool = True
    early_term_no_missiles: bool = False

    # Spawning geometry
    use_controlled_geometry: bool = False
    regime: str = "hot_120km"

    # Debugging / logging
    debug: bool = False
    tacview_logfile: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict) -> SimplifiedConfig:
        n = int(cfg.get("num_agents_per_team", cfg.get("num_agents_per_side", 2)))
        return cls(
            num_agents_per_team=n,
            max_steps=int(cfg.get("max_steps", 600)),
            tick_secs=float(cfg.get("tick_secs", 1.0)),
            map_size_km=float(cfg.get("map_size_km", cfg.get("map_size", 600.0))),
            max_alt=float(cfg.get("max_alt", 25000.0)),
            agent_aircraft_type=cfg.get("agent_aircraft_type", None),
            opponent_aircraft_type=cfg.get("opponent_aircraft_type", None),
            num_ff=int(cfg.get("num_ff", n - 1)),
            num_ef=int(cfg.get("num_ef", n)),
            num_warn=int(cfg.get("num_warn", 4)),
            missile_cooldown_s=float(cfg.get("missile_cooldown_s", 3.0)),
            early_term_all_enemies_dead=bool(cfg.get("early_term_all_enemies_dead", True)),
            early_term_mission_complete=bool(cfg.get("early_term_mission_complete", True)),
            early_term_no_missiles=bool(cfg.get("early_term_no_missiles", False)),
            use_controlled_geometry=bool(cfg.get("use_controlled_geometry", False)),
            regime=cfg.get("regime", "hot_120km"),
            debug=bool(cfg.get("debug", False)),
            tacview_logfile=cfg.get("tacview_logfile", None),
        )

    def to_bvr_config_dict(self) -> dict:
        """Build a BVREnvConfig-compatible dict with AWACS disabled."""
        n = self.num_agents_per_team
        aircraft_types = {}
        if self.agent_aircraft_type:
            aircraft_types["agent"] = self.agent_aircraft_type
        if self.opponent_aircraft_type:
            aircraft_types["opponent"] = self.opponent_aircraft_type

        return {
            "num_agents_per_team": n,
            "max_steps": self.max_steps,
            "simulation_config": {
                "tick_secs": self.tick_secs,
                "early_termination": {
                    "all_enemies_dead": self.early_term_all_enemies_dead,
                    "mission_complete": self.early_term_mission_complete,
                    "no_missiles_remaining": self.early_term_no_missiles,
                },
                "fix_radar_lock_after_first_obs": False,
                "force_locks_all_enemies": False,
            },
            "map_size": self.map_size_km,
            "max_alt": self.max_alt,
            "aircraft_types": aircraft_types,
            # Observation slots
            "num_fm": 0,
            "num_ff": self.num_ff,
            "num_em": self.num_warn,  # reuse em_slots for warning builder config
            "num_enemy": self.num_ef,
            "warn_sectors": self.num_warn,
            "pr_slots": 0,
            # Disable all AWACS
            "scenario_config": {
                "awacs_config": {
                    "agent_awacs": False,
                    "opponent_awacs": False,
                },
                "randomize_regime": not self.use_controlled_geometry,
                "allowed_regimes": [self.regime] if self.use_controlled_geometry else [],
            },
            "geometry_config": {
                "use_controlled_geometry": self.use_controlled_geometry,
                "regime": self.regime,
            },
            # Disable gun
            "weapon_config": {"enable_gun": False},
            "debug": self.debug,
            "tacview_logfile": self.tacview_logfile,
        }


class SimplifiedMultiAgentEnv(BaseMultiAgentEnv):
    """
    Lightweight multi-agent air combat environment.

    Observation keys (per agent):
        own_state              (7,)
        friendly_fighters      (ff_slots * 4,)
        mask_friendly_fighters (ff_slots,)
        enemy_fighters         (ef_slots * 4,)
        mask_enemy_fighters    (ef_slots,)
        missile_warning_flag   (1,)
        missile_warning_dirs   (em_slots * (1 + warn_sectors),)
        mask_warning_dirs      (em_slots,)

    Action space (per agent):
        Box([0,1]^4): [energy_cmd, lift_phi, lift_N, missile_trigger]
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.simplified_cfg = SimplifiedConfig.from_dict(config)
        bvr_dict = self.simplified_cfg.to_bvr_config_dict()

        # Parse into BVREnvConfig (used by EpisodeManager, TerminationChecker, etc.)
        self.bvr_config = BVREnvConfig.from_dict(bvr_dict)

        # Episode tracking
        self.current_step = 0
        self.max_steps = self.bvr_config.max_steps
        self.tick_secs = self.bvr_config.tick_secs

        # PettingZoo-like attributes
        self.possible_agents = self.bvr_config.possible_agents
        self.agents: list[str] = []

        # Tacview logging (optional)
        from air_to_air_rl.tacview.logger import TacviewLogger

        self.tacview_logger = (
            TacviewLogger(self.simplified_cfg.tacview_logfile)
            if self.simplified_cfg.tacview_logfile
            else None
        )

        # Reward calculator (shared infrastructure)
        self.reward_calculator = create_reward_calculator(bvr_dict)

        # --- Simplified spaces ---
        simp_cfg = SimplifiedEnvConfig(
            ff_slots=self.bvr_config.num_ff,
            ef_slots=self.bvr_config.num_ef,
            em_slots=self.bvr_config.num_em,
            warn_sectors=self.bvr_config.num_warn,
            all_agent_ids=tuple(self.bvr_config.all_agent_ids),
        )
        self.obs_space_mgr = SimplifiedObservationSpaceManager(
            self.bvr_config.all_agent_ids, simp_cfg
        )
        self.obs_builder = SimplifiedObservationBuilder(
            self.simulator, self.bvr_config.all_agent_ids, simp_cfg
        )
        self.action_processor = SimplifiedActionProcessor(self.simulator)
        self.action_processor.weapon_cooldowns.missile_cooldown_s = (
            self.simplified_cfg.missile_cooldown_s
        )

        # Link action processor to simulator
        self.simulator.action_processor = self.action_processor

        # 4-dim action space
        self.action_space_mgr = ActionSpaceManager(
            self.bvr_config.all_agent_ids, shape=SIMPLIFIED_ACTION_DIM
        )
        self.action_space = self.action_space_mgr.all()

        # Modular components (reused from full env)
        self.state_tracker = StateTracker()
        self.episode_manager = EpisodeManager(
            self.simulator,
            self.obs_builder,
            self.bvr_config,
            bvr_dict,
            self.obs_space_mgr,
        )
        self.helpers = AgentHelpers(
            self.simulator,
            self.bvr_config.agent_ids,
            self.bvr_config.opponent_ids,
            self.episode_manager.agent_to_unit_id,
            force_locks_all_enemies=False,
        )
        self.step_processor = StepProcessor(
            self.simulator,
            self.obs_builder,
            self.action_processor,
            self.reward_calculator,
            self.bvr_config,
        )
        self.termination_checker = TerminationChecker(self.simulator, self.bvr_config)

        # Observation space (updated each reset via episode_manager)
        self.observation_space = self.episode_manager.observation_space

        # IDs
        self.agent_ids = self.bvr_config.agent_ids
        self.opponent_ids = self.bvr_config.opponent_ids
        self.all_agent_ids = self.bvr_config.all_agent_ids
        self.agent_to_unit_id = self.episode_manager.agent_to_unit_id

        # Required by RLlib new API stack (MultiAgentEnvRunner validation)
        self.possible_agents = list(self.all_agent_ids)
        self.agents: list = []  # populated on first reset()

        # Map limits for visualization
        self.map_limits = self.episode_manager.map_limits
        self.full_map_limits = self.episode_manager.full_map_limits

    # Gym interface

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset environment for a new episode."""
        self.current_step = 0
        self.state_tracker.reset()
        self.termination_checker.reset_timing(self.bvr_config.tick_secs)

        # Clear per-agent action-processor state so stale cooldowns don't carry over.
        self.action_processor.agent_states.clear()
        self.action_processor.missile_auto.missiles_per_target.clear()

        obs, infos = self.episode_manager.reset(seed=seed)

        self.observation_space = self.episode_manager.observation_space

        for aid in self.bvr_config.all_agent_ids:
            position = self.episode_manager.get_agent_position(aid)
            self.state_tracker.initialize_agent(aid, position)

        self.agents = self.episode_manager.agents
        return obs, infos

    def step(self, actions: dict[str, np.ndarray] | None = None):
        """Execute one environment step."""
        self.current_step += 1
        actions = actions or {}

        current_agents = list(self.agents) if self.agents else list(self.bvr_config.all_agent_ids)

        # Process step (advance sim, apply actions, compute obs + rewards)
        obs, rewards, terminateds, truncateds, infos, _ = self.step_processor.process_step(
            actions,
            current_agents,
            self.episode_manager.agent_to_unit_id,
            self.episode_manager.observation_space,
            self.state_tracker,
            self.helpers,
            self.bvr_config.tick_secs,
        )

        # Update simulation time
        self.termination_checker.update_simulation_time(self.bvr_config.tick_secs)

        # Check termination
        terminated, truncated, end_reasons, _ = self.termination_checker.check_termination(
            self.current_step,
            self.episode_manager.agent_to_unit_id,
            self.bvr_config.agent_ids,
            self.bvr_config.opponent_ids,
        )

        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        if terminated:
            for aid in current_agents:
                terminateds[aid] = True
                truncateds[aid] = False
        elif truncated:
            for aid in current_agents:
                if not terminateds.get(aid, False):
                    truncateds[aid] = True

        # Episode-end info
        if terminated or truncated:
            agents_alive = any(
                self.episode_manager.agent_to_unit_id.get(aid) in self.simulator.active_units
                for aid in self.bvr_config.agent_ids
                if self.episode_manager.agent_to_unit_id.get(aid) is not None
            )
            opponents_alive = any(
                self.episode_manager.agent_to_unit_id.get(aid) in self.simulator.active_units
                for aid in self.bvr_config.opponent_ids
                if self.episode_manager.agent_to_unit_id.get(aid) is not None
            )
            episode_info = self.termination_checker.compute_episode_info(
                end_reasons,
                agents_alive,
                opponents_alive,
                self.current_step,
                self.state_tracker,
            )
            for aid in list(infos.keys()):
                infos[aid].update(episode_info)
                infos[aid].update(
                    {
                        "missiles_fired": self.state_tracker.episode_missiles_fired.get(aid, 0),
                        "kills": self.state_tracker.episode_kills.get(aid, 0),
                        "died": self.state_tracker.episode_deaths.get(aid, 0),
                    }
                )

        # Update alive agents
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
