"""Stable public RL extension API for bvr-marl.

Extension packages must import from this module
rather than from deep internal paths.  This module is the single stable
surface for RL infrastructure; all other ``bvr_marl_core.rl.*`` paths are
considered internal and subject to change without notice.

Exported symbols
----------------
Environments
    BVRMultiAgentEnv, SimpleOracleEnv, SimplifiedMultiAgentEnv (compatibility alias)

Environment utilities
    create_env_creator, create_simplified_env_creator,
    extract_env_cfg, set_gymnasium_spec,
    RewardNormalizationWrapper, resolve_aircraft_config,
    ActionProcessor, EnergyLiftVectorActionProcessor,
    EnergyProcessor, LiftVectorProcessor

Training
    build_ppo_config, configure_automation_level,
    policies_to_train_from_cfg, policy_ids_from_cfg,
    EpisodeMetricsCallback, ProgressCallback,
    SmartCheckpointCallback, WeightLoadingCallback,
    process_checkpoint_resume, process_weight_loading
"""

from __future__ import annotations

# ── Environments ──────────────────────────────────────────────────────────────
from bvr_marl_core.aircraft.systems.fire_veto import (
    TEAM_A_WASTED_VETO_KEYS,
    WASTED_VETO_CATEGORIES,
)
from bvr_marl_core.domain.launch_geometry import (
    MISSILE_DIAGNOSTIC_KEYS,
    SENSOR_DIAGNOSTIC_KEYS,
)
from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from bvr_marl_core.rl.environment.gym.simplified_env import (
    SimpleOracleEnv,
    SimplifiedMultiAgentEnv,
)
from bvr_marl_core.rl.environment.rewards.information import (
    RewardInformationClass,
    ensure_reward_information_allowed,
    resolve_reward_information_mode,
)
from bvr_marl_core.rl.environment.spaces.action_space import (
    ACTION_SCHEMA_VERSION,
    ActionProcessor,
    EnergyLiftVectorActionProcessor,
)
from bvr_marl_core.rl.environment.spaces.action_space.processors import (
    EnergyProcessor,
    LiftVectorProcessor,
)

# ── Observation extension points ──────────────────────────────────────────────
# An extension package widens the enemy-fighter token by subclassing these two and
# declaring the extra width as ``ef_extra_dim`` on the env config. Exported so that
# widening does not require reaching into core's internal modules.
from bvr_marl_core.rl.environment.spaces.observation.builder import ObservationBuilder

# Observation layout indices an extension may need to slice out of a flat observation.
# Exported here rather than left in the internal constants module so the policy side
# does not hardcode a magic offset that silently breaks when the layout changes.
from bvr_marl_core.rl.environment.spaces.observation.constants import (
    EF_IDX_CONFIDENCE,
    OBS_SCHEMA_VERSION,
    OWN_IDX_CAN_FIRE,
    d_EF,
    ef_token_dim,
)
from bvr_marl_core.rl.environment.spaces.observation.enemy_info_builder import EnemyInfoBuilder
from bvr_marl_core.rl.environment.spaces.observation.helpers.covariance import (
    body_frame_rotation,
    rotate_cov_to_body,
)

# ── Training ──────────────────────────────────────────────────────────────────
from bvr_marl_core.rl.training.callbacks import (
    EpisodeMetricsCallback,
    ProgressCallback,
    SmartCheckpointCallback,
    WeightLoadingCallback,
)
from bvr_marl_core.rl.training.checkpoint_utils import (
    process_checkpoint_resume,
    process_weight_loading,
)
from bvr_marl_core.rl.training.config_builder import (
    build_ppo_config,
    configure_automation_level,
    policies_to_train_from_cfg,
    policy_ids_from_cfg,
)

# ── Environment utilities ─────────────────────────────────────────────────────
from bvr_marl_core.rl.utils import (
    RewardNormalizationWrapper,
    create_env_creator,
    create_simplified_env_creator,
)
from bvr_marl_core.rl.utils.env_creator import (
    extract_env_cfg,
    set_gymnasium_spec,
)
from bvr_marl_core.rl.utils.type_maps import resolve_aircraft_config

__all__ = [
    "MISSILE_DIAGNOSTIC_KEYS",
    "SENSOR_DIAGNOSTIC_KEYS",
    "TEAM_A_WASTED_VETO_KEYS",
    "WASTED_VETO_CATEGORIES",
    "OBS_SCHEMA_VERSION",
    "ACTION_SCHEMA_VERSION",
    "OWN_IDX_CAN_FIRE",
    "EF_IDX_CONFIDENCE",
    "d_EF",
    "ef_token_dim",
    # Observation extension points
    "ObservationBuilder",
    "EnemyInfoBuilder",
    "body_frame_rotation",
    "rotate_cov_to_body",
    # Environments
    "BVRMultiAgentEnv",
    "SimpleOracleEnv",
    "SimplifiedMultiAgentEnv",
    # Environment utilities
    "create_env_creator",
    "create_simplified_env_creator",
    "extract_env_cfg",
    "set_gymnasium_spec",
    "RewardNormalizationWrapper",
    "resolve_aircraft_config",
    "ActionProcessor",
    "EnergyLiftVectorActionProcessor",
    "EnergyProcessor",
    "LiftVectorProcessor",
    "RewardInformationClass",
    "ensure_reward_information_allowed",
    "resolve_reward_information_mode",
    # Training
    "build_ppo_config",
    "configure_automation_level",
    "policies_to_train_from_cfg",
    "policy_ids_from_cfg",
    "EpisodeMetricsCallback",
    "ProgressCallback",
    "SmartCheckpointCallback",
    "WeightLoadingCallback",
    "process_checkpoint_resume",
    "process_weight_loading",
]
