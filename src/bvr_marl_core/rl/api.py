"""Stable public RL extension API for bvr-marl-core.

Extension packages must import from this module
rather than from deep internal paths.  This module is the single stable
surface for RL infrastructure; all other ``bvr_marl_core.rl.*`` paths are
considered internal and subject to change without notice.

Exported symbols
----------------
Environments
    BVRMultiAgentEnv, SimplifiedMultiAgentEnv

Environment utilities
    create_env_creator, create_simplified_env_creator,
    extract_env_cfg, set_gymnasium_spec,
    RewardNormalizationWrapper, resolve_aircraft_config,
    ActionProcessor, EnergyLiftVectorActionProcessor,
    EnergyProcessor, LiftVectorProcessor

Training
    build_ppo_config, configure_automation_level,
    EpisodeMetricsCallback, ProgressCallback,
    SmartCheckpointCallback, WeightLoadingCallback,
    process_checkpoint_resume, process_weight_loading
"""

from __future__ import annotations

# ── Environments ──────────────────────────────────────────────────────────────
from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv  # noqa: F401
from bvr_marl_core.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv  # noqa: F401
from bvr_marl_core.rl.environment.spaces.action_space import (  # noqa: F401
    ActionProcessor,
    EnergyLiftVectorActionProcessor,
)
from bvr_marl_core.rl.environment.spaces.action_space.processors import (  # noqa: F401
    EnergyProcessor,
    LiftVectorProcessor,
)

# ── Training ──────────────────────────────────────────────────────────────────
from bvr_marl_core.rl.training.callbacks import (  # noqa: F401
    EpisodeMetricsCallback,
    ProgressCallback,
    SmartCheckpointCallback,
    WeightLoadingCallback,
)
from bvr_marl_core.rl.training.checkpoint_utils import (  # noqa: F401
    process_checkpoint_resume,
    process_weight_loading,
)
from bvr_marl_core.rl.training.config_builder import (  # noqa: F401
    build_ppo_config,
    configure_automation_level,
    policies_to_train_from_cfg,
    policy_ids_from_cfg,
)

# ── Environment utilities ─────────────────────────────────────────────────────
from bvr_marl_core.rl.utils import (  # noqa: F401
    RewardNormalizationWrapper,
    create_env_creator,
    create_simplified_env_creator,
)
from bvr_marl_core.rl.utils.env_creator import (  # noqa: F401
    _extract_env_cfg as extract_env_cfg,
)
from bvr_marl_core.rl.utils.env_creator import (
    _set_gymnasium_spec as set_gymnasium_spec,
)
from bvr_marl_core.rl.utils.type_maps import resolve_aircraft_config  # noqa: F401

__all__ = [
    # Environments
    "BVRMultiAgentEnv",
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
