"""
Modular components for BVR Multi-Agent Environment.

This package contains the refactored components of BVRMultiAgentEnv:
- config: Environment configuration parsing and validation
- state_tracker: State tracking for rewards and episode statistics
- episode_manager: Episode reset and initialization logic
- step_processor: Step execution and reward computation
- observation_builder: Observation and info building for agents
- termination: Episode termination logic
- helpers: Utility functions for agent/enemy management
- reward_config: Reward calculator configuration helper
"""

from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig
from bvr_marl_core.rl.environment.gym.gym_components.episode_manager import EpisodeManager
from bvr_marl_core.rl.environment.gym.gym_components.helpers import AgentHelpers
from bvr_marl_core.rl.environment.gym.gym_components.observation_builder import (
    ObservationInfoBuilder,
)
from bvr_marl_core.rl.environment.gym.gym_components.reward_config import create_reward_calculator
from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker
from bvr_marl_core.rl.environment.gym.gym_components.step_processor import StepProcessor
from bvr_marl_core.rl.environment.gym.gym_components.termination import TerminationChecker

__all__ = [
    "BVREnvConfig",
    "StateTracker",
    "EpisodeManager",
    "StepProcessor",
    "ObservationInfoBuilder",
    "TerminationChecker",
    "AgentHelpers",
    "create_reward_calculator",
]
