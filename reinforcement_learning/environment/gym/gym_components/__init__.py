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

from .config import BVREnvConfig
from .state_tracker import StateTracker
from .episode_manager import EpisodeManager
from .step_processor import StepProcessor
from .observation_builder import ObservationInfoBuilder
from .termination import TerminationChecker
from .helpers import AgentHelpers
from .reward_config import create_reward_calculator

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
