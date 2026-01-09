"""Custom RLModule components for multi-agent PPO training.

This module contains all components for the CustomMultiAgentRLModule:
- Main RLModule class
- Policy and value heads
- Action wrapper for curriculum learning
- Utility functions
"""

from .custom_multi_agent_model import CustomMultiAgentRLModule
from .policy_heads import ContinuousPolicyHead, DiscretePolicyHead, ValueHead
from .action_wrapper import ActionWrapper
from .utils import space_flatdim, flatten_obs_tensor, get_action_mask_from_obs

__all__ = [
    'CustomMultiAgentRLModule',
    'ContinuousPolicyHead',
    'DiscretePolicyHead',
    'ValueHead',
    'ActionWrapper',
    'space_flatdim',
    'flatten_obs_tensor',
    'get_action_mask_from_obs',
]
