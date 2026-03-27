"""
Gym environments for air combat RL training.

  BVRMultiAgentEnv     — full-fidelity environment with radar pipeline, AWACS,
                         passive radar, NEZ/SQI features, 10-dim action space.

  SimplifiedMultiAgentEnv — lightweight environment with truth observations,
                             no radar pipeline, no AWACS, 4-dim action space.
                             Use this for fast initial training.
"""

from .bvr_multi_agent_env import BVRMultiAgentEnv
from .simplified_env import SimplifiedConfig, SimplifiedMultiAgentEnv

__all__ = [
    "BVRMultiAgentEnv",
    "SimplifiedMultiAgentEnv",
    "SimplifiedConfig",
]
