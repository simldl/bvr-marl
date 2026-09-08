"""
Gym environments for air combat RL training.

  BVRMultiAgentEnv     — full-fidelity environment with radar pipeline, AWACS,
                         passive radar, NEZ features, 10-dim action space.

  SimpleOracleEnv       — oracle/debug/curriculum environment with truth
                          observations, no radar pipeline, and no AWACS.
  SimplifiedMultiAgentEnv — backward-compatible alias for SimpleOracleEnv.
"""

from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from bvr_marl_core.rl.environment.gym.simplified_env import (
    SimpleOracleEnv,
    SimplifiedConfig,
    SimplifiedMultiAgentEnv,
)

__all__ = [
    "BVRMultiAgentEnv",
    "SimpleOracleEnv",
    "SimplifiedMultiAgentEnv",
    "SimplifiedConfig",
]
