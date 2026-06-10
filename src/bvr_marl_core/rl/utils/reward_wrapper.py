"""Reward normalization wrapper for multi-agent environments.

Normalizes rewards using running statistics to stabilize training.
"""

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RewardNormalizationWrapper(MultiAgentEnv):
    """
    Wraps a multi-agent environment to normalize rewards using running statistics.

    Inherits from MultiAgentEnv so that RLlib's MultiAgentEnvRunner correctly
    identifies it as a multi-agent environment. The new RLlib API stack also
    requires `env.agents` (currently active) and `env.possible_agents` (all
    possible) to be populated — we maintain these in sync with the inner env.
    """

    def __init__(self, env, clip_reward=10.0, epsilon=1e-8):
        # Do NOT call super().__init__() — it would set self.agents = [] which
        # we want to derive from the inner env instead.
        self.env = env
        self.clip_reward = clip_reward
        self.epsilon = epsilon

        # Derive possible_agents from the inner env's all_agent_ids.
        # agents (currently active) starts empty and is populated on first reset().
        _all = list(getattr(env, "all_agent_ids", []))
        self.possible_agents = list(getattr(env, "possible_agents", _all))
        self.agents: list = []  # instance attr shadows MultiAgentEnv class attr

        # Running statistics per agent
        self.reward_mean = {}
        self.reward_var = {}
        self.reward_count = {}

    # ── unwrapped: let validation reach the inner env directly ────────────────
    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    # ── Delegate observation / action spaces to the wrapped env ──────────────
    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.env.observation_space = value

    @property
    def action_space(self):
        return self.env.action_space

    @action_space.setter
    def action_space(self, value):
        self.env.action_space = value

    # ── Delegate unknown attributes to the wrapped env ───────────────────────
    def __getattr__(self, name):
        # Only reached when 'name' is not found on self or its class hierarchy
        return getattr(self.env, name)

    # ── Core env interface ────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        """Reset the environment and sync agent lists."""
        result = self.env.reset(seed=seed, options=options)
        # Keep agents in sync so MultiAgentEnvRunner can read env.agents
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        return result

    def step(self, actions):
        """Step the environment and return normalised rewards."""
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        normalized_rewards = {}
        for agent_id, reward in rewards.items():
            self._update_stats(agent_id, reward)
            normalized_rewards[agent_id] = self._normalize_reward(agent_id, reward)

        # Keep agents in sync (inner env may remove terminated agents)
        self.agents = list(getattr(self.env, "agents", self.agents))

        return obs, normalized_rewards, terminateds, truncateds, infos

    # ── Running-statistics helpers ────────────────────────────────────────────
    def _update_stats(self, agent_id, reward):
        """Update running mean and variance (Welford's algorithm)."""
        if agent_id not in self.reward_mean:
            self.reward_mean[agent_id] = 0.0
            self.reward_var[agent_id] = 1.0
            self.reward_count[agent_id] = 0

        self.reward_count[agent_id] += 1
        n = self.reward_count[agent_id]
        delta = reward - self.reward_mean[agent_id]
        self.reward_mean[agent_id] += delta / n
        delta2 = reward - self.reward_mean[agent_id]
        self.reward_var[agent_id] += delta * delta2

    def _normalize_reward(self, agent_id, reward):
        """Normalize reward using running statistics."""
        if agent_id not in self.reward_mean or self.reward_count[agent_id] < 2:
            return reward

        mean = self.reward_mean[agent_id]
        var = self.reward_var[agent_id] / (self.reward_count[agent_id] - 1)
        std = np.sqrt(var + self.epsilon)
        normalized = (reward - mean) / std

        if self.clip_reward is not None:
            normalized = np.clip(normalized, -self.clip_reward, self.clip_reward)

        return normalized
