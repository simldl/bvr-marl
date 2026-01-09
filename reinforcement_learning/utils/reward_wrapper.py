"""Reward normalization wrapper for multi-agent environments.

Normalizes rewards using running statistics to stabilize training.
"""

import numpy as np
import gymnasium as gym


class RewardNormalizationWrapper(gym.Wrapper):
    """
    Wraps a multi-agent environment to normalize rewards using running statistics.
    Helps stabilize training by keeping reward scale consistent.
    """

    def __init__(self, env, clip_reward=10.0, epsilon=1e-8):
        """
        Initialize reward normalization wrapper.

        Args:
            env: Multi-agent environment to wrap
            clip_reward: Maximum absolute value for normalized rewards
            epsilon: Small constant to prevent division by zero
        """
        super().__init__(env)
        self.clip_reward = clip_reward
        self.epsilon = epsilon

        # Running statistics per agent
        self.reward_mean = {}
        self.reward_var = {}
        self.reward_count = {}

    def _update_stats(self, agent_id, reward):
        """
        Update running mean and variance for an agent's rewards.

        Uses Welford's online algorithm for numerical stability.

        Args:
            agent_id: Agent identifier
            reward: Raw reward value
        """
        if agent_id not in self.reward_mean:
            self.reward_mean[agent_id] = 0.0
            self.reward_var[agent_id] = 1.0
            self.reward_count[agent_id] = 0

        self.reward_count[agent_id] += 1
        n = self.reward_count[agent_id]

        # Welford's online algorithm for running mean/variance
        delta = reward - self.reward_mean[agent_id]
        self.reward_mean[agent_id] += delta / n
        delta2 = reward - self.reward_mean[agent_id]
        self.reward_var[agent_id] += delta * delta2

    def _normalize_reward(self, agent_id, reward):
        """
        Normalize reward using running statistics.

        Args:
            agent_id: Agent identifier
            reward: Raw reward value

        Returns:
            Normalized and clipped reward
        """
        if agent_id not in self.reward_mean or self.reward_count[agent_id] < 2:
            return reward

        mean = self.reward_mean[agent_id]
        var = self.reward_var[agent_id] / (self.reward_count[agent_id] - 1)
        std = np.sqrt(var + self.epsilon)

        normalized = (reward - mean) / std

        # Clip to prevent extreme values
        if self.clip_reward is not None:
            normalized = np.clip(normalized, -self.clip_reward, self.clip_reward)

        return normalized

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)

    def step(self, actions):
        """
        Step the environment and normalize rewards.

        Args:
            actions: Actions for all agents

        Returns:
            Tuple of (observations, normalized_rewards, terminateds, truncateds, infos)
        """
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # Normalize rewards for each agent
        normalized_rewards = {}
        for agent_id, reward in rewards.items():
            self._update_stats(agent_id, reward)
            normalized_rewards[agent_id] = self._normalize_reward(agent_id, reward)

        return obs, normalized_rewards, terminateds, truncateds, infos
