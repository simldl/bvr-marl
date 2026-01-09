"""Main RL Module for multi-agent PPO training with energy-based control.

This module implements the RLlib RLModule interface with ValueFunctionAPI for PPO.
It combines:
- MLP observation encoding
- Policy and value heads
- Neural action wrapper for curriculum learning
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from gymnasium.spaces import Space, Box, Discrete

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI

from reinforcement_learning.networks.rl_module.utils import space_flatdim, flatten_obs_tensor
from reinforcement_learning.networks.rl_module.action_wrapper import ActionWrapper
from reinforcement_learning.networks.rl_module.policy_heads import ContinuousPolicyHead, DiscretePolicyHead, ValueHead


class CustomMultiAgentRLModule(TorchRLModule, ValueFunctionAPI):
    """
    RLModule for multi-agent PPO with MLP encoding.

    Action Space (10 actions total when use_neural_wrapper=False):
    - [0] Ps: Specific energy rate command (climb/dive/accelerate)
    - [1] n:  Normal load factor command (turn intensity)
    - [2] φ:  Bank angle command (turn direction)
    - [3] Target selection
    - [4] Missile firing
    - [5] Gun firing
    - [6-9] Countermeasures (flares, chaff, ECM, decoys)

    When use_neural_wrapper=True (default):
    - Network controls active actions only (e.g., [0,1,2,4])
    - Automation handles inactive actions with defaults
    - Enables curriculum learning by gradually expanding control

    Features:
    - MLP encoder for observations
    - Continuous and discrete action space support
    - Action masking for discrete actions
    - Neural wrapper for phased training
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Dict[str, Any] = None,
        inference_only: bool = False,
        learner_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=dict(model_config or {}),
            inference_only=inference_only,
            learner_only=learner_only,
            **kwargs,
        )

        # ============================================================
        # 1. CONFIGURE ACTION SPACE
        # ============================================================
        obs_dim = space_flatdim(self.observation_space)
        self.is_continuous = isinstance(self.action_space, Box)

        # Neural wrapper configuration
        use_wrapper = bool(self.model_config.get("use_neural_wrapper", True))
        wrapped_action_dim = int(self.model_config.get("wrapped_action_dim", 4))
        full_action_dim = int(self.model_config.get("full_action_dim", 10))
        active_indices = self.model_config.get("active_indices", None)

        self.action_wrapper = ActionWrapper(
            wrapped_action_dim=wrapped_action_dim,
            full_action_dim=full_action_dim,
            active_indices=active_indices,
            use_wrapper=use_wrapper
        )

        # Determine network output dimension
        if self.is_continuous:
            action_dim = self.action_wrapper.get_active_action_dim()
        else:
            assert isinstance(self.action_space, Discrete), \
                f"Unsupported action space: {type(self.action_space)}"
            action_dim = int(self.model_config.get("action_dim", self.action_space.n))

        # ============================================================
        # 2. BUILD MLP OBSERVATION ENCODER
        # ============================================================
        hidden_dim = int(self.model_config.get("hidden_dim", 256))

        self.encoder = self._build_encoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim
        )

        feature_dim = hidden_dim

        # ============================================================
        # 3. BUILD POLICY AND VALUE HEADS
        # ============================================================
        if self.is_continuous:
            init_log_std = float(self.model_config.get("init_log_std", -0.5))
            self.policy_head = ContinuousPolicyHead(
                input_dim=feature_dim,
                action_dim=action_dim,
                init_log_std=init_log_std
            )
        else:
            action_mask_key = str(self.model_config.get("action_mask_key", "action_mask"))
            self.policy_head = DiscretePolicyHead(
                input_dim=feature_dim,
                action_dim=action_dim,
                action_mask_key=action_mask_key
            )

        self.value_head = ValueHead(input_dim=feature_dim)

        # ============================================================
        # 4. INITIALIZE WEIGHTS
        # ============================================================
        self._initialize_weights()

    def _build_encoder(
        self,
        obs_dim: int,
        hidden_dim: int
    ) -> nn.Module:
        """
        Build MLP observation encoder.

        Args:
            obs_dim: Observation dimension
            hidden_dim: Hidden layer dimension

        Returns:
            Encoder module
        """
        num_layers = int(self.model_config.get("num_hidden_layers", 2))
        act_name = str(self.model_config.get("activation", "relu")).lower()
        act_layer = nn.Tanh() if act_name == "tanh" else nn.ReLU()

        layers = []
        in_dim = obs_dim
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_layer])
            in_dim = hidden_dim

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def _encode(
        self,
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Encode observations through MLP encoder.

        Args:
            batch: Input batch containing observations

        Returns:
            Encoded features
        """
        obs = batch[Columns.OBS]

        # Flatten observation features
        obs_flat = flatten_obs_tensor(obs)

        # Encode observations
        features = self.encoder(obs_flat)

        return features

    def _compute_values(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute state value estimates.

        Args:
            features: Encoded state features

        Returns:
            Value estimates (batch, 1)
        """
        return self.value_head(features)

    def _policy_dist_inputs_and_actions(
        self,
        features: torch.Tensor,
        batch: Dict[str, Any],
        sample_actions: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Compute policy distribution and optionally sample actions.

        Args:
            features: Encoded state features
            batch: Input batch (for action masking)
            sample_actions: Whether to sample actions

        Returns:
            Dict with ACTION_DIST_INPUTS and optionally ACTIONS, ACTION_LOGP
        """
        if self.is_continuous:
            return self.policy_head(
                features,
                action_wrapper=self.action_wrapper,
                sample_actions=sample_actions
            )
        else:
            return self.policy_head(
                features,
                batch=batch,
                sample_actions=sample_actions
            )

    # ============================================================
    # RLLIB API METHODS
    # ============================================================

    @torch.no_grad()
    def forward_inference(self, batch: Dict[str, Any], **kwargs):
        """
        Forward pass for inference (action selection during rollout).

        Args:
            batch: Input batch with observations

        Returns:
            Dict with actions
        """
        h = self._encode(batch)

        # Get policy outputs and actions
        out = self._policy_dist_inputs_and_actions(
            h, batch, sample_actions=True
        )

        # Add value function predictions for GAE computation
        out[Columns.VF_PREDS] = self._compute_values(h)

        return out

    @torch.no_grad()
    def forward_exploration(self, batch: Dict[str, Any], **kwargs):
        """
        Forward pass for exploration (same as inference for PPO).

        Args:
            batch: Input batch with observations

        Returns:
            Dict with actions
        """
        h = self._encode(batch)

        # Get policy outputs and actions
        out = self._policy_dist_inputs_and_actions(
            h, batch, sample_actions=True
        )

        # Add value function predictions for GAE computation
        out[Columns.VF_PREDS] = self._compute_values(h)

        return out

    def forward_train(self, batch: Dict[str, Any], **kwargs):
        """
        Forward pass for training (computes policy distribution and values).

        Args:
            batch: Input batch with observations

        Returns:
            Dict with action distribution and values
        """
        h = self._encode(batch)

        # Get policy outputs (no sampling during training)
        out = self._policy_dist_inputs_and_actions(
            h, batch, sample_actions=False
        )

        # CRITICAL: Always provide value function predictions for PPO training
        out[Columns.VF_PREDS] = self._compute_values(h)

        return out

    def compute_values(self, batch: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Compute state value estimates (ValueFunctionAPI).

        This method is called by RLlib's GAE connector to compute advantages.

        Args:
            batch: Input batch with observations

        Returns:
            Value estimates (batch,)
        """
        h = self._encode(batch)
        return self._compute_values(h).squeeze(-1)

    def _forward(
        self,
        batch: Dict[str, Any],
        sample_actions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Internal forward method (for compatibility).

        Args:
            batch: Input batch
            sample_actions: Whether to sample actions

        Returns:
            Dict with outputs
        """
        h = self._encode(batch)

        # Always sample actions for the connector pipeline and ensure ACTIONS column is present
        out = self._policy_dist_inputs_and_actions(
            h, batch, sample_actions=sample_actions
        )

        # ALWAYS provide value function predictions - needed for GAE during sampling AND training
        out[Columns.VF_PREDS] = self._compute_values(h)

        return out
