"""Policy and value function heads for RL agent."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns

from .utils import get_action_mask_from_obs
from .action_wrapper import ActionWrapper


class ContinuousPolicyHead(nn.Module):
    """
    Policy head for continuous action spaces with Gaussian distribution.

    Outputs mean and log_std for each action dimension. Actions are
    squashed to [0,1] range using sigmoid activation.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        init_log_std: float = -0.5
    ):
        """
        Initialize continuous policy head.

        Args:
            input_dim: Input feature dimension
            action_dim: Number of continuous actions
            init_log_std: Initial value for log standard deviation
        """
        super().__init__()
        self.pi_mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

        # Initialize with small weights for stable exploration
        nn.init.orthogonal_(self.pi_mean.weight, gain=0.01)
        # Center at 0.5 for proper exploration in [0,1] action space
        nn.init.constant_(self.pi_mean.bias, 0.5)

    def forward(
        self,
        features: torch.Tensor,
        action_wrapper: Optional[ActionWrapper] = None,
        sample_actions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute policy distribution and optionally sample actions.

        Args:
            features: Encoded state features (batch, input_dim)
            action_wrapper: Optional action wrapper for active/inactive actions
            sample_actions: Whether to sample actions or just return distribution

        Returns:
            Dict with ACTION_DIST_INPUTS and optionally ACTIONS, ACTION_LOGP
        """
        out = {}

        # Network outputs unbounded mean
        mean_unbounded = self.pi_mean(features)

        # Squash to [0,1] using sigmoid (differentiable)
        mean_active = torch.sigmoid(mean_unbounded)

        # Clamp log_std to safe range: [-2.0, 0.5] => std ∈ [0.135, 1.65]
        log_std_active = self.log_std.expand_as(mean_active)
        log_std_active = torch.clamp(log_std_active, min=-2.0, max=0.5)

        # Expand to full action space if using wrapper
        if action_wrapper and action_wrapper.use_wrapper:
            mean_full, log_std_full = action_wrapper.expand_to_full_space(
                mean_active, log_std_active
            )
            out[Columns.ACTION_DIST_INPUTS] = torch.cat([mean_full, log_std_full], dim=-1)

            if sample_actions:
                std_full = torch.exp(log_std_full)
                actions_raw = torch.normal(mean=mean_full, std=std_full)
                actions = torch.clamp(actions_raw, min=0.0, max=1.0)
                out[Columns.ACTIONS] = actions

                # Compute log prob only for active actions
                actions_active = action_wrapper.extract_active_actions(actions)
                dist_active = torch.distributions.Normal(mean_active, torch.exp(log_std_active))
                action_logp = dist_active.log_prob(actions_active).sum(dim=-1)
                out[Columns.ACTION_LOGP] = action_logp
        else:
            # Full action space - no expansion
            out[Columns.ACTION_DIST_INPUTS] = torch.cat([mean_active, log_std_active], dim=-1)

            if sample_actions:
                std = torch.exp(log_std_active)
                actions_raw = torch.normal(mean=mean_active, std=std)
                actions = torch.clamp(actions_raw, min=0.0, max=1.0)
                out[Columns.ACTIONS] = actions

                dist = torch.distributions.Normal(mean_active, std)
                action_logp = dist.log_prob(actions).sum(dim=-1)
                out[Columns.ACTION_LOGP] = action_logp

        return out


class DiscretePolicyHead(nn.Module):
    """
    Policy head for discrete action spaces with Categorical distribution.

    Supports action masking to prevent invalid actions.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_mask_key: str = "action_mask"
    ):
        """
        Initialize discrete policy head.

        Args:
            input_dim: Input feature dimension
            action_dim: Number of discrete actions
            action_mask_key: Key for action mask in observations
        """
        super().__init__()
        self.pi_logits = nn.Linear(input_dim, action_dim)
        self.action_mask_key = action_mask_key

        # Initialize with small weights
        nn.init.orthogonal_(self.pi_logits.weight, gain=0.01)
        nn.init.constant_(self.pi_logits.bias, 0.0)

    def forward(
        self,
        features: torch.Tensor,
        batch: Dict[str, Any],
        sample_actions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute policy distribution and optionally sample actions.

        Args:
            features: Encoded state features (batch, input_dim)
            batch: Batch dict containing observations (for action mask)
            sample_actions: Whether to sample actions

        Returns:
            Dict with ACTION_DIST_INPUTS and optionally ACTIONS, ACTION_LOGP
        """
        out = {}

        logits = self.pi_logits(features)

        # Apply action mask if present
        mask = get_action_mask_from_obs(batch.get(Columns.OBS), self.action_mask_key)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand_as(logits)
            logits = logits.masked_fill(mask <= 0, float("-inf"))

        out[Columns.ACTION_DIST_INPUTS] = logits

        if sample_actions:
            probs = torch.softmax(logits, dim=-1)

            # Handle NaN or zero probabilities (fallback to uniform)
            nan_or_zero = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) == 0)
            if nan_or_zero.any():
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs = torch.where(nan_or_zero.unsqueeze(-1), uniform, probs)

            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()
            out[Columns.ACTIONS] = actions
            out[Columns.ACTION_LOGP] = dist.log_prob(actions)

        return out


class ValueHead(nn.Module):
    """Value function head for estimating state values."""

    def __init__(self, input_dim: int):
        """
        Initialize value head.

        Args:
            input_dim: Input feature dimension
        """
        super().__init__()
        self.v_head = nn.Linear(input_dim, 1)

        # Initialize with small weights
        nn.init.orthogonal_(self.v_head.weight, gain=0.01)
        nn.init.constant_(self.v_head.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute state value estimate.

        Args:
            features: Encoded state features (batch, input_dim)

        Returns:
            Value estimates (batch, 1)
        """
        return self.v_head(features)
