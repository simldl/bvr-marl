"""Simple MLP encoder for flat observations."""

import torch.nn as nn
from ..components import MLPEncoder


class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for flat observation vectors.

    Args:
        obs_dim: Observation dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        activation: Activation function (default: ReLU)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        obs_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: nn.Module = None,
        dropout: float = 0.0
    ):
        super().__init__()
        activation = activation or nn.ReLU()
        hidden_dims = [hidden_dim] * (num_layers - 1) if num_layers > 1 else []

        self.encoder = MLPEncoder(
            input_dim=obs_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=True
        )

    def forward(self, obs_flat):
        """Encode flat observations."""
        return self.encoder(obs_flat)
