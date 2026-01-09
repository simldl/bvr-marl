"""Basic encoding components for feature extraction."""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder with layer normalization and optional dropout.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions (default: [])
        activation: Activation function (default: ReLU)
        dropout: Dropout rate (default: 0.0)
        use_layer_norm: Whether to use layer normalization (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = None,
        activation: nn.Module = None,
        dropout: float = 0.0,
        use_layer_norm: bool = True
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        activation = activation or nn.ReLU()

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if i < len(dims) - 2:  # No activation/dropout after last layer
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features."""
        return self.encoder(x)
