"""Neural action wrapper for managing active/inactive action subsets."""

from typing import Dict, List, Optional, Tuple
import torch


class ActionWrapper:
    """
    Manages mapping between neural network outputs and full action space.

    The neural wrapper allows the network to control only a subset of actions
    (e.g., basic flight controls) while automation handles the rest (e.g.,
    targeting, countermeasures). This enables curriculum learning by gradually
    expanding the action space.

    Action Space (full 10 actions):
    - [0] Ps: Specific energy rate command (climb/dive/accelerate)
    - [1] n:  Normal load factor command (turn intensity)
    - [2] φ:  Bank angle command (turn direction)
    - [3] Target selection
    - [4] Missile firing
    - [5] Gun firing
    - [6-9] Countermeasures (flares, chaff, ECM, decoys)

    Example configurations:
    - Phase 1 (basic flight): active_indices=[0,1,2,4] controls Ps, n, φ, missile
    - Phase 2 (with targeting): active_indices=[0,1,2,3,4] adds target selection
    - Full control: active_indices=[0,1,2,3,4,5,6,7,8,9]
    """

    def __init__(
        self,
        wrapped_action_dim: int,
        full_action_dim: int,
        active_indices: Optional[List[int]] = None,
        use_wrapper: bool = True
    ):
        """
        Initialize action wrapper.

        Args:
            wrapped_action_dim: Dimension of network output (number of active actions)
            full_action_dim: Total action space dimension (10 for full control)
            active_indices: Indices that network controls (default: [0,1,2,4])
            use_wrapper: Whether to use wrapper or full action space
        """
        self.wrapped_action_dim = wrapped_action_dim
        self.full_action_dim = full_action_dim
        self.use_wrapper = use_wrapper

        if use_wrapper:
            # Default: basic flight controls + missile
            if active_indices is None:
                active_indices = [0, 1, 2, 4]  # Ps, n, φ, missile_fire

            self.active_indices = list(active_indices)
            self.inactive_indices = [
                i for i in range(full_action_dim) if i not in active_indices
            ]

            # Default values for automated actions
            # 0.5 for target selection (middle of range), 0.0 for triggers
            self.inactive_defaults = [0.5 if i == 3 else 0.0 for i in self.inactive_indices]
        else:
            # No wrapping - network controls all actions
            self.active_indices = list(range(full_action_dim))
            self.inactive_indices = []
            self.inactive_defaults = []

    def expand_to_full_space(
        self,
        mean_active: torch.Tensor,
        log_std_active: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand active action distribution to full action space.

        Inserts network-controlled actions at active_indices and fills
        inactive actions with automation defaults.

        Args:
            mean_active: Network output means (batch, wrapped_action_dim)
            log_std_active: Network output log stds (batch, wrapped_action_dim)

        Returns:
            Tuple of (mean_full, log_std_full) with shape (batch, full_action_dim)
        """
        if not self.use_wrapper:
            return mean_active, log_std_active

        batch_size = mean_active.shape[0]
        device = mean_active.device

        # Create full tensors
        mean_full = torch.zeros(batch_size, self.full_action_dim, device=device)
        log_std_full = torch.full(
            (batch_size, self.full_action_dim), -2.0, device=device
        )

        # Fill in active actions (network outputs)
        mean_full[:, self.active_indices] = mean_active
        log_std_full[:, self.active_indices] = log_std_active

        # Fill in inactive actions (automation defaults)
        for i, idx in enumerate(self.inactive_indices):
            mean_full[:, idx] = self.inactive_defaults[i]

        return mean_full, log_std_full

    def extract_active_actions(self, actions_full: torch.Tensor) -> torch.Tensor:
        """
        Extract only the active (network-controlled) actions from full action tensor.

        Args:
            actions_full: Full action tensor (batch, full_action_dim)

        Returns:
            Active actions (batch, wrapped_action_dim)
        """
        if not self.use_wrapper:
            return actions_full

        return actions_full[:, self.active_indices]

    def get_active_action_dim(self) -> int:
        """Get the dimension of actions that the network actually outputs."""
        return self.wrapped_action_dim if self.use_wrapper else self.full_action_dim

    def __repr__(self) -> str:
        if self.use_wrapper:
            return (
                f"ActionWrapper(active={self.active_indices}, "
                f"inactive={self.inactive_indices}, "
                f"defaults={self.inactive_defaults})"
            )
        return "ActionWrapper(full_control)"
