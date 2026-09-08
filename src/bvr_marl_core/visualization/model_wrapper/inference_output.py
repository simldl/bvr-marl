"""Shared extraction of deterministic actions from RLlib inference output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def deterministic_actions(output: Mapping[str, Any], columns: Any) -> Any:
    """Return one unbatched action vector from supported RLlib output schemas."""
    actions = output.get(columns.ACTIONS, output.get("actions"))
    if actions is not None:
        return actions[0, 0] if actions.dim() == 3 else actions[0]

    distribution = output.get(
        columns.ACTION_DIST_INPUTS,
        output.get("action_dist_inputs"),
    )
    if distribution is None:
        raise ValueError(
            f"Unexpected output format from forward_inference. Keys: {list(output.keys())}"
        )
    distribution = distribution[0, 0] if distribution.dim() == 3 else distribution[0]
    return distribution[: distribution.shape[0] // 2]
