"""Stable public visualization bridge API for bvr-marl.

Extension packages that implement RL/BT
explainability views must import visualization infrastructure from this
module rather than from deep internal paths.  All other
``bvr_marl_core.visualization.*`` paths are considered internal and
subject to change without notice.

Exported symbols
----------------
CombinedVisualizer
    Combined 2D/attention live visualizer for RL explainability.
DefaultModel
    Dummy model that returns random actions (for visualization without a
    trained checkpoint).
TrainedModelWrapper
    Wraps a trained RLlib checkpoint for inference during visualization.
"""

from __future__ import annotations

from bvr_marl_core.visualization.model_wrapper.model_wrapper import (
    DefaultModel,
    TrainedModelWrapper,
)
from bvr_marl_core.visualization.scenario_overlays import (
    normalize_map_extents_mode,
    normalize_visualization_scenario,
    select_display_limits,
)
from bvr_marl_core.visualization.utils.combined_visualizer import CombinedVisualizer

__all__ = [
    "CombinedVisualizer",
    "DefaultModel",
    "TrainedModelWrapper",
    "normalize_map_extents_mode",
    "normalize_visualization_scenario",
    "select_display_limits",
]
