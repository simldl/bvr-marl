"""Public analysis/plotting API surface.

Stable entry point that extension packages may import. The implementation
lives in ``bvr_marl_core.analysis.plotting``; import
from here rather than reaching into those submodules so the public surface stays
explicit and the import-boundary checks pass.

This mirrors the ``bvr_marl_core.rl.api`` / ``bvr_marl_core.visualization.api``
pattern.
"""

from __future__ import annotations

from bvr_marl_core.analysis.plotting.matcom_style import (
    MATCOM_COLORS,
    set_matcom_style,
)
from bvr_marl_core.analysis.plotting.plot_tensorboard_metric import (
    RLLIB_TAG_PREFIXES,
    TAG_LABELS,
    has_known_label,
    labels_for_tag,
    normalize_tag,
    plot_metric,
)

# Rendering helpers shared with the GUI plot exporter. Underscore-prefixed in the
# implementation module, re-exported here under their original names because the
# exporter builds figures with them directly.
from bvr_marl_core.analysis.plotting.plot_tensorboard_metric import (
    _run_color_map as run_color_map,
)
from bvr_marl_core.analysis.plotting.plot_tensorboard_metric import (
    _smooth as smooth_series,
)
from bvr_marl_core.analysis.plotting.plot_training_dashboard import (
    DASHBOARD_TAGS,
    TACTICAL_DASHBOARD_TAGS,
    plot_tactical_dashboard,
    plot_training_dashboard,
)
from bvr_marl_core.analysis.plotting.tensorboard_loader import load_tensorboard_scalars
from bvr_marl_core.analysis.plotting.unibw_style import (
    RUN_COLOR_ORDER,
    UNIBW_COLORS,
    set_unibw_style,
)

__all__ = [
    # catalog + tag helpers
    "TAG_LABELS",
    "RLLIB_TAG_PREFIXES",
    "normalize_tag",
    "labels_for_tag",
    "has_known_label",
    # single-metric + dashboards
    "plot_metric",
    "plot_training_dashboard",
    "plot_tactical_dashboard",
    "DASHBOARD_TAGS",
    "TACTICAL_DASHBOARD_TAGS",
    # data loading
    "load_tensorboard_scalars",
    # styles + rendering helpers
    "set_unibw_style",
    "set_matcom_style",
    "UNIBW_COLORS",
    "MATCOM_COLORS",
    "RUN_COLOR_ORDER",
    "run_color_map",
    "smooth_series",
]
