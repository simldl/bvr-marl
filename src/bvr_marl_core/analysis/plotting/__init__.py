"""UniBw-style TensorBoard plotting utilities."""

from bvr_marl_core.analysis.plotting.plot_tensorboard_metric import TAG_LABELS, plot_metric
from bvr_marl_core.analysis.plotting.plot_training_dashboard import (
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
    "UNIBW_COLORS",
    "RUN_COLOR_ORDER",
    "TAG_LABELS",
    "TACTICAL_DASHBOARD_TAGS",
    "set_unibw_style",
    "load_tensorboard_scalars",
    "plot_metric",
    "plot_training_dashboard",
    "plot_tactical_dashboard",
]
