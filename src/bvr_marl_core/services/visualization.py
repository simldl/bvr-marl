"""
Visualization service — shared command construction for CLI and GUI.

Provides ``build_visualization_cmd`` so the GUI panel and any future CLI
wrapper use identical launch mechanics.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Maps friendly mode names to canonical mode keys.
VISUALIZATION_MODES: dict[str, str] = {
    "Standard 2D Live View": "standard",
    "Behavior Tree Controllers": "behavior-tree",
    "RL Commands Panel": "commands",
}

#: Maps canonical mode keys to their Python module path.
_MODE_MODULES: dict[str, str] = {
    "standard": "bvr_marl_core.visualization.live_view",
    "commands": "bvr_marl_core.visualization.live_view_commands",
}

_MODE_COMMANDS: dict[str, list[str]] = {
    "behavior-tree": ["bvr-view-bt"],
}

# Args supported by each module (only these are passed to avoid "unrecognized arguments").
# standard:      --checkpoint --model-config --train-config --viz-config
#                --save-rewards/--no-save-rewards --random --real-time --show-text/--no-show-text
# behavior-tree: --train-config --viz-config --frames --interval --save-video
#                --real-time --show-text/--no-show-text
# commands:      --checkpoint --model-config --train-config --viz-config
#                --real-time --show-text/--no-show-text
_MODE_SUPPORTS_FRAMES = {"standard", "behavior-tree"}
_MODE_SUPPORTS_INTERVAL = {"standard", "behavior-tree"}
_MODE_SUPPORTS_SAVE_VIDEO = {"standard", "behavior-tree"}
_MODE_SUPPORTS_CHECKPOINT = {"standard", "commands"}
_MODE_SUPPORTS_SAVE_REWARDS = {"standard"}
_MODE_SUPPORTS_SCENARIO = {"standard", "behavior-tree", "commands"}
_MODE_SUPPORTS_LINE_OVERLAY = {"standard", "behavior-tree", "commands"}
_MODE_SUPPORTS_MAP_EXTENTS = {"standard", "behavior-tree", "commands"}
_MODE_SUPPORTS_SHOW_TEXT = {"standard", "behavior-tree", "commands"}


def build_visualization_cmd(
    mode: str = "standard",
    checkpoint: str | None = None,
    train_config: str | None = None,
    viz_config: str | None = None,
    scenario: str | None = None,
    show_line_overlay: bool = False,
    line_east_km: float | None = None,
    map_extents_mode: str | None = None,
    frames: int = 100,
    interval: int = 100,
    save_video: bool = False,
    save_rewards: bool = True,
    aircraft_type: str | None = None,
    real_time: bool = False,
    show_text: bool | None = None,
) -> list[str]:
    """Build the command list for launching a visualization process.

    Each visualization mode is a separate module with its own argument set.
    This function selects the correct module and only passes arguments that
    the module actually supports.

    Args:
        mode: One of ``"standard"``, ``"behavior-tree"``, ``"commands"`` — or a
              friendly display name from :data:`VISUALIZATION_MODES`.
        checkpoint: Path to a trained model checkpoint (optional).
        train_config: Name or path of the training config (optional).
        viz_config: Name or path of the visualization config (optional).
        scenario: Scenario overlay key that overrides the viz config (optional).
        show_line_overlay: Force a line-of-engagement overlay even for scenarios
            without a built-in line.
        line_east_km: Optional LOE x-axis offset in kilometers.
        map_extents_mode: One of ``auto``, ``combat``, or ``full``.
        frames: Number of frames to render (behavior-tree only).
        interval: Animation interval in ms (behavior-tree only).
        save_video: Save animation as video (behavior-tree only).
        save_rewards: Save reward logs to CSV/JSON (standard only).
        aircraft_type: Unused (kept for API compatibility).
        real_time: Run at simulation tick rate (all modes).
        show_text: Show map labels and status text. ``False`` passes
            ``--no-show-text`` for modes that support it.

    Returns:
        Command list suitable for :class:`subprocess.Popen`.
    """
    # Accept either friendly display names or raw mode keys
    resolved_mode = VISUALIZATION_MODES.get(mode, mode)
    if resolved_mode in _MODE_COMMANDS:
        cmd = list(_MODE_COMMANDS[resolved_mode])
    else:
        module = _MODE_MODULES.get(resolved_mode, _MODE_MODULES["standard"])
        cmd = [sys.executable, "-m", module]

    if resolved_mode in _MODE_SUPPORTS_CHECKPOINT and checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if train_config:
        cmd += ["--train-config", train_config]
    if viz_config:
        cmd += ["--viz-config", viz_config]
    if resolved_mode in _MODE_SUPPORTS_SCENARIO and scenario:
        cmd += ["--scenario", scenario]
    if resolved_mode in _MODE_SUPPORTS_LINE_OVERLAY and show_line_overlay:
        cmd.append("--show-line")
    if resolved_mode in _MODE_SUPPORTS_LINE_OVERLAY and line_east_km is not None:
        cmd += ["--line-east-km", str(line_east_km)]
    if resolved_mode in _MODE_SUPPORTS_MAP_EXTENTS and map_extents_mode:
        cmd += ["--map-extents", map_extents_mode]
    if resolved_mode in _MODE_SUPPORTS_FRAMES:
        cmd += ["--frames", str(frames)]
    if resolved_mode in _MODE_SUPPORTS_INTERVAL and not real_time:
        cmd += ["--interval", str(interval)]
    if resolved_mode in _MODE_SUPPORTS_SAVE_VIDEO and save_video:
        cmd.append("--save-video")
    if resolved_mode in _MODE_SUPPORTS_SAVE_REWARDS:
        cmd.append("--save-rewards" if save_rewards else "--no-save-rewards")
    if resolved_mode in _MODE_SUPPORTS_SHOW_TEXT and show_text is not None:
        cmd.append("--show-text" if show_text else "--no-show-text")
    if real_time:
        cmd.append("--real-time")

    return cmd


def find_checkpoint_files(search_root: str | Path | None = None) -> list[str]:
    """Return sorted checkpoint file paths under *search_root* (default: project root).

    Searches common output directories: ``models/``, ``outputs/``,
    ``checkpoints/``.
    """
    from bvr_marl_core.utils.paths import core_project_root

    root = Path(search_root) if search_root else core_project_root()
    patterns = [
        "models/**/*checkpoint*",
        "outputs/*/checkpoint*",
        "checkpoints/*",
        "*.pkl",
        "*.pth",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return [str(f) for f in sorted(set(files))]
