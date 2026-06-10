"""Tacview command construction shared by the CLI and GUI."""

from __future__ import annotations

import sys


def build_tacview_cmd(
    controller: str = "rl",
    checkpoint: str | None = None,
    model_config: str | None = None,
    train_config: str | None = None,
    aircraft: str | None = None,
    frames: int = 500,
    acmi_path: str | None = None,
    num_scenarios: int = 1,
    seed_start: int = 0,
) -> list[str]:
    """Build the command list for Tacview scenario generation.

    ``controller="rl"`` uses the core Tacview generator.  The
    ``"behavior-tree"`` controller delegates to the optional
    ``bvr-tacview-bt`` command provided by the behavior package.
    """
    if controller == "behavior-tree":
        cmd = ["bvr-tacview-bt"]
        if aircraft:
            cmd += ["--aircraft", aircraft]
    else:
        cmd = [sys.executable, "-m", "bvr_marl_core.tacview.generate"]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        if model_config:
            cmd += ["--model-config", model_config]
        if train_config:
            cmd += ["--train-config", train_config]

    cmd += [
        "--frames",
        str(frames),
        "--num-scenarios",
        str(num_scenarios),
        "--seed-start",
        str(seed_start),
    ]
    if acmi_path:
        cmd += ["--acmi", acmi_path]

    return cmd
