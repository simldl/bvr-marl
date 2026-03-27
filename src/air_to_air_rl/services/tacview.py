"""
Tacview service — shared command construction for CLI and GUI.

Provides ``build_tacview_cmd`` so the GUI Tacview generator and any future
CLI wrapper use identical launch mechanics.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Default script path (relative to project root) used by the GUI.
GENERATE_SCRIPT = "scripts/tacview/generate_scenario.py"


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
    use_script: bool = True,
) -> list[str]:
    """Build the command list for Tacview scenario generation.

    Args:
        controller: ``"rl"`` or ``"behavior-tree"``.
        checkpoint: Path to a trained model checkpoint (RL only, optional).
        model_config: Path to model config (RL only, optional).
        train_config: Name or path of the training config (RL only, optional).
        aircraft: Aircraft type override (behavior-tree only, optional).
        frames: Number of simulation frames per scenario.
        acmi_path: Output ACMI path (auto-generated if None).
        num_scenarios: Number of scenarios to generate.
        seed_start: Starting random seed.
        use_script: If True, invoke ``scripts/tacview/generate_scenario.py``
                    (GUI mode).  If False, invoke the installed package module
                    ``air_to_air_rl.tacview.generate`` directly (CLI mode).

    Returns:
        Command list suitable for :class:`subprocess.Popen`.
    """
    if use_script:
        cmd = [sys.executable, GENERATE_SCRIPT]
    else:
        cmd = [sys.executable, "-m", "air_to_air_rl.tacview.generate"]

    cmd += ["--controller", controller]

    if controller == "rl":
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        if model_config:
            cmd += ["--model-config", model_config]
        if train_config:
            cmd += ["--train-config", train_config]
    elif controller == "behavior-tree":
        if aircraft:
            cmd += ["--aircraft", aircraft]

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
