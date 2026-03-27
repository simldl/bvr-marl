"""
Smoke tests for the training pipeline.

Verifies that:
- Training configs can be loaded
- The BVR environment can be created, reset, and stepped
- A minimal subprocess training run completes without error (slow)
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_training_config_loads():
    """Default training config can be loaded and contains expected keys."""
    from air_to_air_rl.utils.config_loader import load_train_config

    cfg = load_train_config()
    assert isinstance(cfg, dict), "Config must be a dict"
    assert "env" in cfg, "Config must have an 'env' section"


def test_training_config_resolve():
    """resolve_training_config returns a Path that exists for the default config."""
    from air_to_air_rl.core.paths import rl_configs_root
    from air_to_air_rl.utils.config_loader import resolve_training_config

    # The package-bundled default should always resolve
    default = rl_configs_root() / "train_config.yaml"
    assert default.exists(), f"Default training config not found: {default}"


def test_environment_create_reset_step():
    """BVR environment can be created, reset, and stepped with random actions."""
    from air_to_air_rl.rl.utils.env_creator import create_env_creator

    cfg = {
        "env": {
            "num_agents_per_team": 1,
            "max_steps": 5,
        }
    }
    env_creator = create_env_creator(cfg)
    env = env_creator({})

    obs, info = env.reset()
    assert obs is not None, "reset() must return observations"

    # Sample and apply one action for each agent
    actions = {agent_id: env.action_space[agent_id].sample() for agent_id in obs}
    obs2, rewards, terminateds, truncateds, infos = env.step(actions)

    assert isinstance(rewards, dict), "step() must return a rewards dict"
    assert isinstance(terminateds, dict), "step() must return a terminateds dict"

    env.close()


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Ray startup overhead too slow for CI runners"
)
def test_training_subprocess_runs():
    """A minimal training run completes via subprocess (2 steps, no GPU)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "air_to_air_rl.training.train",
            "--overrides",
            "training.steps=2",
            "num_env_runners=1",
            "num_gpus=0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Training subprocess failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
