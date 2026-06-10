"""
Smoke tests for the training pipeline.

Verifies that:
- The BVR environment can be created, reset, and stepped
- A minimal subprocess training run completes without error (slow)

Note: Training config tests live in bvr-marl-behavior, where the full
YAML config suite (with reward parameters) is maintained.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_environment_create_reset_step():
    """BVR environment can be created, reset, and stepped with random actions."""
    from bvr_marl_core.rl.utils.env_creator import create_env_creator

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
            "bvr_marl_core.training.train",
            "--config",
            "basic",
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
