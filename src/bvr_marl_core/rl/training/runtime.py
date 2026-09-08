"""Shared runtime helpers for RLlib training entry points."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from ray import tune


def set_random_seeds(seed: int, *, announce: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if announce:
        print("=" * 80)
        print(f"RANDOM SEED SET TO: {seed}")
        print("=" * 80)


def shared_policy_mapping_fn(_agent_id: Any, *_args: Any, **_kwargs: Any) -> str:
    """All agents share a single policy for symmetric self-play."""
    return "shared_policy"


def create_tuner(ppo_config: Any, run_config: tune.RunConfig, experiment_dir: str | None):
    if experiment_dir:
        return tune.Tuner.restore(
            path=experiment_dir,
            trainable="PPO",
            param_space=ppo_config.to_dict(),
            resume_unfinished=True,
            resume_errored=False,
        )
    return tune.Tuner(
        trainable="PPO",
        param_space=ppo_config.to_dict(),
        run_config=run_config,
    )
