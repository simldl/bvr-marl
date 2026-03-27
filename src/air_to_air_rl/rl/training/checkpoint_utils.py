"""Checkpoint loading and restoration utilities."""

import os
from typing import Optional


def resolve_checkpoint_path(checkpoint_path: str, project_root: str) -> str:
    """Convert relative checkpoint path to absolute path (relative to project root)."""
    if not os.path.isabs(checkpoint_path):
        return os.path.abspath(os.path.join(project_root, checkpoint_path))
    return checkpoint_path


def find_experiment_directory(checkpoint_path: str) -> str | None:
    """
    Find experiment directory from checkpoint path.

    Returns experiment directory if found, otherwise None.
    """
    # Check if this is already an experiment directory (has tuner.pkl)
    if os.path.isfile(os.path.join(checkpoint_path, "tuner.pkl")):
        return checkpoint_path

    # Check if this is a checkpoint directory
    if "checkpoint_" in os.path.basename(checkpoint_path):
        trial_dir = os.path.dirname(checkpoint_path)
        potential_exp_dir = os.path.dirname(trial_dir)
        if os.path.isfile(os.path.join(potential_exp_dir, "tuner.pkl")):
            return potential_exp_dir
        return None

    # Check if this is a trial directory
    potential_exp_dir = os.path.dirname(checkpoint_path)
    if os.path.isfile(os.path.join(potential_exp_dir, "tuner.pkl")):
        return potential_exp_dir

    return None


def process_checkpoint_resume(
    resume_checkpoint: str | None, project_root: str
) -> tuple[str | None, str | None]:
    """
    Process checkpoint resumption logic.

    Returns:
        tuple of (experiment_dir, load_weights_from)
        - experiment_dir: Directory to resume full experiment from
        - load_weights_from: Checkpoint to load weights from for new training
    """
    if not resume_checkpoint:
        return None, None

    checkpoint_path = resolve_checkpoint_path(resume_checkpoint, project_root)

    if not os.path.exists(checkpoint_path):
        print("=" * 80)
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Starting from scratch instead.")
        print("=" * 80)
        return None, None

    # Try to find experiment directory
    experiment_dir = find_experiment_directory(checkpoint_path)

    if experiment_dir:
        print("=" * 80)
        if "checkpoint_" in os.path.basename(checkpoint_path):
            print(f"RESUMING FROM CHECKPOINT: {os.path.basename(checkpoint_path)}")
        else:
            print(f"RESUMING FROM EXPERIMENT: {experiment_dir}")
        print("=" * 80)
        return experiment_dir, None
    else:
        # No experiment directory found - use as weight loading
        print("=" * 80)
        print("No experiment directory found - loading from single checkpoint")
        print(f"CHECKPOINT: {os.path.basename(checkpoint_path)}")
        print("Will start NEW training with weights from this checkpoint")
        print("=" * 80)
        return None, checkpoint_path


def process_weight_loading(load_weights_from: str | None, project_root: str) -> str | None:
    """
    Process weight loading from checkpoint.

    Returns checkpoint path if valid, otherwise None.
    """
    if not load_weights_from:
        return None

    checkpoint_path = resolve_checkpoint_path(load_weights_from, project_root)

    if os.path.exists(checkpoint_path):
        print("=" * 80)
        print("STARTING NEW TRAINING WITH WEIGHTS FROM:")
        print(f"  {checkpoint_path}")
        print("=" * 80)
        return checkpoint_path
    else:
        print("=" * 80)
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Starting with random weights instead.")
        print("=" * 80)
        return None
