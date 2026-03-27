"""Callback for loading weights from a checkpoint at training start."""

import os

from ray.tune import Callback


class WeightLoadingCallback(Callback):
    """
    Callback that loads model weights from a previous checkpoint.

    This allows continuing training from a saved checkpoint.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize the weight loading callback.

        Args:
            checkpoint_path: Path to the checkpoint directory
        """
        self.checkpoint_path = checkpoint_path
        self.weights_loaded = False

    def on_trial_start(self, iteration: int, trials: list, trial, **info):
        """
        Load weights when trial starts (after algorithm initialization).
        """
        if self.weights_loaded:
            return

        if not hasattr(trial, "runner") or trial.runner is None:
            return

        algorithm = trial.runner

        try:
            print(f"[WEIGHT LOADING] Loading weights from: {self.checkpoint_path}")

            # Use Ray's built-in checkpoint restoration
            algorithm.restore(self.checkpoint_path)

            print("[WEIGHT LOADING] ✓ Weights loaded successfully!")
            self.weights_loaded = True

        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}")
            print("[WARNING] Continuing with random weights")
            self.weights_loaded = True
