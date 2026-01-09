"""Callback for loading weights from a checkpoint at training start."""

import os
import pickle
from ray.tune import Callback


class WeightLoadingCallback(Callback):
    """
    Callback that loads RLModule weights from a checkpoint after trial setup.

    This allows starting a new training run with weights from a previous checkpoint
    without the TensorBoard path issues that come from custom trainable wrappers.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize the weight loading callback.

        Args:
            checkpoint_path: Path to the checkpoint directory containing learner_group/
        """
        self.checkpoint_path = checkpoint_path
        self.weights_loaded = False

    def on_trial_start(self, iteration: int, trials: list, trial, **info):
        """
        Load weights when trial starts (after algorithm initialization).

        This runs after PPO is fully initialized, including TensorBoard setup,
        so we don't interfere with the normal trainable name/path logic.
        """
        if self.weights_loaded:
            return  # Only load once

        # Get the algorithm instance from the trial
        if not hasattr(trial, 'runner') or trial.runner is None:
            return  # Not ready yet

        algorithm = trial.runner

        # Find the RLModule checkpoint
        rl_module_path = os.path.join(
            self.checkpoint_path,
            "learner_group",
            "learner",
            "rl_module",
            "module_state.pkl"
        )

        if not os.path.exists(rl_module_path):
            print(f"[WARNING] RLModule checkpoint not found at: {rl_module_path}")
            print("[WARNING] Starting with random weights")
            self.weights_loaded = True  # Mark as attempted
            return

        try:
            print(f"[WEIGHT LOADING] Loading weights from: {self.checkpoint_path}")

            # Load the module state
            with open(rl_module_path, "rb") as f:
                module_state = pickle.load(f)

            # Get the RLModule from the learner group
            if hasattr(algorithm, 'learner_group'):
                learner = algorithm.learner_group._learner
                if hasattr(learner, 'module'):
                    multi_rl_module = learner.module

                    # Load weights for each policy
                    if isinstance(module_state, dict):
                        for policy_id, policy_state in module_state.items():
                            if policy_id in multi_rl_module:
                                multi_rl_module[policy_id].set_state(policy_state)
                                print(f"[WEIGHT LOADING] ✓ Loaded weights for policy: {policy_id}")
                    else:
                        # Direct loading if not a dict
                        multi_rl_module.set_state(module_state)
                        print(f"[WEIGHT LOADING] ✓ Loaded module state")

                    print("[WEIGHT LOADING] ✓ All weights loaded successfully!")
                    self.weights_loaded = True
                else:
                    print("[WARNING] Could not access learner module")
            else:
                print("[WARNING] Algorithm doesn't have learner_group")

        except Exception as e:
            print(f"[ERROR] Could not load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("[WARNING] Continuing with random weights")

        self.weights_loaded = True  # Mark as attempted regardless of success
