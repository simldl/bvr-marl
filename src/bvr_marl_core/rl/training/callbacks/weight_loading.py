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

    def __init__(
        self,
        checkpoint_path: str,
        policy_mapping: dict | None = None,
        strict_policy_mapping: bool = False,
    ):
        """
        Initialize the weight loading callback.

        Args:
            checkpoint_path: Path to the checkpoint directory containing learner_group/.
            policy_mapping: Optional source-to-target policy transfer mapping.
            strict_policy_mapping: Raise instead of falling back to random weights.
        """
        self.checkpoint_path = checkpoint_path
        self.policy_mapping = dict(policy_mapping or {})
        self.strict_policy_mapping = bool(strict_policy_mapping)
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
        if not hasattr(trial, "runner") or trial.runner is None:
            return  # Not ready yet

        algorithm = trial.runner

        # Find the RLModule checkpoint
        rl_module_path = os.path.join(
            self.checkpoint_path, "learner_group", "learner", "rl_module", "module_state.pkl"
        )

        if not os.path.exists(rl_module_path):
            message = f"RLModule checkpoint not found at: {rl_module_path}"
            if self.strict_policy_mapping:
                raise RuntimeError(message)
            print(f"[WARNING] {message}")
            print("[WARNING] Starting with random weights")
            self.weights_loaded = True  # Mark as attempted
            return

        try:
            print(f"[WEIGHT LOADING] Loading weights from: {self.checkpoint_path}")

            # Load the module state
            with open(rl_module_path, "rb") as f:
                module_state = pickle.load(f)

            # Get the RLModule from the learner group
            if hasattr(algorithm, "learner_group"):
                learner = algorithm.learner_group._learner
                if hasattr(learner, "module"):
                    self._load_policy_states(learner.module, module_state)
                    print("[WEIGHT LOADING] All requested weights loaded successfully!")
                    self.weights_loaded = True
                else:
                    if self.strict_policy_mapping:
                        raise RuntimeError("Could not access learner module for weight loading.")
                    print("[WARNING] Could not access learner module")
            else:
                if self.strict_policy_mapping:
                    raise RuntimeError("Algorithm does not have learner_group for weight loading.")
                print("[WARNING] Algorithm doesn't have learner_group")

        except Exception as e:
            print(f"[ERROR] Could not load checkpoint: {e}")
            import traceback

            traceback.print_exc()
            if self.strict_policy_mapping:
                raise
            print("[WARNING] Continuing with random weights")

        self.weights_loaded = True  # Mark as attempted regardless of success

    def _load_policy_states(self, multi_rl_module, module_state) -> None:
        """Load checkpoint policy states into target RLModules."""
        if not isinstance(module_state, dict):
            multi_rl_module.set_state(module_state)
            print("[WEIGHT LOADING] Loaded module state")
            return

        mapping = self.policy_mapping or {"mode": "direct_policy_id_match"}
        mode = str(mapping.get("mode") or "")
        if mode == "direct_policy_id_match" or not mapping:
            loaded = self._load_direct_policy_matches(multi_rl_module, module_state)
        elif mapping.get("from_checkpoint_policy") and mapping.get("to_stage_policies"):
            loaded = self._load_mapped_policy_states(multi_rl_module, module_state, mapping)
        else:
            raise ValueError(
                "Unsupported policy_weight_mapping. Use mode=direct_policy_id_match or "
                "from_checkpoint_policy/to_stage_policies."
            )

        if self.strict_policy_mapping and loaded <= 0:
            raise RuntimeError(
                "Policy weight mapping did not initialize any target policy. "
                f"source_policies={list(module_state.keys())}"
            )

    def _load_direct_policy_matches(self, multi_rl_module, module_state: dict) -> int:
        loaded = 0
        for policy_id, policy_state in module_state.items():
            if policy_id not in multi_rl_module:
                continue
            multi_rl_module[policy_id].set_state(policy_state)
            loaded += 1
            print(f"[CURRICULUM] Loaded source policy: {policy_id}")
            print(f"[CURRICULUM] Initialized target policy: {policy_id}")
        if loaded > 0:
            print("[CURRICULUM] Weight transfer complete.")
        target_ids = self._target_policy_ids(multi_rl_module)
        missing = [policy_id for policy_id in target_ids if policy_id not in module_state]
        if self.strict_policy_mapping and missing:
            raise RuntimeError(
                "Direct policy-id checkpoint transfer did not find source weights for "
                f"target policies: {missing}"
            )
        return loaded

    def _load_mapped_policy_states(
        self,
        multi_rl_module,
        module_state: dict,
        mapping: dict,
    ) -> int:
        source_policy = str(mapping["from_checkpoint_policy"])
        target_policies = [str(policy_id) for policy_id in mapping["to_stage_policies"]]
        if source_policy not in module_state:
            raise KeyError(
                f"Source policy {source_policy!r} not found in checkpoint. "
                f"Available source policies: {list(module_state.keys())}"
            )

        loaded = 0
        for target_policy in target_policies:
            if target_policy not in multi_rl_module:
                raise KeyError(f"Target policy {target_policy!r} not present in target module.")
            multi_rl_module[target_policy].set_state(module_state[source_policy])
            loaded += 1
            print(f"[CURRICULUM] Loaded source policy: {source_policy}")
            print(f"[CURRICULUM] Initialized target policy: {target_policy}")
        if loaded > 0:
            print("[CURRICULUM] Weight transfer complete.")
        return loaded

    @staticmethod
    def _target_policy_ids(multi_rl_module) -> list[str]:
        keys = getattr(multi_rl_module, "keys", None)
        if callable(keys):
            return [str(policy_id) for policy_id in keys()]
        return []
