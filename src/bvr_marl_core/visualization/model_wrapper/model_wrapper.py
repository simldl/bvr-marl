"""Model wrappers for inference."""

import numpy as np

from bvr_marl_core.visualization.model_wrapper.inference_output import deterministic_actions
from bvr_marl_core.visualization.model_wrapper.policy_selection import resolve_policy_id


class DefaultModel:
    """Default model that returns random actions."""

    def __init__(self, env):
        self.env = env

    def compute_single_action(self, observation, agent_id):
        """Compute a random action."""
        return np.random.rand(10).astype(np.float32)  # Updated for gun firing action


class TrainedModelWrapper:
    """Wrapper for trained RLlib models."""

    def __init__(self, checkpoint_path, env, model_config_path=None, train_config=None):
        import logging
        from pathlib import Path

        import ray
        from ray.rllib.core.rl_module import RLModule

        self.env = env
        self.train_config = train_config or {}

        # Suppress Ray core worker warnings
        logging.getLogger("ray.rllib.utils.deprecation").setLevel(logging.ERROR)

        # Initialize Ray for potential remote inference (though we'll use local)
        ray.init(ignore_reinit_error=True, log_to_driver=False)

        # For inference, we can directly load the RLModule from the checkpoint
        # This avoids all the config deserialization issues with refactored code
        # The RLModule is stored in: checkpoint_path/learner_group/learner/rl_module/

        # For multi-agent, we need to load the MultiRLModule (parent of all policies)
        rl_module_path = Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"

        if not rl_module_path.exists():
            raise FileNotFoundError(f"RLModule not found at {rl_module_path}")

        # Check what policies are available
        policy_dirs = [d for d in rl_module_path.iterdir() if d.is_dir()]
        print(f"Found policies in checkpoint: {[d.name for d in policy_dirs]}")

        # Load the MultiRLModule directly - this bypasses all the Algorithm config issues
        # RLModule.from_checkpoint will automatically detect if it's a MultiRLModule
        self.rl_module = RLModule.from_checkpoint(rl_module_path)

        print(f"Loaded RLModule: {type(self.rl_module)}")

        # For MultiRLModule, we need to check if it has the keys() method
        # If not, it means we loaded a single-agent module by mistake
        if hasattr(self.rl_module, "keys"):
            print(f"Available policies: {list(self.rl_module.keys())}")
        else:
            # This is a single RLModule, wrap it in a dict-like structure
            print("Warning: Loaded single RLModule instead of MultiRLModule")
            print("Creating wrapper for single policy...")
            # Store the single module and create a simple dict-like interface
            self._single_module = self.rl_module

            # Create a simple wrapper that acts like a MultiRLModule
            class SingleModuleWrapper:
                def __init__(self, module):
                    self._module = module
                    self._policy_name = "shared_policy"

                def keys(self):
                    return [self._policy_name]

                def __getitem__(self, key):
                    if key == self._policy_name:
                        return self._module
                    raise KeyError(f"Policy {key} not found")

            self.rl_module = SingleModuleWrapper(self._single_module)
            print(f"Wrapped module, available policies: {list(self.rl_module.keys())}")

    def compute_single_action(self, observation, agent_id=None):
        """Compute action using the trained model."""
        available_policies = list(self.rl_module.keys())
        policy_id = resolve_policy_id(agent_id, available_policies, self.train_config)

        # Use the new RLModule API for inference
        # Convert observation to batch format
        # If observation is a dict, batch each component separately
        if isinstance(observation, dict):
            obs_batch = {key: np.expand_dims(value, axis=0) for key, value in observation.items()}
        else:
            obs_batch = np.expand_dims(observation, axis=0)

        # Get the specific policy module for this agent
        policy_module = self.rl_module[policy_id]

        # Ensure model is in eval mode for inference
        if hasattr(policy_module, "eval"):
            policy_module.eval()

        # Convert observation to torch tensor if needed
        import torch

        if isinstance(obs_batch, dict):
            obs_batch_torch = {}
            for key, value in obs_batch.items():
                if not isinstance(value, torch.Tensor):
                    obs_batch_torch[key] = torch.from_numpy(value).float()
                else:
                    obs_batch_torch[key] = value.float()
            obs_batch = obs_batch_torch
        elif not isinstance(obs_batch, torch.Tensor):
            obs_batch = torch.from_numpy(obs_batch).float()

        # Run inference through the policy's RLModule
        # The public method is forward_inference (not _forward_inference)
        # Use RLlib's Columns constant for the observation key
        from ray.rllib.core.columns import Columns

        batch = {Columns.OBS: obs_batch}
        output = policy_module.forward_inference(batch)

        actions = deterministic_actions(output, Columns)

        # Convert from torch tensor to numpy if needed
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        elif hasattr(actions, "numpy"):
            actions = actions.numpy()

        # The action processor expects normalized inputs in energy-space mode.
        actions = np.clip(actions, 0.0, 1.0)

        return actions
