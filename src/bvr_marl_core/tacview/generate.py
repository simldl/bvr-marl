"""
Tacview scenario generation for BVR simulation with optional trained model support.

"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bvr_marl_core.registry import get_aircraft_class as _get_aircraft_class
from bvr_marl_core.rl.environment.env_registry import resolve_env_class
from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from bvr_marl_core.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv
from bvr_marl_core.rl.utils.type_maps import resolve_aircraft_config
from bvr_marl_core.simulator import MapLimits, Simulator
from bvr_marl_core.tacview.logger import TacviewLogger
from bvr_marl_core.utils.config_loader import load_train_config
from bvr_marl_core.utils.paths import tacview_output_root
from bvr_marl_core.visualization.model_wrapper.inference_output import deterministic_actions
from bvr_marl_core.visualization.model_wrapper.policy_selection import resolve_policy_id


class DefaultModel:
    """Default model that returns random actions."""

    def __init__(self, env, seed=None):
        self.env = env
        self._rng = np.random.default_rng(seed)

    def compute_single_action(self, observation, agent_id):
        return self._rng.random(10, dtype=np.float32)


class TrainedModelWrapper:
    """Wrapper for trained RLlib models."""

    def __init__(self, checkpoint_path, env, model_config_path=None, train_config=None):
        import logging
        from pathlib import Path

        import ray
        from ray.rllib.core.rl_module import RLModule

        self.env = env
        self.train_config = train_config or {}
        self._states: dict[tuple[str, str | None], Any] = {}
        self.ray_initialized_here = False

        # Suppress Ray core worker warnings
        logging.getLogger("ray.rllib.utils.deprecation").setLevel(logging.ERROR)

        # Only initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
            self.ray_initialized_here = True

        # Load the MultiRLModule directly.  This mirrors the visualization wrapper
        # and avoids restoring a full Algorithm with stale env/config references.
        checkpoint = Path(checkpoint_path)
        rl_module_candidates = [
            checkpoint / "learner_group" / "learner" / "rl_module",
            checkpoint / "rl_module",
        ]
        self.rl_module = None
        for candidate in rl_module_candidates:
            if not candidate.exists():
                continue
            try:
                self.rl_module = RLModule.from_checkpoint(str(candidate))
                break
            except Exception as exc:
                print(f"Warning: could not load RLModule from {candidate}: {exc}")

        if self.rl_module is None:
            self.rl_module = self._load_algorithm_module(checkpoint_path, model_config_path)

        print(f"Loaded RLModule: {type(self.rl_module)}")
        if hasattr(self.rl_module, "keys"):
            print(f"Available policies: {list(self.rl_module.keys())}")
        else:
            self._single_module = self.rl_module

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
            print(f"Wrapped single module, available policies: {list(self.rl_module.keys())}")

    def _load_algorithm_module(self, checkpoint_path, model_config_path):
        from ray import tune
        from ray.rllib.algorithms.ppo import PPO

        def env_creator(config):
            simulator = Simulator(tick_secs=0.1)
            default_config = {
                "simulator": simulator,
                "num_agents_per_team": config.get("num_agents_per_team", 2),
                "map_size": config.get("map_size", 300),
                "max_steps": config.get("max_steps", 128),
                "debug": config.get("debug", True),
                "simulation_config": config.get("simulation_config", {}),
                "weapon_config": config.get("weapon_config", {}),
                "aircraft_types": {
                    "agent": _get_aircraft_class(config.get("agent_aircraft", "DebugPlane")),
                    "opponent": _get_aircraft_class(config.get("opponent_aircraft", "DebugPlane")),
                },
            }
            return BVRMultiAgentEnv(default_config)

        tune.register_env("BVRMultiAgentEnv", env_creator)

        override_config = {}
        if model_config_path:
            with open(model_config_path, encoding="utf-8") as f:
                override_config = yaml.safe_load(f) or {}

        algo = PPO.from_checkpoint(checkpoint_path, override_config=override_config)
        if hasattr(algo, "env_runner_group"):
            return algo.env_runner_group.local_env_runner.module
        if hasattr(algo, "workers"):
            return algo.workers.local_worker().module
        raise RuntimeError("Could not find RLModule in the loaded checkpoint")

    def compute_single_action(self, observation, agent_id=None):
        """Compute action using the trained model."""
        available_policies = list(self.rl_module.keys())
        policy_id = resolve_policy_id(agent_id, available_policies, self.train_config)

        from ray.rllib.core.columns import Columns

        policy_module = self.rl_module[policy_id]
        if hasattr(policy_module, "eval"):
            policy_module.eval()

        batch = {Columns.OBS: _tensor_tree(observation, module=policy_module, add_batch_dim=True)}
        state_key = (policy_id, agent_id)
        if state_key in self._states:
            batch[Columns.STATE_IN] = _tensor_tree(
                self._states[state_key], module=policy_module, add_batch_dim=False
            )

        output = policy_module.forward_inference(batch)
        if Columns.STATE_OUT in output:
            self._states[state_key] = output[Columns.STATE_OUT]

        actions = deterministic_actions(output, Columns)

        # Convert from torch tensor to numpy if needed
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        elif hasattr(actions, "numpy"):
            actions = actions.numpy()

        # Match the normalized training-time action contract.
        actions = np.clip(actions, 0.0, 1.0)

        return actions

    def cleanup(self):
        """Clean up resources."""
        pass


def infer_train_config_for_checkpoint(checkpoint_path: str | Path | None) -> str | None:
    """Find the train config saved next to a checkpoint, if present."""
    if not checkpoint_path:
        return None

    checkpoint = Path(checkpoint_path)
    search_dirs: list[Path] = []
    if checkpoint.parent != checkpoint:
        search_dirs.append(checkpoint.parent)
    if checkpoint.parent.parent != checkpoint.parent:
        search_dirs.append(checkpoint.parent.parent)
    search_dirs.extend(checkpoint.parents)

    seen: set[Path] = set()
    for directory in search_dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            resolved = directory
        if resolved in seen:
            continue
        seen.add(resolved)
        for name in ("train_config.yaml", "autotune_train_config.yaml"):
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    return None


def _module_device(module) -> str:
    try:
        return str(next(module.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return "cpu"


def _tensor_tree(value, *, module, add_batch_dim: bool):
    import torch

    if isinstance(value, dict):
        return {
            key: _tensor_tree(item, module=module, add_batch_dim=add_batch_dim)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _tensor_tree(item, module=module, add_batch_dim=add_batch_dim) for item in value
        )

    tensor = torch.as_tensor(value, device=_module_device(module))
    tensor = tensor.float() if tensor.is_floating_point() else tensor
    return tensor.unsqueeze(0) if add_batch_dim else tensor


def _line_objective_enabled(train_config: dict, env_config: dict) -> bool:
    categories = env_config.get("reward_categories", {})
    scenario_cfg = env_config.get("scenario_config", {})
    line_cfg = scenario_cfg.get("line_objective", {})
    return bool(categories.get("line_objective", False)) and bool(line_cfg.get("enabled", False))


def generate_unique_filename(base_path=None):
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_path is None:
        base_path = tacview_output_root() / "scenario.acmi"
    path = Path(base_path)
    # Extract directory and filename
    directory = path.parent
    stem = path.stem if path.stem != "scenario" else "scenario"
    extension = path.suffix

    # Create unique filename
    unique_path = directory / f"{stem}_{timestamp}{extension}"

    # Ensure directory exists
    unique_path.parent.mkdir(parents=True, exist_ok=True)

    return str(unique_path)


def run_tacview_scenario(
    frames=200,
    checkpoint_path=None,
    model_config_path=None,
    acmi_path=None,
    train_config_path=None,
    progress_interval=50,
    seed=None,
    model=None,
):
    """
    Run Tacview scenario generation with optional trained model.

    Args:
        frames: Number of simulation steps to run
        checkpoint_path: Path to trained model checkpoint (optional)
        model_config_path: Path to model config file (optional)
        acmi_path: Path to save ACMI file (if None, auto-generates unique name)
        train_config_path: Path to training config for environment settings (optional)
        progress_interval: Print progress every N steps (optional)
        seed: Random seed for environment reset (optional)
        model: Pre-loaded model to reuse (optional, avoids reloading for multiple scenarios)
    """
    # Generate unique filename if acmi_path is None
    if acmi_path is None:
        acmi_path = generate_unique_filename()
        print(f"Auto-generated output path: {acmi_path}")

    # Load training config if provided, or infer it from checkpoint artifacts.
    train_config = {}
    if train_config_path is None and checkpoint_path:
        train_config_path = infer_train_config_for_checkpoint(checkpoint_path)
        if train_config_path:
            print(f"Auto-detected train config from checkpoint: {train_config_path}")

    if train_config_path:
        train_config = load_train_config(train_config_path) or {}

    env_config = train_config.get("env", {})
    env_config = dict(env_config) if env_config else {}

    # Initialize simulator and environment
    sim_config = env_config.get("simulation_config", {})
    tick_secs = float(sim_config.get("tick_secs", 1.0))
    sim = Simulator(tick_secs=tick_secs, weapon_config=env_config.get("weapon_config", {}))
    _map_size_km = float(env_config.get("map_size", 300))
    _half = (_map_size_km / 2.0) / 111.0
    _max_alt = float(env_config.get("max_alt", 15000))
    map_limits = MapLimits(
        left_lon=-_half,
        bottom_lat=-_half,
        right_lon=_half,
        top_lat=_half,
        min_alt=0.0,
        max_alt=_max_alt,
    )

    # Configure environment
    default_env_config = dict(env_config)
    default_env_config.update(
        {
            "simulator": sim,
            "map_limits": map_limits,
            "num_agents_per_team": env_config.get(
                "num_agents_per_team", env_config.get("num_agents_per_side", 2)
            ),
            "map_size": env_config.get("map_size", 300),
            "max_steps": env_config.get("max_steps", frames),
            "debug": env_config.get("debug", True),
            "simulation_config": env_config.get("simulation_config", {}),
            "weapon_config": env_config.get("weapon_config", {}),
        }
    )
    if "aircraft_types" not in default_env_config:
        default_env_config.update(resolve_aircraft_config(default_env_config))
    default_env_config.setdefault(
        "aircraft_types",
        {
            "agent": _get_aircraft_class("Eurofighter"),
            "opponent": _get_aircraft_class("Eurofighter"),
        },
    )

    # Initialize environment
    if train_config.get("training_mode") == "simplified" or (
        checkpoint_path and "Simplified" in str(checkpoint_path)
    ):
        print("Using SimplifiedMultiAgentEnv")
        env = SimplifiedMultiAgentEnv(default_env_config)
    elif (
        _line_objective_enabled(train_config, default_env_config)
        and (line_env_cls := resolve_env_class("line_objective")) is not None
    ):
        print(f"Using {line_env_cls.__name__}")
        env = line_env_cls(default_env_config)
    else:
        print("Using BVRMultiAgentEnv")
        env = BVRMultiAgentEnv(default_env_config)

    # Initialize Tacview logger
    tacview_logger = TacviewLogger(acmi_path)

    # Register the Tacview logger with the environment path that writes each step.
    env.tacview_logger = tacview_logger
    sim.tacview_logger = tacview_logger

    # Load or initialize model
    model_to_return = model
    if model is None:
        if checkpoint_path:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = TrainedModelWrapper(
                checkpoint_path,
                env,
                model_config_path,
                train_config=train_config,
            )
            print("Trained model loaded successfully!")
            model_to_return = model
        else:
            print("No checkpoint provided - using random actions")
            model = DefaultModel(env, seed=seed)
    else:
        print("Reusing pre-loaded model from previous scenario...")
        model_to_return = model

    # Run simulation
    obs, _ = env.reset(seed=seed)
    if seed is not None:
        print(f"Environment reset with seed: {seed}")

    print("\nActive units after reset:")
    for uid, unit in sim.active_units.items():
        print(f"  - Unit ID {uid}: {getattr(unit, 'name', None)}")

    print(f"\nRunning simulation for {frames} steps...")
    dones = {"__all__": False}
    steps = 0

    while steps < frames and not dones.get("__all__", False):
        actions = {}
        for aid in env.all_agent_ids:
            # Only get actions for agents that are still active (have observations)
            if aid in obs:
                obs_agent = obs[aid]
                actions[aid] = model.compute_single_action(obs_agent, aid)

        obs, rewards, dones, trunc, infos = env.step(actions)
        steps += 1

        if steps % progress_interval == 0:
            print(f"  Step {steps}/{frames}")

    print(f"\nSimulation completed after {steps} steps")

    if acmi_path:
        print(f"Tacview file saved to: {acmi_path}")

    # Return the model so it can be reused across scenarios
    return model_to_return


def load_tacview_config(config_path):
    """Load Tacview configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for air2air-tacview command."""
    parser = argparse.ArgumentParser(
        description="Tacview scenario generation for BVR simulation with optional trained model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single scenario with specific seed
  air2air-tacview --seed 42

  # Generate 10 scenarios with different seeds (0-9)
  air2air-tacview --num-scenarios 10

  # Generate scenario with trained model
  air2air-tacview --frames 500 --checkpoint path/to/checkpoint --train-config config.yaml

  # Use configuration file
  air2air-tacview --config configs/tacview/default.yaml
        """,
    )
    parser.add_argument("--config", type=str, help="Path to Tacview configuration YAML file")
    parser.add_argument("--frames", type=int, default=500, help="Number of simulation steps to run")
    parser.add_argument(
        "--acmi", type=str, help="ACMI output file path (auto-generates if not provided)"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--model-config", type=str, help="Path to model config YAML file")
    parser.add_argument("--train-config", type=str, help="Path to training config YAML file")
    parser.add_argument("--seed", type=int, help="Random seed for environment reset")
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=1,
        help="Number of scenarios to generate with different seeds",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed value when generating multiple scenarios",
    )

    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config and Path(args.config).exists():
        config = load_tacview_config(args.config)

        # Extract values from config with command-line overrides
        frames = (
            args.frames
            if hasattr(args, "frames")
            else config.get("simulation", {}).get("frames", 500)
        )
        acmi_path = args.acmi or config.get("simulation", {}).get("acmi_output_path")
        checkpoint_path = args.checkpoint or config.get("model", {}).get("checkpoint_path")
        model_config_path = args.model_config or config.get("model", {}).get("model_config_path")
        train_config_path = args.train_config or config.get("model", {}).get("train_config_path")
        progress_interval = config.get("output", {}).get("progress_interval", 50)
    else:
        # Use command-line arguments
        frames = args.frames
        acmi_path = args.acmi
        checkpoint_path = args.checkpoint
        model_config_path = args.model_config
        train_config_path = args.train_config
        progress_interval = 50

    # Determine number of scenarios and seeds
    num_scenarios = args.num_scenarios
    seed_start = args.seed_start

    # If --seed is provided, use it (overrides num_scenarios to 1)
    if args.seed is not None:
        seeds = [args.seed]
        num_scenarios = 1
    elif num_scenarios > 1:
        # Generate multiple scenarios with sequential seeds
        seeds = list(range(seed_start, seed_start + num_scenarios))
    else:
        # Single scenario with no specific seed
        seeds = [None]

    print("=" * 60)
    print("Tacview Scenario Generation")
    print("=" * 60)
    print(f"Scenarios to generate: {num_scenarios}")
    if num_scenarios > 1:
        print(f"Seeds: {seeds[0]} to {seeds[-1]}")
    elif seeds[0] is not None:
        print(f"Seed: {seeds[0]}")
    print(f"Frames per scenario: {frames}")
    print(f"Output: {acmi_path if acmi_path else 'Auto-generated (timestamp-based)'}")
    print(f"Checkpoint: {checkpoint_path if checkpoint_path else 'Random actions'}")
    print(f"Train config: {train_config_path if train_config_path else 'Default settings'}")
    print("=" * 60 + "\n")

    # Generate scenarios
    loaded_model = None
    for i, seed in enumerate(seeds, 1):
        if num_scenarios > 1:
            print(f"\n{'=' * 60}")
            print(f"Generating scenario {i}/{num_scenarios} (seed={seed})")
            print(f"{'=' * 60}\n")

            # Generate unique output path for each scenario
            if acmi_path is None:
                # Auto-generate with seed in filename
                scenario_acmi = generate_unique_filename(
                    tacview_output_root() / f"scenario_seed{seed}.acmi"
                )
            else:
                # User-specified path - add seed suffix
                path = Path(acmi_path)
                scenario_acmi = str(path.parent / f"{path.stem}_seed{seed}{path.suffix}")
        else:
            scenario_acmi = acmi_path

        # Reuse model across scenarios
        returned_model = run_tacview_scenario(
            frames=frames,
            checkpoint_path=checkpoint_path,
            model_config_path=model_config_path,
            acmi_path=scenario_acmi,
            train_config_path=train_config_path,
            progress_interval=progress_interval,
            seed=seed,
            model=loaded_model,
        )

        # Save the model for reuse in subsequent scenarios
        if returned_model is not None:
            loaded_model = returned_model

    # Clean up Ray if it was initialized
    if checkpoint_path:
        try:
            import ray

            if ray.is_initialized():
                print("\nShutting down Ray...")
                ray.shutdown()
                print("Ray shutdown complete.")
        except Exception as e:
            print(f"Warning: Error during Ray shutdown: {e}")

    if num_scenarios > 1:
        print(f"\n{'=' * 60}")
        print(f"Successfully generated {num_scenarios} scenarios!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
