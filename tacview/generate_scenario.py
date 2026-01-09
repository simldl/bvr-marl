"""
Tacview scenario generation for BVR air combat simulation.

This script automatically uses the correct config files and relative paths.
Just run: python tacview/generate_scenario.py

The script will:
1. Automatically find and use tacview/tacview_config.yaml
2. Use relative paths for all file loading
3. Work from any location as long as you run it from the project root

Optional arguments:
  --checkpoint PATH    Override checkpoint path from config
  --frames N          Override number of frames to simulate
  --num-scenarios N   Generate N scenarios with different seeds
"""
import sys
from pathlib import Path

# Setup project root path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from datetime import datetime
import argparse
import yaml

from simulator.simulator import Simulator
from simulator.utils.map_limits import MapLimits
from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from aircrafts.types.eurofighter import Eurofighter
from tacview.logger import TacviewLogger


class DefaultModel:
    """Default model that returns random actions."""
    def __init__(self, env):
        self.env = env

    def compute_single_action(self, observation, agent_id):
        return np.random.rand(10).astype(np.float32)


class TrainedModelWrapper:
    """Wrapper for trained RLlib models."""

    def __init__(self, checkpoint_path, env, model_config_path=None):
        import ray
        from ray import tune
        from ray.rllib.algorithms.ppo import PPO
        import logging

        self.env = env
        self.ray_initialized_here = False

        # Suppress Ray core worker warnings
        logging.getLogger("ray.rllib.utils.deprecation").setLevel(logging.ERROR)

        # Only initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
            self.ray_initialized_here = True

        def env_creator(config):
            from simulator.simulator import Simulator
            from aircrafts.types.eurofighter import Eurofighter

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
                    "agent": config.get("agent_aircraft", Eurofighter),
                    "opponent": config.get("opponent_aircraft", Eurofighter)
                }
            }
            return BVRMultiAgentEnv(default_config)

        tune.register_env("BVRMultiAgentEnv", env_creator)

        override_config = {}
        if model_config_path:
            with open(model_config_path, "r") as f:
                override_config = yaml.safe_load(f)

        self.algo = PPO.from_checkpoint(checkpoint_path, override_config=override_config)

        # Get the multi-agent RLModule for inference
        if hasattr(self.algo, 'env_runner_group'):
            self.rl_module = self.algo.env_runner_group.local_env_runner.module
        elif hasattr(self.algo, 'workers'):
            self.rl_module = self.algo.workers.local_worker().module
        else:
            raise RuntimeError("Could not find RLModule in the loaded checkpoint")

        print(f"Loaded RLModule: {type(self.rl_module)}")
        if hasattr(self.rl_module, 'keys'):
            print(f"Available policies: {list(self.rl_module.keys())}")

    def compute_single_action(self, observation, agent_id=None):
        """Compute action using the trained model."""
        import torch
        from ray.rllib.core.columns import Columns

        available_policies = list(self.rl_module.keys())

        if "shared_policy" in available_policies:
            policy_id = "shared_policy"
        else:
            if agent_id and str(agent_id).startswith("A"):
                policy_id = "agent_policy"
            else:
                policy_id = "opponent_policy"

        # Convert observation to batch format
        if isinstance(observation, dict):
            obs_batch = {
                key: np.expand_dims(value, axis=0)
                for key, value in observation.items()
            }
        else:
            obs_batch = np.expand_dims(observation, axis=0)

        # Get the specific policy module
        policy_module = self.rl_module[policy_id]

        if hasattr(policy_module, 'eval'):
            policy_module.eval()

        # Convert to torch tensor
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

        # Run inference
        batch = {Columns.OBS: obs_batch}
        output = policy_module.forward_inference(batch)

        # Extract action
        if Columns.ACTIONS in output:
            actions = output[Columns.ACTIONS]
            if actions.dim() == 3:
                actions = actions[0, 0]
            else:
                actions = actions[0]
        elif Columns.ACTION_DIST_INPUTS in output:
            action_dist_inputs = output[Columns.ACTION_DIST_INPUTS]
            if action_dist_inputs.dim() == 3:
                action_dist_inputs = action_dist_inputs[0, 0]
            else:
                action_dist_inputs = action_dist_inputs[0]
            action_dim = action_dist_inputs.shape[0] // 2
            actions = action_dist_inputs[:action_dim]
        else:
            raise ValueError(f"Unexpected output format. Keys: {list(output.keys())}")

        # Convert to numpy
        if hasattr(actions, 'cpu'):
            actions = actions.cpu().numpy()
        elif hasattr(actions, 'numpy'):
            actions = actions.numpy()

        # Clip to [0,1]
        actions = np.clip(actions, 0.0, 1.0)

        return actions


def generate_unique_filename(base_path="tacview/logs/scenario.acmi"):
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_path)
    directory = path.parent
    stem = path.stem if path.stem != "scenario" else "scenario"
    extension = path.suffix

    unique_path = directory / f"{stem}_{timestamp}{extension}"
    unique_path.parent.mkdir(parents=True, exist_ok=True)

    return str(unique_path)


def run_scenario(
    frames=1000,
    checkpoint_path=None,
    model_config_path=None,
    acmi_path=None,
    train_config_path=None,
    progress_interval=50,
    seed=None,
    model=None
):
    """
    Run Tacview scenario generation.

    Args:
        frames: Number of simulation steps
        checkpoint_path: Path to trained model checkpoint (optional)
        model_config_path: Path to model config (optional)
        acmi_path: Path to save ACMI file (if None, auto-generates)
        train_config_path: Path to training config (optional)
        progress_interval: Print progress every N steps
        seed: Random seed for environment reset (optional)
        model: Pre-loaded model to reuse (optional)

    Returns:
        Loaded model (for reuse across multiple scenarios)
    """
    # Generate unique filename if needed
    if acmi_path is None:
        acmi_path = generate_unique_filename("tacview/logs/scenario.acmi")
        print(f"Auto-generated output path: {acmi_path}")

    # Load training config if provided
    train_config = {}
    if train_config_path:
        with open(train_config_path, 'r', encoding='utf-8') as f:
            train_config = yaml.safe_load(f)

    env_config = train_config.get('env', {})

    # Get configuration from train_config or use defaults
    sim = Simulator()
    map_size = env_config.get('map_size', 300)
    max_alt = env_config.get('max_alt', 25000)
    num_agents_per_side = env_config.get('num_agents_per_side', 2)

    map_limits = MapLimits(
        left_lon=-map_size/222.0,
        right_lon=map_size/222.0,
        bottom_lat=-map_size/222.0,
        top_lat=map_size/222.0,
        min_alt=0,
        max_alt=max_alt
    )

    config = {
        "simulator": sim,
        "num_agents_per_team": num_agents_per_side,
        "map_size": map_size,
        "max_steps": frames,
        "debug": env_config.get('debug', True),
        "aircraft_types": {
            "agent": Eurofighter,
            "opponent": Eurofighter,
        },
        "num_fm": env_config.get('num_fm', 4),
        "num_ff": env_config.get('num_ff', 2),
        "num_em": env_config.get('num_em', 4),
        "num_enemy": env_config.get('num_enemy', 2),
        "pr_slots": env_config.get('pr_slots', 4),
        "warn_sectors": env_config.get('warn_sectors', 4),
        "tacview_logfile": acmi_path,
        "simulation_config": env_config.get('simulation_config', {}),
    }

    env = BVRMultiAgentEnv(config)

    # Load or reuse model
    model_to_return = None
    if model is None:
        if checkpoint_path:
            checkpoint_path_abs = str(Path(checkpoint_path).resolve())
            print(f"Loading model from: {checkpoint_path}")
            model = TrainedModelWrapper(checkpoint_path_abs, env, model_config_path)
            print("Model loaded successfully!")
            model_to_return = model
        else:
            print("No checkpoint provided - using random actions")
            model = DefaultModel(env)
    else:
        print("Reusing pre-loaded model...")
        model_to_return = model

    # Run simulation
    obs, _ = env.reset(seed=seed)
    if seed is not None:
        print(f"Environment reset with seed: {seed}")

    print(f"\nRunning simulation for {frames} steps...")
    dones = {"__all__": False}
    steps = 0
    while steps < frames and not dones.get("__all__", False):
        actions = {}
        for aid in env.all_agent_ids:
            if aid in obs:
                obs_agent = obs[aid]
                actions[aid] = model.compute_single_action(obs_agent, aid)
        obs, rewards, dones, trunc, infos = env.step(actions)
        steps += 1

        if steps % progress_interval == 0:
            print(f"  Step {steps}/{frames}")

    print(f"\nSimulation completed after {steps} steps")
    print(f"Tacview file saved to: {acmi_path}")

    return model_to_return


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tacview Scenario Generation for BVR Air Combat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config (random actions)
  python tacview/generate_scenario.py

  # Run with specific checkpoint
  python tacview/generate_scenario.py --checkpoint models/my_model/checkpoint_000010

  # Generate 10 scenarios with different seeds
  python tacview/generate_scenario.py --num-scenarios 10
        """
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained model checkpoint (relative to project root)")
    parser.add_argument("--frames", type=int, default=None,
                       help="Number of simulation steps")
    parser.add_argument("--num-scenarios", type=int, default=1,
                       help="Number of scenarios to generate with different seeds")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for environment reset")
    parser.add_argument("--seed-start", type=int, default=0,
                       help="Starting seed value when generating multiple scenarios")

    args = parser.parse_args()

    print("=" * 80)
    print("BVR AIR COMBAT TACVIEW GENERATION")
    print("=" * 80)
    print(f"Project root: {project_root}")

    # Load tacview config
    tacview_config_path = project_root / "tacview" / "tacview_config.yaml"
    train_config_path = project_root / "reinforcement_learning" / "configs" / "train_config.yaml"

    with open(tacview_config_path, 'r', encoding='utf-8') as f:
        tacview_config = yaml.safe_load(f)

    print(f"Config file: {tacview_config_path}")

    # Get settings from config, allow command-line override
    frames = args.frames if args.frames is not None else tacview_config.get('simulation', {}).get('frames', 1000)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        config_checkpoint = tacview_config.get('model', {}).get('checkpoint_path')
        if config_checkpoint:
            checkpoint_path = project_root / config_checkpoint

    print(f"Training config: {train_config_path}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint - using random actions")
    print(f"Frames per scenario: {frames}")

    # Determine scenarios and seeds
    num_scenarios = args.num_scenarios
    seed_start = args.seed_start

    if args.seed is not None:
        seeds = [args.seed]
        num_scenarios = 1
    elif num_scenarios > 1:
        seeds = list(range(seed_start, seed_start + num_scenarios))
    else:
        seeds = [None]

    print(f"Scenarios to generate: {num_scenarios}")
    if num_scenarios > 1:
        print(f"Seeds: {seeds[0]} to {seeds[-1]}")
    elif seeds[0] is not None:
        print(f"Seed: {seeds[0]}")
    print("=" * 80)

    # Generate scenarios
    loaded_model = None
    for i, seed in enumerate(seeds, 1):
        if num_scenarios > 1:
            print(f"\n{'=' * 80}")
            print(f"Generating scenario {i}/{num_scenarios} (seed={seed})")
            print(f"{'=' * 80}\n")

        returned_model = run_scenario(
            frames=frames,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            acmi_path=None,  # Auto-generate unique name
            train_config_path=str(train_config_path),
            progress_interval=50,
            seed=seed,
            model=loaded_model
        )

        if returned_model is not None:
            loaded_model = returned_model

    # Clean up Ray if needed
    if checkpoint_path:
        try:
            import ray
            if ray.is_initialized():
                print("\nShutting down Ray...")
                ray.shutdown()
        except Exception as e:
            print(f"Warning: Error during Ray shutdown: {e}")

    if num_scenarios > 1:
        print(f"\n{'=' * 80}")
        print(f"Successfully generated {num_scenarios} scenarios!")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
