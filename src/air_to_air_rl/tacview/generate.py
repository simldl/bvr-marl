"""
Tacview scenario generation for BVR simulation with optional trained model support.

"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.simulator.utils.map_limits import MapLimits
from air_to_air_rl.tacview.logger import TacviewLogger


class DefaultModel:
    """Default model that returns random actions."""

    def __init__(self, env):
        self.env = env

    def compute_single_action(self, observation, agent_id):
        return np.random.rand(10).astype(np.float32)


class TrainedModelWrapper:
    """Wrapper for trained RLlib models."""

    def __init__(self, checkpoint_path, env, model_config_path=None):
        import logging

        import ray
        from ray import tune
        from ray.rllib.algorithms.ppo import PPO

        from air_to_air_rl.aircrafts.types.debug_plane import DebugPlane
        from air_to_air_rl.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
        from air_to_air_rl.simulator.simulator import Simulator

        self.env = env
        self.ray_initialized_here = False

        # Suppress Ray core worker warnings
        logging.getLogger("ray.rllib.utils.deprecation").setLevel(logging.ERROR)

        # Only initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
            self.ray_initialized_here = True

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
                    "agent": config.get("agent_aircraft", DebugPlane),
                    "opponent": config.get("opponent_aircraft", DebugPlane),
                },
            }
            return BVRMultiAgentEnv(default_config)

        tune.register_env("BVRMultiAgentEnv", env_creator)

        override_config = {}
        if model_config_path:
            with open(model_config_path) as f:
                override_config = yaml.safe_load(f)

        self.algo = PPO.from_checkpoint(checkpoint_path, override_config=override_config)

        # Get the multi-agent policy module for inference
        if hasattr(self.algo, "env_runner_group"):
            # New API stack
            self.rl_module = self.algo.env_runner_group.local_env_runner.module
        elif hasattr(self.algo, "workers"):
            # Old API stack fallback
            self.rl_module = self.algo.workers.local_worker().module
        else:
            raise RuntimeError("Could not find policy module in the loaded checkpoint")

        print(f"Loaded policy module: {type(self.rl_module)}")
        if hasattr(self.rl_module, "keys"):
            print(f"Available policies: {list(self.rl_module.keys())}")

    def compute_single_action(self, observation, agent_id=None):
        """Compute action using the trained model."""
        # Determine which policy to use
        available_policies = list(self.rl_module.keys())

        if "shared_policy" in available_policies:
            # Shared policy mode - all agents use the same policy
            policy_id = "shared_policy"
        else:
            # Separate policies mode (legacy)
            if agent_id and str(agent_id).startswith("A"):
                policy_id = "blue_policy"
            else:
                policy_id = "red_policy"

        # Prepare input batch

        # Get the RL module for the policy
        from ray.rllib.core.columns import Columns

        batch = {
            Columns.OBS: {agent_id: np.expand_dims(observation, axis=0)},
            Columns.STATE_IN: {},
        }

        # Handle stateful models
        if hasattr(self.rl_module[policy_id], "get_initial_state"):
            initial_state = self.rl_module[policy_id].get_initial_state()
            if initial_state:
                batch[Columns.STATE_IN] = {
                    agent_id: {k: np.expand_dims(v, axis=0) for k, v in initial_state.items()}
                }

        # Forward pass
        output = self.rl_module[policy_id].forward_inference(batch)

        # Extract actions from output
        if Columns.ACTIONS in output:
            # Standard action output
            actions = output[Columns.ACTIONS]
            # Handle both (batch, action_dim) and (batch, time, action_dim) shapes
            if actions.dim() == 3:
                actions = actions[0, 0]  # Remove batch and time dimensions
            else:
                actions = actions[0]  # Remove batch dimension only
        elif "actions" in output:
            # Fallback: lowercase actions
            actions = output["actions"]
            if actions.dim() == 3:
                actions = actions[0, 0]
            else:
                actions = actions[0]
        elif Columns.ACTION_DIST_INPUTS in output:
            # Distribution parameters - extract means
            action_dist_inputs = output[Columns.ACTION_DIST_INPUTS]
            # Handle both (batch, dist_params) and (batch, time, dist_params) shapes
            if action_dist_inputs.dim() == 3:
                action_dist_inputs = action_dist_inputs[0, 0]
            else:
                action_dist_inputs = action_dist_inputs[0]
            # Extract just the means (first half) for deterministic behavior
            action_dim = action_dist_inputs.shape[0] // 2
            actions = action_dist_inputs[:action_dim]
        elif "action_dist_inputs" in output:
            # Fallback: lowercase
            action_dist_inputs = output["action_dist_inputs"]
            if action_dist_inputs.dim() == 3:
                action_dist_inputs = action_dist_inputs[0, 0]
            else:
                action_dist_inputs = action_dist_inputs[0]
            action_dim = action_dist_inputs.shape[0] // 2
            actions = action_dist_inputs[:action_dim]
        else:
            raise ValueError(
                f"Unexpected output format from forward_inference. Keys: {list(output.keys())}"
            )

        # Convert from torch tensor to numpy if needed
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        elif hasattr(actions, "numpy"):
            actions = actions.numpy()

        # CRITICAL: Clip actions to [0,1] to match training behavior
        actions = np.clip(actions, 0.0, 1.0)

        return actions

    def cleanup(self):
        """Clean up resources."""
        pass


def generate_unique_filename(base_path="tacview/logs/scenario.acmi"):
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        acmi_path = generate_unique_filename("tacview/logs/scenario.acmi")
        print(f"Auto-generated output path: {acmi_path}")

    # Load training config if provided
    train_config = {}
    if train_config_path:
        with open(train_config_path, encoding="utf-8") as f:
            train_config = yaml.safe_load(f)

    env_config = train_config.get("env", {})

    # Initialize simulator and environment
    sim = Simulator(tick_secs=0.1)
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
    default_env_config = {
        "simulator": sim,
        "map_limits": map_limits,
        "num_agents_per_team": env_config.get("num_agents_per_team", 2),
        "map_size": env_config.get("map_size", 300),
        "max_steps": env_config.get("max_steps", frames),
        "debug": env_config.get("debug", True),
        "simulation_config": env_config.get("simulation_config", {}),
        "weapon_config": env_config.get("weapon_config", {}),
        "aircraft_types": {"agent": Eurofighter, "opponent": Eurofighter},
    }

    # Initialize environment
    env = BVRMultiAgentEnv(default_env_config)

    # Initialize Tacview logger
    tacview_logger = TacviewLogger(acmi_path)

    # Register the Tacview logger with the simulator
    sim.tacview_logger = tacview_logger

    # Load or initialize model
    model_to_return = model
    if model is None:
        if checkpoint_path:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = TrainedModelWrapper(checkpoint_path, env, model_config_path)
            print("Trained model loaded successfully!")
            model_to_return = model
        else:
            print("No checkpoint provided - using random actions")
            model = DefaultModel(env)
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
                scenario_acmi = generate_unique_filename(f"tacview/logs/scenario_seed{seed}.acmi")
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
