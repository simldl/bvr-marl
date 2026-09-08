"""
Adaptive training configuration system.

Automatically adjusts training parameters based on system resources
and handles training failures by scaling down configuration.
"""

import platform
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from bvr_marl_core.rl.environment.spaces.action_space import FULL_ACTION_DIM


class SystemResourceChecker:
    """Check system resources and capabilities."""

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "percent_used": psutil.cpu_percent(interval=1),
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
                "percent_used": psutil.virtual_memory().percent,
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "devices": [],
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
            },
        }

        # Get GPU details if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu"]["devices"].append(
                    {
                        "id": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 1),
                        "major": props.major,
                        "minor": props.minor,
                    }
                )

        return info

    @staticmethod
    def assess_training_capability() -> tuple[str, dict[str, Any]]:
        """Assess system capability for training and recommend configuration tier.

        Returns:
            (tier, recommendations) where tier is one of "minimal", "low",
            "medium", "high" and recommendations is a dict of suggested
            training parameters for that tier.
        """
        info = SystemResourceChecker.get_system_info()

        recommendations: dict[str, dict[str, Any]] = {
            "minimal": {
                "description": "Basic training - single CPU worker, small batch",
                "num_env_runners": 1,
                "train_batch_size": 2000,
                "sgd_minibatch_size": 128,
                "num_sgd_iter": 10,
                "framework": "torch",
                "environment": "SimplifiedMultiAgentEnv",
            },
            "low": {
                "description": "Light training - few workers, modest batch",
                "num_env_runners": 2,
                "train_batch_size": 4000,
                "sgd_minibatch_size": 256,
                "num_sgd_iter": 15,
                "framework": "torch",
                "environment": "SimplifiedMultiAgentEnv",
            },
            "medium": {
                "description": "Standard training - balanced configuration",
                "num_env_runners": 4,
                "train_batch_size": 8000,
                "sgd_minibatch_size": 512,
                "num_sgd_iter": 20,
                "framework": "torch",
                "environment": "BVRMultiAgentEnv",
            },
            "high": {
                "description": "Performance training - full environment, GPU acceleration",
                "num_env_runners": 6,
                "train_batch_size": 12000,
                "sgd_minibatch_size": 1024,
                "num_sgd_iter": 25,
                "framework": "torch",
                "environment": "BVRMultiAgentEnv",
            },
        }

        # Determine tier based on system specs
        memory_gb = info["memory"]["total_gb"]
        cpu_threads = info["cpu"]["threads"]
        has_gpu = info["gpu"]["available"]
        gpu_memory = max([gpu["total_memory_gb"] for gpu in info["gpu"]["devices"]], default=0)

        # Account for Ray overhead: need num_env_runners + learner + driver + overhead (~2-3x workers)
        # Conservative estimate: need at least 3x the number of workers in total CPU threads
        if memory_gb >= 32 and cpu_threads >= 24 and has_gpu and gpu_memory >= 8:
            tier = "high"
        elif memory_gb >= 16 and cpu_threads >= 12 and (has_gpu or cpu_threads >= 16):
            tier = "medium"
        elif memory_gb >= 8 and cpu_threads >= 6:
            tier = "low"
        else:
            tier = "minimal"

        return tier, recommendations[tier]


_TIER_CONFIGS: dict[str, dict[str, Any]] = {
    "minimal": {
        "description": "Basic training - single CPU worker, small batch",
        "num_env_runners": 1,
        "train_batch_size": 2000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "framework": "torch",
        "environment": "SimplifiedMultiAgentEnv",
    },
    "low": {
        "description": "Light training - few workers, modest batch",
        "num_env_runners": 2,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 15,
        "framework": "torch",
        "environment": "SimplifiedMultiAgentEnv",
    },
    "medium": {
        "description": "Standard training - balanced configuration",
        "num_env_runners": 4,
        "train_batch_size": 8000,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 20,
        "framework": "torch",
        "environment": "BVRMultiAgentEnv",
    },
    "high": {
        "description": "Performance training - full environment, GPU acceleration",
        "num_env_runners": 6,
        "train_batch_size": 12000,
        "sgd_minibatch_size": 1024,
        "num_sgd_iter": 25,
        "framework": "torch",
        "environment": "BVRMultiAgentEnv",
    },
}


class AdaptiveTrainingConfig:
    """Adaptive training configuration that scales based on system capabilities."""

    def __init__(self, base_config_path: str):
        """Initialize with a base configuration file."""
        self.base_config_path = Path(base_config_path)
        with open(self.base_config_path) as f:
            self.base_config = yaml.safe_load(f)

        self.system_info = SystemResourceChecker.get_system_info()
        self.current_tier: str | None = None
        self.fallback_attempts = 0
        self.max_fallback_attempts = 6  # matches the number of fallback strategies

    def get_adaptive_config(self, target_tier: str | None = None) -> dict[str, Any]:
        """Get training config adapted for the current system or specified tier.

        Args:
            target_tier: Force a specific tier (``"minimal"``, ``"low"``,
                ``"medium"``, ``"high"``).  When *None* the tier is
                auto-detected from the live system resources.

        Returns:
            A fully-populated configuration dict ready to be passed to the
            PPO config builder.
        """
        if target_tier is None:
            tier, recommendations = SystemResourceChecker.assess_training_capability()
        else:
            tier = target_tier
            recommendations = _TIER_CONFIGS.get(tier, _TIER_CONFIGS["minimal"])

        self.current_tier = tier

        # Create adapted config
        adapted_config = self.base_config.copy()

        # Apply tier-specific training parameters
        if "training" not in adapted_config:
            adapted_config["training"] = {}

        # Dynamically adjust num_env_runners based on available CPU threads and current load
        total_threads = self.system_info["cpu"]["threads"]
        cpu_usage = self.system_info["cpu"]["percent_used"]

        # Reserve: 1 for system + 1 for learner + 1 for driver + 1 buffer = 4 minimum
        # If system is already busy (>50% CPU), be more conservative
        if cpu_usage > 50:
            reserved_threads = 6  # Extra conservative when system is busy
            print(f"System CPU usage is {cpu_usage:.1f}% - using conservative allocation")
        else:
            reserved_threads = 4

        available_for_workers = max(0, total_threads - reserved_threads)

        # Each worker typically uses 1 thread, but be conservative
        max_workers = max(1, available_for_workers)
        actual_workers = min(recommendations["num_env_runners"], max_workers)

        print(
            f"CPU allocation: {total_threads} total threads, {reserved_threads} reserved, "
            f"{available_for_workers} available for workers"
        )
        print(f"Current CPU usage: {cpu_usage:.1f}%")
        print(
            f"Recommended workers: {recommendations['num_env_runners']}, "
            f"actual workers: {actual_workers}"
        )
        print(
            f"Expected total CPU usage after training starts: "
            f"~{actual_workers + reserved_threads} threads"
        )

        # Set comprehensive training parameters with defaults
        training_defaults: dict[str, Any] = {
            "steps": self.base_config.get("training", {}).get("steps", 30000),
            "learning_rate": 0.0003,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "vf_loss_coeff": 0.5,
            "num_epochs": 5,
            "grad_clip": 1.0,
            "use_gae": True,
            "lambda": 0.95,
            "rollout_fragment_length": 200,
            "batch_size": 512,
            "n_envs": 1,
            # Reward config
            "reward_config": {
                "kill_reward": 200.0,
                "destruction_penalty": -200.0,
                "enable_terminal": True,
                "enable_tactical": False,
                "enable_control": False,
                "enable_defensive": False,
                "enable_energy": False,
            },
        }

        # Apply training defaults
        for key, default_value in training_defaults.items():
            if key == "reward_config":
                if "reward_config" not in adapted_config["training"]:
                    adapted_config["training"]["reward_config"] = {}
                for rkey, rvalue in default_value.items():
                    adapted_config["training"]["reward_config"].setdefault(rkey, rvalue)
            else:
                adapted_config["training"].setdefault(key, default_value)

        # Override with adaptive parameters
        adapted_config["training"].update(
            {
                "train_batch_size_per_learner": recommendations["train_batch_size"],
                "sgd_minibatch_size": recommendations["sgd_minibatch_size"],
                "num_sgd_iter": recommendations["num_sgd_iter"],
            }
        )

        # Set num_env_runners at root level where PPO config expects it
        adapted_config["num_env_runners"] = actual_workers

        # API stack compatibility - use legacy API by default to avoid encoder configuration issues
        # This can be overridden if the custom model is available
        adapted_config["use_new_api_stack"] = adapted_config.get("use_new_api_stack", False)

        # Ensure env section exists
        if "env" not in adapted_config:
            adapted_config["env"] = {}

        # Set comprehensive environment defaults for both environments
        env_defaults: dict[str, Any] = {
            # Team setup
            "num_agents_per_team": 2,
            "map_size_km": 300,
            "max_steps": 1200,
            "tick_secs": 1.0,
            "debug": False,
            # Aircraft types
            "agent_aircraft_type": "Eurofighter",
            "opponent_aircraft_type": "Eurofighter",
            # Observation slots
            "num_ff": 1,  # friendly fighters (excluding self)
            "num_ef": 2,  # enemy fighters
            "num_warn": 4,  # missile warning sectors
            # Missile settings
            "missile_cooldown_s": 3.0,
            # Termination
            "early_term_all_enemies_dead": True,
            "early_term_mission_complete": True,
            "early_term_no_missiles": False,
            # Spawning
            "use_controlled_geometry": False,
            "regime": "hot_200km",
            # Logging
            "tacview_logfile": None,
        }

        # Apply defaults for missing keys
        for key, default_value in env_defaults.items():
            adapted_config["env"].setdefault(key, default_value)

        # Switch environment for lower tiers and reduce complexity
        if recommendations["environment"] == "SimplifiedMultiAgentEnv":
            adapted_config["env"]["environment_class"] = "SimplifiedMultiAgentEnv"
            # Reduce environment complexity for lower tiers
            adapted_config["env"]["num_agents_per_team"] = min(
                adapted_config["env"]["num_agents_per_team"], 2
            )
            adapted_config["env"]["max_steps"] = min(adapted_config["env"]["max_steps"], 800)
            adapted_config["env"]["map_size_km"] = min(adapted_config["env"]["map_size_km"], 200)

        print(
            f"Environment configuration: "
            f"{adapted_config['env'].get('environment_class', 'BVRMultiAgentEnv')}"
        )
        print(f"Agents per team: {adapted_config['env'].get('num_agents_per_team', 2)}")

        # Add model configuration defaults
        if "model" not in adapted_config:
            adapted_config["model"] = {}

        model_defaults = {
            "model_config": {
                "action_dim": 4,
                "wrapped_action_dim": 4,
                "full_action_dim": FULL_ACTION_DIM,
                "active_indices": [0, 1, 2, 3],
                "hidden_dim": 128,
                "num_hidden_layers": 2,
                "activation": "relu",
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
                "free_log_std": False,
            },
        }

        # Apply model defaults
        for key, default_value in model_defaults.items():
            if key == "model_config":
                if "model_config" not in adapted_config["model"]:
                    adapted_config["model"]["model_config"] = {}
                for mkey, mvalue in default_value.items():
                    adapted_config["model"]["model_config"].setdefault(mkey, mvalue)
            else:
                adapted_config["model"].setdefault(key, default_value)

        # Add logging defaults
        if "logging" not in adapted_config:
            adapted_config["logging"] = {}
        logging_defaults = {
            "log_dir": "logs",
            "save_dir": "models",
            "model_name": "adaptive_model",
            "use_tensorboard": True,
        }
        for key, default_value in logging_defaults.items():
            adapted_config["logging"].setdefault(key, default_value)

        # GPU-specific adjustments
        if not self.system_info["gpu"]["available"]:
            adapted_config["num_gpus"] = 0
            adapted_config["framework"] = "torch"
        else:
            # Set GPU usage at root level where PPO config expects it
            adapted_config["num_gpus"] = min(1, self.system_info["gpu"]["count"])
            adapted_config["framework"] = "torch"

        return adapted_config

    def get_fallback_config(self) -> dict[str, Any] | None:
        """Get a more conservative config if training fails.

        Returns the next fallback configuration in the escalation hierarchy,
        or *None* when all fallback attempts are exhausted.  The hierarchy is:

        1. Disable new Ray RLlib API stack (API compatibility).
        2. Reduce to medium tier.
        3. Reduce to low tier.
        4. Force SimplifiedMultiAgentEnv at current tier.
        5. Force simplified environment + disable new API stack.
        6. Minimal tier + simplified environment + legacy API (last resort).
        """
        if self.fallback_attempts >= self.max_fallback_attempts:
            return None

        self.fallback_attempts += 1

        fallback_strategies = [
            {"tier": self.current_tier, "force_simplified": False, "use_legacy_api": True},
            {"tier": "medium", "force_simplified": False, "use_legacy_api": False},
            {"tier": "low", "force_simplified": False, "use_legacy_api": False},
            {"tier": self.current_tier, "force_simplified": True, "use_legacy_api": False},
            {"tier": self.current_tier, "force_simplified": True, "use_legacy_api": True},
            {"tier": "minimal", "force_simplified": True, "use_legacy_api": True},
        ]

        if self.fallback_attempts <= len(fallback_strategies):
            strategy = fallback_strategies[self.fallback_attempts - 1]
            config = self.get_adaptive_config(target_tier=strategy["tier"])

            if strategy["use_legacy_api"]:
                config["use_new_api_stack"] = False
                print("Fallback: Disabling new Ray RLlib API stack for compatibility")

            if strategy["force_simplified"]:
                if "env" not in config:
                    config["env"] = {}
                config["env"]["environment_class"] = "SimplifiedMultiAgentEnv"
                print("Fallback: Switching to SimplifiedMultiAgentEnv for better compatibility")

            return config

        return None

    def print_system_summary(self):
        """Print a summary of system capabilities."""
        info = self.system_info
        tier, recommendations = SystemResourceChecker.assess_training_capability()

        print("=" * 60)
        print("SYSTEM RESOURCE ASSESSMENT")
        print("=" * 60)
        print(f"CPU: {info['cpu']['threads']} threads ({info['cpu']['cores']} cores)")
        print(f"CPU Usage: {info['cpu']['percent_used']:.1f}%")
        print(
            f"Memory: {info['memory']['total_gb']} GB total, "
            f"{info['memory']['available_gb']} GB available "
            f"({info['memory']['percent_used']:.1f}% used)"
        )

        if info["gpu"]["available"]:
            for i, gpu in enumerate(info["gpu"]["devices"]):
                print(f"GPU {i}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
        else:
            print("GPU: Not available")

        print(f"Platform: {info['platform']['system']} {info['platform']['machine']}")
        print()
        print(f"RECOMMENDED TIER: {tier.upper()}")
        print(f"Description: {recommendations['description']}")
        print(f"Environment: {recommendations['environment']}")
        print(f"Workers: {recommendations['num_env_runners']}")
        print(f"Batch size: {recommendations['train_batch_size']}")
        print("=" * 60)


def create_adaptive_trainer(config_path: str, auto_detect: bool = True) -> AdaptiveTrainingConfig:
    """Create an adaptive training configuration.

    When *auto_detect* is True, prints a system summary and prompts the user
    for confirmation before returning the trainer.  Returns *None* if the user
    declines the recommended configuration.
    """
    trainer = AdaptiveTrainingConfig(config_path)

    if auto_detect:
        trainer.print_system_summary()

        try:
            response = (
                input("\nUse recommended configuration? (y/n/s for system info): ").lower().strip()
            )
            if response == "s":
                import json

                print("\nDetailed System Information:")
                print(json.dumps(trainer.system_info, indent=2))
                response = input("\nUse recommended configuration? (y/n): ").lower().strip()

            if response in ["n", "no"]:
                print("Using base configuration without adaptation.")
                return None

        except KeyboardInterrupt:
            print("\nUsing recommended configuration.")

    return trainer


def save_adaptive_config(config: dict[str, Any], output_path: str, tier: str):
    """Save an adaptive configuration to a YAML file.

    Adds a ``_adaptive_metadata`` block with the tier name and generation
    source so users can trace where the config came from.
    """
    output_path = Path(output_path)

    config["_adaptive_metadata"] = {
        "tier": tier,
        "generated_by": "AdaptiveTrainingConfig",
        "system_optimized": True,
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Adaptive configuration saved to: {output_path}")


if __name__ == "__main__":
    checker = SystemResourceChecker()
    info = checker.get_system_info()
    tier, recommendations = checker.assess_training_capability()

    print("System Assessment Demo")
    print("=====================")
    print(f"Detected tier: {tier}")
    print(f"Recommendations: {recommendations}")
