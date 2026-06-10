"""
Unit tests for bvr_marl_core.rl.training.adaptive_config.

Tests validate adaptive configuration generation under different mocked
resource scenarios:
  - CPU-only (no GPU)
  - GPU-enabled
  - Constrained resources (minimal tier)
  - Unsupported / missing accelerator (torch.cuda not available)

All tests mock psutil and torch.cuda so they run without real hardware
and without side-effects on the host system.
"""

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_base_config(tmp_path: Path, extra: dict | None = None) -> Path:
    """Write a minimal base training YAML config and return its path."""
    config: dict[str, Any] = {
        "training": {
            "steps": 10000,
        },
        "env": {
            "num_agents_per_team": 2,
            "max_steps": 1200,
        },
    }
    if extra:
        config.update(extra)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.dump(config))
    return path


def _make_cpu_info(threads: int = 4, cores: int = 2, memory_gb: float = 8.0, cpu_pct: float = 10.0):
    """Return a psutil-shaped info dict for a CPU-only system."""
    return {
        "cpu": {
            "cores": cores,
            "threads": threads,
            "freq_max": 3000.0,
            "percent_used": cpu_pct,
        },
        "memory": {
            "total_gb": memory_gb,
            "available_gb": memory_gb * 0.6,
            "percent_used": 40.0,
        },
        "gpu": {
            "available": False,
            "count": 0,
            "devices": [],
        },
        "platform": {
            "system": "Linux",
            "machine": "x86_64",
            "python_version": "3.12.0",
        },
    }


def _make_gpu_info(threads: int = 32, memory_gb: float = 64.0, gpu_memory_gb: float = 16.0):
    """Return a system info dict for a GPU-enabled high-end system."""
    info = _make_cpu_info(threads=threads, cores=16, memory_gb=memory_gb)
    info["gpu"] = {
        "available": True,
        "count": 1,
        "devices": [
            {
                "id": 0,
                "name": "NVIDIA RTX 4090",
                "total_memory_gb": gpu_memory_gb,
                "major": 8,
                "minor": 9,
            }
        ],
    }
    return info


# ---------------------------------------------------------------------------
# SystemResourceChecker
# ---------------------------------------------------------------------------


class TestSystemResourceChecker:
    def test_get_system_info_cpu_only(self, monkeypatch):
        """get_system_info returns a valid structure on a CPU-only system."""
        from bvr_marl_core.rl.training.adaptive_config import SystemResourceChecker

        cpu_count = MagicMock(side_effect=lambda logical=True: 8 if logical else 4)
        cpu_freq = MagicMock(return_value=MagicMock(max=3600.0))
        cpu_percent = MagicMock(return_value=20.0)
        vmem = MagicMock()
        vmem.total = 16 * 1024**3
        vmem.available = 8 * 1024**3
        vmem.percent = 50.0

        with (
            patch("psutil.cpu_count", cpu_count),
            patch("psutil.cpu_freq", cpu_freq),
            patch("psutil.cpu_percent", cpu_percent),
            patch("psutil.virtual_memory", return_value=vmem),
            patch("torch.cuda.is_available", return_value=False),
        ):
            info = SystemResourceChecker.get_system_info()

        assert info["cpu"]["threads"] == 8
        assert info["cpu"]["cores"] == 4
        assert info["memory"]["total_gb"] == pytest.approx(16.0, abs=0.2)
        assert info["gpu"]["available"] is False
        assert info["gpu"]["devices"] == []

    def test_assess_capability_returns_valid_tier(self, monkeypatch):
        """assess_training_capability always returns a known tier string."""
        from bvr_marl_core.rl.training.adaptive_config import SystemResourceChecker

        with patch.object(SystemResourceChecker, "get_system_info", return_value=_make_cpu_info()):
            tier, recs = SystemResourceChecker.assess_training_capability()

        assert tier in {"minimal", "low", "medium", "high"}
        assert "num_env_runners" in recs
        assert "train_batch_size" in recs
        assert "sgd_minibatch_size" in recs

    def test_constrained_system_selects_minimal_tier(self):
        """A system with 4 GB RAM and 2 threads lands in the minimal tier."""
        from bvr_marl_core.rl.training.adaptive_config import SystemResourceChecker

        constrained = _make_cpu_info(threads=2, cores=1, memory_gb=4.0)
        with patch.object(SystemResourceChecker, "get_system_info", return_value=constrained):
            tier, _ = SystemResourceChecker.assess_training_capability()

        assert tier == "minimal"

    def test_high_end_gpu_system_selects_high_tier(self):
        """A system with 64 GB RAM, 32 threads, and an 8 GB+ GPU lands in the high tier."""
        from bvr_marl_core.rl.training.adaptive_config import SystemResourceChecker

        gpu_system = _make_gpu_info(threads=32, memory_gb=64.0, gpu_memory_gb=16.0)
        with patch.object(SystemResourceChecker, "get_system_info", return_value=gpu_system):
            tier, _ = SystemResourceChecker.assess_training_capability()

        assert tier == "high"

    def test_missing_gpu_does_not_raise(self):
        """get_system_info succeeds even when torch.cuda.is_available() returns False."""
        from bvr_marl_core.rl.training.adaptive_config import SystemResourceChecker

        with (
            patch("psutil.cpu_count", return_value=4),
            patch("psutil.cpu_freq", return_value=MagicMock(max=2400.0)),
            patch("psutil.cpu_percent", return_value=5.0),
            patch(
                "psutil.virtual_memory",
                return_value=MagicMock(total=8 * 1024**3, available=4 * 1024**3, percent=50.0),
            ),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.device_count", return_value=0),
        ):
            info = SystemResourceChecker.get_system_info()

        assert info["gpu"]["available"] is False
        assert info["gpu"]["count"] == 0


# ---------------------------------------------------------------------------
# AdaptiveTrainingConfig
# ---------------------------------------------------------------------------


class TestAdaptiveTrainingConfig:
    def test_get_adaptive_config_cpu_only(self, tmp_path):
        """CPU-only scenario: num_gpus must be 0, config must be valid."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        cpu_info = _make_cpu_info(threads=8, memory_gb=16.0)

        with (
            patch(
                "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
                return_value=cpu_info,
            ),
            patch(
                "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker"
                ".assess_training_capability",
                return_value=(
                    "low",
                    {
                        "description": "Light training",
                        "num_env_runners": 2,
                        "train_batch_size": 4000,
                        "sgd_minibatch_size": 256,
                        "num_sgd_iter": 15,
                        "framework": "torch",
                        "environment": "SimplifiedMultiAgentEnv",
                    },
                ),
            ),
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            config = trainer.get_adaptive_config()

        assert config["num_gpus"] == 0
        assert config["framework"] == "torch"
        assert "training" in config
        assert "env" in config
        assert config["training"]["train_batch_size_per_learner"] == 4000
        assert config["training"]["sgd_minibatch_size"] == 256

    def test_get_adaptive_config_gpu_enabled(self, tmp_path):
        """GPU scenario: num_gpus must be >= 1 and tier must be high."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        gpu_info = _make_gpu_info()

        with (
            patch(
                "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
                return_value=gpu_info,
            ),
            patch(
                "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker"
                ".assess_training_capability",
                return_value=(
                    "high",
                    {
                        "description": "Performance training",
                        "num_env_runners": 6,
                        "train_batch_size": 12000,
                        "sgd_minibatch_size": 1024,
                        "num_sgd_iter": 25,
                        "framework": "torch",
                        "environment": "BVRMultiAgentEnv",
                    },
                ),
            ),
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            config = trainer.get_adaptive_config()

        assert config["num_gpus"] >= 1
        assert config["framework"] == "torch"
        assert config["training"]["train_batch_size_per_learner"] == 12000
        assert config["training"]["sgd_minibatch_size"] == 1024

    def test_get_adaptive_config_constrained_resources(self, tmp_path):
        """Constrained system: minimal tier, SimplifiedMultiAgentEnv, small batches."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        constrained_info = _make_cpu_info(threads=2, cores=1, memory_gb=4.0)

        with (
            patch(
                "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
                return_value=constrained_info,
            ),
            patch(
                "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker"
                ".assess_training_capability",
                return_value=(
                    "minimal",
                    {
                        "description": "Basic training",
                        "num_env_runners": 1,
                        "train_batch_size": 2000,
                        "sgd_minibatch_size": 128,
                        "num_sgd_iter": 10,
                        "framework": "torch",
                        "environment": "SimplifiedMultiAgentEnv",
                    },
                ),
            ),
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            config = trainer.get_adaptive_config()

        assert config["training"]["train_batch_size_per_learner"] == 2000
        assert config["training"]["sgd_minibatch_size"] == 128
        assert config["env"].get("environment_class") == "SimplifiedMultiAgentEnv"
        assert config["env"]["max_steps"] <= 800
        assert config["env"]["map_size_km"] <= 200

    def test_get_adaptive_config_forced_tier(self, tmp_path):
        """target_tier parameter overrides auto-detection."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        # Even if the system is GPU-capable, forcing "minimal" must produce minimal params
        gpu_info = _make_gpu_info()

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=gpu_info,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            config = trainer.get_adaptive_config(target_tier="minimal")

        assert config["training"]["train_batch_size_per_learner"] == 2000
        assert config["env"].get("environment_class") == "SimplifiedMultiAgentEnv"

    def test_config_contains_required_sections(self, tmp_path):
        """Adaptive config always contains training, env, model, and logging sections."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        cpu_info = _make_cpu_info()

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=cpu_info,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            config = trainer.get_adaptive_config(target_tier="low")

        for section in ("training", "env", "model", "logging"):
            assert section in config, f"Missing required section: {section!r}"

    def test_reward_config_defaults_applied(self, tmp_path):
        """Reward config defaults are populated when not present in base config."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        cpu_info = _make_cpu_info()

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=cpu_info,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            config = trainer.get_adaptive_config(target_tier="low")

        reward = config["training"]["reward_config"]
        assert "kill_reward" in reward
        assert "destruction_penalty" in reward
        assert reward["enable_terminal"] is True

    def test_base_config_values_are_not_overwritten(self, tmp_path):
        """User-defined base config values take priority over defaults."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path, extra={"use_new_api_stack": True})
        cpu_info = _make_cpu_info()

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=cpu_info,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            config = trainer.get_adaptive_config(target_tier="low")

        # The base config explicitly sets use_new_api_stack=True; it must not be overridden
        assert config["use_new_api_stack"] is True

    def test_num_env_runners_capped_by_available_threads(self, tmp_path):
        """num_env_runners is capped to available CPU threads minus reserved overhead."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        # Only 5 threads total — after reserving 4 overhead, only 1 can be a worker
        tight_cpu = _make_cpu_info(threads=5, memory_gb=16.0)

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=tight_cpu,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            # medium recommends 4 workers, but we only have 1 thread available
            config = trainer.get_adaptive_config(target_tier="medium")

        assert config["num_env_runners"] <= 1


# ---------------------------------------------------------------------------
# Fallback hierarchy
# ---------------------------------------------------------------------------


class TestAdaptiveTrainingConfigFallback:
    def test_fallback_disables_new_api_stack_first(self, tmp_path):
        """First fallback attempt disables the new Ray RLlib API stack."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        cpu_info = _make_cpu_info(threads=8, memory_gb=16.0)

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=cpu_info,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            trainer.current_tier = "low"  # simulate a previous training attempt
            fallback = trainer.get_fallback_config()

        assert fallback is not None
        assert fallback["use_new_api_stack"] is False

    def test_fallback_returns_none_after_exhaustion(self, tmp_path):
        """get_fallback_config returns None once max_fallback_attempts is reached."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        cpu_info = _make_cpu_info()

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=cpu_info,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            trainer.current_tier = "low"
            trainer.fallback_attempts = trainer.max_fallback_attempts  # already exhausted

        result = trainer.get_fallback_config()
        assert result is None

    def test_fallback_eventually_forces_simplified_env(self, tmp_path):
        """Later fallback attempts switch to SimplifiedMultiAgentEnv."""
        from bvr_marl_core.rl.training.adaptive_config import AdaptiveTrainingConfig

        cfg_path = _write_base_config(tmp_path)
        cpu_info = _make_cpu_info(threads=8, memory_gb=16.0)

        with patch(
            "bvr_marl_core.rl.training.adaptive_config.SystemResourceChecker.get_system_info",
            return_value=cpu_info,
        ):
            trainer = AdaptiveTrainingConfig(str(cfg_path))
            trainer.current_tier = "high"
            # Advance to attempt 4 (force_simplified=True, use_legacy_api=False)
            trainer.fallback_attempts = 3
            fallback = trainer.get_fallback_config()

        assert fallback is not None
        assert fallback["env"].get("environment_class") == "SimplifiedMultiAgentEnv"


# ---------------------------------------------------------------------------
# save_adaptive_config
# ---------------------------------------------------------------------------


class TestSaveAdaptiveConfig:
    def test_saves_yaml_with_metadata(self, tmp_path):
        """save_adaptive_config writes a valid YAML file with _adaptive_metadata."""
        from bvr_marl_core.rl.training.adaptive_config import save_adaptive_config

        config = {"training": {"steps": 1000}, "env": {}}
        output = str(tmp_path / "saved.yaml")
        save_adaptive_config(config, output, tier="low")

        loaded = yaml.safe_load(Path(output).read_text())
        assert "_adaptive_metadata" in loaded
        assert loaded["_adaptive_metadata"]["tier"] == "low"
        assert loaded["_adaptive_metadata"]["system_optimized"] is True
        assert loaded["training"]["steps"] == 1000
