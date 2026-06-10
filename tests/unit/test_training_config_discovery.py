from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def test_public_training_configs_are_training_options(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths_module = types.ModuleType("bvr_marl_core.utils.paths")
    paths_module.core_project_root = lambda: repo_root
    paths_module.core_runtime_root = lambda: repo_root / "runtime"
    paths_module.rl_configs_root = lambda: repo_root / "src" / "bvr_marl_core" / "rl" / "configs"

    monkeypatch.setitem(sys.modules, "bvr_marl_core", types.ModuleType("bvr_marl_core"))
    monkeypatch.setitem(sys.modules, "bvr_marl_core.utils", types.ModuleType("bvr_marl_core.utils"))
    monkeypatch.setitem(sys.modules, "bvr_marl_core.utils.paths", paths_module)

    spec = importlib.util.spec_from_file_location(
        "training_service_under_test",
        repo_root / "src" / "bvr_marl_core" / "services" / "training.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    configs = [entry.replace("\\", "/") for entry in module.list_training_configs()]

    assert all(not entry.startswith("configs/campaign/") for entry in configs)
    assert "basic.yaml" in configs
