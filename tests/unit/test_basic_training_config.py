from pathlib import Path

import yaml


def test_basic_yaml_uses_canonical_reward_schema() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "training" / "basic.yaml"
    cfg = yaml.safe_load(path.read_text())

    assert "reward_config" not in cfg.get("training", {})
    assert isinstance(cfg["env"].get("reward_categories"), dict)
    assert isinstance(cfg["env"].get("reward_magnitudes"), dict)
    assert cfg["env"]["reward_normalization"]["enabled"] is False
    assert cfg["training"]["multi_agent"]["policy_mode"] == "shared"
