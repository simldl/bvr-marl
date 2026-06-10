import pytest

pytest.importorskip("ray")

from bvr_marl_core.rl.training.config_builder import (
    policies_to_train_from_cfg,
    policy_ids_from_cfg,
)


def test_policy_ids_shared_mode_uses_configured_id() -> None:
    cfg = {"training": {"multi_agent": {"policy_mode": "shared", "shared_policy_id": "one"}}}

    assert policy_ids_from_cfg(cfg) == {"one"}
    assert policies_to_train_from_cfg(cfg) == ["one"]


def test_policy_ids_team_separate_honors_freeze_flags() -> None:
    cfg = {
        "training": {
            "multi_agent": {
                "policy_mode": "team_separate",
                "attacker_policy_id": "attack",
                "defender_policy_id": "defend",
                "train_attacker": False,
                "train_defender": True,
            }
        }
    }

    assert policy_ids_from_cfg(cfg) == {"attack", "defend"}
    assert policies_to_train_from_cfg(cfg) == ["defend"]
