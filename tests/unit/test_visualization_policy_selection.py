import pytest

from bvr_marl_core.visualization.model_wrapper.policy_selection import resolve_policy_id


def test_resolve_policy_id_prefers_shared_policy() -> None:
    assert resolve_policy_id("A0", ["shared_policy"]) == "shared_policy"
    assert (
        resolve_policy_id(
            "B0",
            ["custom_shared"],
            {"training": {"multi_agent": {"shared_policy_id": "custom_shared"}}},
        )
        == "custom_shared"
    )


def test_resolve_policy_id_maps_current_team_separate_defaults() -> None:
    policies = ["attacker_policy", "defender_policy"]

    assert resolve_policy_id("A0", policies) == "attacker_policy"
    assert resolve_policy_id("B0", policies) == "defender_policy"


def test_resolve_policy_id_honors_line_objective_team_roles() -> None:
    cfg = {
        "env": {
            "scenario_config": {
                "line_objective": {
                    "attacker_team": "B",
                    "defender_team": "A",
                }
            }
        },
        "training": {
            "multi_agent": {
                "policy_mode": "team_separate",
                "attacker_policy_id": "attack",
                "defender_policy_id": "defend",
            }
        },
    }

    assert resolve_policy_id("B0", ["attack", "defend"], cfg) == "attack"
    assert resolve_policy_id("A0", ["attack", "defend"], cfg) == "defend"


def test_resolve_policy_id_keeps_legacy_pairs() -> None:
    assert resolve_policy_id("A0", ["agent_policy", "opponent_policy"]) == "agent_policy"
    assert resolve_policy_id("B0", ["agent_policy", "opponent_policy"]) == "opponent_policy"
    assert resolve_policy_id("A0", ["blue_policy", "red_policy"]) == "blue_policy"
    assert resolve_policy_id("B0", ["blue_policy", "red_policy"]) == "red_policy"


def test_resolve_policy_id_raises_with_clear_available_modules() -> None:
    with pytest.raises(KeyError, match="Available modules"):
        resolve_policy_id("C0", ["attacker_policy", "defender_policy"])
