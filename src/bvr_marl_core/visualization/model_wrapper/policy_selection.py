"""Policy selection helpers for visualization-time inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def resolve_policy_id(
    agent_id: str | None,
    available_policies: Iterable[str],
    train_config: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the checkpoint policy module to use for an agent.

    Training can save checkpoints with several naming schemes:
    ``shared_policy`` for symmetric self-play, ``attacker_policy`` /
    ``defender_policy`` for curriculum team-separate runs, and older
    ``agent_policy`` / ``opponent_policy`` or ``blue_policy`` / ``red_policy``
    pairs.  Visualization should follow the same mapping instead of assuming
    one hard-coded pair.
    """
    policies = [str(policy_id) for policy_id in available_policies]
    if not policies:
        raise KeyError("Cannot resolve policy ID because the checkpoint has no policy modules.")

    policy_set = set(policies)
    cfg = train_config or {}
    ma_cfg = _nested_mapping(cfg, "training", "multi_agent")

    shared_id = str(ma_cfg.get("shared_policy_id", "shared_policy") or "shared_policy")
    shared_match = _first_available(policy_set, shared_id, "shared_policy")
    if shared_match is not None:
        return shared_match

    if len(policies) == 1:
        return policies[0]

    aid = str(agent_id or "")
    if not aid:
        raise KeyError(
            "Cannot choose among multiple policy modules without an agent_id. "
            f"Available modules: {sorted(policy_set)}"
        )

    current_match = _resolve_attacker_defender_policy(aid, policy_set, ma_cfg, cfg)
    if current_match is not None:
        return current_match

    legacy_match = _resolve_side_pair_policy(
        aid,
        policy_set,
        team_a_policy="agent_policy",
        team_b_policy="opponent_policy",
    )
    if legacy_match is not None:
        return legacy_match

    legacy_color_match = _resolve_side_pair_policy(
        aid,
        policy_set,
        team_a_policy="blue_policy",
        team_b_policy="red_policy",
    )
    if legacy_color_match is not None:
        return legacy_color_match

    raise KeyError(
        f"Cannot resolve policy for agent_id={aid!r}. Available modules: {sorted(policy_set)}"
    )


def _resolve_attacker_defender_policy(
    agent_id: str,
    policy_set: set[str],
    ma_cfg: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> str | None:
    attacker_id = str(ma_cfg.get("attacker_policy_id", "attacker_policy") or "attacker_policy")
    defender_id = str(ma_cfg.get("defender_policy_id", "defender_policy") or "defender_policy")
    attacker_id = attacker_id if attacker_id in policy_set else "attacker_policy"
    defender_id = defender_id if defender_id in policy_set else "defender_policy"
    if attacker_id not in policy_set or defender_id not in policy_set:
        return None

    line_cfg = _nested_mapping(cfg, "env", "scenario_config", "line_objective")
    attacker_team = str(line_cfg.get("attacker_team", "A") or "A")
    defender_team = str(line_cfg.get("defender_team", "B") or "B")
    if _agent_matches_team(agent_id, attacker_team):
        return attacker_id
    if _agent_matches_team(agent_id, defender_team):
        return defender_id
    return None


def _resolve_side_pair_policy(
    agent_id: str,
    policy_set: set[str],
    *,
    team_a_policy: str,
    team_b_policy: str,
) -> str | None:
    if team_a_policy not in policy_set or team_b_policy not in policy_set:
        return None
    if _agent_matches_team(agent_id, "A"):
        return team_a_policy
    if _agent_matches_team(agent_id, "B"):
        return team_b_policy
    return None


def _agent_matches_team(agent_id: str, team: str) -> bool:
    return agent_id.upper().startswith(str(team).upper())


def _first_available(policy_set: set[str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in policy_set:
            return candidate
    return None


def _nested_mapping(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, Mapping) else {}
