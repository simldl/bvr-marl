"""Reward information classes and fail-closed access policy."""

from __future__ import annotations

from enum import StrEnum


class RewardInformationClass(StrEnum):
    OBSERVATION_ONLY = "observation_only"
    TEAM_SHARED = "team_shared"
    PRIVILEGED_TRAINING = "privileged_training"
    EVALUATOR_TERMINAL_ONLY = "evaluator_terminal_only"


_ALLOWED_BY_MODE = {
    RewardInformationClass.OBSERVATION_ONLY: {
        RewardInformationClass.OBSERVATION_ONLY,
        RewardInformationClass.EVALUATOR_TERMINAL_ONLY,
    },
    RewardInformationClass.TEAM_SHARED: {
        RewardInformationClass.OBSERVATION_ONLY,
        RewardInformationClass.TEAM_SHARED,
        RewardInformationClass.EVALUATOR_TERMINAL_ONLY,
    },
    RewardInformationClass.PRIVILEGED_TRAINING: set(RewardInformationClass),
    RewardInformationClass.EVALUATOR_TERMINAL_ONLY: {
        RewardInformationClass.EVALUATOR_TERMINAL_ONLY
    },
}


def resolve_reward_information_mode(value: object) -> RewardInformationClass:
    raw = "observation_only" if value is None else str(value).strip().lower()
    try:
        return RewardInformationClass(raw)
    except ValueError as exc:
        choices = ", ".join(item.value for item in RewardInformationClass)
        raise ValueError(f"reward_information_mode must be one of: {choices}") from exc


def ensure_reward_information_allowed(
    mode: RewardInformationClass, component: str, information_class: RewardInformationClass
) -> None:
    if information_class not in _ALLOWED_BY_MODE[mode]:
        raise ValueError(
            f"Reward component {component!r} requires {information_class.value!r} "
            f"information, forbidden by reward_information_mode={mode.value!r}."
        )
