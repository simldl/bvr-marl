import pytest

pytest.importorskip("ray")

from bvr_marl_core.rl.utils import env_creator as env_creator_mod


class _DummyEnv:
    pass


class _DummyWrapper:
    def __init__(self, env, clip_reward):
        self.env = env
        self.clip_reward = clip_reward


def test_core_env_creator_helper_leaves_rewards_raw_by_default(monkeypatch) -> None:
    monkeypatch.setattr(env_creator_mod, "RewardNormalizationWrapper", _DummyWrapper)
    env = _DummyEnv()

    assert env_creator_mod._apply_reward_normalization(env, {}) is env


def test_core_env_creator_helper_wraps_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(env_creator_mod, "RewardNormalizationWrapper", _DummyWrapper)
    env = _DummyEnv()

    wrapped = env_creator_mod._apply_reward_normalization(
        env,
        {"reward_normalization": {"enabled": True, "clip_reward": 7.5}},
    )

    assert isinstance(wrapped, _DummyWrapper)
    assert wrapped.env is env
    assert wrapped.clip_reward == 7.5
