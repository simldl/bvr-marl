"""Env-runner fault-tolerance defaults in build_ppo_config.

Regression guard for Ray issue #53727: RLlib passes the config to each
env-runner actor via the object store, and with restart_failed_env_runners=True
the actor is created with max_restarts>0, so a lost object hangs the restart
indefinitely and strands the whole run/campaign. build_ppo_config must default to
NOT restarting env runners (-> actor max_restarts=0) so a failure degrades the run
instead of hanging it.
"""

from __future__ import annotations

import pytest
from gymnasium import spaces

from bvr_marl_core.rl.training.config_builder import build_ppo_config

pytest.importorskip("ray", reason="ray is not installed in this environment")


def _build(cfg_extra: dict | None = None):
    cfg = {
        "env": {},
        "seed": 1,
        "num_env_runners": 2,
        "multi_agent": {"policy_mode": "shared", "shared_policy_id": "shared_policy"},
        "training": dict(cfg_extra or {}),
    }
    obs = {"shared_policy": spaces.Box(-1, 1, (4,))}
    act = {"shared_policy": spaces.Box(-1, 1, (2,))}
    return build_ppo_config(cfg, obs, act, None, lambda aid, *a, **k: "shared_policy", None)


def _actor_max_restarts(config) -> int:
    # Mirrors ray.rllib.env.env_runner_group.EnvRunnerGroup actor creation.
    return config.max_num_env_runner_restarts if config.restart_failed_env_runners else 0


def test_env_runners_do_not_restart_by_default():
    config = _build()

    assert config.restart_failed_env_runners is False
    assert config.ignore_env_runner_failures is True
    # The #53727 footgun is max_restarts>0 on an actor with object-store ctor args.
    assert _actor_max_restarts(config) == 0
    # Restore stall is bounded well below RLlib's 1800s default.
    assert config.env_runner_restore_timeout_s == 120.0


def test_restart_can_be_re_enabled_for_clusters():
    config = _build({"restart_failed_env_runners": True})

    assert config.restart_failed_env_runners is True
    assert config.ignore_env_runner_failures is False
    assert _actor_max_restarts(config) > 0
