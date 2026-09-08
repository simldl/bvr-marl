"""Value-function loss clamp and PPO regularisation plumbing in build_ppo_config.

Regression guard for the silent critic-death bug. RLlib clamps the *squared* value
error at ``vf_clip_param`` and ``torch.clamp`` has zero gradient above its bound::

    vf_loss = torch.pow(value_fn_out - value_targets, 2.0)
    vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)

RLlib's default of 10.0 is an error threshold of ~3.16. BVR rewards are not
normalised and carry terminal magnitudes of 100-200, so value targets run to
O(100) and squared errors to O(1e5): every sample lands outside the clamp, the
critic trains on nothing, and the unopposed entropy bonus diffuses the policy.
build_ppo_config must therefore not leave this at RLlib's default.
"""

from __future__ import annotations

import warnings

import pytest
from gymnasium import spaces

from bvr_marl_core.rl.training.config_builder import (
    DEFAULT_GRAD_CLIP,
    DEFAULT_VF_CLIP_PARAM,
    build_ppo_config,
    required_vf_clip_param,
    resolve_reward_magnitudes,
    validate_value_scale,
)

pytest.importorskip("ray", reason="ray is not installed in this environment")

# Squared value error for a policy whose returns are dominated by a 200-point kill
# reward: the critic must still receive gradient at this magnitude.
BVR_SCALE_SQUARED_ERROR = 200.0**2


# The BVR reward scale that broke the campaign: unnormalised, 200-point kill reward.
BVR_REWARD_CONFIG = {
    "kill_reward": 200.0,
    "destruction_penalty": -100.0,
    "boundary_violation_penalty": -100.0,
}


def _build(training: dict | None = None, reward_config: dict | None = None, key="reward_config"):
    cfg = {
        "env": {key: dict(reward_config)} if reward_config else {},
        "seed": 1,
        "num_env_runners": 2,
        "multi_agent": {"policy_mode": "shared", "shared_policy_id": "shared_policy"},
        "training": dict(training or {}),
    }
    obs = {"shared_policy": spaces.Box(-1, 1, (4,))}
    act = {"shared_policy": spaces.Box(-1, 1, (2,))}
    return build_ppo_config(cfg, obs, act, None, lambda aid, *a, **k: "shared_policy", None)


def test_vf_clip_param_does_not_fall_back_to_rllib_default():
    config = _build()

    # 10.0 is RLlib's default and the value that killed the critic.
    assert config.vf_clip_param != 10.0
    assert config.vf_clip_param == DEFAULT_VF_CLIP_PARAM


def test_default_clamp_leaves_gradient_at_bvr_reward_scale():
    config = _build()

    # The clamp must not bind for errors the size of a missed kill reward, or the
    # critic gets exactly zero gradient on precisely the samples that matter.
    assert config.vf_clip_param > BVR_SCALE_SQUARED_ERROR


def test_default_clamp_still_bounds_a_diverged_critic():
    # The clamp must not be disabled outright either. The value head shares its trunk
    # with the policy and grad_clip_by="global_norm" rescales all parameters together,
    # so an unbounded VF term dominates the global norm and starves the policy update.
    # A boundary-only stage measured vf_loss_unclipped at 4.5e8 (RMS value error ~21k); that is a
    # diverged critic, not a real return, and it must still be clamped.
    diverged_squared_error = 4.5e8

    assert _build().vf_clip_param < diverged_squared_error


def test_vf_clip_param_is_configurable_for_normalised_rewards():
    # A config that normalises rewards may legitimately want a tight clamp back.
    config = _build({"vf_clip_param": 10.0})

    assert config.vf_clip_param == 10.0


def test_kl_target_is_plumbed_through():
    assert _build().kl_target == 0.01
    assert _build({"kl_target": 0.03}).kl_target == 0.03


def test_entropy_schedule_is_on_by_default_and_explicitly_overridable():
    # Previously this asserted `entropy_coeff_schedule` was None by default. Two
    # things changed: a flat coefficient is what let entropy climb unopposed once
    # the advantage signal weakened, so the default is now a decaying schedule; and
    # the schedule rides on `entropy_coeff` because RLlib's new API stack raises on
    # any non-None `entropy_coeff_schedule`. See test_config_builder_entropy_kl.py.
    default = _build().entropy_coeff
    assert isinstance(default, list)
    assert default[-1][1] < default[0][1]

    schedule = [[0, 0.01], [1_000_000, 0.0]]
    assert _build({"entropy_coef_schedule": schedule}).entropy_coeff == schedule


def test_deprecated_entropy_coeff_schedule_setting_is_never_populated():
    # Non-None here is a hard ValueError at build time on the new API stack.
    assert _build().entropy_coeff_schedule is None
    assert _build({"entropy_coef_schedule": [[0, 0.01], [10, 0.0]]}).entropy_coeff_schedule is None


def test_existing_ppo_knobs_still_resolve_from_config():
    config = _build(
        {
            "learning_rate": 0.0001,
            "entropy_coef": 0.02,
            "kl_coeff": 0.3,
            "clip_param": 0.25,
            "grad_clip": 0.5,
            "vf_loss_coeff": 0.7,
        }
    )

    assert config.lr == 0.0001
    # entropy_coeff now carries the schedule; the configured value is its first knot.
    assert config.entropy_coeff[0][1] == 0.02
    assert config.kl_coeff == 0.3
    assert config.clip_param == 0.25
    assert config.grad_clip == 0.5
    assert config.vf_loss_coeff == 0.7
    # Gradient magnitude, not the VF loss clamp, is what bounds updates here.
    assert config.grad_clip_by == "global_norm"


def test_gradients_are_bounded_even_when_the_config_omits_grad_clip():
    # The VF clamp bounds the loss; grad_clip bounds the resulting update. Leaving this
    # at None pairs an unbounded update with a large VF clamp -- one bad batch then
    # moves the encoder the policy shares with the critic.
    config = _build()

    assert config.grad_clip == DEFAULT_GRAD_CLIP
    assert config.grad_clip is not None
    assert config.grad_clip > 0.0


def test_grad_clip_is_not_a_lever_for_a_stalled_policy():
    # Regression guard for a WRONG diagnosis, not a bug. A run raised this
    # 0.5 -> 50.0 believing a ~1e-4 global-norm scale factor was starving the policy.
    # It cannot: the optimizer is Adam, whose update is invariant to multiplying every
    # gradient by one scalar -- which is all global-norm clipping does. At a 170x
    # difference in clip coefficient the stage-0 curves were superimposable. Keep the
    # default tight; look at the environment instead.
    assert DEFAULT_GRAD_CLIP == 0.5


def test_explicit_grad_clip_still_wins():
    assert _build({"grad_clip": 1.0}).grad_clip == 1.0


def test_required_clip_scales_with_terminal_reward_magnitude():
    small = required_vf_clip_param({"kill_reward": 10.0})
    large = required_vf_clip_param({"kill_reward": 200.0})

    assert small is not None and large is not None
    assert large > small
    # A reward config with no terminal magnitudes gives nothing to validate against.
    assert required_vf_clip_param({}) is None
    assert required_vf_clip_param(None) is None


def test_rllib_default_clamp_is_rejected_at_bvr_reward_scale():
    # This is the exact combination that ran for three campaigns undetected.
    message = validate_value_scale({"vf_clip_param": 10.0}, BVR_REWARD_CONFIG)

    assert message is not None
    assert "vf_clip_param" in message


def test_shipped_default_passes_validation_at_bvr_reward_scale():
    assert validate_value_scale({}, BVR_REWARD_CONFIG) is None
    assert DEFAULT_VF_CLIP_PARAM >= required_vf_clip_param(BVR_REWARD_CONFIG)


def test_building_with_a_too_tight_clamp_warns_loudly():
    with pytest.warns(RuntimeWarning, match="vf_clip_param"):
        _build({"vf_clip_param": 10.0}, reward_config=BVR_REWARD_CONFIG)


def test_building_at_the_default_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _build(reward_config=BVR_REWARD_CONFIG)


def test_campaign_stage_yamls_declare_reward_magnitudes_not_reward_config():
    # Campaign stage YAMLs use env.reward_magnitudes. A validator that only knew
    # env.reward_config would pass every one of them without checking anything.
    assert resolve_reward_magnitudes({"reward_magnitudes": BVR_REWARD_CONFIG}) == BVR_REWARD_CONFIG
    assert resolve_reward_magnitudes({"reward_config": BVR_REWARD_CONFIG}) == BVR_REWARD_CONFIG
    assert resolve_reward_magnitudes({"reward_magnitudes": {}}) is None
    assert resolve_reward_magnitudes(None) is None

    with pytest.warns(RuntimeWarning, match="vf_clip_param"):
        _build({"vf_clip_param": 10.0}, BVR_REWARD_CONFIG, key="reward_magnitudes")
