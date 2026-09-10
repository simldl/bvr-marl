"""The actor must not be rescaled by the critic's gradient magnitude.

RLlib takes ONE global norm over an RLModule's parameters and rescales them all by
``grad_clip / ||g||``. At BVR's unnormalized reward scale that norm IS the critic's:
over a measured 500-iteration stage the norm ran to a median of
489,312 against ``grad_clip: 0.5``, so the actor was multiplied by ~1e-6 every step and
``mean_kl_loss`` sat at 9.2e-8 against a 1e-2 target -- a policy that is not moving --
while ``vf_explained_var`` climbed to 0.62 and entropy rose monotonically.

These tests pin the properties that make the split actually work:
  1. the actor's scaling does not depend on the critic's gradient magnitude,
  2. the critic is still bounded (this is not "remove the clip"),
  3. it is off unless configured, so an old run still reproduces, and
  4. the critic-warmup mask still runs, and runs FIRST.
"""

from __future__ import annotations

import pytest
import torch

from bvr_marl_core.rl.training.critic_warmup_learner import (
    CRITIC_WARMUP_ITERATIONS_KEY,
    CriticWarmupPPOTorchLearner,
)
from bvr_marl_core.rl.training.decoupled_grad_clip_learner import (
    POLICY_GRAD_CLIP_KEY,
    POLICY_GRAD_NORM_KEY,
    VALUE_GRAD_CLIP_KEY,
    VALUE_GRAD_NORM_KEY,
    DecoupledGradClipPPOTorchLearner,
)


class _Module(torch.nn.Module):
    """Shapes match the real model once `separate_value_encoder` is on."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.policy_head = torch.nn.Linear(4, 2)
        self.value_encoder = torch.nn.Linear(4, 4)
        self.value_head = torch.nn.Linear(4, 1)


class _Metrics:
    def __init__(self):
        self.logged = {}

    def log_value(self, key, value, **kwargs):
        self.logged[key] = value


class _Config:
    def __init__(self, *, policy_clip=None, value_clip=None, warmup=None, grad_clip=0.5):
        self.grad_clip = grad_clip
        self.grad_clip_by = "global_norm"
        self.learner_config_dict = {}
        if policy_clip is not None:
            self.learner_config_dict[POLICY_GRAD_CLIP_KEY] = policy_clip
        if value_clip is not None:
            self.learner_config_dict[VALUE_GRAD_CLIP_KEY] = value_clip
        if warmup is not None:
            self.learner_config_dict[CRITIC_WARMUP_ITERATIONS_KEY] = warmup


def _learner():
    obj = object.__new__(DecoupledGradClipPPOTorchLearner)
    obj._critic_warmup_iterations_seen = 0
    obj._critic_warmup_value_params = {}
    obj._decoupled_policy_params = {}
    obj._policy_norm_log_ema = {}
    obj.metrics = _Metrics()
    obj._module = {"attacker_policy": _Module()}
    return obj


def _clip_fn(self):
    from ray.rllib.utils.torch_utils import clip_gradients

    return clip_gradients


DecoupledGradClipPPOTorchLearner._get_clip_function = _clip_fn


def _grads(module, *, policy_scale, value_scale):
    """Gradients whose actor and critic magnitudes can be varied independently."""
    out = {}
    for name, p in module.named_parameters():
        scale = value_scale if ("value_head" in name or "value_encoder" in name) else policy_scale
        out[p] = torch.full_like(p, float(scale))
    return out


def _policy_grads(module, out):
    names = {id(p): n for n, p in module.named_parameters()}
    return {
        names[id(p)]: g
        for p, g in out.items()
        if not (names[id(p)].startswith(("value_head.", "value_encoder.")))
    }


def test_actor_scaling_is_independent_of_critic_magnitude():
    """The whole point: a huge critic gradient must not shrink the actor's update."""
    results = []
    for value_scale in (1.0, 1_000_000.0):
        learner = _learner()
        module = learner.module["attacker_policy"]
        config = _Config(policy_clip=10.0)
        out = learner.postprocess_gradients_for_module(
            module_id="attacker_policy",
            config=config,
            module_gradients_dict=_grads(module, policy_scale=1.0, value_scale=value_scale),
        )
        pg = _policy_grads(module, out)
        results.append(torch.cat([g.flatten() for g in pg.values()]).norm().item())

    assert results[0] == results[1], (
        "actor gradients changed when only the CRITIC's magnitude changed -- "
        f"{results[0]} vs {results[1]}; the norms are still coupled"
    )


def test_the_critic_is_still_bounded():
    """This is a split, not a removal: a diverged critic must still be capped."""
    learner = _learner()
    module = learner.module["attacker_policy"]
    out = learner.postprocess_gradients_for_module(
        module_id="attacker_policy",
        config=_Config(policy_clip=10.0, value_clip=0.5),
        module_gradients_dict=_grads(module, policy_scale=1.0, value_scale=1_000_000.0),
    )
    names = {id(p): n for n, p in module.named_parameters()}
    value_grads = [
        g for p, g in out.items() if names[id(p)].startswith(("value_head.", "value_encoder."))
    ]
    norm = torch.cat([g.flatten() for g in value_grads]).norm().item()
    assert norm <= 0.5 + 1e-4, f"critic global norm {norm} exceeded its 0.5 bound"


def test_both_norms_are_logged_separately():
    """The combined norm is what hid this for so long."""
    learner = _learner()
    module = learner.module["attacker_policy"]
    learner.postprocess_gradients_for_module(
        module_id="attacker_policy",
        config=_Config(policy_clip=10.0),
        module_gradients_dict=_grads(module, policy_scale=1.0, value_scale=500.0),
    )
    logged = learner.metrics.logged
    assert ("attacker_policy", POLICY_GRAD_NORM_KEY) in logged
    assert ("attacker_policy", VALUE_GRAD_NORM_KEY) in logged
    assert (
        logged[("attacker_policy", VALUE_GRAD_NORM_KEY)]
        > logged[("attacker_policy", POLICY_GRAD_NORM_KEY)]
    )


def test_unset_policy_clip_falls_back_to_stock_behaviour():
    """An old run must still reproduce exactly."""
    learner = _learner()
    module = learner.module["attacker_policy"]
    calls = []
    original = CriticWarmupPPOTorchLearner.postprocess_gradients_for_module
    try:
        CriticWarmupPPOTorchLearner.postprocess_gradients_for_module = (
            lambda self, *, module_id, config, module_gradients_dict: (
                calls.append(module_id) or module_gradients_dict
            )
        )
        learner.postprocess_gradients_for_module(
            module_id="attacker_policy",
            config=_Config(policy_clip=None),
            module_gradients_dict=_grads(module, policy_scale=1.0, value_scale=1.0),
        )
    finally:
        CriticWarmupPPOTorchLearner.postprocess_gradients_for_module = original
    assert calls == ["attacker_policy"], "must delegate to the parent when unconfigured"


def test_warmup_mask_still_applies_and_runs_first():
    """Order matters: a frozen actor's gradients must not inflate the critic's norm."""
    learner = _learner()
    module = learner.module["attacker_policy"]
    learner.before_gradient_based_update(timesteps={})
    out = learner.postprocess_gradients_for_module(
        module_id="attacker_policy",
        config=_Config(policy_clip=10.0, warmup=5),
        module_gradients_dict=_grads(module, policy_scale=1.0, value_scale=1.0),
    )
    names = {id(p): n for n, p in module.named_parameters()}
    for p, g in out.items():
        if not names[id(p)].startswith(("value_head.", "value_encoder.")):
            assert g.abs().sum() == 0, f"{names[id(p)]} must be frozen during warmup"


def test_value_encoder_trains_during_warmup():
    """`separate_value_encoder` is most of the critic; freezing it crippled the warmup."""
    learner = _learner()
    module = learner.module["attacker_policy"]
    learner.before_gradient_based_update(timesteps={})
    out = learner.postprocess_gradients_for_module(
        module_id="attacker_policy",
        config=_Config(policy_clip=10.0, warmup=5),
        module_gradients_dict=_grads(module, policy_scale=1.0, value_scale=1.0),
    )
    names = {id(p): n for n, p in module.named_parameters()}
    trained = {names[id(p)] for p, g in out.items() if g.abs().sum() > 0}
    assert any(n.startswith("value_encoder.") for n in trained), (
        "the separate value encoder must TRAIN during critic warmup, not be frozen"
    )


# --- adaptive actor bound ----------------------------------------------------------
#
# A CONSTANT bound cannot serve this problem: over 650 real iterations the actor norm ran
# median 669 / p90 3087 / max 23730, so the 10.0 calibrated on a 30-iteration smoke run
# was ~65x too tight and `mean_kl` reached EXACTLY 0.0 on 117 of the last 200 iterations.
# It also has to serve two policies whose norms differ ~30x at the same iteration.


def _bound_for(learner, module_id, norms, *, multiplier=3.0, ema=0.01):
    from bvr_marl_core.rl.training.decoupled_grad_clip_learner import (
        EMA_KEY,
        MULTIPLIER_KEY,
    )

    config = _Config(policy_clip="adaptive")
    config.learner_config_dict[MULTIPLIER_KEY] = multiplier
    config.learner_config_dict[EMA_KEY] = ema
    last = None
    for norm in norms:
        last = learner._adaptive_policy_bound(module_id, float(norm), config)
    return last


def test_adaptive_bound_seeds_from_the_first_observation():
    """Seeding at log(1.0) would clip a norm of 600 to ~3.0 for a whole horizon."""
    learner = _learner()
    bound = _bound_for(learner, "attacker_policy", [669.0])

    assert bound == pytest.approx(3.0 * 669.0, rel=1e-6)


def test_adaptive_bound_does_not_clip_the_ordinary_step():
    """The factor must be 1.0 on typical steps, or Adam's scale-invariance never holds."""
    learner = _learner()
    bound = _bound_for(learner, "attacker_policy", [669.0] * 200)

    assert bound > 669.0, "a bound at or below the typical norm rescales every step"
    assert bound == pytest.approx(3.0 * 669.0, rel=1e-3)


def test_adaptive_bound_tracks_growth_over_a_stage():
    """The norm grows by orders of magnitude; a bound calibrated early must follow."""
    learner = _learner()
    early = _bound_for(learner, "attacker_policy", [20.0] * 50)
    late = _bound_for(learner, "attacker_policy", [2000.0] * 500)

    assert early < 100.0
    assert late > 1000.0, "bound stayed stranded at its early calibration"


def test_adaptive_bound_is_per_module():
    """Attacker median 646 vs defender 58 at the same iteration: one constant froze one."""
    learner = _learner()
    attacker = _bound_for(learner, "attacker_policy", [646.0] * 100)
    defender = _bound_for(learner, "defender_policy", [58.0] * 100)

    assert attacker > 10.0 * defender, "the two policies must not share one reference"


def test_adaptive_bound_collapses_the_clip_factor_spread():
    """Replay of the measured run-0001 distribution.

    Under the fixed 10.0 the factor spanned 4.2e-04 .. 0.71 (~1700x within this sample).
    The adaptive bound must leave the bulk unclipped and only rescale genuine spikes.
    """
    learner = _learner()
    from bvr_marl_core.rl.training.decoupled_grad_clip_learner import (
        EMA_KEY,
        MULTIPLIER_KEY,
    )

    config = _Config(policy_clip="adaptive")
    config.learner_config_dict[MULTIPLIER_KEY] = 3.0
    config.learner_config_dict[EMA_KEY] = 0.01

    # median 669, p90 3087, occasional 23730 -- the shape that was measured.
    norms = ([669.0] * 90 + [3087.0] * 9 + [23729.5]) * 5
    factors = []
    for norm in norms:
        bound = learner._adaptive_policy_bound("attacker_policy", norm, config)
        factors.append(min(1.0, bound / norm))

    settled = factors[len(factors) // 2 :]
    unclipped = [f for f in settled if f >= 1.0]
    assert len(unclipped) / len(settled) > 0.8, "most ordinary steps must pass untouched"

    fixed_spread = 0.71 / 4.2e-04
    adaptive_spread = max(settled) / min(settled)
    assert adaptive_spread < fixed_spread / 50.0, (
        f"factor spread {adaptive_spread:.1f} is not materially better than {fixed_spread:.1f}"
    )


def test_adaptive_ignores_a_degenerate_norm_without_poisoning_the_reference():
    """log(0) is -inf; a zero-gradient step must not drag the bound to nothing."""
    learner = _learner()
    settled = _bound_for(learner, "attacker_policy", [669.0] * 100)
    after_zero = _bound_for(learner, "attacker_policy", [0.0])

    assert after_zero == pytest.approx(settled, rel=1e-6)


def test_a_bad_policy_grad_clip_string_is_rejected_not_ignored():
    """Silently disabling the bound on a typo is how a run is lost without a trace."""
    from bvr_marl_core.rl.training.decoupled_grad_clip_learner import _bound

    with pytest.raises(ValueError, match="adaptive"):
        _bound(_Config(policy_clip="adaptiv"), POLICY_GRAD_CLIP_KEY)
