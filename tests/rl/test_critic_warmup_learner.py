"""The critic must get the first N iterations of a stage to itself.

Every curriculum promotion warm-starts from the promoted checkpoint, which carries a
value function calibrated to the PREVIOUS stage's return scale. Measured on
a measured curriculum, stage 0 returned a mean of -727 and stage 1 a mean of -65, so the
transferred critic was wrong by ~660 and explained 7-13% of return variance. At 32
gradient updates per training iteration the competent transferred policy
(missile_effectiveness 0.192, above the 0.12 gate) was destroyed in four iterations.

These tests pin the two properties that make the warmup actually work:
  1. only value-head parameters are updated while it is active, and
  2. the freeze covers the SHARED ENCODER -- the failure mode a "skip the policy loss"
     implementation would silently reintroduce.
"""

from __future__ import annotations

import torch

from bvr_marl_core.rl.training.critic_warmup_learner import (
    CRITIC_WARMUP_ACTIVE_KEY,
    CRITIC_WARMUP_ITERATIONS_KEY,
    DEFAULT_CRITIC_WARMUP_ITERATIONS,
    CriticWarmupPPOTorchLearner,
)
from bvr_marl_core.rl.training.kl_floor_learner import KLFloorPPOTorchLearner


class _Module(torch.nn.Module):
    """Minimal stand-in with the real model's shape: one shared encoder, two heads."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.policy_head = torch.nn.Linear(4, 2)
        self.value_head = torch.nn.Linear(4, 1)


class _SeparateValueEncoderModule(torch.nn.Module):
    """The shape produced by `separate_value_encoder: true`.

    `value_encoder` is value-only, so -- exactly like RLlib's `critic_encoder` -- it
    must TRAIN during warmup rather than be frozen with the actor.
    """

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

    def peek(self, key, default=None):
        return self.logged.get(key, default)


class _Config:
    def __init__(self, warmup=None):
        self.learner_config_dict = {}
        if warmup is not None:
            self.learner_config_dict[CRITIC_WARMUP_ITERATIONS_KEY] = warmup


def _learner(warmup=None):
    """A learner with only the state the warmup path touches."""
    obj = object.__new__(CriticWarmupPPOTorchLearner)
    obj._critic_warmup_iterations_seen = 0
    obj._critic_warmup_value_params = {}
    obj.metrics = _Metrics()
    # `module` is a read-only property backed by `_module`.
    obj._module = {"attacker_policy": _Module()}
    # postprocess_gradients_for_module chains to super(); stub it out so these tests
    # isolate the masking rather than RLlib's global-norm clip.
    obj._super_calls = []
    return obj, _Config(warmup)


def _grads(module):
    return {p: torch.ones_like(p) for p in module.parameters()}


def _run(learner, config, module_id="attacker_policy", monkeypatch=None):
    grads = _grads(learner.module[module_id])
    # Bypass the RLlib base implementation, which needs a full Learner.
    import bvr_marl_core.rl.training.critic_warmup_learner as mod

    original = KLFloorPPOTorchLearner.postprocess_gradients_for_module
    try:
        KLFloorPPOTorchLearner.postprocess_gradients_for_module = (
            lambda self, *, module_id, config, module_gradients_dict: module_gradients_dict
        )
        return mod.CriticWarmupPPOTorchLearner.postprocess_gradients_for_module(
            learner,
            module_id=module_id,
            config=config,
            module_gradients_dict=grads,
        )
    finally:
        KLFloorPPOTorchLearner.postprocess_gradients_for_module = original


def _named(module):
    return {id(p): n for n, p in module.named_parameters()}


def test_only_the_value_head_is_updated_during_warmup():
    learner, config = _learner(warmup=5)
    learner.before_gradient_based_update(timesteps={})

    out = _run(learner, config)
    names = _named(learner.module["attacker_policy"])

    for param, grad in out.items():
        name = names[id(param)]
        if name.startswith("value_head."):
            assert grad.abs().sum() > 0, f"{name} must still train"
        else:
            assert grad.abs().sum() == 0, f"{name} must be frozen during warmup"


def test_the_shared_encoder_is_frozen_too():
    """The corruption route a 'skip the policy loss' implementation would leave open.

    The encoder feeds both heads, so value-loss gradients reach it. If it is not frozen
    the value fit still moves the features the policy depends on -- the same damage, by
    a quieter route.
    """
    learner, config = _learner(warmup=5)
    learner.before_gradient_based_update(timesteps={})

    out = _run(learner, config)
    names = _named(learner.module["attacker_policy"])
    encoder = [g for p, g in out.items() if names[id(p)].startswith("encoder.")]

    assert encoder, "test would pass vacuously without encoder params"
    assert all(g.abs().sum() == 0 for g in encoder)


def test_everything_trains_again_once_warmup_ends():
    learner, config = _learner(warmup=3)
    for _ in range(5):
        learner.before_gradient_based_update(timesteps={})

    out = _run(learner, config)

    assert all(g.abs().sum() > 0 for g in out.values())


def test_warmup_covers_exactly_the_configured_iterations():
    learner, config = _learner(warmup=3)
    names = None
    frozen_on = []
    for it in range(1, 7):
        learner.before_gradient_based_update(timesteps={})
        out = _run(learner, config)
        names = names or _named(learner.module["attacker_policy"])
        enc = [g for p, g in out.items() if names[id(p)].startswith("encoder.")]
        frozen_on.append(all(g.abs().sum() == 0 for g in enc))

    assert frozen_on == [True, True, True, False, False, False]


def test_zero_disables_the_warmup_entirely():
    learner, config = _learner(warmup=0)
    learner.before_gradient_based_update(timesteps={})

    out = _run(learner, config)

    assert all(g.abs().sum() > 0 for g in out.values())


def test_default_is_applied_when_the_stage_does_not_set_one():
    learner, config = _learner(warmup=None)

    assert learner._warmup_length(config) == DEFAULT_CRITIC_WARMUP_ITERATIONS


def test_a_module_without_a_value_head_degrades_to_stock_ppo():
    """Failing open beats failing closed: an empty match would train nothing at all."""
    learner, config = _learner(warmup=5)
    learner._module["attacker_policy"] = torch.nn.Linear(4, 2)
    learner.before_gradient_based_update(timesteps={})

    out = _run(learner, config)

    assert all(g.abs().sum() > 0 for g in out.values())


def test_warmup_state_is_visible_in_metrics():
    """Otherwise a warmup and a stalled run look identical in progress.csv."""
    learner, config = _learner(warmup=2)
    learner.before_gradient_based_update(timesteps={})
    _run(learner, config)

    assert learner.metrics.peek(("attacker_policy", CRITIC_WARMUP_ACTIVE_KEY)) == 1.0

    for _ in range(3):
        learner.before_gradient_based_update(timesteps={})
    _run(learner, config)

    assert learner.metrics.peek(("attacker_policy", CRITIC_WARMUP_ACTIVE_KEY)) == 0.0


def test_it_still_bounds_the_kl_coefficient():
    """The warmup is additive: the KL floor that v13 needed must survive."""
    assert issubclass(CriticWarmupPPOTorchLearner, KLFloorPPOTorchLearner)


def test_separate_value_encoder_trains_during_warmup():
    """Regression: `value_encoder` was frozen with the actor, crippling the warmup.

    `separate_value_encoder` gives the critic its own trunk, which is the large majority
    of its parameters -- the value HEAD alone is a single Linear. Freezing the trunk left
    the warmup refitting ~2% of the critic while `critic_warmup_active` still logged 1.0,
    so it looked like it was working.
    """
    learner, config = _learner(warmup=5)
    learner._module = {"attacker_policy": _SeparateValueEncoderModule()}
    learner.before_gradient_based_update(timesteps={})

    out = _run(learner, config)
    names = _named(learner.module["attacker_policy"])

    trained = {names[id(p)] for p, g in out.items() if g.abs().sum() > 0}
    assert any(n.startswith("value_encoder.") for n in trained), (
        "the separate value encoder is value-only and must train during warmup"
    )
    assert any(n.startswith("value_head.") for n in trained)
    assert not any(n.startswith(("encoder.", "policy_head.")) for n in trained), (
        "the actor must still be frozen"
    )
