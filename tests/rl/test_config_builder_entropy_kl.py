"""Entropy decay and KL-coefficient bounds -- the two levers that keep a run stable.

Both guard measured training failures:

* ``entropy_coeff`` was flat, so once the advantage signal weakened the entropy
  bonus became the dominant gradient term and entropy rose monotonically instead
  of sharpening (0.774 -> 0.984 over 25 iterations while missile
  effectiveness fell 0.90 -> 0.27 kills/episode).
* ``kl_coeff`` decayed geometrically to nothing (0.2 -> 1.4e-7 by iteration 25,
  3.9e-19 by iteration 64) because RLlib's adaptive controller halves it whenever
  measured KL sits below half the target and never floors the result.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from gymnasium import spaces

pytest.importorskip("ray", reason="ray is not installed in this environment")

from bvr_marl_core.domain import (  # noqa: E402
    PINNED_ACTION_LOG_STD,
    entropy_of_pinned_axes,
)
from bvr_marl_core.rl.training.config_builder import (  # noqa: E402
    DEFAULT_ENTROPY_DECAY_FRACTION,
    DEFAULT_ENTROPY_DECAY_MAX_ITERATIONS,
    DEFAULT_ENTROPY_FINAL_FRACTION,
    build_entropy_coeff_schedule,
    build_ppo_config,
    pinned_axes_entropy_offset,
)
from bvr_marl_core.rl.training.kl_floor_learner import (  # noqa: E402
    DEFAULT_KL_COEFF_CEILING,
    DEFAULT_KL_COEFF_FLOOR,
    ENTROPY_PINNED_AXES_OFFSET_KEY,
    KL_COEFF_CEILING_KEY,
    KL_COEFF_FLOOR_KEY,
    KLFloorPPOTorchLearner,
)

torch = pytest.importorskip("torch", reason="torch is not installed in this environment")


def _build(training: dict | None = None):
    cfg = {
        "env": {},
        "seed": 1,
        "num_env_runners": 2,
        "multi_agent": {"policy_mode": "shared", "shared_policy_id": "shared_policy"},
        "training": dict(training or {}),
    }
    obs = {"shared_policy": spaces.Box(-1, 1, (4,))}
    act = {"shared_policy": spaces.Box(-1, 1, (2,))}
    return build_ppo_config(cfg, obs, act, None, lambda aid, *a, **k: "shared_policy", None)


# --------------------------------------------------------------------------
# Entropy schedule
# --------------------------------------------------------------------------


def test_schedule_decays_from_the_configured_coefficient():
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.02, "steps": 100}, train_batch_size=32768
    )

    assert schedule is not None
    # RLlib's Scheduler requires the first knot to sit at timestep 0.
    assert schedule[0][0] == 0
    assert schedule[0][1] == 0.02
    assert schedule[-1][1] == pytest.approx(0.02 * DEFAULT_ENTROPY_FINAL_FRACTION)


def test_schedule_horizon_tracks_the_planned_run_length():
    # The horizon is sized in ENV STEPS because that is what RLlib's Scheduler keys
    # on (NUM_ENV_STEPS_SAMPLED_LIFETIME), and that counter resets every autotune
    # run -- so it must be derived per run, not per campaign.
    batch = 32768
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.01, "steps": 256}, train_batch_size=batch
    )

    assert schedule[-1][0] == int(256 * DEFAULT_ENTROPY_DECAY_FRACTION * batch)


def test_decay_completes_before_the_run_ends():
    # The tail of the run should train at the final coefficient rather than still be
    # annealing when the run stops, or the last checkpoint is taken mid-decay.
    steps, batch = 256, 32768
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.01, "steps": steps}, train_batch_size=batch
    )

    assert schedule[-1][0] < steps * batch


def test_long_stage_budgets_do_not_stretch_the_exploration_horizon():
    # stage_01 budgets 3072 iterations. Scaling the horizon with that would hold a
    # high coefficient for ~1200 iterations -- the diffusion window the fix exists to
    # close. Exploration length is a property of the task, not of the budget.
    batch = 32768
    long_stage = build_entropy_coeff_schedule(
        {"entropy_coef": 0.02, "steps": 3072}, train_batch_size=batch
    )

    assert long_stage[-1][0] == int(DEFAULT_ENTROPY_DECAY_MAX_ITERATIONS * batch)


def test_short_stage_budgets_still_use_the_fraction():
    # The cap must not become the horizon for every stage: a 256-iteration stage
    # should still finish decaying well inside its own budget.
    batch = 32768
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.01, "steps": 256}, train_batch_size=batch
    )

    assert schedule[-1][0] < int(DEFAULT_ENTROPY_DECAY_MAX_ITERATIONS * batch)


def test_decay_horizon_cap_is_configurable():
    batch = 4096
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.01, "steps": 3072, "entropy_coef_decay_max_iterations": 40},
        train_batch_size=batch,
    )

    assert schedule[-1][0] == 40 * batch


def test_explicit_schedule_wins_over_the_derived_one():
    explicit = [[0, 0.05], [123, 0.0]]
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.01, "entropy_coef_schedule": explicit, "steps": 256},
        train_batch_size=4096,
    )

    assert schedule == explicit


def test_a_stage_opts_out_by_setting_final_equal_to_initial():
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.01, "entropy_coef_final": 0.01, "steps": 256},
        train_batch_size=4096,
    )

    # Still a schedule, but a flat one -- no special sentinel needed.
    assert schedule[0][1] == schedule[-1][1] == 0.01


def test_explicit_final_coefficient_is_honoured():
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.02, "entropy_coef_final": 0.005, "steps": 256},
        train_batch_size=4096,
    )

    assert schedule[-1][1] == 0.005


def test_unschedulable_configs_fall_back_to_a_fixed_coefficient():
    # A zero coefficient has nothing to decay, and a run with no length cannot be
    # sized. Both must yield the plain float rather than a degenerate schedule,
    # because PiecewiseSchedule rejects two knots at the same timestep.
    assert build_entropy_coeff_schedule({"entropy_coef": 0.0}, 4096) == 0.0
    assert build_entropy_coeff_schedule({"entropy_coef": 0.01, "steps": 0}, 4096) == 0.01
    assert build_entropy_coeff_schedule({"entropy_coef": 0.01, "steps": 256}, 0) == 0.01


def test_schedule_rides_on_entropy_coeff_not_the_deprecated_setting():
    # RLlib's new API stack raises "`entropy_coeff_schedule` is deprecated and must
    # be None!" at build time, and PPOLearner feeds `entropy_coeff` straight into a
    # Scheduler that accepts a schedule list. Plumbing the old setting would turn
    # every stage that enabled decay into an immediate build failure.
    config = _build({"entropy_coef": 0.02, "steps": 128})

    assert config.entropy_coeff_schedule is None
    assert isinstance(config.entropy_coeff, list)


def test_degenerate_decay_fraction_cannot_collapse_the_schedule():
    schedule = build_entropy_coeff_schedule(
        {"entropy_coef": 0.01, "steps": 256, "entropy_coef_decay_fraction": 0.0},
        train_batch_size=4096,
    )

    # Two knots at timestep 0 would raise inside RLlib at build time.
    assert schedule[-1][0] > 0


def test_built_config_carries_the_decaying_schedule():
    config = _build({"entropy_coef": 0.02, "steps": 128})

    assert config.entropy_coeff[0][1] == 0.02
    assert config.entropy_coeff[-1][1] < 0.02


# --------------------------------------------------------------------------
# KL coefficient bounds
# --------------------------------------------------------------------------


def test_build_installs_the_bounded_learner_and_its_thresholds():
    config = _build()

    # The installed learner is the critic-warmup subclass; it IS a KLFloorPPOTorchLearner,
    # so the KL bounds this test covers still apply.
    assert issubclass(config.learner_class, KLFloorPPOTorchLearner)
    assert config.learner_config_dict[KL_COEFF_FLOOR_KEY] == DEFAULT_KL_COEFF_FLOOR
    assert config.learner_config_dict[KL_COEFF_CEILING_KEY] == DEFAULT_KL_COEFF_CEILING


def test_kl_bounds_are_configurable_per_stage():
    config = _build({"kl_coeff_floor": 0.01, "kl_coeff_ceiling": 5.0})

    assert config.learner_config_dict[KL_COEFF_FLOOR_KEY] == 0.01
    assert config.learner_config_dict[KL_COEFF_CEILING_KEY] == 5.0


# --------------------------------------------------------------------------
# Reported entropy excludes automation-pinned axes
# --------------------------------------------------------------------------


def test_pinned_axes_offset_matches_the_closed_form():
    # 5 pinned axes at log_std -6.0. Each contributes 0.5*log(2*pi*e) + log_std.
    expected = 5 * (0.5 * math.log(2 * math.pi * math.e) + PINNED_ACTION_LOG_STD)

    assert entropy_of_pinned_axes(5) == pytest.approx(expected)
    assert entropy_of_pinned_axes(5) == pytest.approx(-22.905, abs=1e-3)


def test_offset_restores_the_number_a_reader_expects():
    # The measured case: a run reported entropy -19.221 on a policy whose
    # 4 learned axes sat at the -0.5 init log-std, i.e. a healthy 3.68.
    reported_by_rllib = -19.221
    healthy_active_entropy = 4 * (0.5 * math.log(2 * math.pi * math.e) - 0.5)

    corrected = reported_by_rllib - entropy_of_pinned_axes(5)

    assert corrected == pytest.approx(healthy_active_entropy, abs=0.01)


def test_offset_is_derived_from_the_model_config():
    cfg = {"model": {"model_config": {"active_indices": [0, 1, 2, 3], "full_action_dim": 9}}}

    assert pinned_axes_entropy_offset(cfg) == pytest.approx(entropy_of_pinned_axes(5))


def test_no_offset_when_the_policy_controls_every_axis():
    # Nothing is pinned, so there is nothing to subtract and the correction must be
    # inert rather than silently shifting a correct number.
    cfg = {"model": {"model_config": {"active_indices": list(range(9)), "full_action_dim": 9}}}

    assert pinned_axes_entropy_offset(cfg) == 0.0


def test_no_offset_when_the_model_config_is_silent():
    assert pinned_axes_entropy_offset({}) == 0.0
    assert pinned_axes_entropy_offset({"model": {"model_config": {}}}) == 0.0


def test_built_config_carries_the_offset_for_the_learner():
    cfg = {
        "env": {},
        "seed": 1,
        "num_env_runners": 2,
        "multi_agent": {"policy_mode": "shared", "shared_policy_id": "shared_policy"},
        "training": {},
        "model": {"model_config": {"active_indices": [0, 1, 2, 3], "full_action_dim": 9}},
    }
    obs = {"shared_policy": spaces.Box(-1, 1, (4,))}
    act = {"shared_policy": spaces.Box(-1, 1, (2,))}
    config = build_ppo_config(cfg, obs, act, None, lambda aid, *a, **k: "shared_policy", None)

    assert config.learner_config_dict[ENTROPY_PINNED_AXES_OFFSET_KEY] == pytest.approx(
        entropy_of_pinned_axes(5)
    )


class _Metrics:
    """Stand-in for Learner.metrics; records the last value logged per key."""

    def __init__(self):
        self.logged = {}

    def log_value(self, key, value, **kwargs):
        self.logged[key] = value


def _learner(initial_kl_coeff: float, bounds: dict | None = None):
    """A KLFloorPPOTorchLearner with only the state _update_module_kl_coeff touches.

    Built via __new__ because a real Learner needs an RLModule, an optimizer and a
    device. The method under test reads exactly three attributes, and going through
    the real class (rather than a duck-typed fake) keeps the zero-arg super() call
    in the override bound to the genuine RLlib implementation.
    """
    learner = object.__new__(KLFloorPPOTorchLearner)
    learner.curr_kl_coeffs_per_module = {"p": torch.tensor(initial_kl_coeff)}
    learner.metrics = _Metrics()
    config = SimpleNamespace(
        kl_target=0.01,
        learner_config_dict=bounds if bounds is not None else {},
    )
    return learner, config


def _update(learner, config, kl_loss):
    learner._update_module_kl_coeff(module_id="p", config=config, kl_loss=kl_loss)
    return float(learner.curr_kl_coeffs_per_module["p"].item())


def test_tiny_kl_no_longer_decays_the_coefficient_to_zero():
    # The exact failure: measured KL ~1e-5 against kl_target 0.01 trips RLlib's
    # halving branch every iteration, with nothing to stop it.
    learner, config = _learner(DEFAULT_KL_COEFF_FLOOR)

    for _ in range(200):
        value = _update(learner, config, kl_loss=1e-5)

    assert value == pytest.approx(DEFAULT_KL_COEFF_FLOOR)


def test_stock_rllib_would_have_collapsed_under_the_same_input():
    # Guards the premise of the fix rather than the fix: if Ray ever bounds the
    # controller itself, this fails and the override can be reconsidered.
    kl_target, kl_loss, coeff = 0.01, 1e-5, 0.2
    for _ in range(64):
        # Verbatim shape of PPOTorchLearner._update_module_kl_coeff: the branch is
        # on the MEASURED KL, so a persistently small KL halves the coeff forever.
        if kl_loss > 2.0 * kl_target:
            coeff *= 1.5
        elif kl_loss < 0.5 * kl_target:
            coeff *= 0.5
    # The campaign measured 3.9e-19 at this iteration count.
    assert coeff < 1e-15


def test_coefficient_still_falls_toward_the_floor():
    # Flooring must not pin the coefficient at its starting value -- the controller
    # still needs to relax when KL is genuinely small.
    learner, config = _learner(0.2)

    first = _update(learner, config, kl_loss=1e-5)

    assert first < 0.2
    assert first >= DEFAULT_KL_COEFF_FLOOR


def test_coefficient_still_rises_on_a_kl_spike():
    # 1.26 is the measured iteration-1 KL from run-0002.
    learner, config = _learner(DEFAULT_KL_COEFF_FLOOR)

    raised = _update(learner, config, kl_loss=1.26)

    assert raised > DEFAULT_KL_COEFF_FLOOR


def test_sustained_spike_cannot_let_the_kl_term_take_over():
    learner, config = _learner(0.2)

    for _ in range(100):
        value = _update(learner, config, kl_loss=1.26)

    assert value == pytest.approx(DEFAULT_KL_COEFF_CEILING)


def test_reported_coefficient_matches_the_bounded_value():
    # super() logs the pre-clamp value; the override must correct it or the metric
    # reports a coefficient the next iteration never uses.
    learner, config = _learner(DEFAULT_KL_COEFF_FLOOR)

    value = _update(learner, config, kl_loss=1e-5)

    assert learner.metrics.logged[("p", "curr_kl_coeff")] == pytest.approx(value)


def test_zero_floor_restores_stock_rllib_behaviour():
    learner, config = _learner(0.2, bounds={KL_COEFF_FLOOR_KEY: 0.0})

    for _ in range(200):
        value = _update(learner, config, kl_loss=1e-5)

    assert value < 1e-15


def test_a_floor_above_the_ceiling_resolves_to_the_floor():
    learner, config = _learner(0.2, bounds={KL_COEFF_FLOOR_KEY: 5.0, KL_COEFF_CEILING_KEY: 1.0})

    assert _update(learner, config, kl_loss=1e-5) == pytest.approx(5.0)


def test_non_numeric_bounds_fall_back_to_the_defaults():
    # learner_config_dict round-trips through YAML and Ray serialisation; a bad
    # value must not take the trust region out with it.
    learner, config = _learner(0.2, bounds={KL_COEFF_FLOOR_KEY: "not-a-number"})

    for _ in range(200):
        value = _update(learner, config, kl_loss=1e-5)

    assert value == pytest.approx(DEFAULT_KL_COEFF_FLOOR)
