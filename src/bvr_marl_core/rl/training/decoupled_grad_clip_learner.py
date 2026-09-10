"""PPO learner that clips the actor's and the critic's gradients separately.

RLlib computes ONE global norm over every parameter of an RLModule and rescales them
all by ``grad_clip / ||g||``. With unnormalized BVR rewards the critic's gradient is
legitimately enormous -- ``vf_loss`` runs to O(1e3) because value targets are O(100-500)
-- so that single norm is the critic's norm, and the actor is rescaled by it.

Measured over a 500-iteration stage with ``grad_clip: 0.5``:

    gradients_default_optimizer_global_norm   median 489,312   min 930   max 10,357,426
    resulting scale factor 0.5/||g||          median 1.0e-06   range 4.8e-08 .. 5.4e-04
    mean_kl_loss                              median 9.2e-08   (target_kl is 1e-2)
    vf_explained_var                          -0.10 -> 0.62    (the critic was fine)
    entropy                                   2.763 -> 3.209   (monotonic, coeff at floor)

A ``mean_kl`` five orders of magnitude below target is a policy that is not moving. The
critic learned; the actor was rescaled into the noise floor, and the entropy bonus --
acting on ``log_std``, which is a bare parameter rather than something behind the trunk
-- was the only term still able to move it. That is the "entropy climbing while every
combat metric decays" signature, arriving through the optimizer rather than through the
reward.

Why Adam does not save you here
-------------------------------
The standing objection is that Adam's update ``lr * m_hat/(sqrt(v_hat)+eps)`` is
invariant to multiplying every gradient by a constant, so a global-norm clip cannot
starve anything. That is true for a CONSTANT factor. The factor above moves over four
orders of magnitude between consecutive iterations, and ``m`` and ``v`` are running
averages ACROSS iterations: they end up accumulating differently-scaled gradients, so
``m_hat/sqrt(v_hat)`` is not invariant and the ratio collapses. Check the norm's
VARIANCE before reusing the scale-invariance argument.

Why not just raise ``grad_clip``
--------------------------------
Raising it to sit above the critic's typical norm is the same thing as removing the
bound, and the bound is doing real work: the encoder is shared (or, with
``separate_value_encoder``, the optimizer still is), and one diverged batch at a value
scale of O(500) moves weights far enough to undo the policy. PPO's ratio clip does not
protect against that, because the damage arrives through the parameters rather than
through the objective. So keep a bound on each -- just not the SAME bound, computed
over the critic's magnitudes and applied to the actor.

``separate_value_encoder`` is NOT a substitute for this. It stops the value gradient
reaching the policy ENCODER's parameters; it does not stop the value gradient dominating
the single global NORM those parameters are then rescaled by.

Set ``policy_grad_clip`` (and optionally ``value_grad_clip``) in
``learner_config_dict``. Leave ``policy_grad_clip`` unset for stock single-norm
behaviour.

Measured effect
---------------
A 30-iteration self-play smoke run under this learner, post-warmup:

    gradients_policy_global_norm   16.3 .. 318.3   (clip factor 0.031 .. 0.614 at 10.0)
    gradients_value_global_norm    median 146
    mean_kl_loss                   median 9.7e-04  (shared norm: 9.2e-08)

The actor bound still binds nearly every step. The point is not that clipping stopped,
but that its factor is now set by the actor's own gradient -- a ~20x spread -- instead
of the critic's ~11,000x one.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable
from typing import Any

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.utils.typing import ModuleID

from bvr_marl_core.rl.training.critic_warmup_learner import (
    CriticWarmupPPOTorchLearner,
    _is_value_param,
)

logger = logging.getLogger(__name__)

POLICY_GRAD_CLIP_KEY = "policy_grad_clip"
"""Global-norm bound applied to the ACTOR's parameters alone.

Read from ``learner_config_dict``. ``None``/absent disables decoupled clipping entirely
and restores RLlib's single-norm behaviour, so this file is inert unless asked for."""

VALUE_GRAD_CLIP_KEY = "value_grad_clip"
"""Global-norm bound applied to the CRITIC's parameters alone.

Defaults to ``config.grad_clip`` -- the critic is what the existing bound was actually
sized against, since it always dominated the norm."""

POLICY_GRAD_NORM_KEY = "gradients_policy_global_norm"
VALUE_GRAD_NORM_KEY = "gradients_value_global_norm"
"""Logged separately so the two are legible.

The stock ``gradients_default_optimizer_global_norm`` reports their combination, which
is why the actor being rescaled by the critic was invisible for so long."""


ADAPTIVE = "adaptive"
"""``policy_grad_clip: adaptive`` -- track the actor's own norm instead of guessing it.

A CONSTANT actor bound cannot work, and the follow-on to the measurement in the module
docstring is why. ``policy_grad_clip: 10.0`` was calibrated on the 30-iteration smoke run
quoted below, where the actor norm was 16-318. Over 650 real iterations of
a self-play run it was:

    |g|policy   median 669   p90 3087   max 23730   -> clip factor 4.2e-04 .. 0.71

    band          mean_kl median     mean_kl EXACTLY 0.0
    it   0- 50       5.4e-04             0/50
    it 150-300       3.0e-05             0/150
    it 300-450       3.9e-07            31/150
    it 450-650       0.00e+00          117/200

So the bound was ~65x too tight and reproduced the original pathology one level down --
`mean_kl` of exactly 0.0 means the updated policy is bit-identical to the sampling one.
Two further facts a single constant cannot accommodate:

* the norm GROWS by orders of magnitude as training proceeds, so any value calibrated
  early is wrong later;
* it differs ~30x BETWEEN the two self-play policies at the same iteration (attacker
  median 646, defender 58), and whichever one is large gets clipped into a null update
  while its opponent trains freely. Only one policy freezes at a time and which one it
  is flips between runs -- that is how self-play silently became one-sided.

Adaptive mode keeps a geometric running mean of the actor's own norm, per module, and
bounds it at ``policy_grad_clip_multiplier`` times that. Two properties follow:

* the factor is 1.0 (no clipping at all) on the ordinary majority of steps, so Adam's
  scale-invariance actually holds and only genuine outlier spikes are rescaled;
* the reference is per-module and moves with training, so neither the 30x asymmetry nor
  the growth over 650 iterations can strand it.

Geometric, not arithmetic: these norms are heavy-tailed over orders of magnitude
(median 669, max 23730), so a plain mean is dragged around by the spikes it exists to
catch. The EMA runs on ``log(norm)``.
"""

DEFAULT_ADAPTIVE_MULTIPLIER = 3.0
"""Bound = this times the running geometric mean of the actor's norm.

At the measured distribution (median 669) this puts the bound near 2000: the p90 of 3087
is scaled by 0.65 and the 23730 spike by 0.084, leaving a ~12x factor spread instead of
700x, and everything at or below the typical norm passes through untouched."""

DEFAULT_ADAPTIVE_EMA = 0.01
"""Weight on each new observation, i.e. a ~100-iteration horizon.

Slow enough that one batch cannot move the bound it is about to be clipped by -- which
would make the clip self-fulfilling -- and fast enough to follow the order-of-magnitude
growth seen across a real stage."""

ADAPTIVE_BOUND_KEY = "gradients_policy_adaptive_bound"
"""The bound actually applied this iteration. Logged so a run records the schedule it
chose rather than leaving it to be inferred from the norm."""

MULTIPLIER_KEY = "policy_grad_clip_multiplier"
EMA_KEY = "policy_grad_clip_ema"


def _learner_cfg(config: PPOConfig) -> dict:
    return getattr(config, "learner_config_dict", None) or {}


def _is_adaptive(config: PPOConfig) -> bool:
    raw = _learner_cfg(config).get(POLICY_GRAD_CLIP_KEY)
    return isinstance(raw, str) and raw.strip().lower() == ADAPTIVE


def _positive(config: PPOConfig, key: str, default: float) -> float:
    try:
        value = float(_learner_cfg(config).get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0.0 else default


def _bound(config: PPOConfig, key: str) -> float | None:
    raw = _learner_cfg(config).get(key)
    if raw is None:
        return None
    if isinstance(raw, str):
        # "adaptive" is resolved per module in postprocess_gradients_for_module; any
        # other string is a config error and must not silently disable the bound.
        if raw.strip().lower() == ADAPTIVE:
            return None
        raise ValueError(f"{key}: expected a positive number, null, or {ADAPTIVE!r}; got {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


class DecoupledGradClipPPOTorchLearner(CriticWarmupPPOTorchLearner):
    """Clip actor and critic by their own global norms rather than a shared one."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._decoupled_policy_params: dict[ModuleID, set[int]] = {}
        # Per-module geometric running mean of the actor's UNCLIPPED norm, as log(norm).
        self._policy_norm_log_ema: dict[ModuleID, float] = {}

    def _adaptive_policy_bound(
        self, module_id: ModuleID, observed_norm: float, config: PPOConfig
    ) -> float | None:
        """Bound for this module this iteration, from its own norm history.

        Updated with the norm as OBSERVED BEFORE clipping, so the reference tracks what
        the actor's gradient actually is rather than what the previous bound left of it --
        a bound fed its own clipped output would ratchet itself down to nothing, which is
        the failure being fixed.
        """
        if not (observed_norm > 0.0) or not math.isfinite(observed_norm):
            # A zero/NaN norm carries no scale information. Keep the current reference.
            previous = self._policy_norm_log_ema.get(module_id)
            if previous is None:
                return None
            return math.exp(previous) * _positive(
                config, MULTIPLIER_KEY, DEFAULT_ADAPTIVE_MULTIPLIER
            )

        alpha = min(max(_positive(config, EMA_KEY, DEFAULT_ADAPTIVE_EMA), 0.0), 1.0)
        log_norm = math.log(observed_norm)
        previous = self._policy_norm_log_ema.get(module_id)
        # Seed from the first observation rather than from 0.0 (norm 1.0), which would
        # otherwise take a whole horizon to walk up to a norm in the hundreds while
        # clipping everything to ~3.0 in the meantime.
        updated = log_norm if previous is None else (1.0 - alpha) * previous + alpha * log_norm
        self._policy_norm_log_ema[module_id] = updated
        return math.exp(updated) * _positive(config, MULTIPLIER_KEY, DEFAULT_ADAPTIVE_MULTIPLIER)

    @staticmethod
    def _global_norm(grads: dict[Hashable, Any]) -> float:
        """L2 norm over every gradient tensor, matching RLlib's global_norm convention."""
        total = 0.0
        for grad in grads.values():
            if grad is None:
                continue
            try:
                total += float(grad.detach().norm(2).item() ** 2)
            except Exception:
                continue
        return math.sqrt(total)

    def _policy_param_ids(self, module_id: ModuleID) -> set[int]:
        """Identity set of the parameters that are NOT the value function.

        Complement of the critic-warmup rule, so the two can never disagree about which
        tensors belong to the critic.
        """
        cached = self._decoupled_policy_params.get(module_id)
        if cached is not None:
            return cached
        module = self.module[module_id]
        named = getattr(module, "named_parameters", None)
        ids = (
            {id(p) for name, p in named() if not _is_value_param(name)}
            if callable(named)
            else set()
        )
        self._decoupled_policy_params[module_id] = ids
        return ids

    def postprocess_gradients_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig | None = None,
        module_gradients_dict: dict[Hashable, Any],
    ) -> dict[Hashable, Any]:
        adaptive = config is not None and _is_adaptive(config)
        policy_clip = _bound(config, POLICY_GRAD_CLIP_KEY) if config is not None else None
        if config is None or (policy_clip is None and not adaptive):
            # Not configured: behave exactly like the parent.
            return super().postprocess_gradients_for_module(
                module_id=module_id,
                config=config,
                module_gradients_dict=module_gradients_dict,
            )

        # Same order the parent documents: mask first, so a frozen actor's gradients
        # cannot inflate the norm the critic is then scaled by.
        self._apply_critic_warmup_mask(
            module_id=module_id, config=config, module_gradients_dict=module_gradients_dict
        )

        value_clip = _bound(config, VALUE_GRAD_CLIP_KEY)
        if value_clip is None:
            value_clip = float(config.grad_clip) if config.grad_clip else None

        policy_ids = self._policy_param_ids(module_id)
        if not policy_ids:
            # Never silently: an empty split would clip everything as "the critic" and
            # look like a working run. Fall back to the stock single norm.
            logger.warning(
                "decoupled grad clip: no policy parameters matched in module %r; "
                "falling back to the single global-norm clip.",
                module_id,
            )
            return super().postprocess_gradients_for_module(
                module_id=module_id,
                config=config,
                module_gradients_dict=module_gradients_dict,
            )

        policy_grads = {p: g for p, g in module_gradients_dict.items() if id(p) in policy_ids}
        value_grads = {p: g for p, g in module_gradients_dict.items() if id(p) not in policy_ids}

        if adaptive and policy_grads:
            # Measure BEFORE clipping: the reference must track the actor's real gradient.
            observed = self._global_norm(policy_grads)
            policy_clip = self._adaptive_policy_bound(module_id, observed, config)
            if policy_clip is not None:
                self.metrics.log_value((module_id, ADAPTIVE_BOUND_KEY), policy_clip, window=1)

        clip = self._get_clip_function()
        for grads, bound, metric_key in (
            (policy_grads, policy_clip, POLICY_GRAD_NORM_KEY),
            (value_grads, value_clip, VALUE_GRAD_NORM_KEY),
        ):
            if not grads or bound is None:
                continue
            norm = clip(grads, grad_clip=bound, grad_clip_by="global_norm")
            if norm is not None:
                self.metrics.log_value((module_id, metric_key), norm, window=1)

        # `clip` mutates in place, but the dicts above are copies; write the clipped
        # tensors back so the caller sees them.
        module_gradients_dict.update(policy_grads)
        module_gradients_dict.update(value_grads)

        self._log_critic_warmup_active(module_id=module_id, config=config)
        return module_gradients_dict
