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


def _bound(config: PPOConfig, key: str) -> float | None:
    raw = (getattr(config, "learner_config_dict", None) or {}).get(key)
    if raw is None:
        return None
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
        policy_clip = _bound(config, POLICY_GRAD_CLIP_KEY) if config is not None else None
        if config is None or policy_clip is None:
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
