"""PPO learner that fits the critic before letting the policy move.

When a run warm-starts from a checkpoint trained under a different return scale,
the transfer carries the WHOLE module -- policy heads and ``value_head`` together.
The policy generalises across that boundary; the critic does not, because the value
function is calibrated to the return scale it was trained on. The result is a
competent policy being steered by advantages computed from a badly mis-calibrated
critic, and roughly a hundred gradient updates is enough to destroy it. The failure
is self-sustaining: once the policy stops shooting there are no kills, so there is
no gradient pointing back toward shooting.

The fix is to give the critic the first ``critic_warmup_iterations`` of every run to
itself. While warmup is active the gradient of every parameter EXCEPT the value head
is zeroed, so the policy holds still and the data distribution stays stationary while
the value function refits.

Zeroing gradients (rather than skipping the policy loss) is deliberate: the encoder
may be SHARED between the policy heads and ``value_head``. Merely dropping the policy
loss term would still let value-loss gradients flow back through that trunk and move
the features the policy depends on -- the same corruption by a quieter route.
Selecting on the value head's own parameters is the only version that actually holds
the actor still.

``critic_warmup_iterations: 0`` disables this and restores stock behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.utils.typing import ModuleID

from bvr_marl_core.rl.training.kl_floor_learner import KLFloorPPOTorchLearner

logger = logging.getLogger(__name__)

CRITIC_WARMUP_ITERATIONS_KEY = "critic_warmup_iterations"
"""Training iterations during which only the value head is updated.

Read out of ``AlgorithmConfig.learner_config_dict`` -- the supported channel for passing
custom values to a Learner, since PPOConfig has no attribute for this and subclassing the
config would break checkpoint restore against stock RLlib configs."""

DEFAULT_CRITIC_WARMUP_ITERATIONS = 15
"""Long enough to refit, short enough to be cheap.

The value fit converges within a few iterations even while the policy is moving
underneath it; with the actor frozen the target distribution is stationary, so the
fit is strictly easier than that. 15 leaves a generous margin over the observed
requirement while costing a few percent of a typical run's iteration budget."""

CRITIC_WARMUP_ACTIVE_KEY = "critic_warmup_active"
"""Logged 1.0 while the actor is frozen, 0.0 afterwards.

Without this the warmup is invisible in the training metrics: the policy numbers simply
sit still for N iterations, which reads identically to a stalled run."""

# Parameter names that constitute "the critic" -- the only things trained during warmup.
#
# The rule being encoded is "train what ONLY the value function uses; freeze anything the
# policy depends on". That covers two different architectures:
#
#   shared-encoder model         : one SHARED `encoder` -> `policy_head` + `value_head`.
#       Freezing the encoder is required, since training it moves the policy's features.
#   RLlib default PPO module     : `encoder.actor_encoder` + `encoder.critic_encoder`
#       -> `pi` + `vf`. Here `critic_encoder` is value-only, so it must TRAIN or the
#       critic is crippled -- it is most of the critic's capacity.
#   separate-value-encoder model : `value_encoder` -> `value_head`, alongside the
#       policy's own `encoder`. Same rule as `critic_encoder`: value-only, so it must
#       TRAIN. It was missed when `separate_value_encoder` was introduced, which left
#       the warmup training the value HEAD against a frozen trunk -- roughly 2% of the
#       critic's parameters -- while reporting itself as active.
#
# Matching by name is brittle enough that a miss must be loud rather than silent; see
# `_value_param_ids`. Note the prefixes must stay narrower than the substrings: a bare
# "value" substring would also match `value_encoder` inside a POLICY-shared trunk.
_VALUE_PARAM_PREFIXES = ("value_head.", "value_encoder.", "vf.")
_VALUE_PARAM_SUBSTRINGS = ("critic_encoder",)


def _is_value_param(name: str) -> bool:
    return name.startswith(_VALUE_PARAM_PREFIXES) or any(s in name for s in _VALUE_PARAM_SUBSTRINGS)


class CriticWarmupPPOTorchLearner(KLFloorPPOTorchLearner):
    """``KLFloorPPOTorchLearner`` that freezes the actor for the first N iterations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Counts iterations of THIS Learner. Each curriculum stage runs as its own
        # training run in a fresh process, so "iterations since construction" is exactly
        # "iterations since the stage started" -- which is the window that matters.
        self._critic_warmup_iterations_seen = 0
        self._critic_warmup_value_params: dict[ModuleID, set[int]] = {}

    def _warmup_length(self, config: PPOConfig) -> int:
        raw = (config.learner_config_dict or {}).get(
            CRITIC_WARMUP_ITERATIONS_KEY, DEFAULT_CRITIC_WARMUP_ITERATIONS
        )
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return DEFAULT_CRITIC_WARMUP_ITERATIONS

    def _warmup_active(self, config: PPOConfig) -> bool:
        return self._critic_warmup_iterations_seen <= self._warmup_length(config)

    def _value_param_ids(self, module_id: ModuleID) -> set[int]:
        """Identity set of the value head's parameters, resolved once per module.

        ``TorchLearner.get_param_ref`` returns the parameter tensor itself, so the
        gradient dict is keyed by parameter identity and this is what it can be matched
        against.
        """
        cached = self._critic_warmup_value_params.get(module_id)
        if cached is not None:
            return cached

        module = self.module[module_id]
        named = getattr(module, "named_parameters", None)
        ids: set[int] = set()
        if callable(named):
            ids = {id(param) for name, param in named() if _is_value_param(name)}

        if not ids:
            # Fail OPEN (below), but never quietly: a silent miss turns the warmup into
            # a no-op that looks exactly like a working run, which is how the naming
            # mismatch against RLlib's default module (`vf.*`, not `value_head.*`) got
            # past the unit tests in the first place.
            logger.warning(
                "critic warmup: no value-function parameters matched in module %r "
                "(looked for prefixes %s / substrings %s). The warmup is DISABLED for "
                "this module and training falls back to stock PPO. Parameter names "
                "seen: %s",
                module_id,
                _VALUE_PARAM_PREFIXES,
                _VALUE_PARAM_SUBSTRINGS,
                sorted(n for n, _ in named())[:12] if callable(named) else "<none>",
            )

        self._critic_warmup_value_params[module_id] = ids
        return ids

    def before_gradient_based_update(self, *, timesteps: dict[str, Any]) -> None:
        super().before_gradient_based_update(timesteps=timesteps)
        self._critic_warmup_iterations_seen += 1

    def postprocess_gradients_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig | None = None,
        module_gradients_dict: dict[Hashable, Any],
    ) -> dict[Hashable, Any]:
        """Zero every gradient outside the value head while warmup is active.

        Runs BEFORE ``super()``, so the global-norm clip that ``super()`` applies sees
        the already-masked gradients and its norm reflects the critic alone. Clipping
        first would let the frozen parameters' (large) gradients dominate the norm and
        shrink the critic's own update -- the warmup would still freeze the actor, but
        the critic would learn far more slowly than the un-warmed case.
        """
        self._apply_critic_warmup_mask(
            module_id=module_id, config=config, module_gradients_dict=module_gradients_dict
        )

        grads = super().postprocess_gradients_for_module(
            module_id=module_id,
            config=config,
            module_gradients_dict=module_gradients_dict,
        )

        self._log_critic_warmup_active(module_id=module_id, config=config)
        return grads

    def _apply_critic_warmup_mask(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig | None,
        module_gradients_dict: dict[Hashable, Any],
    ) -> None:
        """Zero every gradient outside the value function, in place, while warming up.

        Split out of ``postprocess_gradients_for_module`` so a subclass that replaces
        the CLIPPING step can still apply the mask first, in the documented order.
        """
        if config is None or not self._warmup_active(config):
            return
        value_ids = self._value_param_ids(module_id)
        # An empty set would silently freeze the entire module and train nothing.
        # Skipping the mask is the safe failure: it degrades to stock PPO rather
        # than to a run that cannot learn at all.
        if not value_ids:
            return
        for param, grad in module_gradients_dict.items():
            if grad is not None and id(param) not in value_ids:
                module_gradients_dict[param] = grad.detach().mul(0.0)

    def _log_critic_warmup_active(self, *, module_id: ModuleID, config: PPOConfig | None) -> None:
        if config is None:
            return
        self.metrics.log_value(
            (module_id, CRITIC_WARMUP_ACTIVE_KEY),
            1.0 if self._warmup_active(config) else 0.0,
            window=1,
        )
