"""PPO learner that bounds RLlib's adaptive KL coefficient.

RLlib's adaptive KL controller is a pure multiplicative walk with no bounds
(``ppo_torch_learner._update_module_kl_coeff``)::

    if kl_loss > 2.0 * config.kl_target:
        curr_var.data *= 1.5
    elif kl_loss < 0.5 * config.kl_target:
        curr_var.data *= 0.5

Nothing stops the downward branch. BVR runs sit orders of magnitude below
``kl_target`` almost every iteration, so the halving branch fires nearly every
iteration and ``kl_coeff`` decays geometrically to zero within a few dozen
iterations. Once there, the KL penalty term contributes nothing and PPO is left
with only the ratio clip.

That matters because the measured KL is not *uniformly* small, it is spiky. The
coefficient cannot respond to a spike it has already underflowed away from:
climbing back at 1.5x per iteration takes tens of iterations, long after the
update that needed braking has landed. The observed consequence is the policy
diffusing rather than sharpening.

Flooring the coefficient keeps a live trust-region penalty at all times, so a KL
spike is damped when it happens instead of many iterations later. The ceiling
guards the opposite end: 1.5x growth off a large spike can let the KL term
dominate the total loss and stall policy progress outright.

This bounds only the *coefficient*. It does not change the loss form, and with
``kl_coeff_floor: 0.0`` the behaviour is exactly stock RLlib.
"""

from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_CURR_KL_COEFF_KEY, PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.learner import ENTROPY_KEY
from ray.rllib.utils.typing import ModuleID

# Keys read out of ``AlgorithmConfig.learner_config_dict``. That dict is the
# supported channel for passing custom values to a Learner: PPOConfig has no
# attribute for these, and subclassing the config to add one would break
# checkpoint restore against stock RLlib configs.
KL_COEFF_FLOOR_KEY = "kl_coeff_floor"
KL_COEFF_CEILING_KEY = "kl_coeff_ceiling"

ENTROPY_PINNED_AXES_OFFSET_KEY = "entropy_pinned_axes_offset"
"""Entropy contributed by action axes the policy does not control, to be reported out.

RLlib sums entropy over the FULL action vector. When automation pins some axes to a
near-deterministic log-std, each pinned axis adds a large negative constant, so a
perfectly healthy policy reports a large negative entropy. The term has zero gradient
(pinned axes are constants, not parameters), so removing it changes the reported number
and nothing else. Set by ``config_builder``; 0.0 disables the correction."""

# Small enough to leave the ratio clip as the primary trust region on a healthy
# run, large enough that a KL spike is penalised the iteration it appears.
# kl_coeff * kl_loss at floor and a 1.0 KL spike costs 3e-3 of loss against a
# policy loss of O(1e-2), so it brakes without dominating.
DEFAULT_KL_COEFF_FLOOR = 0.003

# The adaptive controller multiplies by 1.5 per iteration while KL stays above
# 2x target. Unbounded, a sustained excursion drives kl_coeff into the hundreds
# and the KL penalty swamps the policy gradient. 10.0 is ~50x the 0.2 starting
# value -- room for a genuine correction, short of takeover.
DEFAULT_KL_COEFF_CEILING = 10.0


def _bound_from_config(config: PPOConfig, key: str, default: float) -> float:
    """Read a bound out of ``learner_config_dict``, falling back to the default."""
    raw = (config.learner_config_dict or {}).get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class KLFloorPPOTorchLearner(PPOTorchLearner):
    """``PPOTorchLearner`` with a bounded KL coefficient and interpretable entropy."""

    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: dict,
        fwd_out: dict,
    ):
        """Report entropy over the axes the POLICY controls, not the pinned ones.

        The loss itself is untouched: the pinned axes contribute a constant to
        ``curr_entropy``, and a constant has zero gradient, so the entropy bonus already
        acts only on the learned axes. Only the logged metric is misleading -- a
        slope-based check survives a constant offset, but no reader should have to know
        that a large negative number is a healthy one in disguise.
        """
        loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )

        offset = _bound_from_config(config, ENTROPY_PINNED_AXES_OFFSET_KEY, 0.0)
        if offset:
            reported = self.metrics.peek((module_id, ENTROPY_KEY), default=None)
            if reported is not None:
                # Same key/window as the base class, so this replaces rather than
                # accumulates alongside it.
                self.metrics.log_value((module_id, ENTROPY_KEY), reported - offset, window=1)
        return loss

    def _update_module_kl_coeff(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        kl_loss: float,
    ) -> None:
        # Let RLlib apply its own adaptive step first, then bound the result. This
        # keeps the controller's logic (including its NaN warning) intact and means
        # a future Ray change to the update rule still flows through.
        super()._update_module_kl_coeff(module_id=module_id, config=config, kl_loss=kl_loss)

        floor = _bound_from_config(config, KL_COEFF_FLOOR_KEY, DEFAULT_KL_COEFF_FLOOR)
        ceiling = _bound_from_config(config, KL_COEFF_CEILING_KEY, DEFAULT_KL_COEFF_CEILING)
        # A floor above the ceiling would make the clamp order decide the value.
        # Treat the floor as authoritative and widen the ceiling to match.
        if ceiling < floor:
            ceiling = floor

        curr_var = self.curr_kl_coeffs_per_module[module_id]
        current = float(curr_var.item())
        bounded = min(max(current, floor), ceiling)
        if bounded == current:
            return

        curr_var.data.fill_(bounded)
        # Re-log so the reported curr_kl_coeff is the value the next iteration will
        # actually use, not the pre-clamp one super() just logged.
        self.metrics.log_value(
            (module_id, LEARNER_RESULTS_CURR_KL_COEFF_KEY),
            bounded,
            window=1,
        )
