"""PPO configuration builder for BVR combat training."""

import platform
import warnings

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from bvr_marl_core.domain import PINNED_ACTION_LOG_STD, entropy_of_pinned_axes
from bvr_marl_core.rl.environment.spaces.action_space import FULL_ACTION_DIM
from bvr_marl_core.rl.training.critic_warmup_learner import (
    CRITIC_WARMUP_ITERATIONS_KEY,
    DEFAULT_CRITIC_WARMUP_ITERATIONS,
)
from bvr_marl_core.rl.training.decoupled_grad_clip_learner import (
    POLICY_GRAD_CLIP_KEY,
    VALUE_GRAD_CLIP_KEY,
    DecoupledGradClipPPOTorchLearner,
)
from bvr_marl_core.rl.training.kl_floor_learner import (
    DEFAULT_KL_COEFF_CEILING,
    DEFAULT_KL_COEFF_FLOOR,
    ENTROPY_PINNED_AXES_OFFSET_KEY,
    KL_COEFF_CEILING_KEY,
    KL_COEFF_FLOOR_KEY,
)

# RLlib clamps the *squared* value-function error at ``vf_clip_param`` and
# ``torch.clamp`` has ZERO gradient above its bound, so every sample whose value
# error exceeds sqrt(vf_clip_param) trains the critic not at all:
#
#     vf_loss = torch.pow(value_fn_out - value_targets, 2.0)
#     vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
#
# RLlib's default is 10.0 -- an error threshold of ~3.16. BVR rewards are NOT
# normalized and carry terminal magnitudes of 100-200 (kill_reward, destruction and
# boundary penalties), so value targets run to O(100) and squared errors to O(1e5).
# At the default every sample sits outside the clamp for the whole run: the critic
# receives no gradient at all, vf_explained_var pins at -1, GAE advantages become
# noise, and the unopposed entropy bonus diffuses the policy toward uniform random.
# The observable signature is entropy climbing steadily over hundreds of iterations
# while every combat metric decays.
#
# The fix is NOT to disable the clamp. The critic shares its trunk with the policy
# (a shared encoder produces the features that feed both the action-distribution head
# and ``value_head``), and ``grad_clip_by="global_norm"``
# rescales EVERY parameter by ``grad_clip / ||g||``. So an unbounded VF term does not
# stay in the critic: it dominates the global norm and shrinks the policy's share of
# the update toward zero. Measured ``vf_loss_unclipped`` on a boundary-only stage peaked at 4.5e8
# (RMS value error ~21k, i.e. a diverged critic) -- times ``vf_loss_coeff`` that would
# swamp the policy gradient entirely. Removing the clamp trades critic-death for
# policy-death.
#
# Instead size the clamp to the RETURN scale, so it never binds on a legitimate error
# but still caps a diverged critic. Episode returns run to O(500) (kill_reward 200 with
# multiple kills, plus -100 terminal penalties and dense shaping), so a value error of
# ~1000 is already twice anything real: 1000^2 = 1e6. Everything in the legitimate
# range trains at full gradient; only genuine divergence gets bounded.
#
# Configs that normalize rewards can set ``training.vf_clip_param`` back down. Configs
# that raise terminal reward magnitudes should raise this with them.
DEFAULT_VF_CLIP_PARAM = 1.0e6

# The VF clamp bounds the value LOSS; this bounds the resulting UPDATE. It must never be
# ``None``: with no global-norm bound and a value target scale of O(500), a single bad
# batch moves the shared encoder far enough to undo the policy, and PPO's ratio clip
# does not save you because the damage is done through the trunk the critic shares with
# the actor. Every campaign stage YAML already sets 0.5; this makes that the default so
# an entry point that forgets it cannot silently run unbounded.
#
# Do NOT try to fix a stalled policy by raising this. It was tried (0.5 -> 50.0) on the
# theory that a large pre-clip global norm made the resulting small scale factor starve
# the policy. It changes nothing, and cannot: the optimizer is ``torch.optim.Adam``,
# whose update ``lr * m_hat/(sqrt(v_hat)+eps)`` is INVARIANT to multiplying every
# gradient by one scalar -- exactly what global-norm clipping does. Measured over a
# 170x difference in clip coefficient the training curves were superimposable. A small
# ``grad_clip`` here is a bound, not a brake on learning.
DEFAULT_GRAD_CLIP = 0.5

# Bound for the ACTOR's own global norm, used when a config does not set one.
# `grad_clip` continues to bound the critic, which is what it was always measuring.
#
# MEASURED on a 30-iteration self-play smoke run, post-warmup (iterations 15-29):
#
#     policy global norm       16.3 .. 318.3
#     clip factor at 10.0      0.031 .. 0.614      (a ~20x spread)
#     mean_kl_loss             median 9.7e-04      (was 9.2e-08 under the shared norm)
#
# So this DOES bind on essentially every step -- it is a per-step rescale, not the
# outlier-only bound an earlier draft of this comment claimed. What changed is not that
# clipping stopped happening but that the factor is now set by the actor's OWN gradient
# instead of the critic's: a ~20x spread driven by the policy, rather than the ~11,000x
# spread driven by a quantity the policy has nothing to do with. Adam tolerates the
# former (its `m`/`v` averages stay comparable across steps); the latter is what
# collapsed `mean_kl` by five orders of magnitude.
#
# 10.0 is therefore an empirical setting, not a principled ceiling. It produces a healthy
# `mean_kl` an order of magnitude below `kl_target`, so there is room to raise it if the
# policy turns out to move too slowly -- watch `mean_kl_loss` against `kl_target` rather
# than reasoning about the norm.
DEFAULT_POLICY_GRAD_CLIP = 10.0


def _policy_grad_clip(training_cfg: dict) -> float | None:
    """Actor-side global-norm bound, or None to keep RLlib's single shared norm.

    An explicit ``policy_grad_clip: null`` opts a config out (the two-norm split is a
    behaviour change, so it must be possible to reproduce an old run exactly).
    """
    if "policy_grad_clip" in training_cfg:
        raw = training_cfg["policy_grad_clip"]
        return None if raw is None else float(raw)
    return DEFAULT_POLICY_GRAD_CLIP


# The KL the adaptive controller aims for. Left at RLlib's 0.01: BVR runs sit orders
# of magnitude BELOW this (measured 1e-5 to 1e-8), so lowering the target would not
# stop the coefficient from collapsing -- only the floor in KLFloorPPOTorchLearner
# does that. Named here so a stage that wants a tighter trust region has one place
# to look.
DEFAULT_KL_TARGET = 0.01

# A value clamp only protects the critic if it sits ABOVE the squared value error the
# reward scale can legitimately produce. Below that it silently zeroes the critic's
# gradient on exactly the high-return samples that matter, which is expensive to
# discover: it costs whole runs. Terminal rewards bound the return scale, so validate
# the clamp against them at build time rather than finding out 500 iterations in.
_TERMINAL_REWARD_KEYS = (
    "kill_reward",
    "destruction_penalty",
    "boundary_violation_penalty",
    "team_destruction_penalty",
    "team_kill_reward",
)
# Returns accumulate several terminal events plus dense shaping across an episode, so
# the clamp needs headroom over a single terminal magnitude, not parity with it.
_VF_CLIP_HEADROOM = 4.0


def resolve_reward_magnitudes(env_cfg: dict | None) -> dict | None:
    """Pull the terminal-reward block out of an env config.

    Campaign stage YAMLs declare ``env.reward_magnitudes``; the older adaptive-config
    path uses ``env.reward_config``. Both name the same terminal keys, and a validator
    that knows only one of them silently passes every config written in the other.
    """
    if not isinstance(env_cfg, dict):
        return None
    for key in ("reward_magnitudes", "reward_config"):
        block = env_cfg.get(key)
        if isinstance(block, dict) and block:
            return block
    return None


def largest_terminal_magnitude(reward_config: dict | None) -> float:
    """Largest absolute terminal reward, which is what bounds the return scale."""
    if not isinstance(reward_config, dict):
        return 0.0
    return max(
        (
            abs(float(value))
            for key in _TERMINAL_REWARD_KEYS
            if isinstance(value := reward_config.get(key), (int, float))
        ),
        default=0.0,
    )


def required_vf_clip_param(reward_config: dict | None) -> float | None:
    """Smallest ``vf_clip_param`` that still trains the critic at this reward scale.

    Returns ``None`` when the reward config declares no terminal magnitudes, so callers
    can skip the check rather than validate against a fabricated scale.
    """
    largest = largest_terminal_magnitude(reward_config)
    if largest <= 0.0:
        return None
    return (largest * _VF_CLIP_HEADROOM) ** 2


def validate_value_scale(training_cfg: dict, reward_config: dict | None) -> str | None:
    """Return a warning message when the VF clamp is too tight for the reward scale."""
    required = required_vf_clip_param(reward_config)
    if required is None:
        return None
    configured = float(training_cfg.get("vf_clip_param", DEFAULT_VF_CLIP_PARAM))
    if configured >= required:
        return None
    return (
        f"vf_clip_param={configured:g} is below {required:g}, the squared value error a "
        f"reward scale with terminal magnitude {largest_terminal_magnitude(reward_config):g} "
        "produces. torch.clamp has zero gradient above its bound, so the critic will "
        "train on nothing, advantages will become noise, and the entropy bonus will "
        "diffuse the policy toward random. Raise training.vf_clip_param or normalize "
        "rewards."
    )


# A flat ``entropy_coeff`` never stops paying the policy to stay uncertain. In PPO
# the entropy bonus is the one term with a consistent gradient direction regardless
# of returns, so once the advantage signal gets weak -- sparse terminal rewards, a
# noisy critic, or a stage where few actions change the outcome -- it becomes the
# dominant term and entropy rises monotonically instead of sharpening. That is the
# a measured failure (entropy 0.774 -> 0.984 over 25 iterations while
# missile effectiveness fell), and the same shape that ran entropy 0.93 -> 4.70
# during the dead-critic episode. Decaying the coefficient bounds how long
# exploration can outrank the return signal.
#
# The decay completes partway through the run so the tail trains at the final
# coefficient rather than still annealing when the run stops. Sized against how long
# runs ACTUALLY last, not their nominal `steps`: the metric-convergence early stop
# fires from iteration 64 (one run stopped at exactly 64 of a planned 256), and a
# horizon spanning most of the nominal length would leave those runs finishing near
# their starting coefficient -- i.e. still flat, which is the bug. At 0.4 a run that
# stops at 64/256 is already ~60% of the way down.
DEFAULT_ENTROPY_DECAY_FRACTION = 0.4

# Where the schedule lands, as a fraction of the stage's starting coefficient. Not
# zero: some exploration pressure has to survive to the end of a stage or the policy
# collapses onto whatever it found first.
DEFAULT_ENTROPY_FINAL_FRACTION = 0.1

# Ceiling on the decay horizon, in training iterations, applied on top of the
# fraction. Stage budgets span 256 to 3072 iterations, and a pure fraction would let
# stage_01 hold a high coefficient for ~1200 iterations -- far past the point the
# bonus is doing anything useful. The entropy bonus exists to explore until the
# policy finds reward-bearing behaviour, and that horizon is a property of the TASK,
# not of how long we happen to budget: the warmup agents are already firing and
# scoring within the first ~10 iterations. Capping keeps every stage on the same
# exploration timescale instead of scaling it with an unrelated budget.
DEFAULT_ENTROPY_DECAY_MAX_ITERATIONS = 150

# Fallback run length (in training iterations) when a config declares no
# ``training.steps``. Only used to size the schedule horizon.
_DEFAULT_RUN_ITERATIONS = 256


def build_entropy_coeff_schedule(
    training_cfg: dict,
    train_batch_size: int,
) -> float | list[list[float]]:
    """Build the value to pass as RLlib's ``entropy_coeff``.

    Note the return goes to ``entropy_coeff`` itself, NOT to
    ``entropy_coeff_schedule``. On the new API stack that second setting is
    rejected outright ("`entropy_coeff_schedule` is deprecated and must be None!",
    ``algorithm_config._value_error``); ``PPOLearner`` instead feeds ``entropy_coeff``
    straight into a ``Scheduler``, which accepts either a fixed number or a
    ``[[timestep, value], ...]`` schedule. Anything written to the old setting is a
    hard build error, so it must not be plumbed through.

    The schedule keys on ``NUM_ENV_STEPS_SAMPLED_LIFETIME``, linearly interpolates
    between knots, and holds the final value afterwards. That counter resets on
    every autotune run (verified: run-0001 and run-0002 both report ~32.8k steps at
    iteration 1), so the horizon is sized per RUN, not per campaign. Each run
    therefore re-opens exploration at the stage's starting coefficient and anneals
    down again -- warm restarts, which is what we want when a run warm-starts from
    the previous checkpoint and faces a re-tuned reward scale.

    An explicit ``training.entropy_coef_schedule`` wins outright, for configs that
    need a shape this helper does not produce. Setting ``entropy_coef_final`` equal
    to ``entropy_coef`` yields a flat schedule, which is how a stage opts out.

    Falls back to the plain float when no schedule can be sized.
    """
    initial = float(training_cfg.get("entropy_coef", 0.01))

    explicit = training_cfg.get("entropy_coef_schedule")
    if explicit:
        return explicit

    if initial <= 0.0:
        # Nothing to decay; a zero coefficient is already the sharpest setting.
        return initial

    final_raw = training_cfg.get("entropy_coef_final")
    final = (
        float(final_raw)
        if isinstance(final_raw, (int, float))
        else initial * DEFAULT_ENTROPY_FINAL_FRACTION
    )

    iterations = int(training_cfg.get("steps", _DEFAULT_RUN_ITERATIONS) or 0)
    if iterations <= 0 or train_batch_size <= 0:
        return initial

    decay_fraction = float(
        training_cfg.get("entropy_coef_decay_fraction", DEFAULT_ENTROPY_DECAY_FRACTION)
    )
    # Clamp to (0, 1]: a fraction at or below 0 would put both knots at timestep 0,
    # which RLlib's PiecewiseSchedule rejects.
    decay_fraction = min(max(decay_fraction, 1e-3), 1.0)

    max_iterations = float(
        training_cfg.get("entropy_coef_decay_max_iterations", DEFAULT_ENTROPY_DECAY_MAX_ITERATIONS)
    )
    horizon_iterations = min(iterations * decay_fraction, max_iterations)

    horizon = int(horizon_iterations * train_batch_size)
    if horizon <= 0:
        return initial
    return [[0, initial], [horizon, final]]


def pinned_axes_entropy_offset(cfg: dict) -> float:
    """Entropy that automation-pinned action axes add to RLlib's reported entropy.

    Returns 0.0 when the neural wrapper is off or the model config does not say which
    axes the policy controls -- with nothing pinned there is nothing to subtract.
    """
    model_cfg = (cfg.get("model") or {}).get("model_config") or {}
    active = model_cfg.get("active_indices")
    full_dim = model_cfg.get("full_action_dim")
    if not isinstance(active, (list, tuple)) or not isinstance(full_dim, int):
        return 0.0

    pinned = int(full_dim) - len({int(i) for i in active})
    if pinned <= 0:
        return 0.0
    return entropy_of_pinned_axes(pinned, model_cfg.get("inactive_log_std", PINNED_ACTION_LOG_STD))


def print_automation_info(
    automation_level: int,
    wrapped_action_dim: int,
    active_indices: list,
    wrapper_config: dict,
    enable_target_selection: bool,
    enable_gun: bool,
    enable_countermeasures: bool,
):
    """Print information about automation level and action space configuration."""
    print("=" * 80)
    print(f"AUTOMATION LEVEL {automation_level} - Energy + Lift-Vector Control")
    print("=" * 80)
    print(f"  Network controls {wrapped_action_dim} actions:")
    print("    [0] Ps - Specific Energy Rate (climb/dive/accelerate)")
    print("    [1] n  - Normal Load Factor (turn intensity/g-load)")
    print("    [2] phi - Bank Angle (turn direction/roll)")
    print("    [3] Missile firing")
    if 4 in active_indices:
        print("    [4] TARGET SELECTION - choose which enemy to engage")
    print(f"  Remaining {FULL_ACTION_DIM - wrapped_action_dim} actions are NOT policy-controlled:")
    if enable_target_selection:
        print("    [4] Target selection")
    if enable_gun:
        print("    [5] Gun firing")
    if enable_countermeasures:
        # Deliberately not "automation handles these". Core supplies no
        # countermeasure automation: the action wrapper pins every inactive
        # trigger to 0.0 against a 0.5 threshold, so these axes are dead unless
        # an extension fills them -- which the behavior env does only when
        # `env.automate_countermeasures` is set. Claiming otherwise here is how
        # the axes went unnoticed as inert through several campaigns.
        print("    [6-8] Countermeasures: flares, chaff, decoys")
        print("          -> pinned OFF unless the environment automates them")
    if FULL_ACTION_DIM > 10 and 10 not in active_indices:
        print("    [10] EMCON radar on/off (defaults ON)")
    print(f"  Automation behavior: {wrapper_config.get('automation_level', 'balanced')}")
    print("=" * 80)


def configure_automation_level(
    automation_level: int, use_wrapper: bool, model_config_dict: dict, wrapper_config: dict
) -> tuple:
    """
    Configure action space based on automation level.

    Returns:
        Tuple of (wrapped_action_dim, active_indices, enable_target_selection,
                  enable_gun, enable_countermeasures)
    """
    if automation_level == 1:
        # Level 1: Basic flight control (1v1 training)
        wrapped_action_dim = 4
        active_indices = [0, 1, 2, 3]
        enable_target_selection = True
        enable_gun = True
        enable_countermeasures = True
    elif automation_level == 2:
        # Level 2: Add targeting control (multi-agent training)
        wrapped_action_dim = 5
        active_indices = [0, 1, 2, 3, 4]
        enable_target_selection = False  # Network controls it
        enable_gun = True
        enable_countermeasures = True
    elif automation_level == 3:
        # Level 3: Full control (FULL_ACTION_DIM; incl. EMCON toggle only if enabled)
        wrapped_action_dim = FULL_ACTION_DIM
        active_indices = list(range(FULL_ACTION_DIM))
        enable_target_selection = False
        enable_gun = False
        enable_countermeasures = False
    else:
        raise ValueError(f"Invalid automation_level: {automation_level}. Must be 1, 2, or 3.")

    # Update model config
    if use_wrapper and automation_level < 3:
        model_config_dict.update(
            {
                "use_neural_wrapper": True,
                "wrapped_action_dim": wrapped_action_dim,
                "full_action_dim": FULL_ACTION_DIM,
                "active_indices": active_indices,
                "action_dim": wrapped_action_dim,
            }
        )
        print_automation_info(
            automation_level,
            wrapped_action_dim,
            active_indices,
            wrapper_config,
            enable_target_selection,
            enable_gun,
            enable_countermeasures,
        )
    else:
        # Full control mode
        model_config_dict.update(
            {
                "use_neural_wrapper": False,
                "wrapped_action_dim": FULL_ACTION_DIM,
                "full_action_dim": FULL_ACTION_DIM,
                "active_indices": list(range(FULL_ACTION_DIM)),
                "action_dim": FULL_ACTION_DIM,
            }
        )
        print("=" * 80)
        print("AUTOMATION LEVEL 3 - FULL CONTROL MODE")
        print("=" * 80)
        print("  Network controls all 10 actions directly")
        print("=" * 80)

    return (
        wrapped_action_dim,
        active_indices,
        enable_target_selection,
        enable_gun,
        enable_countermeasures,
    )


def _resolve_num_learners(num_learners: int) -> int:
    """Return 0 on Windows: Ray Train uses NCCL for distributed learners, which is Linux-only."""
    if platform.system() == "Windows" and num_learners > 0:
        warnings.warn(
            f"num_learners={num_learners} is not supported on Windows (NCCL unavailable). "
            "Falling back to num_learners=0 (local learner in driver process).",
            stacklevel=3,
        )
        return 0
    return num_learners


def policy_ids_from_cfg(cfg: dict) -> set[str]:
    """Return the policy IDs implied by ``training.multi_agent``."""
    ma_cfg = cfg.get("training", {}).get("multi_agent", {})
    mode = ma_cfg.get("policy_mode", "shared")

    if mode == "shared":
        return {ma_cfg.get("shared_policy_id", "shared_policy")}

    if mode == "team_separate":
        return {
            ma_cfg.get("attacker_policy_id", "attacker_policy"),
            ma_cfg.get("defender_policy_id", "defender_policy"),
        }

    raise ValueError(f"Unknown policy_mode: {mode}")


def policies_to_train_from_cfg(cfg: dict) -> list[str]:
    """Return trainable policy IDs, honoring optional side-freezing flags."""
    ma_cfg = cfg.get("training", {}).get("multi_agent", {})
    mode = ma_cfg.get("policy_mode", "shared")

    if mode == "shared":
        return [ma_cfg.get("shared_policy_id", "shared_policy")]

    if mode == "team_separate":
        policies: list[str] = []
        if bool(ma_cfg.get("train_attacker", True)):
            policies.append(ma_cfg.get("attacker_policy_id", "attacker_policy"))
        if bool(ma_cfg.get("train_defender", True)):
            policies.append(ma_cfg.get("defender_policy_id", "defender_policy"))
        return policies

    raise ValueError(f"Unknown policy_mode: {mode}")


def build_ppo_config(
    cfg: dict,
    obs_spaces: dict,
    act_spaces: dict,
    multi_spec: MultiRLModuleSpec | None,
    policy_mapping_fn,
    episode_callback,
    env_name: str | type = "BVRMultiAgentEnv",
) -> PPOConfig:
    """Build PPO configuration from hydra config.

    Args:
        multi_spec: Custom RLModule spec for the shared policy.  Pass ``None``
                    to use RLlib's default PPO models (no custom network).
        env_name: Registered Ray environment name or importable env class.
                  Defaults to ``"BVRMultiAgentEnv"``. Pass an env class when
                  you want RLlib workers to reconstruct the env without relying
                  on Tune's string registry.
    """
    seed = cfg.get("seed", 42)

    value_scale_warning = validate_value_scale(
        cfg.get("training", {}) or {}, resolve_reward_magnitudes(cfg.get("env"))
    )
    if value_scale_warning is not None:
        warnings.warn(value_scale_warning, RuntimeWarning, stacklevel=2)

    # Use the new API stack only when a custom RLModule spec is provided.
    # Without a custom spec, RLlib falls back to DefaultPPOTorchRLModule which
    # uses PPOCatalog  and PPOCatalog has no encoder for Dict observation spaces.
    # The old API stack has built-in preprocessors that flatten Dict obs to a Box.
    use_new_api = multi_spec is not None
    training_cfg = cfg.get("training", {})
    num_env_runners = cfg.get(
        "num_env_runners",
        training_cfg.get("num_env_runners", training_cfg.get("n_envs", 30)),
    )
    num_envs_per_env_runner = cfg.get(
        "num_envs_per_env_runner",
        training_cfg.get("num_envs_per_env_runner", 1),
    )
    rollout_fragment_length = cfg.get(
        "rollout_fragment_length",
        training_cfg.get("rollout_fragment_length", "auto"),
    )
    train_batch_size = training_cfg.get("train_batch_size", training_cfg.get("batch_size", 4096))
    minibatch_size = training_cfg.get("sgd_minibatch_size", 128)
    policies = policy_ids_from_cfg(cfg)
    policies_to_train = policies_to_train_from_cfg(cfg)

    # Env-runner fault tolerance. RLlib puts the (large, custom-RLModule) config in
    # the object store and passes it as a constructor arg to each env-runner actor
    # (EnvRunnerGroup: ray.put(config) + max_restarts = max_num_env_runner_restarts
    # if restart_failed_env_runners else 0). With RLlib's default
    # restart_failed_env_runners=True the actor gets max_restarts>0, so if that
    # object is lost the restart hangs indefinitely and strands the whole
    # run/campaign (Ray issue #53727, observed on the single-node local setup).
    # Default to NOT restarting (-> actor max_restarts=0, no hanging restart) and
    # instead tolerate/skip a failed runner so a transient failure degrades the run
    # rather than hanging it. A multi-node cluster can re-enable restart via config.
    restart_failed_env_runners = bool(
        cfg.get(
            "restart_failed_env_runners",
            training_cfg.get("restart_failed_env_runners", False),
        )
    )

    ppo_config = (
        PPOConfig()
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=use_new_api,
            enable_env_runner_and_connector_v2=use_new_api,
        )
        .environment(
            env=env_name,
            env_config=cfg["env"],
            normalize_actions=False,
            clip_actions=False,
        )
        .debugging(seed=seed)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="truncate_episodes",
            sample_timeout_s=600.0,
        )
        .fault_tolerance(
            restart_failed_env_runners=restart_failed_env_runners,
            # When not restarting, drop a failed runner and keep training instead of
            # crashing the run. Bounded probe/restore timeouts cap any stall (RLlib
            # defaults env_runner_restore_timeout_s to 1800s) if a cluster re-enables
            # restart.
            ignore_env_runner_failures=not restart_failed_env_runners,
            env_runner_health_probe_timeout_s=float(
                training_cfg.get("env_runner_health_probe_timeout_s", 30.0)
            ),
            env_runner_restore_timeout_s=float(
                training_cfg.get("env_runner_restore_timeout_s", 120.0)
            ),
        )
        .callbacks(episode_callback)
        .learners(
            num_learners=_resolve_num_learners(cfg.get("num_learners", 0)),
            num_gpus_per_learner=cfg.get("num_gpus", 1),
        )
        .resources(
            num_gpus=cfg.get("num_gpus", 1),
        )
        .training(
            lr=training_cfg.get("learning_rate", 0.0003),
            gamma=training_cfg.get("gamma", 0.995),
            lambda_=training_cfg.get("lambda", 0.95),
            use_gae=training_cfg.get("use_gae", True),
            use_critic=True,
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=training_cfg.get("num_epochs", 1),
            vf_loss_coeff=training_cfg.get("vf_loss_coeff", 0.5),
            vf_clip_param=training_cfg.get("vf_clip_param", DEFAULT_VF_CLIP_PARAM),
            # A schedule goes here, not in `entropy_coeff_schedule`: that setting is
            # a hard build error on the new API stack and RLlib reads the schedule
            # off `entropy_coeff` directly.
            entropy_coeff=build_entropy_coeff_schedule(training_cfg, train_batch_size),
            kl_coeff=training_cfg.get("kl_coeff", 0.2),
            kl_target=training_cfg.get("kl_target", DEFAULT_KL_TARGET),
            clip_param=training_cfg.get("clip_param", 0.2),
            grad_clip=training_cfg.get("grad_clip", DEFAULT_GRAD_CLIP),
            grad_clip_by="global_norm",
            # RLlib's adaptive KL controller has no lower bound and decays kl_coeff
            # to zero on this workload; the KL-floor base class bounds it, and the
            # subclass adds the critic warmup every curriculum promotion needs. Set
            # here rather than via `.learners()`: that method's own deprecation notice
            # points at `learners(learner_class=...)`, but the signature in Ray
            # 2.50.1 does not accept it, so `training()` is the only working path.
            learner_class=DecoupledGradClipPPOTorchLearner,
        )
    )

    # learner_config_dict is a plain public attribute and the supported channel for
    # values a custom Learner needs; updating it in place avoids the deprecation
    # path that routing it through `training()` would take.
    ppo_config.learner_config_dict.update(
        {
            KL_COEFF_FLOOR_KEY: float(training_cfg.get("kl_coeff_floor", DEFAULT_KL_COEFF_FLOOR)),
            KL_COEFF_CEILING_KEY: float(
                training_cfg.get("kl_coeff_ceiling", DEFAULT_KL_COEFF_CEILING)
            ),
            ENTROPY_PINNED_AXES_OFFSET_KEY: pinned_axes_entropy_offset(cfg),
            # Every promotion warm-starts from the previous stage's checkpoint, which
            # carries a critic calibrated to the PREVIOUS stage's return scale. Without
            # this the policy is destroyed in ~4 iterations by advantages from a critic
            # that explains ~10% of return variance. See critic_warmup_learner.
            CRITIC_WARMUP_ITERATIONS_KEY: int(
                training_cfg.get("critic_warmup_iterations", DEFAULT_CRITIC_WARMUP_ITERATIONS)
            ),
            # One global norm over policy+value is the critic's norm at this reward
            # scale, and the actor gets rescaled by it -- measured mean_kl 9.2e-8 against
            # a 1e-2 target while the critic was healthy. Clip each by its own bound.
            # See decoupled_grad_clip_learner.
            POLICY_GRAD_CLIP_KEY: _policy_grad_clip(training_cfg),
            VALUE_GRAD_CLIP_KEY: training_cfg.get("value_grad_clip"),
        }
    )

    if multi_spec is not None:
        ppo_config = ppo_config.rl_module(rl_module_spec=multi_spec)

    return ppo_config
