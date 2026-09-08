"""
Weapon firing logic for missiles, guns, and countermeasures.
Handles firing gates, cooldowns, and automation.
"""

import logging

import numpy as np

from bvr_marl_core.aircraft.systems.fire_feasibility import (
    evaluate_fire_gates,
    sync_missile_cooldown,
)
from bvr_marl_core.aircraft.systems.fire_veto import (
    wasted_categories_from_gates,
    wasted_category_key,
)
from bvr_marl_core.domain.launch_geometry import (
    capture_launch_geometry,
    capture_launch_geometry_from_enu,
)
from bvr_marl_core.domain.tactical_contact import TacticalContact

logger = logging.getLogger(__name__)


class WeaponFiringHandler:
    """Handle weapon firing with safety gates and cooldowns."""

    def __init__(self, missile_auto, weapon_cooldowns, trigger_proc):
        """
        Initialize weapon firing handler.

        Args:
            missile_auto: MissileAutomation instance
            weapon_cooldowns: WeaponCooldowns instance
            trigger_proc: TriggerProcessor instance
        """
        self.missile_auto = missile_auto
        self.weapon_cooldowns = weapon_cooldowns
        self.trigger_proc = trigger_proc

    def handle_missile_firing(self, unit, action, selected_target, state, dt):
        """Handle missile firing with automation and cooldowns."""
        aircraft_id = getattr(unit, "id", "unknown")

        # Sensor state for THIS step, written unconditionally before any trigger logic.
        #
        # These two feed `tactical/lock_rate` and `tactical/fov_rate`, which
        # `termination.py` divides by the agent's TOTAL STEP COUNT. They were previously
        # written only on the firing paths below -- a successful launch set them from
        # its diagnostics, a vetoed press cleared them -- so between trigger pulls they
        # held a stale value. That made both metrics a sticky latch driven by shooting,
        # not a measure of sensor coverage: an agent that never fired reported exactly
        # 0.0 no matter how well it held a lock -- landing far below any sensible
        # coverage floor and feeding permanent false "dead run" signals downstream.
        #
        # Both come off the SHARED evaluator (`fire_feasibility`), which is also what
        # produces the observation's `can_fire` bit -- so the metric, the fire mask and
        # the launch gate can no longer drift apart. `lock_rate` intentionally reports
        # the own-radar lock rather than `gates.has_lock`: it measures this aircraft's
        # sensor discipline, so an AWACS-cued track must not flatter it.
        sync_missile_cooldown(unit, state.get("missile_cooldown_left_s", 0.0))
        gates = evaluate_fire_gates(
            unit,
            selected_target,
            simulator=getattr(self, "simulator", None),
            max_missiles_per_target=self.weapon_cooldowns.max_missiles_per_target,
        )
        state["last_lock_ok"] = gates.radar_lock
        state["last_fov_ok"] = gates.target_in_fov

        # Automated missile firing override
        if self.missile_auto.enable and not isinstance(selected_target, TacticalContact):
            should_auto_fire, auto_veto = self.missile_auto.should_auto_fire(unit, selected_target)
            if should_auto_fire:
                action[3] = 1.0
                logger.debug(
                    f"Aircraft {aircraft_id}: AUTO_FIRE_TRIGGERED - target={getattr(selected_target, 'id', 'none')}"
                )
            elif auto_veto and auto_veto not in ("automation_disabled", "no_target"):
                logger.debug(f"Aircraft {aircraft_id}: AUTO_FIRE_VETO - {auto_veto}")

        # Check firing conditions
        trigger_result = self.trigger_proc.apply_trigger(action[3], 3, state)
        has_missiles = bool(unit.missile_types)
        missile_cooldown_ok = self.weapon_cooldowns.is_missile_ready(state)

        # Pre-flight checks
        veto_reason = None
        target_id = self._selection_id(selected_target)
        target_saturated = self._target_saturated(unit, selected_target, target_id)
        if selected_target is None:
            veto_reason = "no_target_selected"
        elif not missile_cooldown_ok:
            veto_reason = f"missile_cooldown({state['missile_cooldown_left_s']:.2f}s)"
        elif target_saturated:
            veto_reason = (
                f"target_saturated(>={self.weapon_cooldowns.max_missiles_per_target}_active)"
            )

        # A viable firing solution on THIS step, independent of the trigger: true exactly
        # when a press WOULD have produced a launch. Counting it every step gives the
        # shot-discipline metrics a denominator of OPPORTUNITIES rather than presses --
        # `trigger_precision_rate` divides by presses, so it cannot tell "never had a
        # shot" apart from "had shots and declined them", which is the distinction that
        # matters for a passive policy.
        #
        # This is `gates.can_fire`, the SAME field the observation exposes as the
        # `can_fire` bit and the fire-gradient mask keys off. Previously it was a second
        # hand-rolled conjunction that omitted the datalink lock, the gimbal limit and
        # the range gate, so with an AWACS in the scenario it undercounted exactly the
        # datalink-cued shots the AWACS exists to create.
        state["shot_opportunity_this_step"] = int(gates.can_fire)

        # Numerator for P(fire | can_fire): the trigger pressed on a step where every
        # launch gate had already passed. Paired with the denominator above, this is the
        # only reading that separates a COLLAPSED fire head from one that shoots
        # whenever it is allowed to.
        #
        # Neither existing counter answers it. The raw fire rate is ~2% of steps, but the
        # infeasible-step pin means the trigger is only EXPRESSIBLE on can_fire steps, so
        # 2% is equally consistent with both stories. `vetoed_missile_*` counts only
        # presses that were REFUSED, which by construction happen on can_fire == 0 steps
        # and so say nothing about the decision being measured here.
        state["fire_attempt_on_opportunity_this_step"] = int(
            bool(trigger_result) and gates.can_fire
        )

        if (
            trigger_result
            and selected_target is not None
            and missile_cooldown_ok
            and has_missiles
            and not target_saturated
        ):
            self._fire_missile(unit, selected_target, state, aircraft_id, gates)
        elif trigger_result:
            # Pre-flight vetoes split by attribution. Cooldown / per-target cap /
            # winchester are doctrine-and-safety constraints, not a policy error, so they
            # stay `suppressed` (unpenalized). Pulling the trigger with NO target selected
            # is a policy error: the target-selection action addressed an empty contact
            # slot. It gets its own counter so the reward can supply a gradient — folding
            # it into `suppressed` is what let the selection head stall on a permanently
            # empty slot with zero corrective signal.
            state["vetoed_missile_attempts_this_step"] = 1
            if selected_target is None:
                state["vetoed_missile_no_target_this_step"] = 1
            else:
                state["vetoed_missile_suppressed_this_step"] = 1
            # NOT last_lock_ok/last_fov_ok: those describe this step's sensor state and
            # are set above. A vetoed press does not mean the radar lost its lock.
            state["last_target_id"] = self._selection_id(selected_target)
            if veto_reason:
                logger.debug(f"Aircraft {aircraft_id}: FIRE_PRE_VETO - {veto_reason}")

    @staticmethod
    def _contact_position(selected_target):
        """Aim point the shot was taken against (estimated for a contact)."""
        if selected_target is None:
            return None
        position = getattr(selected_target, "position", None)
        if position is not None:
            return position
        estimated = getattr(selected_target, "estimated_position", None)
        return estimated

    def _fire_missile(self, unit, selected_target, state, aircraft_id, gates=None):
        """Execute missile firing.

        ``gates`` is this step's shared fire-gate evaluation, carried in so a rejected
        launch can be attributed to the gate that rejected it (see fire_veto).
        """
        try:
            if isinstance(selected_target, TacticalContact):
                missile, veto, diagnostics = unit.weapons.fire_missile_at_contact(
                    self.simulator, selected_target, unit.missile_types[0]
                )
            else:
                missile, veto, diagnostics = unit.weapons.fire_missile(
                    self.simulator, selected_target, unit.missile_types[0]
                )

            # Launch-time gate outcome, kept under its own keys. This is a property of
            # the SHOT (did the launch clear the lock/FOV gates, including the
            # datalink-cued lock), not of the step's sensor coverage, so it must not
            # overwrite last_lock_ok / last_fov_ok.
            state["last_launch_lock_ok"] = diagnostics.get("has_lock", False)
            state["last_launch_fov_ok"] = diagnostics.get("in_fov", False)
            state["last_target_id"] = diagnostics.get("target_id", None)

            if missile is not None:
                target_id = diagnostics.get("target_id")
                self.missile_auto.update_missiles_per_target(target_id, True)
                missiles_at_target = self.missile_auto.get_missiles_at_target(target_id)

                if missiles_at_target >= self.missile_auto.max_per_target:
                    self.weapon_cooldowns.set_missile_cooldown(
                        state, self.missile_auto.long_cooldown_s
                    )
                    logger.info(
                        f"Aircraft {aircraft_id}: MISSILE_FIRED_SUCCESS - LONG_COOLDOWN={self.missile_auto.long_cooldown_s}s"
                    )
                else:
                    self.weapon_cooldowns.set_missile_cooldown(state)
                    logger.info(
                        f"Aircraft {aircraft_id}: MISSILE_FIRED_SUCCESS - missiles_at_target={missiles_at_target}"
                    )
                # Re-publish immediately: the observation for THIS step is built after
                # the action is applied, so without this the `can_fire` bit the policy
                # sees on the step it just fired would still read "ready".
                sync_missile_cooldown(unit, state.get("missile_cooldown_left_s", 0.0))

                state["valid_missile_fires_this_step"] = 1
                # Geometry the shot was actually taken from, against the ESTIMATED
                # contact the policy aimed at. The standalone harness fires hot,
                # co-altitude, at 300 m/s and gets ~90% kills; the campaign gets ~1%.
                # If the difference is the launch conditions the policy chooses,
                # this is where it becomes visible.
                # A TacticalContact carries relative ENU in `state`, not a position;
                # asking it for `.position` silently yields None and loses the
                # geometry for every contact-based launch.
                if isinstance(selected_target, TacticalContact):
                    geometry = capture_launch_geometry_from_enu(
                        unit,
                        selected_target.state[0],
                        selected_target.state[1],
                        selected_target.state[2],
                    )
                else:
                    geometry = capture_launch_geometry(
                        unit, self._contact_position(selected_target)
                    )
                state["last_launch_geometry"] = geometry
                # Mirrored onto the unit so the step processor can pair it with the
                # missile-count delta it uses to detect launches; it sees units, not
                # the per-agent action state.
                try:
                    unit._last_launch_geometry = geometry
                except Exception:
                    pass
                in_envelope = self._shot_in_envelope(unit, selected_target)
                if in_envelope is True:
                    state["in_envelope_missile_fires_this_step"] = 1
                elif in_envelope is False:
                    state["out_of_envelope_missile_fires_this_step"] = 1
            else:
                # Firing was permissible but the launch was rejected by geometry
                # (out of FOV/range/lock): the trigger was pulled on an invalid
                # solution, so classify as a wasted (policy-error) attempt.
                # Also attributed to the gate(s) that were unmet -- the collapsed
                # counter cannot distinguish a pointing failure from a range or
                # sensor failure, and the three have unrelated fixes.
                state["vetoed_missile_attempts_this_step"] = 1
                state["vetoed_missile_wasted_this_step"] = 1
                for category, unmet in wasted_categories_from_gates(gates).items():
                    state[wasted_category_key(category)] = unmet
                logger.debug(f"Aircraft {aircraft_id}: MISSILE_FIRE_VETO - {veto}")

        except Exception as e:
            logger.warning(f"Aircraft {aircraft_id}: MISSILE_FIRE_EXCEPTION - {e}")
            state["vetoed_missile_attempts_this_step"] = 1
            state["vetoed_missile_wasted_this_step"] = 1
            # No gate attribution: the launch raised rather than being refused, so
            # nothing here says which gate was unmet. Left at zero deliberately --
            # the subtotals are bounded by `wasted`, never equal to it.
            state["last_launch_lock_ok"] = False
            state["last_launch_fov_ok"] = False
            state["last_target_id"] = self._selection_id(selected_target)

    @staticmethod
    def _selection_id(selected_target):
        if isinstance(selected_target, TacticalContact):
            return selected_target.track_id
        return getattr(selected_target, "id", None) if selected_target else None

    def _target_saturated(self, unit, selected_target, target_id) -> bool:
        if isinstance(selected_target, TacticalContact):
            count_contact = getattr(self.simulator, "count_missiles_at_contact", None)
            if callable(count_contact):
                return (
                    count_contact(
                        unit.group,
                        getattr(unit, "id", None),
                        target_id,
                        selected_target.report_lineage,
                    )
                    >= self.weapon_cooldowns.max_missiles_per_target
                )
            # Minimal simulator adapters may not expose the team operational registry.
            return (
                self.missile_auto.get_missiles_at_target(target_id)
                >= self.weapon_cooldowns.max_missiles_per_target
            )
        return self.weapon_cooldowns.is_target_saturated(self.simulator, unit.group, target_id)

    @staticmethod
    def _shot_in_envelope(unit, target) -> bool | None:
        """Return True/False if the launch was inside the aero envelope, else None.

        In-envelope = DLZ zone R2 (NEZ) or R3 (pure-pursuit). Out-of-envelope =
        R1 (inside r_min, too close) or R4 (beyond r_pi, too far / low Pk).
        Returns None when the envelope cannot be evaluated (e.g. simplified env
        with no WEZ), so callers can treat it as unscored rather than poor.
        """
        wez = getattr(unit, "wez", None)
        if wez is None or target is None:
            # This early return is the OTHER way a shot ends up unscored, and it was the
            # uninstrumented one: missiles launch with in_envelope AND out_of_envelope
            # both 0.0 and NOT a single ENVELOPE_EVAL_FAILED warning, because the
            # exception branch below never fires and the shots fall out here instead.
            # Logged at warning for the
            # same reason as below: "unscored" must never be silent, because it is
            # indistinguishable from "no shots taken" in every downstream metric.
            logger.warning(
                "Aircraft %s: ENVELOPE_EVAL_SKIPPED (shot recorded as unscored) - wez=%s target=%s",
                getattr(unit, "id", "unknown"),
                "missing" if wez is None else "present",
                "missing" if target is None else type(target).__name__,
            )
            return None
        try:
            if isinstance(target, TacticalContact):
                estimate = wez.compute_dlz_from_track(target.state, target.covariance)
                dlz = estimate.nominal
                slant_range_m = float(np.linalg.norm(np.asarray(target.state[:3], dtype=float)))
            else:
                # Explicit legacy/oracle path. Sensor-limited selection always
                # supplies TacticalContact and therefore never reaches target truth.
                dlz = wez.compute_dlz(target)
                slant_range_m = wez._slant_range_m(unit, target)
            return wez.zone_for_range(slant_range_m, dlz) in ("R2", "R3")
        except Exception as exc:
            # Was a bare `except Exception: return None`. Returning None means the shot
            # is UNSCORED -- neither in_envelope nor out_of_envelope increments -- so a
            # persistent failure here looks exactly like "no shots taken" and is
            # invisible: missiles launch with in_envelope AND out_of_envelope both at
            # 0.0, meaning every envelope evaluation failed silently and the
            # in_envelope_shot_bonus / out_of_envelope_shot_penalty signals never fired.
            # The likely raise is nez.compute_dlz_from_track's
            # "requires a 6D state and 6x6 covariance" guard. Still non-fatal -- an
            # unscorable envelope must not drop the shot -- but no longer silent.
            logger.warning(
                "Aircraft %s: ENVELOPE_EVAL_FAILED (shot recorded as unscored) - %s: %s",
                getattr(unit, "id", "unknown"),
                type(exc).__name__,
                exc,
            )
            return None

    def handle_gun_firing(self, unit, action, selected_target, state):
        """Handle gun firing with FOV and cooldown checks."""
        aircraft_id = getattr(unit, "id", "unknown")
        gun_trigger_result = self.trigger_proc.apply_trigger(action[5], 5, state)
        gun_cooldown_ok = self.weapon_cooldowns.is_gun_ready(state)

        gun_fov_ok = self._check_gun_fov(unit, selected_target)

        if gun_trigger_result and gun_fov_ok and gun_cooldown_ok:
            self._fire_gun(unit, selected_target, state, aircraft_id)
        elif gun_trigger_result:
            state["vetoed_gun_attempts_this_step"] = 1

    def _check_gun_fov(self, unit, selected_target) -> bool:
        """Check if target is in gun FOV."""
        if selected_target is None:
            return True
        if isinstance(selected_target, TacticalContact):
            return bool(unit.weapons.is_contact_in_fov(selected_target))
        if hasattr(unit, "weapons") and unit.weapons and hasattr(unit.weapons, "is_target_in_fov"):
            try:
                return bool(unit.weapons.is_target_in_fov(selected_target))
            except Exception:
                return True
        return True

    def _fire_gun(self, unit, selected_target, state, aircraft_id):
        """Execute gun firing."""
        try:
            if isinstance(selected_target, TacticalContact):
                gun_result = unit.weapons.fire_gun_at_contact(self.simulator, selected_target)
            else:
                gun_result = unit.weapons.fire_gun(self.simulator, selected_target)
            if gun_result:
                self.weapon_cooldowns.set_gun_cooldown(state)
                state["valid_gun_fires_this_step"] = 1
                logger.info(f"Aircraft {aircraft_id}: GUN_FIRED_SUCCESS")
        except Exception as e:
            if "velocity" not in str(e) or "setter" not in str(e):
                logger.warning(f"Aircraft {aircraft_id}: GUN_FIRE_EXCEPTION - {e}")

    def handle_countermeasures(self, unit, action, state):
        """Handle countermeasure deployment. The simulator must be passed so the
        physical countermeasure objects are actually spawned (and can seduce
        missiles); without it they are silently discarded."""
        sim = self.simulator
        if self.trigger_proc.apply_trigger(action[6], 6, state):
            unit.countermeasures.launch_flares(sim)
        if self.trigger_proc.apply_trigger(action[7], 7, state):
            unit.countermeasures.launch_chaff(sim)
        if self.trigger_proc.apply_trigger(action[8], 8, state):
            unit.countermeasures.deploy_decoys(sim)
