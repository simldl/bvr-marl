import math

from bvr_marl_core.missiles.guidance.direct import DirectPursuitGuidance
from bvr_marl_core.missiles.guidance.lead import LeadInterceptGuidance
from bvr_marl_core.missiles.guidance.loft import LoftGuidance
from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance
from bvr_marl_core.missiles.guidance.utils import (
    CircularMovingAverageFilter,
    MovingAverageFilter,
)


class MissileGuidance:
    MA_WINDOW = 5

    def __init__(self, missile, target_provider):
        self.missile = missile
        self.target_provider = target_provider
        self.loft = LoftGuidance(missile)
        self.lead = LeadInterceptGuidance(missile)
        self.direct = DirectPursuitGuidance(missile)
        self.pn = PnPropNavGuidance(missile, pn_law=1)
        self.terminal_tgo_s = 1.5
        self.terminal_track_grace_s = 1.0
        self.terminal_track_grace_range_m = 2500.0
        # Use circular averaging for yaw only; pitch can be linear
        self.yaw_filter = CircularMovingAverageFilter(self.MA_WINDOW)
        self.pitch_filter = MovingAverageFilter(self.MA_WINDOW)
        self.active_tid = None  # track ID of the current guidance target
        self._active_locked_id = None  # unit ID of the radar-locked target

    def compute_guidance(self, missile, target_provider, tracker_manager, dt: float):
        """
        Orchestrator: select phase -> dispatch to compute function -> return angles
        """
        # Invariant: callers must pass the same provider that was injected at construction.
        # A mismatch means two independent code paths are maintaining separate state,
        # which breaks the velocity-frame invariant guaranteed by GuidanceTargetProvider.
        assert target_provider is self.target_provider, (
            f"compute_guidance: target_provider argument ({id(target_provider)}) "
            f"!= self.target_provider ({id(self.target_provider)})"
        )

        new_tid = getattr(target_provider, "get_guidance_tid", lambda: None)()
        if new_tid is None:
            get_coast_tid = getattr(target_provider, "get_coast_guidance_tid", None)
            if callable(get_coast_tid):
                try:
                    new_tid = get_coast_tid(max_age_s=self.terminal_track_grace_s)
                except TypeError:
                    new_tid = get_coast_tid()
        new_locked_id = None
        if getattr(missile, "radar", None) is not None:
            new_locked_id = getattr(missile.radar, "get_locked_target", lambda: None)()

        # Reset PN hold-last-valid when either the tracker TID or the radar
        # locked unit ID changes - velocity cached for a previous target is wrong
        # for the new one.
        if (new_tid is not None and new_tid != self.active_tid) or (
            new_locked_id != self._active_locked_id
        ):
            self.pn._last_valid_vt = None
            self.pn._last_valid_vt_age = 0.0
        self.active_tid = new_tid
        self._active_locked_id = new_locked_id

        phase = self._select_guidance_phase(missile, target_provider, tracker_manager, dt)

        current_yaw = missile.yaw_deg
        current_pitch = missile.pitch_deg
        missile_pos = missile.position

        try:
            target_pos = target_provider.get_guidance_target()
        except Exception:
            target_pos = None

        if phase == "loft":
            if target_pos is not None:
                loft_cmd = self.loft.compute(missile, target_pos)
                if loft_cmd is not None:
                    return loft_cmd

        # Unified PN/terminal dispatch: orchestrator resolves midcourse vs terminal
        # so pn.compute() receives the correct phase and uses ZEM when appropriate.
        if phase in ("pn", "terminal"):
            tgt = target_pos
            if tgt is not None:
                if phase == "terminal":
                    pn_phase = "terminal"
                else:
                    # Sub-gate: switch to ZEM terminal mode when within terminal range
                    dlat = (tgt.lat - missile_pos.lat) * 111000.0
                    dlon = (
                        (tgt.lon - missile_pos.lon)
                        * 111000.0
                        * math.cos(math.radians(missile_pos.lat))
                    )
                    dz = tgt.alt - missile_pos.alt
                    slant_to_tgt = math.sqrt(dlat * dlat + dlon * dlon + dz * dz)
                    pn_phase = "terminal" if slant_to_tgt < self._terminal_range_m() else "pn"
                pn_cmd = self.pn.compute(
                    current_yaw, current_pitch, missile_pos, tgt, dt, phase=pn_phase
                )
                if pn_cmd is not None:
                    return pn_cmd

        if target_pos is None:
            return current_yaw, current_pitch

        if phase == "lead":
            return self.lead.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)
        else:  # direct
            return self.direct.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)

    def _select_guidance_phase(self, missile, target_provider, tracker_manager, dt: float):
        """Clean guidance state machine: terminal PN/ZEM, PN, lead, direct."""
        locked = bool(getattr(self, "_has_lock", lambda *a, **k: False)(missile, target_provider))

        try:
            target_pos = target_provider.get_guidance_target()
        except Exception:
            target_pos = None

        guidance_range_m = (
            self._range_m(missile.position, target_pos) if target_pos is not None else float("inf")
        )
        truth_range_m = self._locked_target_range_m(missile, target_provider)

        has_fresh_fn = getattr(target_provider, "has_fresh_track", None)
        has_fresh_result = has_fresh_fn() if callable(has_fresh_fn) else None
        if isinstance(has_fresh_result, bool):
            has_fresh_track = has_fresh_result
        else:
            has_fresh_track = (
                getattr(target_provider, "get_guidance_tid", lambda: None)() is not None
            )

        coastable = False
        if locked and min(guidance_range_m, truth_range_m) <= self.terminal_track_grace_range_m:
            coast_fn = getattr(target_provider, "has_coastable_track", None)
            if callable(coast_fn):
                try:
                    coast_result = coast_fn(max_age_s=self.terminal_track_grace_s)
                except TypeError:
                    coast_result = coast_fn()
                coastable = coast_result if isinstance(coast_result, bool) else False

        track_for_guidance = has_fresh_track or coastable
        terminal_range_m = self._terminal_range_m()
        terminal_endgame = min(
            guidance_range_m, truth_range_m
        ) <= terminal_range_m or self._terminal_by_time_to_go(missile, target_provider)

        if locked and track_for_guidance and terminal_endgame:
            return "terminal"
        if locked and track_for_guidance:
            return "pn"
        if track_for_guidance and not terminal_endgame:
            return "lead"
        if target_pos is not None:
            return "direct"
        return "direct"

    def _has_lock(self, missile, target_provider) -> bool:
        """
        Returns True if the missile's radar currently has a locked target.
        Used by _select_guidance_phase to gate PN and lead guidance phases.
        """
        radar = getattr(missile, "radar", None)
        if radar is None:
            return False
        get_locked = getattr(radar, "get_locked_target", None)
        if callable(get_locked):
            try:
                return get_locked() is not None
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass
        return getattr(radar, "locked_target", None) is not None

    def _terminal_range_m(self) -> float:
        try:
            terminal_range_m = float(getattr(self.pn, "terminal_range_m", 2000.0))
        except (TypeError, ValueError):
            return 2000.0
        if not math.isfinite(terminal_range_m) or terminal_range_m <= 0.0:
            return 2000.0
        return terminal_range_m

    def _locked_target_range_m(self, missile, target_provider) -> float:
        target = getattr(missile, "target", None)
        if target is None or not hasattr(target, "position"):
            return float("inf")
        if getattr(target, "is_destroyed", False):
            return float("inf")

        target_id = getattr(target, "id", None)
        allowed_ids = set()
        radar = getattr(missile, "radar", None)
        if radar is not None and hasattr(radar, "get_locked_target"):
            try:
                locked_id = radar.get_locked_target()
                if locked_id is not None:
                    allowed_ids.add(locked_id)
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass
        for candidate in (
            getattr(target_provider, "current_target_id", None),
            getattr(missile, "launch_contact_id", None),
            getattr(missile, "designated_target_id", None),
        ):
            if candidate is not None:
                allowed_ids.add(candidate)
        if allowed_ids and target_id not in allowed_ids:
            return float("inf")
        return self._range_m(missile.position, target.position)

    def _terminal_by_time_to_go(self, missile, target_provider) -> bool:
        try:
            target_pos = target_provider.get_guidance_target()
        except Exception:
            target_pos = None
        if target_pos is None:
            return False

        vel = None
        get_vel = getattr(target_provider, "get_guidance_velocity_in_missile_enu", None)
        if callable(get_vel):
            try:
                vel = get_vel(missile.position)
            except Exception:
                vel = None
        if vel is None:
            vel = getattr(target_provider, "get_guidance_velocity", lambda: None)()
        if vel is None:
            return False

        try:
            de = (
                (target_pos.lon - missile.position.lon)
                * 111_000.0
                * math.cos(math.radians(missile.position.lat))
            )
            dn = (target_pos.lat - missile.position.lat) * 111_000.0
            du = target_pos.alt - missile.position.alt
            r_vec = (de, dn, du)
            rng = math.sqrt(de * de + dn * dn + du * du)
            if rng <= 1e-6:
                return True

            vt = [float(v) for v in vel]
            speed = float(getattr(missile, "speed", 0.0))
            yaw = math.radians(float(getattr(missile, "yaw_deg", 0.0)))
            pitch = math.radians(float(getattr(missile, "pitch_deg", 0.0)))
            cp = math.cos(pitch)
            vm = (
                cp * math.sin(yaw) * speed,
                cp * math.cos(yaw) * speed,
                math.sin(pitch) * speed,
            )
            vrel = (vt[0] - vm[0], vt[1] - vm[1], vt[2] - vm[2])
            closing = -sum(vrel[i] * r_vec[i] for i in range(3)) / rng
            return closing > 1e-6 and (rng / closing) <= self.terminal_tgo_s
        except Exception:
            return False

    @staticmethod
    def _range_m(a, b) -> float:
        try:
            de = (b.lon - a.lon) * 111_000.0 * math.cos(math.radians(a.lat))
            dn = (b.lat - a.lat) * 111_000.0
            du = b.alt - a.alt
            return math.sqrt(de * de + dn * dn + du * du)
        except Exception:
            return float("inf")
