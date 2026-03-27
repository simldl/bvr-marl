import math

from air_to_air_rl.missiles.guidance.direct import DirectPursuitGuidance
from air_to_air_rl.missiles.guidance.fov_capture import FovCaptureGuidance
from air_to_air_rl.missiles.guidance.lead import LeadInterceptGuidance
from air_to_air_rl.missiles.guidance.loft import LoftGuidance
from air_to_air_rl.missiles.guidance.pn_propnav import PnPropNavGuidance
from air_to_air_rl.missiles.guidance.utils import (
    CircularMovingAverageFilter,
    MovingAverageFilter,
    distance_m,
)
from air_to_air_rl.radar.core.utils import _angles_dist


class MissileGuidance:
    MA_WINDOW = 5

    def __init__(self, missile, target_provider):
        self.missile = missile
        self.target_provider = target_provider
        self.loft = LoftGuidance(missile)
        self.lead = LeadInterceptGuidance(missile)
        self.fov_capture = FovCaptureGuidance(missile)
        self.direct = DirectPursuitGuidance(missile)
        self.pn = PnPropNavGuidance(missile, pn_law=1)
        self.pn_min_range_m = 1000.0
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
        new_locked_id = None
        if getattr(missile, "radar", None) is not None:
            new_locked_id = getattr(missile.radar, "get_locked_target", lambda: None)()

        # Reset PN hold-last-valid when either the tracker TID or the radar
        # locked unit ID changes — velocity cached for a previous target is wrong
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
                    pn_phase = "terminal" if slant_to_tgt < self.pn.terminal_range_m else "pn"
                pn_cmd = self.pn.compute(
                    current_yaw, current_pitch, missile_pos, tgt, dt, phase=pn_phase
                )
                if pn_cmd is not None:
                    return pn_cmd

        if target_pos is None:
            return current_yaw, current_pitch

        if phase == "lead":
            return self.lead.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)
        elif phase == "fov_capture":
            return self.fov_capture.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)
        else:  # direct
            return self.direct.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)

    def _select_guidance_phase(self, missile, target_provider, tracker_manager, dt: float):
        """
        Decide guidance phase:
        - loft   (boost phase only, long-range climb)
        - pn     (midcourse when locked & in FOV & beyond min PN range)
        - lead   (locked or tracked; position + velocity intercept)
        - fov_capture (no lock, no track — steer toward last known position)
        - direct (fallback)

        NOTE: "terminal" is NOT a separate phase here.  When phase=="pn" and
        the slant to the target falls below pn.terminal_range_m (2 km),
        compute_guidance() passes phase="terminal" directly to pn.compute(),
        which switches from True-PN to ZEM guidance.  There is exactly one
        place where terminal activation is decided: the sub-gate in
        compute_guidance().
        """
        locked = bool(getattr(self, "_has_lock", lambda *a, **k: False)(missile, target_provider))
        tp_pos = None
        try:
            tp_pos = target_provider.get_guidance_target()
        except Exception:
            tp_pos = None
        in_fov = (tp_pos is not None) and self._target_in_radar_fov(missile.position, tp_pos)

        if tp_pos is not None:
            dlat = (tp_pos.lat - missile.position.lat) * 111000.0
            dlon = (
                (tp_pos.lon - missile.position.lon)
                * 111000.0
                * math.cos(math.radians(missile.position.lat))
            )
            dz = tp_pos.alt - missile.position.alt
            slant = math.sqrt(dlat * dlat + dlon * dlon + dz * dz)
        else:
            slant = 1e9  # unknown — treat as far

        # Tracked target with velocity enables a softer FOV gate
        has_tid = getattr(target_provider, "get_guidance_tid", lambda: None)() is not None

        # PN midcourse: locked + beyond min range + (in FOV or tracker has velocity).
        # Allowing PN when has_tid but slightly outside FOV prevents degradation to
        # position-only guidance for lateral movers whose intercept point is still reachable.
        if locked and slant >= self.pn_min_range_m and (in_fov or has_tid):
            return "pn"

        if locked or has_tid:
            return "lead"

        if not in_fov:
            return "fov_capture"

        return "direct"

    def _has_lock(self, missile, target_provider) -> bool:
        """
        Returns True if the missile's radar currently has a locked target.
        Used by _select_guidance_phase to gate PN and lead guidance phases.
        """
        radar = getattr(missile, "radar", None)
        if radar is None:
            return False
        return bool(getattr(radar, "locked_target", None) is not None)

    def _target_in_radar_fov(self, missile_position, target_position):
        radar = getattr(self.missile, "radar", None)
        if radar is None:
            return True
        if target_position is None:
            return False
        # radar.yaw_deg is a property: owner.yaw_deg + yaw_offset_deg.
        # Adding yaw_offset_deg again would double-count the gimbal offset.
        yaw_ref = float(getattr(radar, "yaw_deg", getattr(self.missile, "yaw_deg", 0.0)))
        pitch_ref = float(getattr(radar, "pitch_deg", getattr(self.missile, "pitch_deg", 0.0)))
        az, el, _ = _angles_dist(missile_position, yaw_ref, pitch_ref, target_position)
        h_fov = float(getattr(radar, "h_fov_deg", 60.0))
        v_fov = float(getattr(radar, "v_fov_deg", 30.0))
        return abs(az) < 0.5 * h_fov and abs(el) < 0.5 * v_fov
