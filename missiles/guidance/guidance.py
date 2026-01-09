import math
import numpy as np
from missiles.guidance.loft import LoftGuidance
from missiles.guidance.lead import LeadInterceptGuidance
from missiles.guidance.fov_capture import FovCaptureGuidance
from missiles.guidance.terminal import TerminalGuidance
from missiles.guidance.direct import DirectPursuitGuidance
from missiles.guidance.pn_propnav import PnPropNavGuidance
from missiles.guidance.utils import MovingAverageFilter, CircularMovingAverageFilter, distance_m
from radar.core.utils import _angles_dist

class MissileGuidance:
    MA_WINDOW = 5

    def __init__(self, missile, target_provider):
        self.missile = missile
        self.target_provider = target_provider
        self.loft = LoftGuidance(missile)
        self.lead = LeadInterceptGuidance(missile)
        self.fov_capture = FovCaptureGuidance(missile)
        self.terminal = TerminalGuidance(missile)
        self.direct = DirectPursuitGuidance(missile)
        self.pn = PnPropNavGuidance(missile, pn_law=1)
        self.pn_min_range_m = 1000.0  # Minimum range for PN guidance
        # Use circular averaging for yaw only; pitch can be linear
        self.yaw_filter = CircularMovingAverageFilter(self.MA_WINDOW)
        self.pitch_filter = MovingAverageFilter(self.MA_WINDOW)

    def compute_guidance(self, missile, target_provider, tracker_manager, dt: float):
        """
        Orchestrator: select phase -> dispatch to compute function -> return angles
        """
        phase = self._select_guidance_phase(missile, target_provider, tracker_manager, dt)

        # Get current state for guidance algorithms
        current_yaw = missile.yaw_deg
        current_pitch = missile.pitch_deg
        missile_pos = missile.position
        
        # Get target position
        try:
            target_pos = target_provider.get_guidance_target()
        except Exception:
            target_pos = None

        if phase == "loft":
            if target_pos is not None:
                loft_cmd = self.loft.compute(missile, target_pos)
                if loft_cmd is not None:
                    return loft_cmd

        # PN with FOV check
        if phase == "pn":
            try:
                tgt = target_provider.get_guidance_target()
                vel = target_provider.get_guidance_velocity()
                
                # Build state dict for PN
                R_vec = None; Vt_vec = None
                if tgt is not None:
                    dE = (tgt.lon - missile.position.lon) * 111000.0 * math.cos(math.radians(missile.position.lat))
                    dN = (tgt.lat - missile.position.lat) * 111000.0
                    dU = (tgt.alt - missile.position.alt)
                    R_vec = np.array([dE, dN, dU], dtype=float)
                if vel is not None:
                    Vt_vec = np.array(vel, dtype=float)
                else:
                    # Fallback: estimate R from geodetic deltas; set Vt to zero if unknown
                    if tgt is not None:
                        dE = (tgt.lon - missile.position.lon) * 111000.0 * math.cos(math.radians(missile.position.lat))
                        dN = (tgt.lat - missile.position.lat) * 111000.0
                        dU = tgt.alt - missile.position.alt
                        R_vec = np.array([dE, dN, dU], dtype=float)
                    Vt_vec = np.zeros(3, dtype=float)
                
                if R_vec is not None:
                    state = {"R_vec": R_vec, "Vt_vec": Vt_vec}
                    pn_cmd = self.pn.compute(missile, state, tracker_manager, dt)
                    if pn_cmd is not None:
                        return pn_cmd
            except Exception as e:
                pass

        # Fallback phases - all use the standard signature
        if target_pos is None:
            # No target position available, return current angles
            return current_yaw, current_pitch
            
        if phase == "terminal":
            return self.terminal.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)
        elif phase == "lead":
            return self.lead.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)
        elif phase == "fov_capture":
            return self.fov_capture.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)
        else:  # direct
            return self.direct.compute(current_yaw, current_pitch, missile_pos, target_pos, dt)


    def _select_guidance_phase(self, missile, target_provider, tracker_manager, dt: float):
        """
        Decide guidance phase:
        - terminal
        - loft (simple heuristic with checks in loft.compute)
        - pn (midcourse default when locked & in FOV & beyond min PN range)
        - lead (if locked but PN gate fails or FOV marginal)
        - fov_capture (if not in FOV / no lock yet)
        - direct (fallback)
        """
        # 1) Terminal?
        if getattr(self.terminal, "should_activate", lambda *a, **k: False)(missile, target_provider):
            return "terminal"

        # 2) Loft? (we'll call loft.compute and fall through if it returns None)
        want_loft = True

        # Basic state
        locked = bool(getattr(self, "_has_lock", lambda *a, **k: False)(missile, target_provider))
       # Use actual positions for FOV check
        tp_pos = None
        try:
            tp_pos = target_provider.get_guidance_target()
        except Exception:
            tp_pos = None
        in_fov = (tp_pos is not None) and self._target_in_radar_fov(missile.position, tp_pos)

        if tp_pos is not None:
            dlat = (tp_pos.lat - missile.position.lat) * 111000.0
            dlon = (tp_pos.lon - missile.position.lon) * 111000.0 * math.cos(math.radians(missile.position.lat))
            dz   = tp_pos.alt - missile.position.alt
            slant = math.sqrt(dlat*dlat + dlon*dlon + dz*dz)
        else:
            slant = 1e9  # unknown → treat as farr

        # 3) PN midcourse if we have lock, target in FOV, and not too close
        if locked and in_fov and slant >= self.pn_min_range_m:
            return "pn"

        # 4) If locked but PN gate fails (too close, borderline), try lead intercept
        if locked:
            return "lead"

        # 5) If not locked or not in FOV, use FOV capture
        if not in_fov:
            return "fov_capture"

        # 6) Fallback
        return "direct"

    def _target_in_radar_fov(self, missile_position, target_position):
        radar = getattr(self.missile, "radar", None)
        if radar is None:
            return True
        if target_position is None:
            return False
        yaw_ref   = float(getattr(radar, "yaw_deg",   getattr(self.missile, "yaw_deg", 0.0))) \
                + float(getattr(radar, "yaw_offset_deg", 0.0))
        pitch_ref = float(getattr(radar, "pitch_deg", getattr(self.missile, "pitch_deg", 0.0))) \
                + float(getattr(radar, "pitch_offset_deg", 0.0))
        az, el, _ = _angles_dist(missile_position, yaw_ref, pitch_ref, target_position)
        h_fov = float(getattr(radar, "h_fov_deg", 60.0))
        v_fov = float(getattr(radar, "v_fov_deg", 30.0))
        return abs(az) < 0.5 * h_fov and abs(el) < 0.5 * v_fov