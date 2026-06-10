"""
AWACS (Airborne Warning and Control System) aircraft type.

Provides long-range radar coverage and enhanced situational awareness
for the team. AWACS is a support asset, not a combat aircraft.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from bvr_marl_core.aircraft.aircraft import Aircraft
from bvr_marl_core.simulator.core.helpers import Position


class AWACS(Aircraft):
    """
    AWACS aircraft providing long-range radar surveillance and datalink support.

    Based on general characteristics of systems like E-3 Sentry and E-7 Wedgetail.

    Key distinction: AWACS can DETECT targets in 360° (rotating dome radar),
    but can only LOCK/ENGAGE targets within a narrower forward FOV (lock_fov_deg).
    This simulates the need to point the aircraft toward targets for weapons-quality tracking.
    """

    @dataclass
    class Config:
        mass_kg: float = 150_000.0
        reference_area_m2: float = 283.0
        aspect_ratio: float = 7.1
        oswald_e: float = 0.80

        max_speed_mps: float = 250.0
        min_speed_mps: float = 80.0
        n_max: float = 2.5
        stall0_mps: float = 70.0

        max_climb_angle_deg: float = 15.0
        min_alt_m: float = 0.0
        max_alt_m: float = 12_000.0

        # Rotating dome gives 360° detection; lock is limited to a narrower FOV
        radar_horizontal_fov_deg: float = 360.0
        radar_vertical_fov_deg: float = 60.0
        radar_max_range_m: float = 400_000.0
        radar_frequency_hz: float = 3.0e9
        radar_tx_power_w: float = 100e3
        radar_antenna_gain_db: float = 45.0
        radar_snr_threshold_db: float = 6.0
        radar_beam_rate_hz: float = 6.0
        radar_beam_rate_p_hz: float = 4.0
        # Doppler-notch half-width (m/s) for AWACS detection. Made explicit so the
        # behaviour is deliberate: because AWACS shares its track over datalink and
        # views the fight from a different aspect, beaming a single fighter rarely
        # breaks the fused picture while AWACS is up. Set to 0.0 (or remove AWACS)
        # for scenarios where notch/beaming evasion should actually be learnable.
        radar_notch_velocity_mps: float = 50.0
        # Surveillance-grade measurement noise: AWACS reports a coarse track whose
        # cross-range error grows with range (angular noise x range), so it gives
        # awareness/cueing but not weapons-quality fire-control. A missile flown on
        # the AWACS picture misses a maneuvering target; the fighter/missile own
        # radar (noise ~ 0) stays crisp. Tune up to make AWACS-reliant shots harder.
        radar_meas_angular_noise_deg: float = 0.5
        radar_meas_range_noise_m: float = 300.0
        # Datalink staleness: the AWACS track a consumer receives is delayed by
        # base + per_km * (range to AWACS). This systematic lag (unlike zero-mean
        # noise, which the KF smooths) makes a missile flown on the AWACS picture
        # aim where a maneuvering target *was* -> opens the miss. At ~50 km that's
        # ~1.5 s; at 250 m/s target that is ~375 m of lead error. Tunable.
        radar_dl_delay_base_s: float = 1.0
        radar_dl_delay_per_km_s: float = 0.01

        lock_fov_deg: float = 60.0

        rcs: float = 100.0

        passive_radar_angular_error_deg: float = 2.0
        passive_radar_range_error_m: float = 1000.0
        passive_radar_max_age_s: float = 5.0

        mws_detection_delay_s: float = 1.0
        mws_detection_delay_std: float = 0.5

        missile_types: tuple = ()
        max_missiles: int = 0
        flares: int = 60
        chaff: int = 120
        ecm: int = 10
        decoys: int = 10

    def __init__(
        self,
        position: Position,
        yaw_deg: float,
        speed_mps: float,
        group: str,
        map_limits: Any,
        min_alt_m: float,
        max_alt_m: float,
        name: str = "AWACS",
        lock_fov_deg: float | None = None,
    ):
        cfg = self.Config()
        cfg.min_alt_m = min_alt_m
        cfg.max_alt_m = min(max_alt_m, cfg.max_alt_m)  # AWACS has lower ceiling

        # Allow lock FOV to be overridden at instantiation
        if lock_fov_deg is not None:
            cfg.lock_fov_deg = lock_fov_deg

        super().__init__(
            name=name,
            position=position,
            yaw_deg=yaw_deg,
            speed_mps=min(speed_mps, cfg.max_speed_mps),  # Ensure speed within limits
            group=group,
            map_limits=map_limits,
            config=cfg,
        )

        # Mark as support asset
        self.is_support_asset = True
        self.is_high_value_target = True
        self.is_non_engageable = True  # By default, AWACS cannot be targeted

        # Lock FOV - separate from radar detection FOV
        # Detection: 360° (can see all around)
        # Lock: narrower FOV (must face target to share weapons-quality track)
        self.lock_fov_deg = cfg.lock_fov_deg

        # RCS pattern for large aircraft
        self.rcs_pattern = {
            "front_floor": 0.8,  # Large and visible from all angles
            "tail_floor": 0.9,
            "n_az": 0.5,  # Relatively uniform
            "k_top": 0.7,
            "k_bottom": 0.8,
            "n_el": 0.8,
            "sigma_min": 50.0,
            "sigma_max": 200.0,
        }

    def update(self, tick_secs: float, sim) -> list:
        """
        Update AWACS state, applying the orbit controller before physics.

        If an orbit_controller is attached, its heading/speed/altitude commands
        are fed into the control system each tick so the AWACS actually flies
        the configured pattern. When the controller has fighter-trailing enabled,
        its orbit centre is first moved to keep the AWACS a safe distance behind
        the team's fighters.
        """
        if getattr(self, "orbit_controller", None) is not None:
            self._update_trailing_center(tick_secs, sim)
            controls = self.orbit_controller.get_commanded_controls(self, tick_secs)

            # Desired heading (physics turns gradually toward this)
            self.control.set_yaw_deg(controls.get("target_heading_deg", self.yaw_deg))

            # Throttle
            self.control.set_throttle(controls.get("throttle", 1.0))

            # Altitude control via pitch angle
            target_alt = controls.get("target_altitude_m", self.position.alt)
            alt_error_m = target_alt - self.position.alt
            desired_pitch = float(np.clip(alt_error_m / 500.0 * 5.0, -10.0, 10.0))
            self.control.set_pitch_deg(desired_pitch)

        return super().update(tick_secs, sim)

    def _update_trailing_center(self, tick_secs: float, sim) -> None:
        """
        Move the orbit centre to keep the AWACS behind its team's fighters.

        Only acts when the attached controller has trailing enabled and exposes a
        ``recenter`` method. The centre is advanced toward the desired trailing
        point but capped at the AWACS's own ground speed (with a little headroom)
        per tick, so the centre never teleports when the fighter centroid jumps
        (e.g. a fighter is destroyed).
        """
        from bvr_marl_core.aircraft.control.awacs_controller import compute_trailing_center

        controller = self.orbit_controller
        config = getattr(controller, "config", None)
        if config is None or not getattr(config, "trail_fighters", False):
            return
        if not hasattr(controller, "recenter"):
            return

        target = compute_trailing_center(self, sim, config)
        if target is None:
            return

        km_per_deg = 111.0
        cur_lat, cur_lon = config.center_lat, config.center_lon
        target_lat, target_lon = target

        # Cap how far the centre can move this tick to ~1.5x the AWACS ground speed
        # so it tracks smoothly rather than snapping to the target.
        max_step_deg = (config.speed_mps * 1.5 * tick_secs) / (km_per_deg * 1000.0)
        dlat = target_lat - cur_lat
        dlon = target_lon - cur_lon
        dist = float(np.hypot(dlat, dlon))
        if dist > max_step_deg and dist > 1e-9:
            scale = max_step_deg / dist
            target_lat = cur_lat + dlat * scale
            target_lon = cur_lon + dlon * scale

        controller.recenter(target_lat, target_lon)

    def can_lock_target(self, target) -> bool:
        """
        Check if this AWACS can establish a weapons-quality lock on a target.

        AWACS can DETECT targets in 360° but can only LOCK targets within
        its lock_fov_deg of its current heading.

        Args:
            target: The target unit to check.

        Returns:
            True if target is within lock FOV, False otherwise.
        """
        from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff
        from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg

        if target is None or not hasattr(target, "position"):
            return False

        # Calculate bearing to target
        target_bearing = geodetic_bearing_deg(
            self.position.lat, self.position.lon, target.position.lat, target.position.lon
        )

        # Calculate angle difference from current heading
        angle_diff = abs(signed_yaw_deg_diff(self.yaw_deg, target_bearing))

        # Check if within lock FOV
        return angle_diff <= (self.lock_fov_deg / 2.0)
