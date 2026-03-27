"""
AWACS (Airborne Warning and Control System) aircraft type.

Provides long-range radar coverage and enhanced situational awareness
for the team. AWACS is a support asset, not a combat aircraft.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from air_to_air_rl.aircrafts.aircraft import Aircraft
from air_to_air_rl.simulator.core.helpers import Position


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
        the configured pattern.
        """
        if getattr(self, "orbit_controller", None) is not None:
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
        from air_to_air_rl.simulator.utils.angles import signed_yaw_deg_diff
        from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg

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
