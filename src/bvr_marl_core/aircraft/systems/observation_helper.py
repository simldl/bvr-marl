"""
Observation Helper for BVR AI.

Exposes first-class features from aircraft systems to BT/RL policy:
- DLZ/NEZ & launch envelope (from Range/NEZ calculators)
- Geometry & kinematics (ATA, aspect, closure, bearing)
- Track/target quality (priority scores, selected target score)
- Fire feasibility flags (radar lock, FOV, gimbal, range gates)
- Threat & warnings (missile warnings, time since first warning)
- Weapons state (remaining missiles, cooldowns)
"""

from typing import Any

import numpy as np

from bvr_marl_core.aircraft.systems.fire_feasibility import (
    NO_TARGET_GATES,
    evaluate_fire_gates,
)
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff
from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


class ObservationHelper:
    """
    Centralized observation builder for BVR combat.
    Aggregates features from all aircraft systems into structured observations.
    """

    def __init__(self, aircraft):
        self.aircraft = aircraft

    def get_dlz_nez_features(self, target) -> dict[str, Any]:
        """
        Get DLZ/NEZ & launch envelope features for target.

        Returns:
        - DLZ bands: r_min, r_tr, r_pi, r_aero (meters)
        - NEZ band: r_nez_in, r_nez_out (meters)
        - DLZ zone: R1/R2/R3/R4 for current slant range
        - NEZ visible: bool (in R2/R3 zones)
        - SQI: instantaneous intercept quality [0..1]
        """
        if not target:
            return {
                "valid": False,
                "dlz": None,
                "zone": "R4",
                "nez_visible": False,
                "sqi": 0.0,
                "slant_range_m": 0.0,
            }

        try:
            wez = self.aircraft.wez
            if isinstance(target, TacticalContact):
                estimate = wez.compute_dlz_from_track(target.state, target.covariance)
                dlz = estimate.nominal
                slant_range = float(np.linalg.norm(np.asarray(target.state[:3], dtype=float)))
                sqi = wez.sqi_from_estimate(slant_range, estimate.closing_speed_mps, dlz)
            else:
                # Compatibility/oracle diagnostic path.
                dlz = wez.compute_dlz(target)
                slant_range = wez._slant_range_m(self.aircraft, target)
                sqi = wez.sqi(self.aircraft, target, missile=None, dlz=dlz)
            zone = wez.zone_for_range(slant_range, dlz)
            nez_vis = wez.nez_visible(slant_range, dlz, show_in=("R2", "R3"))

            return {
                "valid": True,
                "dlz": {
                    "r_min_m": dlz.r_min_m,
                    "r_tr_m": dlz.r_tr_m,
                    "r_pi_m": dlz.r_pi_m,
                    "r_aero_m": dlz.r_aero_m,
                    "r_nez_in_m": dlz.r_nez_in_m,
                    "r_nez_out_m": dlz.r_nez_out_m,
                },
                "zone": zone,
                "nez_visible": nez_vis,
                "sqi": sqi,
                "slant_range_m": slant_range,
            }
        except Exception as e:
            return {
                "valid": False,
                "dlz": None,
                "zone": "R4",
                "nez_visible": False,
                "sqi": 0.0,
                "slant_range_m": 0.0,
                "error": str(e),
            }

    def get_geometry_kinematics(self, target) -> dict[str, Any]:
        """
        Get geometry & kinematics features for target.

        Returns:
        - ATA (angle-to-target from own nose) [deg]
        - Aspect (target's nose-to-us angle) [deg]
        - AOT (angle-off-tail) [deg]
        - Closure rate [m/s]
        - Bearing to target [deg]
        - Slant range [m]
        - Altitude difference [m]
        """
        if not target:
            return {
                "valid": False,
                "ata_deg": 0.0,
                "aspect_deg": 0.0,
                "aot_deg": 0.0,
                "closure_rate_mps": 0.0,
                "bearing_deg": 0.0,
                "slant_range_m": 0.0,
                "altitude_diff_m": 0.0,
            }

        try:
            if isinstance(target, TacticalContact):
                state = np.asarray(target.state, dtype=float)
                relative_position = state[:3]
                slant_range = float(np.linalg.norm(relative_position))
                bearing = float(np.degrees(np.arctan2(relative_position[0], relative_position[1])))
                ata = abs(signed_yaw_deg_diff(self.aircraft.yaw_deg, bearing))
                target_velocity = state[3:6]
                target_speed = float(np.linalg.norm(target_velocity))
                target_yaw = (
                    float(np.degrees(np.arctan2(target_velocity[0], target_velocity[1])))
                    if target_speed > 1e-9
                    else 0.0
                )
                bearing_from_target = (bearing + 180.0) % 360.0
                aspect = abs(signed_yaw_deg_diff(target_yaw, bearing_from_target))
                own_velocity = np.array(
                    [
                        float(getattr(getattr(self.aircraft, "velocity", None), axis, 0.0))
                        for axis in ("vx", "vy", "vz")
                    ]
                )
                los = relative_position / max(slant_range, 1.0)
                closure = -float(np.dot(los, target_velocity - own_velocity))
                return {
                    "valid": True,
                    "ata_deg": ata,
                    "aspect_deg": aspect,
                    "aot_deg": 180.0 - aspect,
                    "closure_rate_mps": closure,
                    "bearing_deg": bearing,
                    "slant_range_m": slant_range,
                    "altitude_diff_m": -float(relative_position[2]),
                }
            bearing = geodetic_bearing_deg(
                self.aircraft.position.lat,
                self.aircraft.position.lon,
                target.position.lat,
                target.position.lon,
            )
            ata = abs(signed_yaw_deg_diff(self.aircraft.yaw_deg, bearing))

            bearing_from_target = geodetic_bearing_deg(
                target.position.lat,
                target.position.lon,
                self.aircraft.position.lat,
                self.aircraft.position.lon,
            )
            aspect = abs(signed_yaw_deg_diff(getattr(target, "yaw_deg", 0.0), bearing_from_target))
            aot = 180.0 - aspect

            own_speed = float(self.aircraft.speed)
            tgt_speed = float(getattr(target, "speed", 250.0))

            # Closure rate: project both velocities onto the line-of-sight (own→target).
            # own_los = own_speed * cos(ATA)
            # tgt_los = -tgt_speed * cos(angle target→own), sign flipped because heading away reduces closure
            own_los = own_speed * np.cos(np.radians(ata))
            tgt_yaw = float(getattr(target, "yaw_deg", 0.0))
            tgt_to_own_err = abs(signed_yaw_deg_diff(tgt_yaw, bearing_from_target))
            tgt_los = -tgt_speed * np.cos(np.radians(tgt_to_own_err))
            closure = own_los - tgt_los

            slant_range = (
                geodetic_distance_km(
                    self.aircraft.position.lat,
                    self.aircraft.position.lon,
                    self.aircraft.position.alt,
                    target.position.lat,
                    target.position.lon,
                    target.position.alt,
                )
                * 1000.0
            )

            alt_diff = self.aircraft.position.alt - getattr(target.position, "alt", 0.0)

            return {
                "valid": True,
                "ata_deg": ata,
                "aspect_deg": aspect,
                "aot_deg": aot,
                "closure_rate_mps": closure,
                "bearing_deg": bearing,
                "slant_range_m": slant_range,
                "altitude_diff_m": alt_diff,
            }
        except Exception as e:
            return {
                "valid": False,
                "ata_deg": 0.0,
                "aspect_deg": 0.0,
                "aot_deg": 0.0,
                "closure_rate_mps": 0.0,
                "bearing_deg": 0.0,
                "slant_range_m": 0.0,
                "altitude_diff_m": 0.0,
                "error": str(e),
            }

    def get_track_target_quality(self, selected_target=None) -> dict[str, Any]:
        """
        Get track & target quality features.

        Returns:
        - Top-N prioritized tracks with scores
        - Selected target's priority score
        - Number of tracks
        """
        if not hasattr(self.aircraft, "sensor") or not self.aircraft.sensor:
            return {
                "valid": False,
                "num_tracks": 0,
                "prioritized_tracks": [],
                "selected_target_score": 0.0,
            }

        try:
            prioritized = self.aircraft.sensor.get_prio_tracks()
            num_tracks = len(prioritized)

            top_n = min(5, num_tracks)
            top_tracks = [
                {"state": state.tolist(), "score": score} for state, score in prioritized[:top_n]
            ]

            selected_score = 0.0
            if selected_target:
                # Match by nearest track position (state[:3]) to selected target's relative ENU.
                # Track states are assumed to be in own-ENU; degrades gracefully otherwise.
                try:
                    if isinstance(selected_target, TacticalContact):
                        rel_enu = np.asarray(selected_target.state[:3], dtype=float)
                    else:
                        dlat = (
                            selected_target.position.lat - self.aircraft.position.lat
                        ) * 111000.0
                        dlon = (
                            (selected_target.position.lon - self.aircraft.position.lon)
                            * 111000.0
                            * np.cos(np.radians(self.aircraft.position.lat))
                        )
                        dalt = selected_target.position.alt - self.aircraft.position.alt
                        rel_enu = np.array([dlon, dlat, dalt], dtype=float)

                    _best_idx, best_dist = -1, float("inf")
                    for i, (state, score) in enumerate(prioritized):
                        if len(state) >= 3:
                            dist = float(np.linalg.norm(np.array(state[:3]) - rel_enu))
                            if dist < best_dist:
                                best_dist = dist
                                selected_score = float(score)
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    ZeroDivisionError,
                ):
                    pass

            return {
                "valid": True,
                "num_tracks": num_tracks,
                "prioritized_tracks": top_tracks,
                "selected_target_score": selected_score,
            }
        except Exception as e:
            return {
                "valid": False,
                "num_tracks": 0,
                "prioritized_tracks": [],
                "selected_target_score": 0.0,
                "error": str(e),
            }

    def get_fire_feasibility(self, target, simulator=None) -> dict[str, Any]:
        """
        Get fire feasibility flags.

        Thin adapter over :func:`~bvr_marl_core.aircraft.systems.fire_feasibility.evaluate_fire_gates`,
        which is the single definition of "may this aircraft launch right now?" shared
        with the shot-opportunity counter in the action-space layer. See that module for
        why the two used to disagree and what it cost.

        Returns:
        - radar_lock: bool (own radar only)
        - datalink_lock: bool
        - target_in_fov: bool
        - gimbal_ok: bool (computed from ATA vs h_fov)
        - radar_range_ok: bool (own-radar range gate)
        - launch_range_ok: bool (datalink shots may launch beyond own radar range)
        - inventory_ok: bool
        - cooldown_ok: bool (post-launch cooldown expired)
        - target_not_saturated: bool (below the per-target missile cap)
        - can_fire: bool (all gates passed)
        - veto_reason: str or None
        """
        if not target:
            return NO_TARGET_GATES.as_dict()

        return evaluate_fire_gates(
            self.aircraft, target, simulator=simulator, obs_helper=self
        ).as_dict()

    def get_lock_quality(self, target) -> dict[str, Any]:
        """Return lock quality metrics for the selected target."""
        if isinstance(target, TacticalContact):
            geom = self.get_geometry_kinematics(target)
            locked_ids = set(self.aircraft.sensor.get_locked_targets() or ())
            has_lock = target.track_id in locked_ids
            fov_half = float(getattr(self.aircraft.radar, "h_fov_deg", 90.0)) * 0.5
            radar_max = float(getattr(self.aircraft.radar, "max_range_m", 1.0))
            ata_margin = max(0.0, fov_half - float(geom.get("ata_deg", fov_half)))
            range_margin = max(0.0, radar_max - float(geom.get("slant_range_m", radar_max)))
            geometric_quality = 0.6 * min(1.0, ata_margin / 30.0) + 0.4 * min(
                1.0, range_margin / 20_000.0
            )
            return {
                "valid": True,
                "has_lock": has_lock,
                "ata_margin_deg": ata_margin,
                "range_margin": range_margin,
                "lock_strength": (
                    float(np.clip(target.confidence * geometric_quality, 0.0, 1.0))
                    if has_lock
                    else 0.0
                ),
            }
        metrics = getattr(self.aircraft, "metrics", None)
        if metrics is not None and hasattr(metrics, "get_lock_quality"):
            return metrics.get_lock_quality(target)

        if not target:
            return {
                "valid": False,
                "has_lock": False,
                "ata_margin_deg": 0.0,
                "range_margin": 0.0,
                "lock_strength": 0.0,
            }

        return {
            "valid": False,
            "has_lock": False,
            "ata_margin_deg": 0.0,
            "range_margin": 0.0,
            "lock_strength": 0.0,
        }

    def get_threat_warnings(self) -> dict[str, Any]:
        """
        Get threat & missile warning features.

        Returns:
        - num_warnings: int
        - warning_ids: list[int]
        - time_since_first_warning_s: float (placeholder - needs tracking)
        """
        if not hasattr(self.aircraft, "sensor") or not self.aircraft.sensor:
            return {"num_warnings": 0, "warning_ids": [], "time_since_first_warning_s": 0.0}

        try:
            warnings = self.aircraft.sensor.get_missile_warnings()
            num_warnings = len(warnings)
            warning_ids = [getattr(w, "id", 0) for w in warnings]
            time_since_first = 0.0

            return {
                "num_warnings": num_warnings,
                "warning_ids": warning_ids,
                "time_since_first_warning_s": time_since_first,
            }
        except Exception as e:
            return {
                "num_warnings": 0,
                "warning_ids": [],
                "time_since_first_warning_s": 0.0,
                "error": str(e),
            }

    def get_weapons_state(self) -> dict[str, Any]:
        """
        Get weapons state features.

        Returns:
        - remaining_missiles: int (total)
        - remaining_missiles_by_class: dict[str, int] (Fox-1/2/3 counts)
        - gun_ammo: int
        - gun_cooldown_left_s: float
        - missile_cooldown_left_s: float
        """
        try:
            remaining = getattr(self.aircraft, "remaining_missiles", 0)
            if remaining == 0 and hasattr(self.aircraft, "weapons"):
                remaining = getattr(self.aircraft.weapons, "remaining_missiles", 0)

            remaining_by_class = {
                "fox1": 0,
                "fox2": 0,
                "fox3": remaining,  # Currently assumes all are Fox-3
            }

            gun_ammo = 0
            gun_cooldown = 0.0
            if hasattr(self.aircraft, "weapons") and hasattr(self.aircraft.weapons, "gun"):
                gun_ammo = getattr(self.aircraft.weapons.gun, "current_ammo", 0)

            missile_cooldown = 0.0

            return {
                "remaining_missiles": remaining,
                "remaining_missiles_by_class": remaining_by_class,
                "gun_ammo": gun_ammo,
                "gun_cooldown_left_s": gun_cooldown,
                "missile_cooldown_left_s": missile_cooldown,
            }
        except Exception as e:
            return {
                "remaining_missiles": 0,
                "remaining_missiles_by_class": {"fox1": 0, "fox2": 0, "fox3": 0},
                "gun_ammo": 0,
                "gun_cooldown_left_s": 0.0,
                "missile_cooldown_left_s": 0.0,
                "error": str(e),
            }

    def get_comprehensive_observation(self, target, sim_time: float = 0.0) -> dict[str, Any]:
        """
        Get comprehensive observation combining all subsystems.

        Args:
            target: Selected target (or None)
            sim_time: Simulation time for time-based features

        Returns:
            Structured dict with all observation features
        """
        return {
            "timestamp": sim_time,
            "dlz_nez": self.get_dlz_nez_features(target),
            "geometry": self.get_geometry_kinematics(target),
            "track_quality": self.get_track_target_quality(target),
            "fire_feasibility": self.get_fire_feasibility(target),
            "threats": self.get_threat_warnings(),
            "weapons": self.get_weapons_state(),
        }
