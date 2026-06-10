import logging

import numpy as np

from bvr_marl_core.aircraft.control.gun_projectile import GunSystem
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff
from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg

logger = logging.getLogger(__name__)


class AircraftWeaponSystem:
    def __init__(self, parent):
        self.parent = parent
        self.missile_types = getattr(parent, "missile_types", [])
        self.max_missiles = getattr(parent, "max_missiles", 8)
        self.remaining_missiles = getattr(parent, "max_missiles", 8)
        self.missiles = getattr(parent, "missiles", [])

        # Gun system
        gun_config = getattr(parent, "gun_config", {})
        self.gun = GunSystem(
            parent_aircraft=parent,
            max_ammo=gun_config.get("max_ammo", 500),
            muzzle_velocity_mps=gun_config.get("muzzle_velocity_mps", 1000.0),
            max_range_m=gun_config.get("max_range_m", 800.0),
            burst_size=gun_config.get("burst_size", 10),
            burst_duration_s=gun_config.get("burst_duration_s", 0.1),
        )

    def should_fire(self, fire=1):
        return fire > 0.5

    def is_target_in_fov(self, target):
        """
        Check if target lies within the lock FOV for weapons engagement.

        For most aircraft, this is the radar's horizontal FOV.
        For AWACS (and similar support aircraft), this uses a separate lock_fov_deg
        that is narrower than the detection FOV. AWACS can detect in 360° but
        can only lock/engage targets within lock_fov_deg of their heading.

        Args:
            target: The target unit to check.

        Returns:
            True if target is within the lock FOV.
        """

        target_bearing = geodetic_bearing_deg(
            self.parent.position.lat,
            self.parent.position.lon,
            target.position.lat,
            target.position.lon,
        )
        angle_diff = abs(signed_yaw_deg_diff(self.parent.yaw_deg, target_bearing))

        # AWACS uses a narrower lock_fov_deg than its 360° detection FOV
        if hasattr(self.parent, "lock_fov_deg") and self.parent.lock_fov_deg is not None:
            return angle_diff < (float(self.parent.lock_fov_deg) * 0.5)

        hfov = getattr(getattr(self.parent, "radar", None), "h_fov_deg", 90.0)
        return angle_diff < (float(hfov) * 0.5)

    def check_fire_feasibility(self, target):
        """
        Check if firing at target is feasible (for observations).
        Returns dict with all gate statuses.
        """
        feasibility = {
            "inventory_ok": False,
            "radar_lock_ok": False,
            "fov_ok": False,
            "can_fire": False,
            "remaining_missiles": 0,
            "veto_reason": None,
        }

        remaining = getattr(self.parent, "remaining_missiles", None)
        if remaining is None:
            remaining = self.remaining_missiles
        feasibility["remaining_missiles"] = remaining
        if remaining <= 0:
            feasibility["veto_reason"] = "no_inventory"
            return feasibility
        feasibility["inventory_ok"] = True

        if not hasattr(self.parent, "sensor") or not self.parent.sensor.has_radar_lock(target):
            feasibility["veto_reason"] = "no_radar_lock"
            return feasibility
        feasibility["radar_lock_ok"] = True

        if not self.is_target_in_fov(target):
            feasibility["veto_reason"] = "not_in_fov"
            return feasibility
        feasibility["fov_ok"] = True

        feasibility["can_fire"] = True
        return feasibility

    def select_and_engage_target(
        self, target_candidates, target_selector, fire_action, gun_fire_action, simulator
    ):
        if not target_candidates:
            return None

        selected = self.parent.sensor.select_target(target_candidates, target_selector)
        self.parent.target = selected

        if selected is not None:
            if fire_action > 0.5:
                self.fire_missile(simulator, selected, self.missile_types[0])
            if gun_fire_action > 0.5:
                self.fire_gun(simulator, selected)

        return selected

    def fire_gun(self, sim, target=None):
        """Fire the gun at a target or straight ahead."""
        if not self.gun.can_fire(sim.utc_time):
            return []
        target_position = (
            self._calculate_gun_intercept(target)
            if target is not None and hasattr(target, "position")
            else None
        )
        return self.gun.start_firing(sim, target_position)

    def _calculate_gun_intercept(self, target):
        """Calculate lead-angle intercept point for gun firing."""
        rel_pos = np.array(
            [
                target.position.lon - self.parent.position.lon,
                target.position.lat - self.parent.position.lat,
                target.position.alt - self.parent.position.alt,
            ]
        )
        rel_pos_m = rel_pos * np.array(
            [111000 * np.cos(np.radians(self.parent.position.lat)), 111000, 1]
        )
        rel_vel = np.array(
            [
                target.velocity.vx - self.parent.velocity.vx,
                target.velocity.vy - self.parent.velocity.vy,
                target.velocity.vz - self.parent.velocity.vz,
            ]
        )
        range_to_target = np.linalg.norm(rel_pos_m)
        if range_to_target > 0:
            time_to_intercept = range_to_target / self.gun.muzzle_velocity_mps
            predicted_pos = rel_pos_m + rel_vel * time_to_intercept
            predicted_lat = self.parent.position.lat + predicted_pos[1] / 111000
            predicted_lon = self.parent.position.lon + predicted_pos[0] / (
                111000 * np.cos(np.radians(self.parent.position.lat))
            )
            predicted_alt = self.parent.position.alt + predicted_pos[2]
            return (predicted_lon, predicted_lat, predicted_alt)
        return (target.position.lon, target.position.lat, target.position.alt)

    def fire_missile(self, sim, target, missile_cls):
        """
        Fire a missile with comprehensive gating logic.

        Returns:
            - On success: (missile, None, diagnostics_dict)
            - On failure: (None, veto_reason_str, diagnostics_dict)
        """
        diagnostics = {
            "has_inventory": False,
            "has_lock": False,
            "in_locked_tracks": False,
            "in_fov": False,
            "target_id": getattr(target, "id", None),
            "is_engageable": True,
        }

        # Gate 0: Filter out support assets (e.g. AWACS)
        if getattr(target, "is_non_engageable", False):
            diagnostics["is_engageable"] = False
            veto = f"target_non_engageable(target={getattr(target, 'id', 'unknown')}, is_support={getattr(target, 'is_support_asset', False)})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        remaining = getattr(self.parent, "remaining_missiles", None)
        if remaining is None:
            remaining = self.remaining_missiles
        diagnostics["has_inventory"] = remaining > 0
        diagnostics["remaining"] = remaining

        if remaining <= 0:
            veto = f"no_inventory(remaining={remaining})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        has_sensor = hasattr(self.parent, "sensor") and self.parent.sensor is not None
        has_lock = has_sensor and self.parent.sensor.has_radar_lock(target) if target else False
        diagnostics["has_lock"] = has_lock

        if not has_lock:
            veto = f"no_radar_lock(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        locked_tracks = self.parent.sensor.get_locked_targets()
        tracker_state = None

        if isinstance(locked_tracks, dict):
            if target.id not in locked_tracks:
                veto = f"not_in_locked_tracks_dict(target={target.id},locked={list(locked_tracks.keys())})"
                logger.debug(f"Fire veto: {veto}")
                return None, veto, diagnostics
            diagnostics["in_locked_tracks"] = True
            entry = locked_tracks[target.id]
            tracker_state = entry[0] if isinstance(entry, tuple) else entry
        elif isinstance(locked_tracks, set):
            if target.id not in locked_tracks:
                veto = f"not_in_locked_tracks_set(target={target.id},locked={locked_tracks})"
                logger.debug(f"Fire veto: {veto}")
                return None, veto, diagnostics
            diagnostics["in_locked_tracks"] = True
            # Retrieve the 6-DOF track state from the radar's cached tracks.
            # state[:3] is ENU offset from the shooter position.
            tracker_state = None
            radar = getattr(self.parent, "radar", None)
            if radar is not None:
                for track in getattr(radar, "cached_tracks", None) or []:
                    if len(track) >= 4 and getattr(track[3], "id", None) == target.id:
                        tracker_state = track[1]
                        break
        else:
            veto = f"invalid_locked_tracks_type(type={type(locked_tracks)})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        in_fov = self.is_target_in_fov(target)
        diagnostics["in_fov"] = in_fov

        if not in_fov:
            veto = f"not_in_fov(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        tracker_vel = (
            tracker_state[3:6] if tracker_state is not None and len(tracker_state) >= 6 else None
        )
        tracker_ref = self.parent.position.copy() if tracker_state is not None else None

        missile = missile_cls(
            firing_time_s=sim.utc_time,
            target=target,
            source=self.parent,
            map_limits=self.parent.map_limits,
            group=self.parent.group,
        )
        target_id = getattr(target, "id", None)
        missile.designated_target_id = target_id
        missile.retarget_policy = "locked_override"
        if hasattr(missile, "target_provider"):
            missile.target_provider.current_target_id = target_id

        if tracker_state is not None:
            missile.initial_tracked_position_enu = tracker_state[:3]
            missile.initial_tracked_velocity_enu = tracker_vel
            missile.tracker_reference_pos = tracker_ref
        missile.tracking_lock = True

        if hasattr(missile.radar, "lock_ctrl"):
            missile.radar.lock_ctrl.set_mode("track", target_id)
        if hasattr(missile, "radar"):
            missile.radar.locked_target = target_id

        sim.add_unit(missile)
        self.parent.missiles.append(missile)
        self.remaining_missiles -= 1

        logger.info(
            f"Missile fired successfully: target={target.id}, remaining={self.remaining_missiles}"
        )
        diagnostics["fired"] = True
        return missile, None, diagnostics

    def fire_missile_direct(self, sim, target, missile_cls):
        """
        Fire a missile without requiring a radar lock (for simplified environments).

        Bypasses radar lock / FOV gates. Only checks inventory and target engageability.
        Seeds the missile with the true target position and velocity at launch time so
        PN guidance has a good initial estimate without needing a tracker.

        Returns:
            - On success: (missile, None, diagnostics_dict)
            - On failure: (None, veto_reason_str, diagnostics_dict)
        """
        diagnostics = {
            "has_inventory": False,
            "has_lock": False,
            "in_locked_tracks": False,
            "in_fov": True,
            "target_id": getattr(target, "id", None),
            "is_engageable": True,
        }

        if getattr(target, "is_non_engageable", False):
            diagnostics["is_engageable"] = False
            veto = f"target_non_engageable(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto (direct): {veto}")
            return None, veto, diagnostics

        remaining = getattr(self.parent, "remaining_missiles", None)
        if remaining is None:
            remaining = self.remaining_missiles
        diagnostics["has_inventory"] = remaining > 0
        diagnostics["remaining"] = remaining

        if remaining <= 0:
            veto = f"no_inventory(remaining={remaining})"
            logger.debug(f"Fire veto (direct): {veto}")
            return None, veto, diagnostics

        # Seed missile with truth ENU offset so PN guidance bootstraps correctly
        try:
            from bvr_marl_core.radar.core.utils import geodetic_to_enu

            shooter_pos = self.parent.position
            tgt_pos = target.position
            enu_offset = np.asarray(
                geodetic_to_enu(
                    tgt_pos.lat,
                    tgt_pos.lon,
                    tgt_pos.alt,
                    shooter_pos.lat,
                    shooter_pos.lon,
                    shooter_pos.alt,
                ),
                dtype=np.float32,
            )
        except Exception:
            enu_offset = None

        tgt_vel_enu = None
        if hasattr(target, "velocity_enu"):
            tgt_vel_enu = np.asarray(target.velocity_enu, dtype=np.float32)

        missile = missile_cls(
            firing_time_s=sim.utc_time,
            target=target,
            source=self.parent,
            map_limits=self.parent.map_limits,
            group=self.parent.group,
        )
        target_id = getattr(target, "id", None)
        missile.designated_target_id = target_id
        missile.retarget_policy = "truth_fallback"
        if hasattr(missile, "target_provider"):
            missile.target_provider.current_target_id = target_id

        if enu_offset is not None:
            tracker_state = np.zeros(6, dtype=np.float32)
            tracker_state[:3] = enu_offset
            if tgt_vel_enu is not None:
                tracker_state[3:6] = tgt_vel_enu
            missile.initial_tracked_position_enu = tracker_state[:3]
            missile.initial_tracked_velocity_enu = tracker_state[3:6]
            missile.tracker_reference_pos = self.parent.position.copy()

        missile.tracking_lock = True

        if hasattr(missile, "radar") and hasattr(missile.radar, "lock_ctrl"):
            missile.radar.lock_ctrl.set_mode("track", target_id)
        if hasattr(missile, "radar"):
            missile.radar.locked_target = target_id

        sim.add_unit(missile)
        self.parent.missiles.append(missile)
        self.remaining_missiles -= 1

        logger.info(
            f"Missile fired (direct): target={getattr(target, 'id', 'none')}, remaining={self.remaining_missiles}"
        )
        diagnostics["fired"] = True
        return missile, None, diagnostics
