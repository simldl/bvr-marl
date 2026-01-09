import logging
import numpy as np
logger = logging.getLogger(__name__)
from simulator.utils.angles import signed_yaw_deg_diff
from simulator.utils.geodesics import geodetic_bearing_deg
from aircrafts.control.gun_projectile import GunSystem


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
            burst_duration_s=gun_config.get("burst_duration_s", 0.1)
        )

    def should_fire(self, fire=1):
        return fire > 0.5

    def is_target_in_fov(self, target):
        """Check if target lies within radar FOV (true ATA with guard for missing radar)."""
        from simulator.utils.angles import signed_yaw_deg_diff
        from simulator.utils.geodesics import geodetic_bearing_deg

        target_bearing = geodetic_bearing_deg(
            self.parent.position.lat, self.parent.position.lon,
            target.position.lat, target.position.lon
        )
        angle_diff = abs(signed_yaw_deg_diff(self.parent.yaw_deg, target_bearing))

        # Guard: if no radar object, default to a reasonable HFOV (e.g., 90°)
        hfov = getattr(getattr(self.parent, "radar", None), "h_fov_deg", 90.0)
        return angle_diff < (float(hfov) * 0.5)

    def check_fire_feasibility(self, target):
        """
        Check if firing at target is feasible (for observations).
        Returns dict with all gate statuses.
        """
        feasibility = {
            'inventory_ok': False,
            'radar_lock_ok': False,
            'fov_ok': False,
            'can_fire': False,
            'remaining_missiles': 0,
            'veto_reason': None
        }

        # Check inventory
        remaining = getattr(self.parent, 'remaining_missiles', None)
        if remaining is None:
            remaining = self.remaining_missiles
        feasibility['remaining_missiles'] = remaining
        if remaining <= 0:
            feasibility['veto_reason'] = 'no_inventory'
            return feasibility
        feasibility['inventory_ok'] = True

        # Check radar lock
        if not hasattr(self.parent, "sensor") or not self.parent.sensor.has_radar_lock(target):
            feasibility['veto_reason'] = 'no_radar_lock'
            return feasibility
        feasibility['radar_lock_ok'] = True

        # Check FOV
        if not self.is_target_in_fov(target):
            feasibility['veto_reason'] = 'not_in_fov'
            return feasibility
        feasibility['fov_ok'] = True

        # All gates passed
        feasibility['can_fire'] = True
        return feasibility

    def select_and_engage_target(self, target_candidates, target_selector, fire_action, gun_fire_action, simulator):
        if not target_candidates:
            return None

        selected = self.parent.sensor.select_target(target_candidates, target_selector)
        self.parent.target = selected
        
        if selected is not None:
            # Missile firing
            if fire_action > 0.5:
                self.fire_missile(simulator, selected, self.missile_types[0])
            
            # Gun firing
            if gun_fire_action > 0.5:
                self.fire_gun(simulator, selected)
                
        return selected
    
    def fire_gun(self, sim, target=None):
        """Fire the gun at a target or straight ahead."""
        if not self.gun.can_fire(sim.utc_time):
            return []
            
        # Calculate target position
        target_position = None
        if target is not None and hasattr(target, "position"):
            # Lead the target based on relative velocity and projectile speed
            target_position = self._calculate_gun_intercept(target)
        
        return self.gun.start_firing(sim, target_position)
    
    def _calculate_gun_intercept(self, target):
        """Calculate intercept point for gun firing."""
        # Simple lead calculation
        # Get relative position and velocity
        rel_pos = np.array([
            target.position.lon - self.parent.position.lon,
            target.position.lat - self.parent.position.lat,
            target.position.alt - self.parent.position.alt
        ])
        
        # Convert to meters (rough approximation)
        rel_pos_m = rel_pos * np.array([111000 * np.cos(np.radians(self.parent.position.lat)), 111000, 1])
        
        # Get relative velocity
        rel_vel = np.array([
            target.velocity.vx - self.parent.velocity.vx,
            target.velocity.vy - self.parent.velocity.vy, 
            target.velocity.vz - self.parent.velocity.vz
        ])
        
        # Time to intercept (simplified)
        range_to_target = np.linalg.norm(rel_pos_m)
        projectile_speed = self.gun.muzzle_velocity_mps
        
        if range_to_target > 0:
            time_to_intercept = range_to_target / projectile_speed
            
            # Predicted target position
            predicted_pos = rel_pos_m + rel_vel * time_to_intercept
            
            # Convert back to lat/lon
            predicted_lat = self.parent.position.lat + predicted_pos[1] / 111000
            predicted_lon = self.parent.position.lon + predicted_pos[0] / (111000 * np.cos(np.radians(self.parent.position.lat)))
            predicted_alt = self.parent.position.alt + predicted_pos[2]
            
            return (predicted_lon, predicted_lat, predicted_alt)
        
        # Fallback to current target position
        return (target.position.lon, target.position.lat, target.position.alt)

    def fire_missile(self, sim, target, missile_cls):
        """
        Fire a missile with comprehensive gating logic.
        This is the SINGLE SOURCE OF TRUTH for all fire authorization.

        Returns:
            - On success: (missile, None, diagnostics_dict)
            - On failure: (None, veto_reason_str, diagnostics_dict)

        diagnostics_dict contains gate check results for logging/debugging
        """
        # Initialize diagnostics
        diagnostics = {
            'has_inventory': False,
            'has_lock': False,
            'in_locked_tracks': False,
            'in_fov': False,
            'target_id': getattr(target, 'id', None),
        }

        # Gate 1: Inventory check
        remaining = getattr(self.parent, 'remaining_missiles', None)
        if remaining is None:
            remaining = self.remaining_missiles
        diagnostics['has_inventory'] = remaining > 0
        diagnostics['remaining'] = remaining

        if remaining <= 0:
            veto = f"no_inventory(remaining={remaining})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        # Gate 2: Radar lock check
        has_sensor = hasattr(self.parent, "sensor") and self.parent.sensor is not None
        has_lock = has_sensor and self.parent.sensor.has_radar_lock(target) if target else False
        diagnostics['has_lock'] = has_lock

        if not has_lock:
            veto = f"no_radar_lock(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        # Gate 3: Locked tracks verification
        locked_tracks = self.parent.sensor.get_locked_targets()
        tracker_state = None

        if isinstance(locked_tracks, dict):
            if target.id not in locked_tracks:
                veto = f"not_in_locked_tracks_dict(target={target.id},locked={list(locked_tracks.keys())})"
                logger.debug(f"Fire veto: {veto}")
                return None, veto, diagnostics
            diagnostics['in_locked_tracks'] = True
            entry = locked_tracks[target.id]
            tracker_state = entry[0] if isinstance(entry, tuple) else entry
        elif isinstance(locked_tracks, set):
            if target.id not in locked_tracks:
                veto = f"not_in_locked_tracks_set(target={target.id},locked={locked_tracks})"
                logger.debug(f"Fire veto: {veto}")
                return None, veto, diagnostics
            diagnostics['in_locked_tracks'] = True
            tracker_state = None
        else:
            veto = f"invalid_locked_tracks_type(type={type(locked_tracks)})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        # Gate 4: FOV check (true angle-to-target within radar gimbal)
        in_fov = self.is_target_in_fov(target)
        diagnostics['in_fov'] = in_fov

        if not in_fov:
            veto = f"not_in_fov(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        # All gates passed - create and fire the missile
        tracker_vel = tracker_state[3:6] if tracker_state is not None and len(tracker_state) >= 6 else None
        tracker_ref = self.parent.position.copy() if tracker_state is not None else None

        missile = missile_cls(
            firing_time_s=sim.utc_time,
            target=target,
            source=self.parent,
            map_limits=self.parent.map_limits,
            group=self.parent.group
        )
        missile.designated_target_id = getattr(target, "id", None)
        missile.retarget_policy = "locked_override"

        if tracker_state is not None:
            missile.initial_tracked_position_enu = tracker_state[:3]
            missile.initial_tracked_velocity_enu = tracker_vel
            missile.tracker_reference_pos = tracker_ref
        missile.tracking_lock = True

        if hasattr(missile.radar, "lock_ctrl"):
            missile.radar.lock_ctrl.set_mode("track", target.id)

        sim.add_unit(missile)
        self.parent.missiles.append(missile)
        self.remaining_missiles -= 1

        logger.info(f"Missile fired successfully: target={target.id}, remaining={self.remaining_missiles}")
        diagnostics['fired'] = True
        return missile, None, diagnostics