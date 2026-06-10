"""Gun projectile implementation for close-range air-to-air combat."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from bvr_marl_core.simulator.core.events import Event
from bvr_marl_core.simulator.core.units import FlyingUnit
from bvr_marl_core.simulator.simulator import Simulator


class GunProjectilePhysics:
    """Simple ballistic physics for gun projectiles."""

    def __init__(
        self,
        initial_velocity_mps: float = 1000.0,
        drag_coefficient: float = 0.1,
        mass_kg: float = 0.1,
    ):
        self.initial_velocity_mps = initial_velocity_mps
        self.drag_coefficient = drag_coefficient
        self.mass_kg = mass_kg
        self.gravity_mps2 = 9.81

    def update_position(self, projectile, dt_s: float):
        """Update projectile position using ballistic physics."""
        if hasattr(projectile, "_velocity_override"):
            velocity_vector = projectile._velocity_override.copy()
        else:
            velocity_vector = np.array(
                [projectile.velocity.vx, projectile.velocity.vy, projectile.velocity.vz]
            )

        if not np.all(np.isfinite(velocity_vector)):
            return

        speed = np.linalg.norm(velocity_vector)
        if speed > 1e-6:
            # Clamp drag so it can't reverse velocity direction in one step
            # (Euler stability: max deceleration is speed/dt_s).
            drag_decel = 0.5 * self.drag_coefficient * speed**2 / self.mass_kg
            max_decel = speed / dt_s if dt_s > 0 else speed
            drag_decel = min(drag_decel, max_decel)
            acceleration_drag = -drag_decel * (velocity_vector / speed)
        else:
            acceleration_drag = np.zeros(3)

        total_acceleration = acceleration_drag + np.array([0, 0, -self.gravity_mps2])
        velocity_vector += total_acceleration * dt_s

        if hasattr(projectile, "_velocity_override"):
            projectile._velocity_override = velocity_vector

        # velocity_vector is [East, North, Up] m/s
        projectile.position.lat += velocity_vector[1] * dt_s / 111000.0
        cos_lat = np.cos(np.radians(projectile.position.lat))
        if abs(cos_lat) > 1e-9:
            projectile.position.lon += velocity_vector[0] * dt_s / (111000.0 * cos_lat)
        projectile.position.alt += velocity_vector[2] * dt_s


class GunProjectile(FlyingUnit):
    """Individual gun projectile/bullet."""

    def __init__(
        self,
        firing_time: datetime,
        source_aircraft,
        target_position,
        source_velocity,
        muzzle_velocity_mps: float = 1000.0,
        max_range_m: float = 2000.0,
        group: str = None,
        kill_radius_m: float = 5.0,
    ):

        import time

        simple_name = f"Bullet_{getattr(source_aircraft, 'id', 'unknown')}_{int(time.time() * 1000000) % 1000000}"
        super().__init__(
            name=simple_name,
            position=source_aircraft.position.copy(),
            yaw_deg=source_aircraft.yaw_deg,
            speed=muzzle_velocity_mps,
        )

        self.group = group if group is not None else source_aircraft.group
        self.type = "GunProjectile"
        self.firing_time = firing_time
        self.source_aircraft = source_aircraft
        self.max_range_m = max_range_m
        self.traveled_distance_m = 0.0

        aircraft_velocity = np.array(
            [source_velocity.vx, source_velocity.vy, source_velocity.vz], dtype=float
        )

        # Build ENU delta (meters) from target tuple (lon, lat, alt)
        dE = (
            (target_position[0] - self.position.lon)
            * 111000.0
            * np.cos(np.radians(self.position.lat))
        )
        dN = (target_position[1] - self.position.lat) * 111000.0
        dU = target_position[2] - self.position.alt
        target_dir_m = np.array([dE, dN, dU], dtype=float)

        n = np.linalg.norm(target_dir_m)
        if n < 1e-9:
            yaw = np.radians(self.yaw_deg)
            target_dir_m = np.array([np.sin(yaw), np.cos(yaw), 0.0], dtype=float)
        else:
            target_dir_m = target_dir_m / n

        # velocity property is calculated from speed/yaw/pitch; store ENU velocity separately
        self._velocity_override = aircraft_velocity + target_dir_m * float(muzzle_velocity_mps)

        self.physics = GunProjectilePhysics(
            initial_velocity_mps=muzzle_velocity_mps, drag_coefficient=0.1, mass_kg=0.1
        )

        self.damage = 50.0
        self.kill_radius_m = float(kill_radius_m)

    def __hash__(self):
        """Make GunProjectile hashable for use in sets and dictionaries."""
        return hash(id(self))

    def __eq__(self, other):
        """Equality comparison for GunProjectile."""
        return isinstance(other, GunProjectile) and id(self) == id(other)

    def update(self, tick_secs: float, sim: Simulator) -> list[Event]:
        """Update projectile physics and check for hits/expiration."""
        events = []

        old_position = (self.position.lat, self.position.lon, self.position.alt)
        self.physics.update_position(self, tick_secs)
        new_position = (self.position.lat, self.position.lon, self.position.alt)

        if not all(map(lambda v: v == v and abs(v) < 1e15, new_position)):
            sim.remove_unit(self)
            return events

        dx = (
            (new_position[0] - old_position[0])
            * 111000.0
            * np.cos(np.radians((new_position[1] + old_position[1]) * 0.5))
        )
        dy = (new_position[1] - old_position[1]) * 111000.0
        dz = new_position[2] - old_position[2]
        self.traveled_distance_m += float(np.linalg.norm([dx, dy, dz]))

        if self.traveled_distance_m > self.max_range_m or self.position.alt < 0:
            sim.remove_unit(self)
            return events

        for unit_id, unit in sim.active_units.items():
            if (
                unit.id != self.id
                and unit.group != self.group
                and unit.type == "Aircraft"
                and unit.id != self.source_aircraft.id
            ):
                if self._calculate_distance_to_unit(unit) <= self.kill_radius_m:
                    from bvr_marl_core.simulator.core.events import UnitDestroyedEvent

                    events.append(
                        UnitDestroyedEvent(
                            origin=sim, unit_killer=self.source_aircraft, unit_destroyed=unit
                        )
                    )
                    sim.remove_unit(self)
                    break

        return events

    def _calculate_distance_to_unit(self, unit) -> float:
        dx = (
            (unit.position.lon - self.position.lon) * 111000 * np.cos(np.radians(self.position.lat))
        )
        dy = (unit.position.lat - self.position.lat) * 111000
        dz = unit.position.alt - self.position.alt
        return np.sqrt(dx**2 + dy**2 + dz**2)


class GunSystem:
    """Aircraft gun system managing projectiles and ammunition."""

    def __init__(
        self,
        parent_aircraft,
        max_ammo: int = 500,
        muzzle_velocity_mps: float = 1000.0,
        max_range_m: float = 2000.0,
        burst_size: int = 10,
        burst_duration_s: float = 0.1,
    ):
        self.parent = parent_aircraft
        self.max_ammo = max_ammo
        self.current_ammo = max_ammo
        self.muzzle_velocity_mps = muzzle_velocity_mps
        self.max_range_m = max_range_m
        self.burst_size = burst_size
        self.burst_duration_s = burst_duration_s

        self.is_firing = False
        self.burst_start_time = None
        self.rounds_fired_in_burst = 0
        self.last_fire_time = None
        self.min_fire_interval_s = 0.05

    def can_fire(self, current_time: datetime) -> bool:
        """Check if gun can fire (has ammo and not in cooldown)."""
        if self.current_ammo <= 0:
            return False

        if self.last_fire_time is not None:
            time_since_last = (current_time - self.last_fire_time).total_seconds()
            if time_since_last < self.min_fire_interval_s:
                return False

        return True

    def start_firing(self, sim: Simulator, target_position: tuple = None) -> list[GunProjectile]:
        """Start a gun burst."""
        if not self.can_fire(sim.utc_time):
            return []

        self.is_firing = True
        self.burst_start_time = sim.utc_time
        self.rounds_fired_in_burst = 0
        self.last_fire_time = sim.utc_time

        return self._fire_burst(sim, target_position)

    def _fire_burst(self, sim: Simulator, target_position: tuple = None) -> list[GunProjectile]:
        """Fire a burst of projectiles."""
        projectiles = []

        if target_position is None:
            range_ahead = 1000.0
            target_position = (
                self.parent.position.lon
                + range_ahead * np.cos(np.radians(self.parent.yaw_deg)) / 111000,
                self.parent.position.lat
                + range_ahead * np.sin(np.radians(self.parent.yaw_deg)) / 111000,
                self.parent.position.alt,
            )

        rounds_to_fire = min(self.burst_size, self.current_ammo)

        for i in range(rounds_to_fire):
            spread_factor = 0.02
            spread_x = (np.random.random() - 0.5) * spread_factor * 1000
            spread_y = (np.random.random() - 0.5) * spread_factor * 1000
            spread_z = (np.random.random() - 0.5) * spread_factor * 500

            spread_target = (
                target_position[0]
                + spread_x / (111000.0 * np.cos(np.radians(self.parent.position.lat))),
                target_position[1] + spread_y / 111000.0,
                target_position[2] + spread_z,
            )
            projectile = GunProjectile(
                firing_time=sim.utc_time,
                source_aircraft=self.parent,
                target_position=spread_target,
                source_velocity=self.parent.velocity,
                muzzle_velocity_mps=self.muzzle_velocity_mps,
                max_range_m=self.max_range_m,
                group=self.parent.group,
                kill_radius_m=getattr(sim, "gun_hit_radius_m", 5.0),
            )

            sim.add_unit(projectile)
            projectiles.append(projectile)

        self.current_ammo -= rounds_to_fire
        self.rounds_fired_in_burst += rounds_to_fire
        self.is_firing = False

        return projectiles
