"""
AWACS orbit controller for automated patrol patterns.

Implements racetrack and figure-8 orbit patterns for AWACS aircraft
to maintain station while providing radar coverage.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from air_to_air_rl.simulator.core.helpers import Position


@dataclass
class OrbitConfig:
    """Configuration for an orbit pattern."""

    # Orbit center position
    center_lat: float = 0.0
    center_lon: float = 0.0

    # Orbit dimensions
    leg_length_km: float = 40.0  # Length of straight legs (racetrack)
    turn_radius_km: float = 25.0  # Turn radius at ends (racetrack) / circle radius (figure8)

    # Orbit altitude
    altitude_m: float = 10000.0

    # Patrol speed
    speed_mps: float = 200.0

    # Orbit direction (clockwise or counterclockwise)
    clockwise: bool = True

    # Orbit type
    pattern: Literal["racetrack", "figure8", "circle", "random", "random_all"] = "racetrack"

    # Zone bounds for RandomWaypointController (None → derived from center ± 2*turn_radius)
    zone_min_lat: float | None = None
    zone_max_lat: float | None = None
    zone_min_lon: float | None = None
    zone_max_lon: float | None = None


class RacetrackController:
    """
    Controller for racetrack orbit pattern.

    Implements a standard racetrack holding pattern where the aircraft
    flies straight legs connected by 180-degree turns.
    """

    def __init__(self, config: OrbitConfig | None = None):
        """
        Initialize the racetrack controller.

        Args:
            config: Orbit configuration. Uses defaults if not provided.
        """
        self.config = config or OrbitConfig()

        self.km_per_deg = 111.0
        self.phase = "outbound"
        self.phase_progress = 0.0
        self.target_heading = 90.0  # Start heading east
        self._compute_waypoints()

    def _compute_waypoints(self):
        """Compute waypoint positions for the racetrack pattern (East-West aligned)."""
        half_leg_deg = (self.config.leg_length_km / 2.0) / self.km_per_deg
        turn_deg = self.config.turn_radius_km / self.km_per_deg

        self.wp1 = Position(
            lat=self.config.center_lat,
            lon=self.config.center_lon - half_leg_deg,
            alt=self.config.altitude_m,
        )
        self.wp2 = Position(
            lat=self.config.center_lat,
            lon=self.config.center_lon + half_leg_deg,
            alt=self.config.altitude_m,
        )

        # Turn centers
        if self.config.clockwise:
            self.turn1_center = Position(
                lat=self.config.center_lat - turn_deg,
                lon=self.config.center_lon + half_leg_deg,
                alt=self.config.altitude_m,
            )
            self.turn2_center = Position(
                lat=self.config.center_lat + turn_deg,
                lon=self.config.center_lon - half_leg_deg,
                alt=self.config.altitude_m,
            )
        else:
            self.turn1_center = Position(
                lat=self.config.center_lat + turn_deg,
                lon=self.config.center_lon + half_leg_deg,
                alt=self.config.altitude_m,
            )
            self.turn2_center = Position(
                lat=self.config.center_lat - turn_deg,
                lon=self.config.center_lon - half_leg_deg,
                alt=self.config.altitude_m,
            )

    def update(self, unit, dt: float) -> tuple[float, float, float]:
        """
        Update the controller and compute desired heading/speed/altitude.

        Args:
            unit: The aircraft unit being controlled.
            dt: Time step in seconds.

        Returns:
            tuple of (desired_heading_deg, desired_speed_mps, desired_altitude_m).
        """
        pos = unit.position

        # Determine which phase we're in based on position
        phase, target_heading = self._determine_phase(pos)
        self.phase = phase
        self.target_heading = target_heading

        return target_heading, self.config.speed_mps, self.config.altitude_m

    def _determine_phase(self, pos: Position) -> tuple[str, float]:
        """
        Determine the current phase and target heading based on position.

        Returns:
            tuple of (phase_name, target_heading).
        """
        half_leg_deg = (self.config.leg_length_km / 2.0) / self.km_per_deg
        turn_deg = self.config.turn_radius_km / self.km_per_deg

        in_east_turn = pos.lon > (self.config.center_lon + half_leg_deg - turn_deg)
        in_west_turn = pos.lon < (self.config.center_lon - half_leg_deg + turn_deg)

        if in_east_turn:
            if self.config.clockwise:
                if pos.lat > self.config.center_lat:
                    return "turn1", 180.0
                else:
                    return "inbound", 270.0
            else:
                if pos.lat < self.config.center_lat:
                    return "turn1", 0.0
                else:
                    return "inbound", 270.0

        elif in_west_turn:
            if self.config.clockwise:
                if pos.lat < self.config.center_lat:
                    return "turn2", 0.0
                else:
                    return "outbound", 90.0
            else:
                if pos.lat > self.config.center_lat:
                    return "turn2", 180.0
                else:
                    return "outbound", 90.0

        else:
            if pos.lat > self.config.center_lat:
                return "outbound", 90.0
            else:
                return "inbound", 270.0

    def get_commanded_controls(self, unit, dt: float) -> dict:
        """Get control commands for the aircraft."""
        heading, speed, altitude = self.update(unit, dt)
        max_speed = getattr(unit, "max_speed_mps", 250.0)
        throttle = np.clip(speed / max_speed, 0.0, 1.0)

        return {
            "throttle": throttle,
            "target_altitude_m": altitude,
            "target_heading_deg": heading,
        }


class Figure8Controller:
    """
    Controller for a north-south oriented figure-8 orbit pattern.

    The figure-8 consists of two circles of radius `turn_radius_km` stacked
    vertically (north/south) and centered at the given position. This is used
    for AWACS on the east/west flanks to maximize NS radar coverage while
    remaining within their designated side zone.

    Pattern geometry:
      - North circle center: (center_lat + radius, center_lon)
      - South circle center: (center_lat - radius, center_lon)
      - Crossover point:     (center_lat, center_lon)

    The aircraft crosses through the center point twice per cycle, creating
    the classic figure-8 crossing.
    """

    def __init__(self, config: OrbitConfig | None = None):
        if config is None:
            config = OrbitConfig(pattern="figure8")
        else:
            config.pattern = "figure8"
        self.config = config
        self.km_per_deg = 111.0

        self._generate_waypoints()
        self.current_wp_idx = 0

    def _generate_waypoints(self, num_per_loop: int = 12):
        """
        Generate waypoints tracing the NS-oriented figure-8 path.

        The path is divided into two loops:
          - North loop: counterclockwise circle above center_lat
          - South loop: clockwise circle below center_lat

        Each loop starts at the crossover point (center_lat, center_lon).
        The crossover appears at waypoint index 0 and index num_per_loop,
        giving the aircraft two passes through the center per cycle.

        Args:
            num_per_loop: Number of waypoints per circle loop.
        """
        r_deg = self.config.turn_radius_km / self.km_per_deg
        c_lat = self.config.center_lat
        c_lon = self.config.center_lon

        waypoints = []

        # North loop: counterclockwise circle, center at (c_lat + r_deg, c_lon)
        # angle=-pi/2 puts the starting point at the bottom of the circle = crossover
        for i in range(num_per_loop):
            angle = -np.pi / 2.0 + 2.0 * np.pi * i / num_per_loop
            lat = (c_lat + r_deg) + r_deg * np.sin(angle)
            lon = c_lon + r_deg * np.cos(angle)
            waypoints.append((lat, lon))

        # South loop: clockwise circle, center at (c_lat - r_deg, c_lon)
        # angle=+pi/2 puts the starting point at the top of the circle = crossover
        for i in range(num_per_loop):
            angle = np.pi / 2.0 - 2.0 * np.pi * i / num_per_loop
            lat = (c_lat - r_deg) + r_deg * np.sin(angle)
            lon = c_lon + r_deg * np.cos(angle)
            waypoints.append((lat, lon))

        self.waypoints = waypoints

    def update(self, unit, dt: float) -> tuple[float, float, float]:
        """
        Update the controller and compute desired heading/speed/altitude.

        Advances the current waypoint target when the aircraft is within
        the capture radius, then computes the bearing to the next waypoint.

        Args:
            unit: The aircraft unit being controlled.
            dt: Time step in seconds.

        Returns:
            tuple of (desired_heading_deg, desired_speed_mps, desired_altitude_m).
        """
        pos = unit.position

        target_lat, target_lon = self.waypoints[self.current_wp_idx]

        # Distance to current waypoint in km
        dlat_km = (target_lat - pos.lat) * self.km_per_deg
        dlon_km = (target_lon - pos.lon) * self.km_per_deg
        dist_km = np.sqrt(dlat_km**2 + dlon_km**2)

        # Capture radius: advance when within this distance
        capture_radius_km = max(2.0, self.config.turn_radius_km * 0.15)

        if dist_km < capture_radius_km:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            target_lat, target_lon = self.waypoints[self.current_wp_idx]
            dlat_km = (target_lat - pos.lat) * self.km_per_deg
            dlon_km = (target_lon - pos.lon) * self.km_per_deg

        # Bearing to waypoint: arctan2(east, north) gives bearing from north
        heading = float(np.degrees(np.arctan2(dlon_km, dlat_km)) % 360.0)

        return heading, self.config.speed_mps, self.config.altitude_m

    def get_commanded_controls(self, unit, dt: float) -> dict:
        """
        Get control commands for the aircraft.

        Returns a dictionary with throttle, heading, and altitude commands.
        """
        heading, speed, altitude = self.update(unit, dt)

        max_speed = getattr(unit, "max_speed_mps", 250.0)
        throttle = np.clip(speed / max_speed, 0.0, 1.0)

        return {
            "throttle": throttle,
            "target_altitude_m": altitude,
            "target_heading_deg": heading,
        }


class CircleController:
    """
    Controller for a simple circular orbit around the center point.

    Generates 16 evenly-spaced waypoints on a circle of radius `turn_radius_km`
    and navigates through them sequentially, respecting the `clockwise` flag.
    """

    def __init__(self, config: OrbitConfig | None = None):
        if config is None:
            config = OrbitConfig(pattern="circle")
        else:
            config.pattern = "circle"
        self.config = config
        self.km_per_deg = 111.0

        self._generate_waypoints()
        self.current_wp_idx = 0

    def _generate_waypoints(self, num_points: int = 16):
        """Generate evenly-spaced waypoints on the circle."""
        r_deg = self.config.turn_radius_km / self.km_per_deg
        c_lat = self.config.center_lat
        c_lon = self.config.center_lon

        # Start at the top (north) of the circle; go clockwise or counter-clockwise
        direction = 1.0 if self.config.clockwise else -1.0
        waypoints = []
        for i in range(num_points):
            angle = np.pi / 2.0 - direction * 2.0 * np.pi * i / num_points
            lat = c_lat + r_deg * np.sin(angle)
            lon = c_lon + r_deg * np.cos(angle)
            waypoints.append((lat, lon))
        self.waypoints = waypoints

    def update(self, unit, dt: float) -> tuple[float, float, float]:
        """
        Update the controller and compute desired heading/speed/altitude.

        Args:
            unit: The aircraft unit being controlled.
            dt: Time step in seconds.

        Returns:
            tuple of (desired_heading_deg, desired_speed_mps, desired_altitude_m).
        """
        pos = unit.position
        target_lat, target_lon = self.waypoints[self.current_wp_idx]

        dlat_km = (target_lat - pos.lat) * self.km_per_deg
        dlon_km = (target_lon - pos.lon) * self.km_per_deg
        dist_km = np.sqrt(dlat_km**2 + dlon_km**2)

        capture_radius_km = max(3.0, self.config.turn_radius_km * 0.12)

        if dist_km < capture_radius_km:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            target_lat, target_lon = self.waypoints[self.current_wp_idx]
            dlat_km = (target_lat - pos.lat) * self.km_per_deg
            dlon_km = (target_lon - pos.lon) * self.km_per_deg

        heading = float(np.degrees(np.arctan2(dlon_km, dlat_km)) % 360.0)
        return heading, self.config.speed_mps, self.config.altitude_m

    def get_commanded_controls(self, unit, dt: float) -> dict:
        """Get control commands for the aircraft."""
        heading, speed, altitude = self.update(unit, dt)
        max_speed = getattr(unit, "max_speed_mps", 250.0)
        throttle = np.clip(speed / max_speed, 0.0, 1.0)
        return {
            "throttle": throttle,
            "target_altitude_m": altitude,
            "target_heading_deg": heading,
        }


class RandomWaypointController:
    """
    Controller that navigates to random waypoints within a bounded zone.

    Picks a new random waypoint each time the aircraft comes within the capture
    radius of the current target. Zone bounds are derived from OrbitConfig zone_*
    fields; if those are None, defaults to center ± 2*turn_radius_km.
    """

    def __init__(self, config: OrbitConfig | None = None, rng=None):
        if config is None:
            config = OrbitConfig(pattern="random")
        else:
            config.pattern = "random"
        self.config = config
        self.km_per_deg = 111.0
        self.rng = rng if rng is not None else np.random.default_rng()

        r_deg = config.turn_radius_km / self.km_per_deg
        margin = 0.05  # inward shrink to keep waypoints away from zone edges

        if config.zone_min_lat is not None and config.zone_max_lat is not None:
            lat_lo, lat_hi = config.zone_min_lat, config.zone_max_lat
        else:
            lat_lo = config.center_lat - 2.0 * r_deg
            lat_hi = config.center_lat + 2.0 * r_deg

        if config.zone_min_lon is not None and config.zone_max_lon is not None:
            lon_lo, lon_hi = config.zone_min_lon, config.zone_max_lon
        else:
            lon_lo = config.center_lon - 2.0 * r_deg
            lon_hi = config.center_lon + 2.0 * r_deg

        lat_span = lat_hi - lat_lo
        lon_span = lon_hi - lon_lo
        self._lat_lo = lat_lo + margin * lat_span
        self._lat_hi = lat_hi - margin * lat_span
        self._lon_lo = lon_lo + margin * lon_span
        self._lon_hi = lon_hi - margin * lon_span

        self.current_wp = self._random_wp()

    def _random_wp(self) -> tuple[float, float]:
        lat = float(self.rng.uniform(self._lat_lo, self._lat_hi))
        lon = float(self.rng.uniform(self._lon_lo, self._lon_hi))
        return lat, lon

    def update(self, unit, dt: float) -> tuple[float, float, float]:
        """
        Update the controller and compute desired heading/speed/altitude.

        Args:
            unit: The aircraft unit being controlled.
            dt: Time step in seconds.

        Returns:
            tuple of (desired_heading_deg, desired_speed_mps, desired_altitude_m).
        """
        pos = unit.position
        target_lat, target_lon = self.current_wp

        dlat_km = (target_lat - pos.lat) * self.km_per_deg
        dlon_km = (target_lon - pos.lon) * self.km_per_deg
        dist_km = np.sqrt(dlat_km**2 + dlon_km**2)

        capture_radius_km = max(3.0, self.config.turn_radius_km * 0.12)

        if dist_km < capture_radius_km:
            self.current_wp = self._random_wp()
            target_lat, target_lon = self.current_wp
            dlat_km = (target_lat - pos.lat) * self.km_per_deg
            dlon_km = (target_lon - pos.lon) * self.km_per_deg

        heading = float(np.degrees(np.arctan2(dlon_km, dlat_km)) % 360.0)
        return heading, self.config.speed_mps, self.config.altitude_m

    def get_commanded_controls(self, unit, dt: float) -> dict:
        """Get control commands for the aircraft."""
        heading, speed, altitude = self.update(unit, dt)
        max_speed = getattr(unit, "max_speed_mps", 250.0)
        throttle = np.clip(speed / max_speed, 0.0, 1.0)
        return {
            "throttle": throttle,
            "target_altitude_m": altitude,
            "target_heading_deg": heading,
        }


def create_orbit_controller(
    pattern: str = "racetrack",
    center_lat: float = 0.0,
    center_lon: float = -0.5,
    leg_length_km: float = 40.0,
    turn_radius_km: float = 25.0,
    altitude_m: float = 10000.0,
    speed_mps: float = 200.0,
    clockwise: bool = True,
    zone_min_lat: float | None = None,
    zone_max_lat: float | None = None,
    zone_min_lon: float | None = None,
    zone_max_lon: float | None = None,
    rng=None,
):
    """
    Factory function to create an orbit controller.

    Args:
        pattern: "racetrack", "figure8", "circle", "random", or "random_all".
        center_lat: Latitude of orbit center in degrees.
        center_lon: Longitude of orbit center in degrees.
        leg_length_km: Length of straight legs (racetrack only).
        turn_radius_km: Turn radius / circle radius in km.
        altitude_m: Orbit altitude in meters.
        speed_mps: Patrol speed in m/s.
        clockwise: Whether to orbit clockwise.
        zone_min_lat: Southern bound for random pattern (None → auto-derived).
        zone_max_lat: Northern bound for random pattern (None → auto-derived).
        zone_min_lon: Western bound for random pattern (None → auto-derived).
        zone_max_lon: Eastern bound for random pattern (None → auto-derived).
        rng: Optional numpy Generator for reproducibility (random/random_all).

    Returns:
        Configured orbit controller.
    """
    if pattern == "random_all":
        _rng = rng if rng is not None else np.random.default_rng()
        pattern = str(_rng.choice(["circle", "figure8", "racetrack", "random"]))

    config = OrbitConfig(
        center_lat=center_lat,
        center_lon=center_lon,
        leg_length_km=leg_length_km,
        turn_radius_km=turn_radius_km,
        altitude_m=altitude_m,
        speed_mps=speed_mps,
        clockwise=clockwise,
        pattern=pattern,
        zone_min_lat=zone_min_lat,
        zone_max_lat=zone_max_lat,
        zone_min_lon=zone_min_lon,
        zone_max_lon=zone_max_lon,
    )

    if pattern == "figure8":
        return Figure8Controller(config)
    elif pattern == "circle":
        return CircleController(config)
    elif pattern == "random":
        return RandomWaypointController(config, rng=rng)
    else:
        return RacetrackController(config)
