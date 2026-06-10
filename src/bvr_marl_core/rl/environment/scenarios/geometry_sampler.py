"""
Geometry sampler for controlled BVR scenario initialization.

Provides deterministic and reproducible geometry configurations for study scenarios,
ensuring controlled initial conditions for engagement geometry.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from bvr_marl_core.simulator import Position


@dataclass
class GeometryConfig:
    """Configuration for a scenario geometry."""

    # Engagement range in meters
    range_m: float

    # Aspect angle: "hot" (head-on), "beam" (90 deg), "cold" (tail chase)
    aspect: Literal["hot", "beam", "cold"]

    # Altitude split: "co_alt", "high_low", "low_high"
    altitude_split: Literal["co_alt", "high_low", "low_high"]

    # Lateral offset angle in degrees (0 = head-on, 30 = flanking)
    offset_deg: float = 0.0

    # Altitude values in meters
    agent_altitude_m: float = 8000.0
    opponent_altitude_m: float = 8000.0

    # Speed in m/s
    agent_speed_mps: float = 250.0
    opponent_speed_mps: float = 250.0


# Pre-defined study regimes for reproducible experiments
STUDY_REGIMES: dict[str, GeometryConfig] = {
    # Hot engagements at various ranges
    "hot_80km": GeometryConfig(
        range_m=80_000.0,
        aspect="hot",
        altitude_split="co_alt",
        offset_deg=0.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    "hot_120km": GeometryConfig(
        range_m=120_000.0,
        aspect="hot",
        altitude_split="co_alt",
        offset_deg=0.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    "hot_160km": GeometryConfig(
        range_m=160_000.0,
        aspect="hot",
        altitude_split="co_alt",
        offset_deg=0.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    "hot_200km": GeometryConfig(
        range_m=200_000.0,
        aspect="hot",
        altitude_split="co_alt",
        offset_deg=0.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    # Offset engagements
    "offset_30deg": GeometryConfig(
        range_m=100_000.0,
        aspect="hot",
        altitude_split="co_alt",
        offset_deg=30.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    "offset_45deg": GeometryConfig(
        range_m=100_000.0,
        aspect="hot",
        altitude_split="co_alt",
        offset_deg=45.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    # Altitude split engagements
    "high_low_split": GeometryConfig(
        range_m=100_000.0,
        aspect="hot",
        altitude_split="high_low",
        offset_deg=0.0,
        agent_altitude_m=12000.0,
        opponent_altitude_m=5000.0,
    ),
    "low_high_split": GeometryConfig(
        range_m=100_000.0,
        aspect="hot",
        altitude_split="low_high",
        offset_deg=0.0,
        agent_altitude_m=5000.0,
        opponent_altitude_m=12000.0,
    ),
    # Bracket viable - suitable for bracket tactics
    "bracket_viable": GeometryConfig(
        range_m=80_000.0,
        aspect="hot",
        altitude_split="co_alt",
        offset_deg=0.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    # Beam engagements
    "beam_100km": GeometryConfig(
        range_m=100_000.0,
        aspect="beam",
        altitude_split="co_alt",
        offset_deg=0.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
    # Defensive scenario
    "cold_60km": GeometryConfig(
        range_m=60_000.0,
        aspect="cold",
        altitude_split="co_alt",
        offset_deg=0.0,
        agent_altitude_m=8000.0,
        opponent_altitude_m=8000.0,
    ),
}


class GeometrySampler:
    """
    Samples engagement geometries for controlled BVR scenarios.

    Provides deterministic initial positions for agents and opponents
    based on predefined geometry configurations.
    """

    def __init__(self, map_size_km: float = 200.0):
        """
        Initialize the geometry sampler.

        Args:
            map_size_km: Map size in kilometers.
        """
        self.map_size_km = map_size_km
        # Conversion factor: ~111 km per degree at equator
        self.km_per_deg = 111.0

    def sample(self, regime: str) -> GeometryConfig:
        """
        Get a geometry configuration for a named regime.

        Args:
            regime: Name of the regime (e.g., "hot_120km", "offset_30deg").

        Returns:
            GeometryConfig for the specified regime.

        Raises:
            ValueError: If the regime is not recognized.
        """
        if regime not in STUDY_REGIMES:
            available = ", ".join(STUDY_REGIMES.keys())
            raise ValueError(f"Unknown geometry regime: {regime}. Available: {available}")

        return STUDY_REGIMES[regime]

    def compute_positions(
        self,
        geometry: GeometryConfig,
        num_agents: int = 2,
        num_opponents: int = 2,
        formation_spread_m: float = 2000.0,
    ) -> dict[str, Any]:
        """
        Compute spawn positions for a given geometry configuration.

        Returns positions centered around (0, 0) with agents on the west side
        and opponents on the east side.

        Args:
            geometry: The geometry configuration to use.
            num_agents: Number of agents to spawn.
            num_opponents: Number of opponents to spawn.
            formation_spread_m: Lateral spread between formation members.

        Returns:
            Dictionary with:
                - agent_positions: List of Position objects for agents
                - agent_headings: List of heading angles (degrees) for agents
                - opponent_positions: List of Position objects for opponents
                - opponent_headings: List of heading angles (degrees) for opponents
        """
        half_range_m = geometry.range_m / 2.0
        half_range_km = half_range_m / 1000.0
        half_range_deg = half_range_km / self.km_per_deg

        # Base positions: agents on west, opponents on east
        # Offset angle rotates the engagement axis
        offset_rad = np.radians(geometry.offset_deg)

        # Agent base position (west side)
        agent_base_lon = -half_range_deg * np.cos(offset_rad)
        agent_base_lat = -half_range_deg * np.sin(offset_rad)

        # Opponent base position (east side)
        opponent_base_lon = half_range_deg * np.cos(offset_rad)
        opponent_base_lat = half_range_deg * np.sin(offset_rad)

        # Compute headings based on aspect
        if geometry.aspect == "hot":
            # Head-on: agents face east, opponents face west
            agent_heading = 90.0 + geometry.offset_deg
            opponent_heading = 270.0 + geometry.offset_deg
        elif geometry.aspect == "beam":
            # Beam: perpendicular to engagement axis
            agent_heading = 0.0 + geometry.offset_deg
            opponent_heading = 180.0 + geometry.offset_deg
        elif geometry.aspect == "cold":
            # Tail chase: both facing same direction (agents pursuing)
            agent_heading = 90.0 + geometry.offset_deg
            opponent_heading = 90.0 + geometry.offset_deg
        else:
            agent_heading = 90.0
            opponent_heading = 270.0

        # Normalize headings to [0, 360)
        agent_heading = agent_heading % 360.0
        opponent_heading = opponent_heading % 360.0

        # Compute altitudes
        agent_alt = geometry.agent_altitude_m
        opponent_alt = geometry.opponent_altitude_m

        # Spread formation members laterally (perpendicular to heading)
        spread_deg = (formation_spread_m / 1000.0) / self.km_per_deg

        agent_positions = []
        agent_headings = []
        for i in range(num_agents):
            # Spread laterally relative to heading
            lateral_offset = (i - (num_agents - 1) / 2.0) * spread_deg
            lat_offset = lateral_offset * np.cos(np.radians(agent_heading))
            lon_offset = lateral_offset * np.sin(np.radians(agent_heading))

            pos = Position(
                lat=agent_base_lat + lat_offset,
                lon=agent_base_lon + lon_offset,
                alt=agent_alt,
            )
            agent_positions.append(pos)
            agent_headings.append(agent_heading)

        opponent_positions = []
        opponent_headings = []
        for i in range(num_opponents):
            lateral_offset = (i - (num_opponents - 1) / 2.0) * spread_deg
            lat_offset = lateral_offset * np.cos(np.radians(opponent_heading))
            lon_offset = lateral_offset * np.sin(np.radians(opponent_heading))

            pos = Position(
                lat=opponent_base_lat + lat_offset,
                lon=opponent_base_lon + lon_offset,
                alt=opponent_alt,
            )
            opponent_positions.append(pos)
            opponent_headings.append(opponent_heading)

        return {
            "agent_positions": agent_positions,
            "agent_headings": agent_headings,
            "agent_speed": geometry.agent_speed_mps,
            "opponent_positions": opponent_positions,
            "opponent_headings": opponent_headings,
            "opponent_speed": geometry.opponent_speed_mps,
            "geometry": geometry,
        }

    def sample_with_noise(
        self,
        regime: str,
        position_noise_m: float = 0.0,
        altitude_noise_m: float = 0.0,
        heading_noise_deg: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> GeometryConfig:
        """
        Sample a geometry configuration with optional noise for training variety.

        Args:
            regime: Base regime name.
            position_noise_m: Standard deviation of position noise in meters.
            altitude_noise_m: Standard deviation of altitude noise in meters.
            heading_noise_deg: Standard deviation of heading noise in degrees.
            rng: Optional random number generator for reproducibility.

        Returns:
            GeometryConfig with added noise.
        """
        base = self.sample(regime)

        if rng is None:
            rng = np.random.default_rng()

        # Add noise to range
        range_noise = rng.normal(0, position_noise_m) if position_noise_m > 0 else 0
        new_range = max(10000.0, base.range_m + range_noise)  # Min 10km

        # Add noise to altitudes
        agent_alt_noise = rng.normal(0, altitude_noise_m) if altitude_noise_m > 0 else 0
        opponent_alt_noise = rng.normal(0, altitude_noise_m) if altitude_noise_m > 0 else 0

        new_agent_alt = max(1000.0, min(20000.0, base.agent_altitude_m + agent_alt_noise))
        new_opponent_alt = max(1000.0, min(20000.0, base.opponent_altitude_m + opponent_alt_noise))

        # Add noise to offset
        offset_noise = rng.normal(0, heading_noise_deg) if heading_noise_deg > 0 else 0
        new_offset = base.offset_deg + offset_noise

        return GeometryConfig(
            range_m=new_range,
            aspect=base.aspect,
            altitude_split=base.altitude_split,
            offset_deg=new_offset,
            agent_altitude_m=new_agent_alt,
            opponent_altitude_m=new_opponent_alt,
            agent_speed_mps=base.agent_speed_mps,
            opponent_speed_mps=base.opponent_speed_mps,
        )

    @staticmethod
    def list_regimes() -> list[str]:
        """List all available geometry regimes."""
        return list(STUDY_REGIMES.keys())
