"""
Lightweight missile parameter storage for NEZ/DLZ calculations.

Stores missile specifications without instantiating full missile objects,
avoiding the overhead of object creation during tactical zone computations.
"""

from dataclasses import dataclass


@dataclass
class MissileRadarParameters:
    """Radar configuration parameters for a missile."""

    horizontal_fov_deg: float
    vertical_fov_deg: float
    max_range_m: float
    radar_frequency_hz: float
    tx_power_w: float
    antenna_gain_db: float
    snr_threshold_db: float


@dataclass
class MissileParameters:
    """
    Lightweight storage for missile specifications used in NEZ/DLZ calculations.

    This replaces the need to instantiate full Missile objects just to read
    their configuration parameters during tactical calculations.

    Attributes:
        name: Missile type name (e.g., "METEOR", "AIM-120")
        max_range_m: Maximum kinematic range in meters
        min_range_m: Minimum engagement range in meters
        max_speed_mps: Maximum missile speed in m/s
        max_g: Maximum G-load capability
        seeker_sensitivity: Seeker sensitivity factor for RCS calculations
        fox_type: Missile type (1=SARH, 2=IR, 3=ARH)
        hit_probability: Probability of hit on target
        life_time_s: Maximum flight time in seconds
        motor_burn_s: Motor burn duration in seconds
        mass_kg: Missile mass in kilograms
        radar: Radar configuration parameters
    """

    name: str
    max_range_m: float
    min_range_m: float
    max_speed_mps: float
    max_g: float
    seeker_sensitivity: float
    fox_type: int
    hit_probability: float
    life_time_s: float
    motor_burn_s: float
    mass_kg: float
    radar: MissileRadarParameters

    # Weapon-effectiveness submodel parameters (mirror of the Missile attributes
    # consumed by KillProbabilityModel). Defaults are neutral so a missile that
    # does not set them keeps neutral clean-hit effectiveness.
    lethal_radius_m: float = 250.0
    warhead_effectiveness: float = 1.0
    fuze_reliability: float = 1.0
    guidance_reliability: float = 1.0
    seeker_reliability: float = 1.0
    datalink_reliability: float = 1.0

    @classmethod
    def from_config(cls, name: str, config: dict) -> "MissileParameters":
        """
        Create MissileParameters from a missile configuration dictionary.

        Args:
            name: Missile type name
            config: Configuration dictionary (same format as used by Missile.__init__)

        Returns:
            MissileParameters instance
        """
        radar_config = config.get("radar", {})

        return cls(
            name=name,
            # Kinematic max range (DLZ anchor), distinct from the radar/seeker range.
            max_range_m=config.get("max_range_m", radar_config.get("max_range_m", 75000.0)),
            min_range_m=config.get("min_range_m", 1500.0),
            max_speed_mps=config.get("max_speed_mps", 1000.0),
            max_g=config.get("n_max", 30.0),
            seeker_sensitivity=config.get("seeker_sensitivity", 1.0),
            fox_type=3,  # Default to ARH (will be overridden by subclass if needed)
            hit_probability=config.get("hit_probability", 0.8),
            life_time_s=config.get("life_time_s", 100.0),
            motor_burn_s=config.get("motor_burn_s", 30.0),
            mass_kg=config.get("mass_kg", 150.0),
            radar=MissileRadarParameters(
                horizontal_fov_deg=radar_config.get("horizontal_fov_deg", 60.0),
                vertical_fov_deg=radar_config.get("vertical_fov_deg", 30.0),
                max_range_m=radar_config.get("max_range_m", 75000.0),
                radar_frequency_hz=radar_config.get("radar_frequency_hz", 10e9),
                tx_power_w=radar_config.get("tx_power_w", 5000.0),
                antenna_gain_db=radar_config.get("antenna_gain_db", 30.0),
                snr_threshold_db=radar_config.get("snr_threshold_db", 8.0),
            ),
            lethal_radius_m=config.get("lethal_radius_m", 500.0),
            warhead_effectiveness=config.get("warhead_effectiveness", 1.0),
            fuze_reliability=config.get("fuze_reliability", 1.0),
            guidance_reliability=config.get("guidance_reliability", 1.0),
            seeker_reliability=config.get("seeker_reliability", 1.0),
            datalink_reliability=config.get("datalink_reliability", 1.0),
        )

    @classmethod
    def from_missile_class(cls, missile_class, fox_type: int | None = None) -> "MissileParameters":
        """
        Create MissileParameters from a missile class by extracting its default config.

        Called once per missile type at aircraft initialization, not on every NEZ calculation.

        Args:
            missile_class: The missile class (e.g., Meteor, AIM120_AMRAAM)
            fox_type: Override fox_type if needed (1=SARH, 2=IR, 3=ARH)

        Returns:
            MissileParameters instance
        """
        try:

            class MockPosition:
                def __init__(self):
                    self.lat = 0.0
                    self.lon = 0.0
                    self.alt = 5000.0

                def copy(self):
                    pos = MockPosition()
                    pos.lat = self.lat
                    pos.lon = self.lon
                    pos.alt = self.alt
                    return pos

            class MockSource:
                def __init__(self):
                    self.position = MockPosition()
                    self.yaw_deg = 0.0
                    self.speed = 250.0
                    self.pitch_deg = 0.0
                    self.roll_deg = 0.0
                    self.group = "mock"

            class MockMapLimits:
                def __init__(self):
                    self.min_alt = 0.0
                    self.max_alt = 20000.0

            dummy = missile_class(
                firing_time_s=0.0,
                target=None,
                source=MockSource(),
                map_limits=MockMapLimits(),
                group="mock",
            )

            params = cls(
                name=getattr(dummy, "name", missile_class.__name__),
                max_range_m=getattr(
                    dummy,
                    "max_range_m",
                    dummy.radar.max_range_m if hasattr(dummy, "radar") else 75000.0,
                ),
                min_range_m=getattr(dummy, "min_range_m", 1500.0),
                max_speed_mps=getattr(dummy, "max_speed_mps", 1000.0),
                max_g=dummy.physics.params.n_max if hasattr(dummy, "physics") else 30.0,
                seeker_sensitivity=getattr(dummy, "seeker_sensitivity", 1.0),
                fox_type=fox_type or getattr(dummy, "fox_type", 3),
                hit_probability=getattr(dummy, "hit_probability", 0.8),
                life_time_s=getattr(dummy, "life_time_s", 100.0),
                motor_burn_s=getattr(dummy, "motor_burn_s", 30.0),
                mass_kg=dummy.physics.params.mass_kg if hasattr(dummy, "physics") else 150.0,
                radar=MissileRadarParameters(
                    horizontal_fov_deg=getattr(dummy.radar, "h_fov_deg", 60.0)
                    if hasattr(dummy, "radar")
                    else 60.0,
                    vertical_fov_deg=getattr(dummy.radar, "v_fov_deg", 30.0)
                    if hasattr(dummy, "radar")
                    else 30.0,
                    max_range_m=getattr(dummy.radar, "max_range_m", 75000.0)
                    if hasattr(dummy, "radar")
                    else 75000.0,
                    radar_frequency_hz=getattr(dummy.radar, "freq_hz", 10e9)
                    if hasattr(dummy, "radar")
                    else 10e9,
                    tx_power_w=getattr(dummy.radar, "tx_power_w", 5000.0)
                    if hasattr(dummy, "radar")
                    else 5000.0,
                    antenna_gain_db=getattr(dummy.radar, "antenna_gain_db", 30.0)
                    if hasattr(dummy, "radar")
                    else 30.0,
                    snr_threshold_db=getattr(dummy.radar, "snr_threshold_db", 8.0)
                    if hasattr(dummy, "radar")
                    else 8.0,
                ),
                lethal_radius_m=getattr(dummy, "lethal_radius_m", 500.0),
                warhead_effectiveness=getattr(dummy, "warhead_effectiveness", 1.0),
                fuze_reliability=getattr(dummy, "fuze_reliability", 1.0),
                guidance_reliability=getattr(dummy, "guidance_reliability", 1.0),
                seeker_reliability=getattr(dummy, "seeker_reliability", 1.0),
                datalink_reliability=getattr(dummy, "datalink_reliability", 1.0),
            )

            return params

        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass

        return cls(
            name=missile_class.__name__,
            max_range_m=75000.0,
            min_range_m=1500.0,
            max_speed_mps=1000.0,
            max_g=30.0,
            seeker_sensitivity=1.0,
            fox_type=fox_type or 3,
            hit_probability=0.8,
            life_time_s=100.0,
            motor_burn_s=30.0,
            mass_kg=150.0,
            radar=MissileRadarParameters(
                horizontal_fov_deg=60.0,
                vertical_fov_deg=30.0,
                max_range_m=75000.0,
                radar_frequency_hz=10e9,
                tx_power_w=5000.0,
                antenna_gain_db=30.0,
                snr_threshold_db=8.0,
            ),
        )
