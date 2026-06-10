# Sources:
# - Wikipedia R-37M (mass ~510 kg, length 4.06 m, Ø 0.38 m, range up to 200 km RVV-BD): https://en.wikipedia.org/wiki/R-37_(missile)
# - Missilery info overview (RVV-BD): https://en.missilery.info/missile/r37

from bvr_marl_core.missiles.missile import Missile


class R37M(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        # Ø 380 mm => Sref ≈ π*(0.19)^2 ≈ 0.113 m²; mass ~510 kg.
        config = {
            "mass_kg": 510.0,
            "reference_area_m2": 0.113,
            "aspect_ratio": 2.8,
            "oswald_e": 0.78,
            "drag_coefficient": 0.07,
            "constant_engine_N": 20_000.0,
            "motor_burn_s": 20.0,
            "n_max": 30.0,  # large missile; lower maneuver envelope
            "max_speed_mps": 2000.0,  # Mach ~6 class
            "min_range_m": 5000.0,
            "seeker_sensitivity": 1.2,
            "life_time_s": 180.0,
            "hit_probability": 0.75,
            "radar": {
                "horizontal_fov_deg": 50.0,
                "vertical_fov_deg": 25.0,
                "max_range_m": 50_000.0,  # ARH activation distance
                "radar_frequency_hz": 10e9,
                "tx_power_w": 10e3,
                "antenna_gain_db": 35.0,
                "snr_threshold_db": 8.0,
            },
        }
        super().__init__(
            name="R37M",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="msl_support",
        )
        self.fox_type = 3
