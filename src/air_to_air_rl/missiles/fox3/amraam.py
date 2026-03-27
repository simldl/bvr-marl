# Sources:
# - AIM-120 AMRAAM (AIM-120D range 130-160km, boost time, general data): https://en.wikipedia.org/wiki/AIM-120_AMRAAM
# - Air & Space Forces fact sheet (recent): https://www.airandspaceforces.com/weapons/aim-120/

from air_to_air_rl.missiles.missile import Missile


class AIM120_AMRAAM(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        # Ø 178 mm => Sref ≈ π r^2 ≈ 0.0249 m²; mass ~152–161 kg; boost ~8 s; range 130–160 km.
        config = {
            "mass_kg": 160.0,
            "reference_area_m2": 0.025,  # π*(0.089 m)^2
            "aspect_ratio": 3.0,
            "oswald_e": 0.78,
            "drag_coefficient": 0.08,
            "constant_engine_N": 9000.0,
            "motor_burn_s": 8.0,
            "n_max": 40.0,
            "max_speed_mps": 1370.0,  # ≈ Mach 4 sea-level
            "min_range_m": 1500.0,
            "seeker_sensitivity": 1.2,
            "life_time_s": 120.0,
            "hit_probability": 0.85,
            "radar": {
                "horizontal_fov_deg": 60.0,
                "vertical_fov_deg": 30.0,
                "max_range_m": 150_000.0,
                "radar_frequency_hz": 10e9,  # X-band
                "tx_power_w": 8e3,
                "antenna_gain_db": 32.0,
                "snr_threshold_db": 8.0,
            },
        }
        super().__init__(
            name="AIM120_AMRAAM",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="full",
        )
        self.fox_type = 3
