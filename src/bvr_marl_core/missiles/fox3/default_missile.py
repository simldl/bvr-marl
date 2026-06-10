from bvr_marl_core.missiles.missile import Missile


class LongRangeMissile(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        config = {
            "mass_kg": 380.0,
            "reference_area_m2": 0.28,
            "aspect_ratio": 3.5,
            "oswald_e": 0.75,
            "drag_coefficient": 0.075,
            "constant_engine_N": 24000.0,
            "n_max": 35.0,
            "max_speed_mps": 1500.0,
            "motor_burn_s": 32.0,
            "min_range_m": 2000.0,
            "seeker_sensitivity": 1.3,
            "life_time_s": 220.0,
            "hit_probability": 0.90,
            "radar": {
                "horizontal_fov_deg": 90.0,
                "vertical_fov_deg": 45.0,
                "max_range_m": 120_000.0,
                "radar_frequency_hz": 12e9,
                "tx_power_w": 8e3,
                "antenna_gain_db": 34.0,
                "snr_threshold_db": 8.5,
            },
        }
        super().__init__(
            name="LongRangeMissile",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="full",
        )
