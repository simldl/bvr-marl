from air_to_air_rl.missiles.fox1.base_fox1 import Fox1Missile


class AIM7_Sparrow(Fox1Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        config = {
            "mass_kg": 230.0,
            "reference_area_m2": 0.20,
            "aspect_ratio": 3.0,
            "oswald_e": 0.75,
            "drag_coefficient": 0.12,
            "constant_engine_N": 12000.0,
            "n_max": 25.0,
            "max_speed_mps": 1200.0,
            "motor_burn_s": 8.0,
            "min_range_m": 1000.0,
            "seeker_sensitivity": 1.2,
            "life_time_s": 60.0,
            "hit_probability": 0.85,
            "radar": {
                "fov_deg": 25.0,
                "max_range_m": 45000.0,
                "sensitivity": 1.2,
            },
        }
        super().__init__(
            name="AIM7_Sparrow",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="illumination",
        )
