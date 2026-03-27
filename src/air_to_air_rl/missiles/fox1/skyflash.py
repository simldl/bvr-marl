from air_to_air_rl.missiles.fox1.base_fox1 import Fox1Missile


class Skyflash(Fox1Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        config = {
            "mass_kg": 193.0,
            "reference_area_m2": 0.18,
            "aspect_ratio": 3.2,
            "oswald_e": 0.78,
            "drag_coefficient": 0.11,
            "constant_engine_N": 11000.0,
            "n_max": 28.0,
            "max_speed_mps": 1300.0,
            "motor_burn_s": 9.0,
            "min_range_m": 800.0,
            "seeker_sensitivity": 1.3,
            "life_time_s": 65.0,
            "hit_probability": 0.88,
            "radar": {
                "fov_deg": 22.0,
                "max_range_m": 50000.0,
                "sensitivity": 1.3,
            },
        }
        super().__init__(
            name="Skyflash",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="illumination",
        )
