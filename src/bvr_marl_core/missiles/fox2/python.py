from bvr_marl_core.missiles.fox2.base_fox2 import Fox2Missile


class Python_5(Fox2Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        config = {
            "mass_kg": 100.0,
            "reference_area_m2": 0.16,
            "aspect_ratio": 2.6,
            "oswald_e": 0.82,
            "drag_coefficient": 0.07,
            "constant_engine_N": 9000.0,
            "n_max": 40.0,
            "max_speed_mps": 950.0,
            "motor_burn_s": 7.0,
            "min_range_m": 400.0,
            "seeker_sensitivity": 1.2,
            "life_time_s": 40.0,
            "hit_probability": 0.92,
            "radar": {
                "fov_deg": 18.0,
                "max_range_m": 20000.0,
                "sensitivity": 1.2,
            },
        }
        super().__init__(
            name="Python_5",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="none",
        )
