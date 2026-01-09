# Sources:
# - AIM-120 AMRAAM (AIM-120D range 130-160km, boost time, general data): https://en.wikipedia.org/wiki/AIM-120_AMRAAM
# - Air & Space Forces fact sheet (recent): https://www.airandspaceforces.com/weapons/aim-120/

from missiles.missile import Missile

class AIM120_AMRAAM(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        # Geometry from public dims (Ø 178 mm => Sref ≈ π r^2 ≈ 0.0249 m²).
        # Mass ~152–161 kg (varies by variant). We use 160 kg as a middle, D-like value.
        # Boost time ~7.8–8 s (AIM-120C-5), AIM-120D range 130–160 km (kinematic).
        config = {
            "mass_kg": 160.0,                    # ~AIM-120D typical mass 152–161 kg (public)
            "reference_area_m2": 0.025,          # π*(0.089 m)^2 ≈ 0.0249 m² (cross-section)
            "aspect_ratio": 3.0,
            "oswald_e": 0.78,
            "drag_coefficient": 0.08,

            # Simple constant-thrust surrogate for boost (public boost ~8 s).
            "constant_engine_N": 9000.0,         # surrogate thrust (not publicly specified)
            "motor_burn_s": 8.0,

            "n_max": 40.0,                        # 40 g class (public for C-5/6/7; D similar)
            "max_speed_mps": 1370.0,             # ≈ Mach 4 (sea-level m/s; envelope dependent)
            "min_range_m": 1500.0,

            # Seeker & lifetime: active ARH with mid-course DL; seeker lock-on range < kinematic range.
            "seeker_sensitivity": 1.2,
            "life_time_s": 120.0,
            "hit_probability": 0.85,

            "radar": {
                "horizontal_fov_deg": 60.0,
                "vertical_fov_deg": 30.0,
                "max_range_m": 150_000.0,       # true kinematic range (130-160 km real-world; using 150 km middle estimate)
                "radar_frequency_hz": 10e9,      # X-band
                "tx_power_w": 8e3,               # surrogate
                "antenna_gain_db": 32.0,         # surrogate
                "snr_threshold_db": 8.0,
            }
        }
        super().__init__(
            name="AIM120_AMRAAM",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="full"                 # AMRAAM has mid-course DL (two-way in D)
        )
        self.fox_type = 3