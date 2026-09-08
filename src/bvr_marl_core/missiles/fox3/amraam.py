# Sources:
# - AIM-120 AMRAAM (AIM-120D range 130-160km, boost time, general data): https://en.wikipedia.org/wiki/AIM-120_AMRAAM
# - Air & Space Forces fact sheet (recent): https://www.airandspaceforces.com/weapons/aim-120/

from bvr_marl_core.missiles.missile import Missile


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
            # Slow the post-burnout energy bleed (the intended use of drag_scale) so
            # the missile keeps enough terminal speed to run down an *extending*
            # target: the no-escape zone vs a full-throttle extender reaches ~40-45 km
            # instead of ~30 km, where the missile previously bled to ~590 m/s by 40 km
            # and was out-run. A committed extender past ~55 km still escapes.
            "drag_scale": 0.9,
            "n_max": 40.0,
            "max_speed_mps": 1370.0,  # ≈ Mach 4 sea-level
            "min_range_m": 1500.0,
            # Cited kinematic max range (AIM-120D, head-on, high-altitude launch);
            # anchors the DLZ. drag_scale (above) slows the post-burnout bleed so the
            # no-escape zone vs an extender reaches ~45-50 km.
            "max_range_m": 160_000.0,
            "seeker_sensitivity": 1.2,
            "life_time_s": 120.0,
            "hit_probability": 0.85,
            # Warhead lethal-radius scale (see DEFAULT_LETHAL_RADIUS_M): the Gaussian
            # e-fold of warhead lethality vs terminal miss. Decoupled from the wider
            # proximity-fuze/CCD detonation radius (~500 m) so a fuze-triggering but
            # off-boresight near-miss degrades Pk instead of counting as a clean kill.
            "lethal_radius_m": 250.0,
            # Weapon-effectiveness submodels (see KillProbabilityModel): a modern
            # Western ARH missile with a reliable two-way datalink and seeker.
            "warhead_effectiveness": 1.0,
            "fuze_reliability": 0.95,
            "guidance_reliability": 0.95,
            "seeker_reliability": 0.95,
            "datalink_reliability": 0.95,
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
