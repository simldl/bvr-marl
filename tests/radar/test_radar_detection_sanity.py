import math

import numpy as np

from bvr_marl_core.aircraft.aircraft import Aircraft

# Uses your actual probability path:
#  - σ_eff from bvr_marl_core.radar.core.utils._effective_rcs
#  - Pd = radar.lut.get_probability(dist, σ_eff)
from bvr_marl_core.radar.core.utils import _effective_rcs  # σ_eff(az, el) model used by obsgen
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator
from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km

np.random.seed(1234)


def _mk_aircraft(name, group, pos, yaw_deg, rcs, map_limits, extra_cfg=None):
    cfg = {
        "mass_kg": 18000,
        "reference_area_m2": 50,
        "max_speed_mps": 300,
        "n_max": 9,
        "radar_horizontal_fov_deg": 60,
        "radar_vertical_fov_deg": 30,
        "radar_max_range_m": 120_000,
        "radar_frequency_hz": 10e9,
        "radar_tx_power_w": 1.5e4,
        "radar_antenna_gain_db": 40,
        "radar_snr_threshold_db": 10,
        "rcs": rcs,
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    return Aircraft(
        name=name,
        position=pos,
        yaw_deg=yaw_deg,
        speed_mps=300.0,
        group=group,
        map_limits=map_limits,
        config=cfg,
    )


def test_radar_detection_probabilities_match_lut_and_rcs(tmp_path):
    # Small rectangular map; altitudes in meters
    from types import SimpleNamespace

    map_limits = SimpleNamespace(
        left_lon=-1.0, right_lon=1.0, bottom_lat=-1.0, top_lat=1.0, min_alt=0.0, max_alt=20000.0
    )
    sim = Simulator(tick_secs=1.0)

    # Blue radar platform (center south, facing north-ish)
    blue = _mk_aircraft(
        "Blue",
        "blue",
        Position(lat=map_limits.bottom_lat + 0.20, lon=0.00, alt=8000.0),
        yaw_deg=0.0,
        rcs=10,
        map_limits=map_limits,
    )
    sim.add_unit(blue)

    rng = np.random.default_rng(7)

    # Put two reds at ~30–40 km, slight azimuth offsets so both are inside FoV.
    # Choose RCS values under LUT cap (max_rcs=20) to avoid saturation.
    lat_ns = float(blue.position.lat + rng.uniform(0.27, 0.36))  # ~30–40 km
    lon_ns = float(blue.position.lon + rng.uniform(-0.02, 0.02))
    red_ns = _mk_aircraft(  # non-stealth
        "Red-NS",
        "red",
        Position(lat=lat_ns, lon=lon_ns, alt=9000.0),
        yaw_deg=(geodetic_bearing_deg(lat_ns, lon_ns, blue.position.lat, blue.position.lon) + 10.0)
        % 360.0,
        rcs=10.0,  # < 20 so LUT is effective (no clamp to Pd=1.0)
        map_limits=map_limits,
    )
    sim.add_unit(red_ns)

    lat_st = float(blue.position.lat + rng.uniform(0.27, 0.36))
    lon_st = float(blue.position.lon + rng.uniform(-0.02, 0.02))
    red_st = _mk_aircraft(  # stealth
        "Red-ST",
        "red",
        Position(lat=lat_st, lon=lon_st, alt=9000.0),
        yaw_deg=(geodetic_bearing_deg(lat_st, lon_st, blue.position.lat, blue.position.lon) - 10.0)
        % 360.0,
        rcs=1e-3,  # head-on stealth order
        map_limits=map_limits,
    )
    sim.add_unit(red_st)

    # Aim the boresight between both reds so both are within FoV
    az_to_mid = geodetic_bearing_deg(
        blue.position.lat, blue.position.lon, (lat_ns + lat_st) * 0.5, (lon_ns + lon_st) * 0.5
    )
    blue.yaw_deg = az_to_mid

    def range_m(a, b):
        return geodetic_distance_km(a.lat, a.lon, a.alt, b.lat, b.lon, b.alt) * 1000.0

    # Convenience
    radar = blue.sensor.radar
    assert radar is not None, "Blue must have a radar"

    print(
        f"[INFO] Initial separations: NS={range_m(blue.position, red_ns.position) / 1000:.2f} km, "
        f"ST={range_m(blue.position, red_st.position) / 1000:.2f} km"
    )

    got_any_detection = False
    got_any_track = False
    got_any_lock = False

    for t in range(10):
        # Run your actual radar stack: obsgen → cluster → tracker → lock
        tracks = radar.update(
            tick_secs=sim.tick_secs,
            sim=sim,
            targets=[u for uid, u in sim.active_units.items() if u is not blue],
            owner_position=blue.position,
            steer_h=0.0,
            steer_p=0.0,
            group_radars=None,
        )

        dets = radar.cached_detections or []
        clus = radar.cached_clusters or []
        trks = tracks or []

        # Compute *actual* Pd used by your code for both targets:
        R_ns = range_m(blue.position, red_ns.position)
        R_st = range_m(blue.position, red_st.position)
        sigma_ns = _effective_rcs(red_ns, blue.position)  # your aspect-dependent σ_eff
        sigma_st = _effective_rcs(red_st, blue.position)
        Pd_ns = radar.lut.get_probability(R_ns, sigma_ns)
        Pd_st = radar.lut.get_probability(R_st, sigma_st)

        print(
            f"[t+{t + 1:02d}] Pd(NS)={Pd_ns:.3f} (σ={sigma_ns:.3e}, R={R_ns / 1000:.1f}km) | "
            f"Pd(ST)={Pd_st:.3f} (σ={sigma_st:.3e}, R={R_st / 1000:.1f}km)"
        )
        print(f"[t+{t + 1:02d}] detections={len(dets)} clusters={len(clus)} tracks={len(trks)}")

        if dets:
            got_any_detection = True
            # print a couple of detections (SNR reported by your generator)
            for d in dets[:2]:
                snr = d.get("snr_db", None)
                if snr is not None:
                    print(
                        f"  det: az={d['az']:.1f} el={d['el']:.1f} R={d['d'] / 1000.0:.1f}km  SNR={snr:.1f} dB"
                    )

        if trks:
            got_any_track = True
            for track in trks:
                print(
                    f"  track tid={track.track_id} conf={track.confidence:.2f} "
                    f"n_obs={len(track.source_ids)} life={track.lifetime_s:.1f} "
                    f"engageable={track.engageable}"
                )

        locked_ids = radar.lock_ctrl.locked_target_ids()
        if locked_ids:
            got_any_lock = True
        print(f"  locked={locked_ids}")

        sim.do_tick()

    # Expectations: non-stealth Pd >> stealth Pd; at least one detection/track/lock overall.
    assert Pd_ns >= Pd_st + 0.2, (
        "Expected significantly higher detection probability for non-stealth"
    )
    assert got_any_detection, "Expected at least one radar detection"
    assert got_any_track, "Expected at least one radar track"
    assert got_any_lock, "Expected at least one radar lock"
