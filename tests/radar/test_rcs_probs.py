import math

import numpy as np
import pytest

from air_to_air_rl.aircrafts.aircraft import Aircraft
from air_to_air_rl.radar.core.utils import _effective_rcs  # your aspect-dependent σ_eff
from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


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


@pytest.fixture
def map_limits():
    from types import SimpleNamespace

    return SimpleNamespace(
        left_lon=-1.0, right_lon=1.0, bottom_lat=-1.0, top_lat=1.0, min_alt=0.0, max_alt=20000.0
    )


def _range_m(a_pos: Position, b_pos: Position) -> float:
    return (
        geodetic_distance_km(a_pos.lat, a_pos.lon, a_pos.alt, b_pos.lat, b_pos.lon, b_pos.alt)
        * 1000.0
    )


def test_radar_pd_two_rcs_runs(map_limits):
    # Two runs: stealthish (0.001 m^2) vs non-stealth (10 m^2)
    scenarios = [("ST", 1e-3), ("NS", 10.0)]

    # Accumulators for a sanity comparison at the end
    mean_pd = {}

    for idx, (label, rcs_val) in enumerate(scenarios):
        # Deterministic seeds per scenario
        np.random.seed(1234 + idx)

        sim = Simulator(tick_secs=1.0)

        # Blue radar platform (center south, facing north-ish)
        blue = _mk_aircraft(
            "Blue",
            "blue",
            Position(lat=map_limits.bottom_lat + 0.20, lon=0.00, alt=8000.0),
            yaw_deg=0.0,
            rcs=10.0,
            map_limits=map_limits,
        )
        sim.add_unit(blue)

        rng = np.random.default_rng(7 + idx)

        # One red target at ~30–40 km, slight azimuth so within FoV
        lat_t = float(blue.position.lat + rng.uniform(0.27, 0.36))
        lon_t = float(blue.position.lon + rng.uniform(-0.02, 0.02))
        red = _mk_aircraft(
            f"Red-{label}",
            "red",
            Position(lat=lat_t, lon=lon_t, alt=9000.0),
            yaw_deg=(
                geodetic_bearing_deg(lat_t, lon_t, blue.position.lat, blue.position.lon) + 10.0
            )
            % 360.0,
            rcs=rcs_val,
            map_limits=map_limits,
        )
        sim.add_unit(red)

        # Point Blue exactly toward Red for best FoV coverage
        blue.yaw_deg = geodetic_bearing_deg(
            blue.position.lat, blue.position.lon, red.position.lat, red.position.lon
        )

        radar = blue.sensor.radar
        assert radar is not None, "Blue must have a radar"

        R0 = _range_m(blue.position, red.position)
        print(f"[{label}] Initial separation: {R0 / 1000:.2f} km | base RCS={red.rcs:.6f} m^2")

        pd_list = []
        for t in range(10):
            # Run the actual radar update (obsgen → cluster → tracker → lock)
            tracks = radar.update(
                tick_secs=sim.tick_secs,
                sim=sim,
                targets=[u for uid, u in sim.active_units.items() if u is not blue],
                owner_position=blue.position,
                steer_h=0.0,
                steer_p=0.0,
                group_radars=None,
            )

            # Effective RCS from your model (depends on aspect/elevation)
            sigma_eff = float(_effective_rcs(red, blue.position))
            R = _range_m(blue.position, red.position)

            # Actual probability used by your radar code
            Pd = float(radar.lut.get_probability(R, sigma_eff))
            pd_list.append(Pd)

            # Print both base and effective RCS plus Pd
            print(
                f"[{label} t+{t + 1:02d}] σ0={red.rcs:.6f} m^2 | σ_eff={sigma_eff:.6e} m^2 | R={R / 1000:.1f} km | Pd={Pd:.3f}"
            )

            # Optional: brief state summary (detections/tracks/locks)
            dets = radar.cached_detections or []
            clus = radar.cached_clusters or []
            locked_ids = radar.lock_ctrl.locked_target_ids()
            print(
                f"           detections={len(dets)} clusters={len(clus)} tracks={len(tracks or [])} locked={locked_ids}"
            )

            sim.do_tick()

        mean_pd[label] = float(np.mean(pd_list))

    # Sanity: the non-stealth case should be substantially easier to detect on average
    assert mean_pd["NS"] >= mean_pd["ST"] + 0.3, f"Expected Pd(NS) >> Pd(ST); got means: {mean_pd}"
    print(f"[SUMMARY] mean Pd — ST={mean_pd['ST']:.3f} | NS={mean_pd['NS']:.3f}")
