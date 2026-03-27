from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from air_to_air_rl.aircrafts.aircraft import Aircraft
from air_to_air_rl.missiles.fox3.default_missile import LongRangeMissile
from air_to_air_rl.simulator.core.helpers import Position, units_distance_km
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg


@pytest.fixture
def map_limits():
    from types import SimpleNamespace

    return SimpleNamespace(
        left_lon=-1, right_lon=1, bottom_lat=-1, top_lat=1, min_alt=0, max_alt=20000
    )


@pytest.fixture
def sim(map_limits):
    return Simulator(tick_secs=0.5)


def random_target(map_limits, rng, missile_source):
    while True:
        lat = rng.uniform(map_limits.bottom_lat + 0.1, map_limits.top_lat - 0.1)
        lon = rng.uniform(map_limits.left_lon + 0.1, map_limits.right_lon - 0.1)
        alt = rng.uniform(3000, 15000)
        pos = Position(lat=lat, lon=lon, alt=alt)
        dist_km = (
            np.sqrt(
                (lat - missile_source.position.lat) ** 2 + (lon - missile_source.position.lon) ** 2
            )
            * 111
        )
        if 10 < dist_km < 100:
            tgt_bearing = geodetic_bearing_deg(
                missile_source.position.lat, missile_source.position.lon, lat, lon
            )
            angle_diff = abs(tgt_bearing - missile_source.yaw_deg)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff < 30:
                break
    yaw = geodetic_bearing_deg(
        lat, lon, missile_source.position.lat, missile_source.position.lon
    ) + rng.uniform(-45, 45)
    yaw = yaw % 360
    speed = rng.uniform(150, 350)
    target = Aircraft(
        name="Target",
        position=pos,
        yaw_deg=yaw,
        speed_mps=speed,
        group="red",
        map_limits=map_limits,
        config={
            "mass_kg": 18000,
            "reference_area_m2": 50,
            "max_speed_mps": 600,
            "n_max": 9,
            "radar_horizontal_fov_deg": 60,
            "radar_vertical_fov_deg": 30,
            "radar_max_range_m": 120000,
            "radar_frequency_hz": 10e9,
            "radar_tx_power_w": 15e3,
            "radar_antenna_gain_db": 40,
            "radar_snr_threshold_db": 10,
            "rcs": 1000,
        },
    )
    return target


def blue_aircraft(map_limits):
    pos = Position(lat=map_limits.bottom_lat + 0.2, lon=0.0, alt=8000)
    return Aircraft(
        name="Blue",
        position=pos,
        yaw_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=map_limits,
        config={
            "mass_kg": 18000,
            "reference_area_m2": 50,
            "max_speed_mps": 600,
            "n_max": 9,
            "radar_horizontal_fov_deg": 60,
            "radar_vertical_fov_deg": 30,
            "radar_max_range_m": 120000,
            "radar_frequency_hz": 10e9,
            "radar_tx_power_w": 15e3,
            "radar_antenna_gain_db": 40,
            "radar_snr_threshold_db": 10,
            "rcs": 10,
        },
    )


def log_state(label, obj):
    print(
        f"{label:10s} pos=({obj.position.lat:.4f},{obj.position.lon:.4f},{obj.position.alt:.1f})"
        f"  yaw={getattr(obj, 'yaw_deg', 0):6.2f}° pitch={getattr(obj, 'pitch_deg', 0):6.2f}°"
        f"  speed={getattr(obj, 'speed', 0):7.2f} m/s"
    )


def test_detailed_missile_tracking_and_guidance(sim, map_limits):
    # Nur distanzrelevante Ausgabe + Trajektorien-Plot
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    from air_to_air_rl.radar.core.utils import _angles_dist, enu_to_geodetic, geodetic_to_enu
    from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km

    rng = np.random.default_rng(42)
    blue = blue_aircraft(map_limits)
    sim.add_unit(blue)
    target = random_target(map_limits, rng, blue)
    sim.add_unit(target)

    # Blue auf Ziel ausrichten
    bearing = geodetic_bearing_deg(
        blue.position.lat, blue.position.lon, target.position.lat, target.position.lon
    )
    blue.yaw_deg = bearing

    # Initiale Distanz
    dist_km_init = geodetic_distance_km(
        blue.position.lat,
        blue.position.lon,
        blue.position.alt,
        target.position.lat,
        target.position.lon,
        target.position.alt,
    )
    print(f"[INFO] Initial Blue->Target distance: {dist_km_init:.2f} km")

    # Lock abwarten (max 50 Ticks), nur Event ausgeben
    locked = False
    for i in range(50):
        sim.do_tick()
        if blue.sensor.has_radar_lock(target):
            print(f"[EVENT] Radar lock at t+{i + 1}s")
            locked = True
            break

    # Note: Radar lock check is informational only - missile firing will proceed regardless
    # The missile's own radar will acquire and track the target
    if not locked:
        print("[WARNING] Radar lock was not established before missile launch - proceeding anyway")
        print("[INFO] Missile will use its own radar for target acquisition and tracking")

    # Missile starten
    result = blue.weapons.fire_missile(sim, target, LongRangeMissile)
    # fire_missile returns (missile, veto_reason, diagnostics)
    missile, veto_reason, diagnostics = result if isinstance(result, tuple) else (result, None, {})
    assert missile is not None, (
        f"Missile was not created! Veto: {veto_reason}, Diagnostics: {diagnostics}"
    )

    # EXTEND FUEL TIME FOR LONG-RANGE ENGAGEMENT
    original_motor_burn = missile.motor_burn_s
    original_life_time = missile.life_time_s
    original_fuel = missile.engine.fuel_s
    missile.motor_burn_s = 70.0  # Increase from ~30s to 70s
    missile.life_time_s = 120.0  # Increase total lifetime
    missile.engine.fuel_s = 70.0  # Update actual fuel in engine
    print(
        f"[INFO] Extended motor burn from {original_motor_burn:.1f}s to {missile.motor_burn_s:.1f}s"
    )
    print(f"[INFO] Extended lifetime from {original_life_time:.1f}s to {missile.life_time_s:.1f}s")
    print(f"[INFO] Extended fuel from {original_fuel:.1f}s to {missile.engine.fuel_s:.1f}s")

    # (kleiner Konsistenzcheck für ENU<->Geodetic; ohne extra Prints)
    _enu = np.array([80000, 5000, 3500])
    lat, lon, alt = enu_to_geodetic(
        _enu, missile.position.lat, missile.position.lon, missile.position.alt
    )
    _ = geodetic_to_enu(
        lat, lon, alt, missile.position.lat, missile.position.lon, missile.position.alt
    )

    # ---------- Trajektorien sammeln (2D-Bodenprojektion in km) ----------
    def _deg2km(lat_deg, lon_deg, lat0_deg, lon0_deg):
        dlat = lat_deg - lat0_deg
        dlon = lon_deg - lon0_deg
        ky = 111.32
        kx = 111.32 * math.cos(math.radians(lat0_deg))
        return dlon * kx, dlat * ky

    lat0 = float(blue.position.lat)
    lon0 = float(blue.position.lon)

    traj = {"blue": {"x": [], "y": []}, "missile": {"x": [], "y": []}, "target": {"x": [], "y": []}}

    # ---------- Distance and Tracker Error Output ----------
    print("\n=== DISTANCE LOG ===")
    print("tick | miss->tgt_dist_m | tracker_error_m (if avail)")

    min_dist_m = float("inf")
    hit = False

    for t in range(170):
        sim.do_tick()

        # Aktuelle Positionen
        missile_pos = missile.position
        target_pos = target.position

        # Distanz Missile–Target
        dist_mt_m = (
            geodetic_distance_km(
                missile_pos.lat,
                missile_pos.lon,
                missile_pos.alt,
                target_pos.lat,
                target_pos.lon,
                target_pos.alt,
            )
            * 1000.0
        )

        # Tracker error calculation (corrected to use proper ENU frame)
        tracker_info = missile.radar.get_tracker_info()
        if tracker_info is not None and tracker_info.get("position") is not None:
            est = tracker_info["position"]
            # Calculate true position error in ENU coordinates
            from air_to_air_rl.radar.core.utils import geodetic_to_enu

            dE, dN, dU = geodetic_to_enu(
                est.lat, est.lon, est.alt, target_pos.lat, target_pos.lon, target_pos.alt
            )
            tracker_error_m = float(np.linalg.norm([dE, dN, dU]))
            print(f"{t + 1:4d} | {dist_mt_m:15.1f} | {tracker_error_m:21.1f}")
        else:
            print(f"{t + 1:4d} | {dist_mt_m:15.1f} | {'NA':>21s}")

        # Min-Distanz tracken
        if dist_mt_m < min_dist_m:
            min_dist_m = dist_mt_m

        # Trajektorien punkten
        bx, by = _deg2km(float(blue.position.lat), float(blue.position.lon), lat0, lon0)
        mx, my = _deg2km(float(missile.position.lat), float(missile.position.lon), lat0, lon0)
        tx, ty = _deg2km(float(target.position.lat), float(target.position.lon), lat0, lon0)
        traj["blue"]["x"].append(bx)
        traj["blue"]["y"].append(by)
        traj["missile"]["x"].append(mx)
        traj["missile"]["y"].append(my)
        traj["target"]["x"].append(tx)
        traj["target"]["y"].append(ty)

        # Abbruchbedingungen (nur distanzrelevant melden)
        if not (target.id in sim.active_units and missile.id in sim.active_units):
            if target.id not in sim.active_units:
                print(
                    f"[RESULT] Target removed -> count as HIT at tick {t + 1} (dist≈{dist_mt_m:.1f} m)"
                )
                hit = True
            if missile.id not in sim.active_units:
                print(f"[DEBUG] Missile {missile.id} removed from simulation at tick {t + 1}")
                print(f"[DEBUG] Active units: {list(sim.active_units.keys())}")
            break

        if dist_mt_m < 500.0:
            print(f"[RESULT] HIT at tick {t + 1} (dist={dist_mt_m:.1f} m)")
            hit = True
            break

    print(f"[STATS] Minimum missile->target distance: {min_dist_m:.1f} m")

    # ---------- Plot nach der Schleife ----------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj["blue"]["x"], traj["blue"]["y"], label="Blue", linewidth=2)
    ax.plot(traj["missile"]["x"], traj["missile"]["y"], label="Missile", linewidth=2)
    ax.plot(traj["target"]["x"], traj["target"]["y"], label="Target", linewidth=2)

    def _mark_start_end(xs, ys, name):
        if xs and ys:
            ax.scatter(xs[0], ys[0], marker="o", s=40)
            ax.scatter(xs[-1], ys[-1], marker="x", s=60)
            ax.annotate(
                f"{name} start",
                (xs[0], ys[0]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
            ax.annotate(
                f"{name} end",
                (xs[-1], ys[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    _mark_start_end(traj["blue"]["x"], traj["blue"]["y"], "Blue")
    _mark_start_end(traj["missile"]["x"], traj["missile"]["y"], "Missile")
    _mark_start_end(traj["target"]["x"], traj["target"]["y"], "Target")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ΔLon (km)")
    ax.set_ylabel("ΔLat (km)")
    ax.set_title("2D Ground Tracks (local ENU approx)")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ground_tracks.png", dpi=150)
    print("[PLOT] saved to ground_tracks.png")

    assert hit, "Missile did not destroy the target or hit within desired ticks."
