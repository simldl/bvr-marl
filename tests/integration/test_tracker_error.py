from pathlib import Path

import numpy as np
import pytest

from air_to_air_rl.aircrafts.aircraft import Aircraft
from air_to_air_rl.missiles.fox3.default_missile import LongRangeMissile
from air_to_air_rl.radar.core.utils import geodetic_to_enu
from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km

pytestmark = pytest.mark.integration

# -------------------- Fixtures & Helpers --------------------


@pytest.fixture
def map_limits():
    from types import SimpleNamespace

    return SimpleNamespace(
        left_lon=-1, right_lon=1, bottom_lat=-1, top_lat=1, min_alt=0, max_alt=20000
    )


@pytest.fixture
def sim(map_limits):
    return Simulator(tick_secs=1)


def mk_blue(map_limits):
    pos = Position(lat=map_limits.bottom_lat + 0.20, lon=0.0, alt=8000.0)
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


def mk_target(map_limits, rng, shooter):
    """
    Erzeugt einen Ziel-Flieger in Sicht, 10–100 km entfernt und mit mäßiger
    Aspektabweichung, damit der Lock zuverlässig kommt.
    """
    while True:
        lat = rng.uniform(map_limits.bottom_lat + 0.10, map_limits.top_lat - 0.10)
        lon = rng.uniform(map_limits.left_lon + 0.10, map_limits.right_lon - 0.10)
        alt = rng.uniform(4000, 14000)
        dist_km = np.hypot(lat - shooter.position.lat, lon - shooter.position.lon) * 111.0
        if 10 < dist_km < 100:
            brg = geodetic_bearing_deg(shooter.position.lat, shooter.position.lon, lat, lon)
            if abs(((brg - shooter.yaw_deg + 180) % 360) - 180) < 30:
                break
    yaw = (
        geodetic_bearing_deg(lat, lon, shooter.position.lat, shooter.position.lon)
        + rng.uniform(-30, 30)
    ) % 360
    speed = rng.uniform(180, 320)
    return Aircraft(
        name="Target",
        position=Position(lat=lat, lon=lon, alt=alt),
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


# -------------------- Core Test --------------------


def test_tracker_focus(sim, map_limits):
    rng = np.random.default_rng(42)

    blue = mk_blue(map_limits)
    sim.add_unit(blue)

    target = mk_target(map_limits, rng, blue)
    sim.add_unit(target)

    # Blue auf Ziel ausrichten, damit Lock zuverlässig klappt
    blue.yaw_deg = geodetic_bearing_deg(
        blue.position.lat, blue.position.lon, target.position.lat, target.position.lon
    )

    # Radar-Lock abwarten (max. 50 Ticks)
    locked = False
    for i in range(50):
        sim.do_tick()
        if blue.sensor.has_radar_lock(target):
            locked = True
            break
    assert locked, "No radar lock before missile launch."

    # Abschuss
    result = blue.weapons.fire_missile(sim, target, LongRangeMissile)
    # fire_missile returns (missile, veto_reason, diagnostics)
    missile, veto_reason, diagnostics = result if isinstance(result, tuple) else (result, None, {})
    assert missile is not None, (
        f"Missile creation failed. Veto: {veto_reason}, Diagnostics: {diagnostics}"
    )

    print("\n=== TRACKER DIAGNOSTICS ===")
    print("Columns:")
    print(
        "tick | dist_true_m | est_err_E_m est_err_N_m est_err_U_m | pos_err_m | "
        "v_true_mps | v_est_mps | v_err_mps | n_obs conf upd_cnt life"
    )

    prev_target_llh = (target.position.lat, target.position.lon, target.position.alt)

    # Freeze-Mechanismen für genau 1 Tick über ENU-Switch
    last_valid_v_est = None
    last_valid_est_pos_llh = None
    freeze_next_tick = False  # wenn True: nächsten Tick Werte „einfrieren“

    for tick in range(120):
        sim.do_tick()

        # Abbruch, falls Target verschwunden
        if target.id not in sim.active_units:
            print(f"[RESULT] Target removed @tick {tick + 1} -> HIT")
            break

        # Distanz (Ground Truth)
        d_true_m = (
            geodetic_distance_km(
                missile.position.lat,
                missile.position.lon,
                missile.position.alt,
                target.position.lat,
                target.position.lon,
                target.position.alt,
            )
            * 1000.0
        )

        # Tracker-Infos
        ti = missile.radar.get_tracker_info() or {}
        est_pos = ti.get("position", None)  # erwartet LLH-Objekt o.ä. mit lat/lon/alt
        est_vel = ti.get("velocity", None)  # (vx,vy,vz)
        n_obs = ti.get("n_obs", 0)
        conf = ti.get("confidence", 0.0)
        upd = ti.get("update_count", 0)
        life = ti.get("lifetime", 0)

        # True-Vel über 1s-Ableitung
        dt = float(sim.tick_secs)
        dEt, dNt, dUt = geodetic_to_enu(
            target.position.lat,
            target.position.lon,
            target.position.alt,
            prev_target_llh[0],
            prev_target_llh[1],
            prev_target_llh[2],
        )
        v_true = np.array([dEt, dNt, dUt], dtype=float) / max(dt, 1e-9)
        v_true_mps = float(np.linalg.norm(v_true))

        # --- Positionsfehler (mit 1-Tick-Freeze über Switch) ---
        # Normalfall: Fehler aus aktueller est_pos
        if est_pos is not None:
            dE_cur, dN_cur, dU_cur = geodetic_to_enu(
                est_pos.lat,
                est_pos.lon,
                est_pos.alt,
                target.position.lat,
                target.position.lon,
                target.position.alt,
            )
            pos_err_cur = float(np.linalg.norm([dE_cur, dN_cur, dU_cur]))
        else:
            dE_cur = dN_cur = dU_cur = pos_err_cur = float("nan")

        # Heuristik: Ausreißer/ENU-Switch erkennen (nur für die PRINT-Ausgabe)
        # (a) Position springt extrem ODER (b) wir sind im „Freeze den nächsten Tick“-Modus
        pos_is_outlier = (pos_err_cur > 2000.0) or freeze_next_tick

        # --- Velocity (mit 1-Tick-Freeze) ---
        v_est_flag = " "  # "*" markiert eingefrorene Ausgabe
        if isinstance(est_vel, (list, tuple, np.ndarray)) and len(est_vel) == 3:
            v_est_vec = np.array(est_vel, dtype=float)
            v_est_norm = float(np.linalg.norm(v_est_vec))

            vel_is_outlier = (
                (v_est_norm > (v_true_mps + 200.0))
                or (
                    v_est_norm > 800.0
                    and last_valid_v_est is not None
                    and v_est_norm > 1.5 * float(np.linalg.norm(last_valid_v_est))
                )
                or freeze_next_tick
            )

            if vel_is_outlier and last_valid_v_est is not None:
                v_est_mps = float(np.linalg.norm(last_valid_v_est))
                v_err_mps = float(np.linalg.norm(last_valid_v_est - v_true))
                v_est_flag = "*"
            else:
                v_est_mps = v_est_norm
                v_err_mps = float(np.linalg.norm(v_est_vec - v_true))
                last_valid_v_est = v_est_vec.copy()
        else:
            v_est_mps = float("nan")
            v_err_mps = float("nan")

        # --- Positionswerte ggf. einfrieren (genau 1 Tick) ---
        pos_flag = ""  # optionaler Marker an den Komponenten
        if pos_is_outlier and last_valid_est_pos_llh is not None:
            # Fehler mit der zuletzt validen est_pos drucken
            dE, dN, dU = geodetic_to_enu(
                last_valid_est_pos_llh[0],
                last_valid_est_pos_llh[1],
                last_valid_est_pos_llh[2],
                target.position.lat,
                target.position.lon,
                target.position.alt,
            )
            pos_err_m = float(np.linalg.norm([dE, dN, dU]))
            pos_flag = "*"
        else:
            dE, dN, dU = dE_cur, dN_cur, dU_cur
            pos_err_m = pos_err_cur
            if est_pos is not None:
                last_valid_est_pos_llh = (est_pos.lat, est_pos.lon, est_pos.alt)

        # Nächsten Tick einfrieren, wenn dieser Tick ein starker Pos-Ausreißer war
        freeze_next_tick = bool(pos_err_cur > 2000.0)

        # --- Ausgabe ---
        # Hinweis: wir hängen das Sternchen an v_est_mps (wie gehabt) und
        # markieren Pos-Komponenten nur still (keine Formatänderung nötig)
        print(
            f"{tick + 1:4d} | "
            f"{d_true_m:10.1f} | "
            f"{dE:10.1f} {dN:10.1f} {dU:10.1f} | "
            f"{pos_err_m:9.1f} | "
            f"{v_true_mps:9.1f} | {v_est_mps:9.1f}{v_est_flag} | {v_err_mps:9.1f} | "
            f"{n_obs:5d} {conf:4.2f} {upd:7d} {life:4d}"
        )

        prev_target_llh = (target.position.lat, target.position.lon, target.position.alt)

        if d_true_m < 500.0:
            print(f"[RESULT] HIT @tick {tick + 1} (dist={d_true_m:.1f} m)")
            if v_est_flag == "*" or pos_flag == "*":
                print("[NOTE] Spike im ENU-Switch-Tick wurde für die Ausgabe unterdrückt.")
            break
