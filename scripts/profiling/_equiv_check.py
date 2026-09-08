"""Capture/compare reference outputs of hot-path functions before/after optimization.

Usage:
    python scripts/profiling/_equiv_check.py capture   # writes reference .npz
    python scripts/profiling/_equiv_check.py compare   # compares current code vs reference
"""

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "_equiv_reference.npz"


def _build_cases():
    rng = np.random.default_rng(42)
    # lat/lon around a plausible map area, alt 0..15000
    cases = []
    for _ in range(200):
        lat = 45.0 + rng.uniform(-1.5, 1.5)
        lon = 10.0 + rng.uniform(-1.5, 1.5)
        alt = rng.uniform(0, 15000)
        rlat = 45.0 + rng.uniform(-1.5, 1.5)
        rlon = 10.0 + rng.uniform(-1.5, 1.5)
        ralt = rng.uniform(0, 15000)
        cases.append((lat, lon, alt, rlat, rlon, ralt))
    return cases


def compute_all():
    from bvr_marl_core.radar.core.utils import enu_to_geodetic, geodetic_to_enu
    from bvr_marl_core.radar.obs.cluster import Clusterer
    from bvr_marl_core.radar.tracking.filter.constant_velocity_filter import (
        ConstantVelocityKFFilter,
    )
    from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_basis, enu_rotation
    from bvr_marl_core.radar.tracking.helpers.recenter_logic import (
        transform_state_between_refs,
    )
    from bvr_marl_core.simulator.core.helpers import Position
    from bvr_marl_core.simulator.utils.geodesics import geodetic_direct

    rng = np.random.default_rng(7)
    cases = _build_cases()
    out = {}

    g2e = [geodetic_to_enu(*c) for c in cases]
    out["geodetic_to_enu"] = np.array(g2e)

    e2g = []
    for i, c in enumerate(cases):
        vec = g2e[i]
        e2g.append(enu_to_geodetic(vec, c[3], c[4], c[5]))
    out["enu_to_geodetic"] = np.array(e2g)

    out["enu_basis"] = np.array([enu_basis(c[0], c[1]) for c in cases])
    out["enu_rotation"] = np.array([enu_rotation(c[0], c[1], c[3], c[4]) for c in cases])

    xs, ps = [], []
    for c in cases[:50]:
        state6 = rng.normal(0, 1000, 6)
        a = rng.normal(0, 1, (6, 6))
        cov6 = a @ a.T + np.eye(6)
        x_t, p_t = transform_state_between_refs(
            state6, cov6, Position(c[0], c[1], c[2]), Position(c[3], c[4], c[5])
        )
        xs.append(x_t)
        ps.append(p_t)
    out["tsbr_x"] = np.array(xs)
    out["tsbr_p"] = np.array(ps)

    # CV filter: deterministic predict/update sequence
    kf = ConstantVelocityKFFilter(np.array([100.0, -50.0, 3000.0]), dt=1.0)
    kf.set_measurement_std((30.0, 30.0, 15.0))
    traj_x, traj_p, traj_nis = [], [], []
    for k in range(40):
        kf.predict(1.0 if k % 5 else 0.5)
        z = np.array([100.0 + 200.0 * k, -50.0 + 150.0 * k, 3000.0 + 5.0 * k])
        z += rng.normal(0, 10, 3)
        kf.update(z)
        traj_x.append(kf.get_state())
        traj_p.append(kf.get_covariance())
        traj_nis.append(kf.get_last_update_stats().get("nis", -1.0))
    out["kf_x"] = np.array(traj_x)
    out["kf_p"] = np.array(traj_p)
    out["kf_nis"] = np.array(traj_nis)

    # Clusterer
    cl = Clusterer(angular_res_deg=2.0, range_res_m=150.0)
    dets = []
    for i in range(60):
        dets.append(
            {
                "az": float(rng.uniform(-60, 60)),
                "el": float(rng.uniform(-20, 20)),
                "d": float(rng.uniform(1000, 80000)),
                "dop": float(rng.uniform(-300, 300)),
                "lat": 45.0 + float(rng.uniform(-1, 1)),
                "lon": 10.0 + float(rng.uniform(-1, 1)),
                "alt": float(rng.uniform(0, 12000)),
            }
        )
    dets.append({"az": float("nan"), "el": 0.0, "d": 1000.0})
    clusters = cl.cluster(dets)
    clusters.sort(key=lambda c: (c["az"], c["el"], c["d"]))
    out["clusters"] = np.array(
        [[c["az"], c["el"], c["d"], c["dop"], c["lat"], c["lon"], c["alt"]] for c in clusters]
    )

    # geodetic_direct small steps (movement fast path)
    gd = []
    for c in cases[:100]:
        for dist in (50.0, 400.0, 1200.0, 4000.0):
            gd.append(geodetic_direct(c[0], c[1], c[2], (c[3] * 77) % 360.0, dist, 12.5))
    out["geodetic_direct"] = np.array(gd)

    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    cur = compute_all()
    if mode == "capture":
        np.savez(OUT, **cur)
        print(f"Reference captured -> {OUT}")
        return

    ref = np.load(OUT)
    worst = {}
    for key in ref.files:
        a, b = ref[key], cur[key]
        if a.shape != b.shape:
            print(f"FAIL {key}: shape {a.shape} != {b.shape}")
            continue
        scale = np.maximum(np.abs(a), 1.0)
        rel = np.abs(a - b) / scale
        worst[key] = float(rel.max())
    print(f"{'key':<20} {'max rel diff':>14}")
    ok = True
    # geodetic_direct uses an intentional local-radii fast path for short steps
    # (centimeter-scale); the clusterer now aggregates in float64 instead of float32.
    tol = {"geodetic_direct": 3e-7, "clusters": 5e-6}
    for k, v in worst.items():
        t = tol.get(k, 1e-9)
        status = "ok" if v <= t else "DIFF"
        if v > t:
            ok = False
        print(f"{k:<20} {v:>14.3e}  {status}")
    print("EQUIVALENT" if ok else "DIFFERENCES EXCEED TOLERANCE (see above)")


if __name__ == "__main__":
    main()
