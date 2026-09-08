"""IRST passive sensor, IR-plume launch warning, and battle-damage assessment."""

from types import SimpleNamespace

import numpy as np

from bvr_marl_core.radar.core.utils import geodetic_to_enu
from bvr_marl_core.registry import get_aircraft_class, get_missile_class
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

_ML = SimpleNamespace(left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20000)


def _mk(kind, lat, lon, yaw, grp):
    return get_aircraft_class(kind)(
        Position(lat=lat, lon=lon, alt=9000.0), yaw, 300.0, grp, _ML, 0.0, 20000.0
    )


# --- IRST sensor -----------------------------------------------------------


def test_f22_has_no_irst_but_f35_does():
    assert _mk("F35", 0, 0, 0, "BLUE").irst is not None
    assert _mk("F22", 0, 0, 0, "BLUE").irst is None


def test_irst_range_hotter_from_rear_and_afterburner():
    f35 = _mk("F35", 0, 0, 0, "BLUE")
    irst = f35.irst
    tgt = _mk(
        "Eurofighter", 0.3, 0.0, 0.0, "RED"
    )  # heading north (away) -> rear/hot to a southern IRST
    rear = irst.ir_detection_range(tgt, 0.0, 0.0)
    tgt.yaw_deg = 180.0  # nose-on (cold)
    front = irst.ir_detection_range(tgt, 0.0, 0.0)
    assert rear > front  # exhaust is hotter than the intake


def test_irst_detects_stealth_that_radar_cannot_and_triangulates():
    def track_err(two):
        sim = Simulator(tick_secs=1.0)
        f35 = _mk("F35", 0.0, 0.0, 0.0, "BLUE")
        red = _mk("F22", 0.3, 0.15, 270.0, "RED")  # VLO, no IRST, ~37 km
        sim.add_unit(f35)
        if two:
            sim.add_unit(_mk("F35", 0.0, 0.4, 0.0, "BLUE"))
        sim.add_unit(red)
        for _ in range(8):
            sim.do_tick()
        skin = [
            d
            for d in (f35.radar.cached_detections or [])
            if not d.get("range_denied") and getattr(d.get("T"), "id", None) == red.id
        ]
        assert not skin, "radar must not see the stealth F-22 (IR is the only way)"
        for t in f35.radar.cached_tracks or []:
            if sim.evaluator_truth_id_for_contact(f35.id, t.track_id) == red.id:
                true = np.array(
                    geodetic_to_enu(
                        red.position.lat,
                        red.position.lon,
                        red.position.alt,
                        f35.position.lat,
                        f35.position.lon,
                        f35.position.alt,
                    )
                )
                return np.linalg.norm(np.array(t.state[:3]) - true) / 1000.0
        return None

    lone = track_err(False)
    pair = track_err(True)
    assert lone is not None and pair is not None
    assert lone > 15.0  # one IRST: bearing-only, big range error
    assert pair < 6.0  # two datalinked IRST: triangulated


# --- IR-plume launch warning ----------------------------------------------


def test_launch_plume_warns_untargeted_observer():
    sim = Simulator(tick_secs=0.5, random_seed=1)
    shooter = _mk("Eurofighter", 0.0, 0.0, 0.0, "RED")
    target = _mk("Eurofighter", 0.5, 0.0, 180.0, "BLUE")
    observer = _mk("Eurofighter", 0.3, 0.2, 180.0, "BLUE")  # NOT the missile's target
    for u in (shooter, target, observer):
        sim.add_unit(u)
    for _ in range(6):
        sim.do_tick()
    shooter.weapons.fire_missile_direct(sim, target, get_missile_class("amraam"))
    warned = False
    for _ in range(20):
        sim.do_tick()
        if observer.sensor.missile_warner.get_warning_count() > 0:
            warned = True
            break
    assert warned, "an observer should detect a launch by its boost plume, even if not targeted"


# --- Battle damage assessment ---------------------------------------------


def test_bda_accumulates_observable_persistent_descent_evidence():
    sim = Simulator(tick_secs=0.5, random_seed=3)
    shooter = _mk("Eurofighter", 0.0, 0.0, 0.0, "BLUE")
    tgt = _mk("Eurofighter", 0.25, 0.0, 180.0, "RED")
    sim.add_unit(shooter)
    sim.add_unit(tgt)
    for _ in range(6):
        sim.do_tick()
    initial_probability = shooter.sensor.bda_probability.get(tgt.id, 0.0)
    tgt.control.set_pitch_deg(-30.0)
    for _ in range(60):
        sim.do_tick()
    assert tgt.is_mortally_hit is False
    assert shooter.sensor.bda_probability.get(tgt.id, 0.0) > initial_probability
    # Confirmation is deliberately fallible; this seed may produce a false
    # negative, but hidden mortal-hit state must never force confirmation.
    probability = shooter.sensor.bda_probability[tgt.id]
    threshold = shooter.sensor._bda_thresholds[tgt.id]
    assert (tgt.id in shooter.sensor.bda_confirmed) == (probability >= threshold)
