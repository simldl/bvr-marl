import math

from bvr_marl_core.aircraft.types.f22 import F22
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _dist3d(a, b) -> float:
    dlat = (b.position.lat - a.position.lat) * 111_000.0
    dlon = (
        (b.position.lon - a.position.lon)
        * 111_000.0
        * math.cos(math.radians(0.5 * (a.position.lat + b.position.lat)))
    )
    dalt = b.position.alt - a.position.alt
    return math.sqrt(dlat * dlat + dlon * dlon + dalt * dalt)


def test_fox3_keeps_designated_target_with_multiple_valid_contacts():
    ml = _MapLimits()
    range_deg = 16.0 / 111.0

    shooter = F22(
        Position(0.0, 0.0, 8_000.0),
        yaw_deg=90.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )
    primary = F22(
        Position(0.0, range_deg, 8_000.0),
        yaw_deg=270.0,
        speed_mps=300.0,
        group="red",
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )
    distractor = F22(
        Position(0.035, range_deg * 0.75, 8_000.0),
        yaw_deg=270.0,
        speed_mps=300.0,
        group="red",
        map_limits=ml,
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )

    sim = Simulator(tick_secs=0.1, random_seed=1)
    sim.add_unit(shooter)
    sim.add_unit(primary)
    sim.add_unit(distractor)

    for _ in range(80):
        sim.do_tick()

    missile = AIM120_AMRAAM(
        firing_time_s=sim.elapsed_time_s,
        target=primary,
        source=shooter,
        map_limits=ml,
        group="blue",
    )
    # Deterministic kill on contact (flat Pk) so this identity test does not
    # depend on the range-dependent kill-probability model.
    missile.hit_probability = 1.0
    missile.lethal_radius_m = 0.0
    missile.designated_target_id = primary.id
    missile.retarget_policy = "locked_override"
    missile.target_provider.current_target_id = primary.id
    missile.radar.lock_ctrl.set_mode("track", primary.id)
    missile.radar.locked_target = primary.id
    sim.add_unit(missile)

    target_id_history = set()
    radar_lock_history = set()
    missile_target_history = set()
    closest_primary_m = float("inf")
    closest_distractor_m = float("inf")

    for _ in range(1_200):
        if primary.id in sim.active_units:
            primary.control.set_yaw_deg(270.0)
        if distractor.id in sim.active_units:
            distractor.control.set_yaw_deg(270.0)

        if missile.id in sim.active_units:
            target_id_history.add(missile.target_provider.current_target_id)
            radar_lock_history.add(missile.radar.get_locked_target())
            missile_target_history.add(getattr(missile.target, "id", None))
            if primary.id in sim.active_units:
                closest_primary_m = min(closest_primary_m, _dist3d(missile, primary))
            if distractor.id in sim.active_units:
                closest_distractor_m = min(closest_distractor_m, _dist3d(missile, distractor))

        sim.do_tick()
        if missile.id not in sim.active_units:
            break

    assert target_id_history == {primary.id}
    assert radar_lock_history == {primary.id}
    assert missile_target_history == {primary.id}
    assert primary.id not in sim.active_units
    assert distractor.id in sim.active_units
    assert closest_primary_m < closest_distractor_m
