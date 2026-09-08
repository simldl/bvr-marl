"""Passive IRST bearings must not manufacture ghost engageable tracks.

Two networked fighters each carry a passive (angle-only) IRST. With two opponents,
the two platforms' bearings cross at two real points *and* two spurious ones, and a
replayed multi-second strobe history re-triangulated that temporally-mixed bearing set
every tick -- so a mere 2 aircraft appeared as many network tracks, several of them
engageable. Firing then dumped multiple missiles onto phantom or duplicate contacts.

The shared network picture now (a) excludes replayed IRST bearings (fusing them only
within the current tick) and (b) refuses to mark a bearing-only triangulation engageable
until a genuinely ranged radar return corroborates it.
"""

import math

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _fighter(lat, lon, yaw, group):
    return Eurofighter(
        Position(lat, lon, 9_000.0),
        yaw_deg=yaw,
        speed_mps=280.0,
        group=group,
        map_limits=_MapLimits(),
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def _engageable_enemy_tracks(fighter):
    return [
        track
        for track in getattr(fighter.sensor, "sensor_tracks", ()) or ()
        if track.engageable and not track.suspect_deception
    ]


def test_two_networked_fighters_do_not_hallucinate_engageable_ghosts():
    sim = Simulator(tick_secs=0.5, random_seed=11)
    sim.record_traces = False
    east = (30.0 * 1_000.0) / (111_000.0 * math.cos(0.0))
    blues = [_fighter(0.2, 0.0, 90.0, "blue"), _fighter(-0.2, 0.0, 90.0, "blue")]
    reds = [_fighter(0.2, east, 270.0, "red"), _fighter(-0.2, east, 270.0, "red")]
    for unit in (*blues, *reds):
        sim.add_unit(unit)

    for _ in range(60):
        sim.do_tick()

    # Two clearly separated opponents must resolve as exactly two engageable
    # contacts -- not the handful of IRST-triangulation ghosts seen before.
    for blue in blues:
        engageable = _engageable_enemy_tracks(blue)
        assert len(engageable) == len(reds), (
            f"expected {len(reds)} engageable contacts, got {len(engageable)} "
            "(IRST bearing ghosts leaking through as shootable tracks)"
        )
