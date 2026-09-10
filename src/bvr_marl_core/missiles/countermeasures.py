"""Countermeasure seduction of missiles.

Expendable countermeasures work by seduction: a defeated missile is pulled onto
the false return and flies to the (stationary) countermeasure cloud/object while
the aircraft escapes. Effectiveness is a rate-based (memoryless) per-tick draw so
that sustained exposure builds up a defeat probability:

    p_tick = 1 - exp(-lambda * dt),   lambda = lambda0 * geometry * window

Chaff (radar) is only effective while the target is beaming the missile: its
radial velocity is near zero, so its Doppler return looks like the ~stationary
chaff and the seeker cannot gate it out (the workshop's "chaff at 90 degrees").
A decoy (radar) is effective from any aspect but is scarce. Flares (IR) reuse the
same machinery against IR seekers and are tuned in the IR step. Countermeasures
only bite once the missile is close enough for its seeker to resolve them.
"""

import math

from bvr_marl_core.radar.core.utils import geodetic_to_enu
from bvr_marl_core.simulator.core.helpers import Position

# Seduction hazard rates, per second at full effectiveness.
#
# Measured before this retune: 10 of 16 missiles (63%) in a BT-vs-BT engagement were
# decoyed before reaching a terminal event, while the shots that got through killed at
# 67% (4 kills from 6 detonations). A defender was therefore defeating roughly two thirds
# of incoming fire with expendables, which is far above what chaff or flares achieve
# against a modern active seeker.
#
# At the old rates a 5 s exposure gave P(seduce) = 1 - exp(-0.6*5) = 95%, so the outcome
# was effectively "any countermeasure released in the basket works". Cut by a third, the
# same exposure gives ~63%, and combined with the halved resolve range below the measured
# end-to-end seduction rate lands near a quarter of shots rather than two thirds.
CHAFF_LAMBDA0 = 0.20  # per second at full effectiveness (beaming, fresh)
# Decoys are the outlier and need a lower rate than chaff for the same END effect, not
# the same rate. Chaff only works while the target beams (`_beam_factor`), so its
# effective exposure is a fraction of the approach; a decoy is aspect-independent and
# lives 15 s, so it works for the whole time the missile is inside the resolve range --
# about 10 s at closing speed. At the shared rate that left decoy seduction at 0.75 while
# chaff had already fallen to 0.25. This puts a decoy near 0.30 over the same window.
DECOY_LAMBDA0 = 0.035  # per second (aspect-independent)
FLARE_LAMBDA0 = 0.20  # per second (IR; verified/tuned in the IR step)

# Countermeasures only seduce once the missile is close enough to resolve them.
#
# 20 km gave a decoy the better part of half a minute of flight time to work in. An active
# seeker resolving a chaff bloom from its target at 20 km is generous; halving it both
# raises the bar and shortens the exposure window, which compounds with the rates above.
SEDUCE_MAX_RANGE_M = 10_000.0
# Target radial speed (m/s) at which chaff falls out of the Doppler notch.
CHAFF_NOTCH_MPS = 150.0

_RADAR_CM = frozenset({"chaff", "decoy"})
_IR_CM = frozenset({"flare"})


def _cm_types_for(missile) -> frozenset:
    fox = getattr(missile, "fox_type", 3)
    if fox in (1, 3):
        return _RADAR_CM
    if fox == 2:
        return _IR_CM
    return frozenset()


def _window_factor(cm) -> float:
    """Linear decay of effectiveness over the countermeasure's lifetime."""
    life = float(getattr(cm, "lifetime_s", 0.0))
    if life <= 0.0:
        return 0.0
    return max(0.0, 1.0 - float(getattr(cm, "age_s", 0.0)) / life)


def _beam_factor(target, los_unit) -> float:
    """1 when the target beams the missile (radial velocity ~ 0), 0 when hot/cold."""
    v = getattr(target, "velocity", None)
    if v is None:
        return 0.0
    radial = abs(v.vx * los_unit[0] + v.vy * los_unit[1] + v.vz * los_unit[2])
    return max(0.0, 1.0 - radial / CHAFF_NOTCH_MPS)


def evaluate_seduction(missile, sim, dt: float) -> None:
    """Possibly seduce ``missile`` onto an active countermeasure of its target.

    Once seduced the missile stays committed (``missile.seduced_by``); the guidance
    then homes on the countermeasure instead of the aircraft.
    """
    if (
        getattr(missile, "seduced_by", None) is not None
        or getattr(missile, "seduced_position", None) is not None
    ):
        return  # already committed to a decoy
    target = getattr(missile, "target", None)
    if target is None:
        # Physical countermeasure adjudication shares the evaluator boundary
        # with collision resolution. Only the frozen decoy position crosses
        # back into guidance; target/countermeasure Units never do.
        resolver = getattr(sim, "evaluator_target_for_weapon", None)
        target = resolver(missile) if callable(resolver) else None
    if target is None:
        return
    cm_types = _cm_types_for(missile)
    if not cm_types:
        return
    cms = getattr(getattr(target, "countermeasures", None), "active_countermeasures", None) or []
    if not cms:
        return

    mp, tp = missile.position, target.position
    los = geodetic_to_enu(tp.lat, tp.lon, tp.alt, mp.lat, mp.lon, mp.alt)
    rng = math.sqrt(los[0] ** 2 + los[1] ** 2 + los[2] ** 2)
    if rng > SEDUCE_MAX_RANGE_M or rng < 1e-6:
        return
    los_unit = (los[0] / rng, los[1] / rng, los[2] / rng)

    rnd = getattr(sim, "rnd_gen", None)
    for cm in cms:
        ct = getattr(cm, "cm_type", None)
        if ct not in cm_types:
            continue
        window = _window_factor(cm)
        if window <= 0.0:
            continue
        if ct == "chaff":
            lam = CHAFF_LAMBDA0 * window * _beam_factor(target, los_unit)
        elif ct == "decoy":
            lam = DECOY_LAMBDA0 * window
        else:  # flare (IR)
            lam = FLARE_LAMBDA0 * window
        if lam <= 0.0:
            continue
        p = 1.0 - math.exp(-lam * float(dt))
        roll = rnd.random() if rnd is not None else __import__("random").random()
        if roll < p:
            position = cm.position
            if getattr(missile, "weapon_track", None) is not None:
                missile.seduced_position = Position(
                    float(position.lat), float(position.lon), float(position.alt)
                )
            else:
                missile.seduced_by = cm
            return
