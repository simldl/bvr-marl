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

CHAFF_LAMBDA0 = 0.6  # per second at full effectiveness (beaming, fresh)
DECOY_LAMBDA0 = 0.4  # per second (aspect-independent)
FLARE_LAMBDA0 = 0.6  # per second (IR; verified/tuned in the IR step)

# Countermeasures only seduce once the missile is close enough to resolve them.
SEDUCE_MAX_RANGE_M = 20_000.0
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
