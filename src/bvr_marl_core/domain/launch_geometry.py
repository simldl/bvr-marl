"""Launch-geometry capture, and the metric keys the campaign diagnostics emit.

Lives in ``domain`` because ``weapon_firing`` (action space) captures the geometry
while the env-side collector aggregates it, and anything under ``gym`` drags in the
environment, which imports the action space right back.

Original note:
Why did the shots we took not kill anything? Answered from inside the campaign.

Standalone harness testing put the weapon chain at ~90% kills against exactly the
target an early warmup stage uses -- stationary, 15-100 km, fired through the real tracking chain,
with or without midcourse support. The campaign meanwhile fires ~1200 missiles for ~12
kills. Nothing reproducible from outside a live episode explains that gap, so the
launch conditions the policy actually chooses have to be measured where they happen.

Two things are collected per episode:

1. LAUNCH GEOMETRY at trigger pull -- range, aspect, altitude delta, shooter speed.
   A harness fires hot, co-altitude, at 300 m/s; if the policy is firing cold, slow, or
   from a bad altitude, that shows up here and nowhere else.

2. TERMINAL OUTCOME per missile, mined from ``MissileTerminalEvent``.

The second carries a subtlety worth stating: the event is emitted ONCE PER PROXIMITY
DETONATION. A missile that flies past without ever tripping its fuze -- out of energy,
guidance lost, seeker never acquired -- emits nothing at all. So the headline number is
not the kill rate but

    detonation_rate = terminal events / launches

which separates "the warhead fired and missed" from "the missile never arrived". Those
two have completely different causes and completely different fixes, and no existing
metric distinguishes them.
"""

from __future__ import annotations

import math

# Emitted under `tactical/` so they land beside the existing shot metrics in
# progress.csv and the autotune diagnosis, rather than in a separate namespace.
LAUNCH_RANGE_KM = "tactical/launch_range_km"
LAUNCH_ASPECT_DEG = "tactical/launch_aspect_deg"
LAUNCH_ALT_DELTA_M = "tactical/launch_alt_delta_m"
LAUNCH_SHOOTER_SPEED = "tactical/launch_shooter_speed_mps"
LAUNCH_CLOSURE_MPS = "tactical/launch_closure_mps"

DETONATION_RATE = "tactical/missile_detonation_rate"
TERMINAL_MISS_M = "tactical/missile_terminal_miss_m"
TERMINAL_PK = "tactical/missile_terminal_pk"
NO_TERMINAL_RATE = "tactical/missile_no_terminal_rate"

# Every key this module can emit, so the metrics header can reserve its columns even on
# an episode where nobody fired. Ray Tune's CSV logger locks progress.csv's header from
# the first result and silently drops keys that appear later.
MISSILE_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    LAUNCH_RANGE_KM,
    LAUNCH_ASPECT_DEG,
    LAUNCH_ALT_DELTA_M,
    LAUNCH_SHOOTER_SPEED,
    LAUNCH_CLOSURE_MPS,
    DETONATION_RATE,
    TERMINAL_MISS_M,
    TERMINAL_PK,
    NO_TERMINAL_RATE,
)

_EARTH_M_PER_DEG = 111_320.0


def _enu(origin, target) -> tuple[float, float, float]:
    """Metres east/north/up from ``origin`` to ``target``."""
    lat_scale = math.cos(math.radians(float(origin.lat)))
    return (
        (float(target.lon) - float(origin.lon)) * _EARTH_M_PER_DEG * lat_scale,
        (float(target.lat) - float(origin.lat)) * _EARTH_M_PER_DEG,
        float(target.alt) - float(origin.alt),
    )


def capture_launch_geometry_from_enu(shooter, east: float, north: float, up: float):
    """Launch geometry from a RELATIVE ENU offset, as carried by a TacticalContact.

    A ``TacticalContact`` has no ``.position`` -- it carries ``state`` as a 6-element
    relative ENU vector, deliberately, so controllers cannot reach simulator truth
    through it. The first version of this module looked for ``.position`` and silently
    got ``None``, so every contact-based launch recorded its geometry as absent --
    detonation_rate populated, launch_range_km empty.
    """
    try:
        slant = math.sqrt(east * east + north * north + up * up)
        if slant <= 0.0:
            return None
        yaw = math.radians(float(getattr(shooter, "yaw_deg", 0.0) or 0.0))
        nose_e, nose_n = math.sin(yaw), math.cos(yaw)
        horizontal = math.sqrt(east * east + north * north) or 1.0
        cos_aspect = max(-1.0, min(1.0, (nose_e * east + nose_n * north) / horizontal))

        velocity = getattr(shooter, "velocity", None)
        speed = closure = 0.0
        if velocity is not None:
            vx = float(getattr(velocity, "vx", 0.0))
            vy = float(getattr(velocity, "vy", 0.0))
            vz = float(getattr(velocity, "vz", 0.0))
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            closure = (vx * east + vy * north + vz * up) / slant

        return {
            LAUNCH_RANGE_KM: slant / 1000.0,
            LAUNCH_ASPECT_DEG: math.degrees(math.acos(cos_aspect)),
            LAUNCH_ALT_DELTA_M: up,
            LAUNCH_SHOOTER_SPEED: speed,
            LAUNCH_CLOSURE_MPS: closure,
        }
    except Exception:
        return None


def capture_launch_geometry(shooter, target_position) -> dict[str, float] | None:
    """Snapshot the geometry a shot was taken from, or ``None`` if unavailable.

    ``target_position`` is the ESTIMATED contact position the policy actually shot at,
    not truth -- the question being answered is what the agent believed when it pulled
    the trigger.
    """
    position = getattr(shooter, "position", None)
    if position is None or target_position is None:
        return None
    try:
        east, north, up = _enu(position, target_position)
        slant = math.sqrt(east * east + north * north + up * up)
        if slant <= 0.0:
            return None

        # Aspect: angle between where the shooter is pointing and the line of sight.
        # 0 deg = nose-on. The harness always fires at ~0; a policy firing at 90+ is
        # taking shots the harness never modelled.
        yaw = math.radians(float(getattr(shooter, "yaw_deg", 0.0) or 0.0))
        nose_e, nose_n = math.sin(yaw), math.cos(yaw)
        horizontal = math.sqrt(east * east + north * north) or 1.0
        cos_aspect = max(-1.0, min(1.0, (nose_e * east + nose_n * north) / horizontal))

        velocity = getattr(shooter, "velocity", None)
        speed = 0.0
        closure = 0.0
        if velocity is not None:
            vx = float(getattr(velocity, "vx", 0.0))
            vy = float(getattr(velocity, "vy", 0.0))
            vz = float(getattr(velocity, "vz", 0.0))
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            # Positive = closing on the contact.
            closure = (vx * east + vy * north + vz * up) / slant

        return {
            LAUNCH_RANGE_KM: slant / 1000.0,
            LAUNCH_ASPECT_DEG: math.degrees(math.acos(cos_aspect)),
            LAUNCH_ALT_DELTA_M: up,
            LAUNCH_SHOOTER_SPEED: speed,
            LAUNCH_CLOSURE_MPS: closure,
        }
    except Exception:
        # Diagnostics must never be able to fail an episode.
        return None


# --- Sensor-chain diagnostics -------------------------------------------------
#
# `lock_rate` collapsing to ~0.00 is the single most repeated observation across this
# project's campaigns, and on its own it cannot say WHY. The lock is the last link in a
# chain, and every link fails differently:
#
#     detections -> tentative track -> CONFIRMED track -> engageable -> own radar lock
#
# Nothing in flight sees anything      -> detections 0        (geometry/pointing problem)
# Detections but tracks never confirm  -> tentative > 0, confirmed 0   (SNR / scan rate)
# Tracks confirm then decay            -> coasting > 0        (losing the return)
# Confirmed but not engageable         -> engageable 0        (bearing-only, deception)
# Engageable but no lock               -> lock_rate 0         (the lock controller)
#
# Reported as per-step means over the episode, so a decline is attributable to the link
# that actually broke rather than inferred from the endpoint.
# NOTE: this is fresh returns THIS STEP, and the radar search is round-robin over
# `scan_sector_count` sectors (4 by default) with a dwell/revisit schedule. A target is
# therefore only illuminated a fraction of the time, and a mean around 0.1 is the SCAN
# CADENCE, not a detection failure. Read it against `tracks_total`: tracks persisting at
# ~1.3 on ~0.1 detections/step means the picture is being carried by coasting between
# revisits, which is normal here. A genuine sensing failure shows up as tracks_total
# falling, not as a low detection mean.
SENSOR_DETECTIONS = "tactical/sensor_detections"
SENSOR_TRACKS_TOTAL = "tactical/sensor_tracks_total"
# NOTE: per-lifecycle buckets (tentative/confirmed/coasting) are deliberately NOT
# emitted. TrackSnapshot exposes `state` as the 6-element STATE VECTOR, not the
# lifecycle, so they read 0.0 forever -- a fabricated zero indistinguishable from a real
# measurement, which is the exact trap these diagnostics exist to avoid. `engageable`
# carries the part that matters (CONFIRMED or REACQUIRED, not deception), and
# tracks_total minus tracks_engageable is the informative gap.
SENSOR_TRACKS_ENGAGEABLE = "tactical/sensor_tracks_engageable"
SENSOR_NEAREST_CONTACT_KM = "tactical/sensor_nearest_contact_km"
# Steps on which at least one track existed. This is the correct denominator for
# `nearest_contact_km`: that value is only defined when a track exists, so averaging it
# over ALL steps silently biases it toward zero in proportion to how often the agent
# holds nothing -- which is exactly the regime being diagnosed.
SENSOR_CONTACT_STEPS = "tactical/sensor_contact_steps"
# Fraction of steps on which the policy actually DESIGNATED a contact.
#
# This is the missing link in the lock_rate collapse. `lock_rate` is the own-radar lock
# on the DESIGNATED contact, so it is zero whenever nothing is designated -- regardless
# of how many engageable tracks the radar is holding. v21 stage 1 showed lock_rate
# 0.954 -> 0.000 while tracks_engageable stayed flat at 0.37-0.44, which is the exact
# signature of the target-selection axis drifting onto the empty slot: the contact-slot
# binning reserves bin 0 as an explicit "no target" choice, so a single action axis
# sliding toward it silently switches off locking, firing and every gate downstream.
#
# Without this metric that hypothesis is untestable from a run, because the veto
# attribution counters (no_target / suppressed / wasted) do not currently reach `info` --
# v21 recorded ~46 vetoed trigger pulls per episode with all three sub-counters at 0.00.
TARGET_DESIGNATED_RATE = "tactical/target_designated_rate"

SENSOR_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    SENSOR_DETECTIONS,
    SENSOR_TRACKS_TOTAL,
    SENSOR_TRACKS_ENGAGEABLE,
    SENSOR_NEAREST_CONTACT_KM,
    SENSOR_CONTACT_STEPS,
    TARGET_DESIGNATED_RATE,
)


def sensor_snapshot(unit) -> dict[str, float]:
    """Per-step state of this aircraft's sensor chain.

    Never raises: diagnostics must not be able to fail an episode.
    """
    out = {
        SENSOR_DETECTIONS: 0.0,
        SENSOR_TRACKS_TOTAL: 0.0,
        SENSOR_TRACKS_ENGAGEABLE: 0.0,
    }
    try:
        radar = getattr(unit, "radar", None)
        detections = getattr(radar, "cached_detections", None) or ()
        out[SENSOR_DETECTIONS] = float(len(detections))

        tracks = getattr(getattr(unit, "sensor", None), "sensor_tracks", None) or ()
        out[SENSOR_TRACKS_TOTAL] = float(len(tracks))
        nearest = None
        for track in tracks:
            if getattr(track, "engageable", False):
                out[SENSOR_TRACKS_ENGAGEABLE] += 1.0
            vector = getattr(track, "state", None)
            if isinstance(vector, (tuple, list)) and len(vector) >= 3:
                import math as _math

                distance = _math.sqrt(
                    float(vector[0]) ** 2 + float(vector[1]) ** 2 + float(vector[2]) ** 2
                )
                nearest = distance if nearest is None else min(nearest, distance)
        if nearest is not None:
            out[SENSOR_NEAREST_CONTACT_KM] = nearest / 1000.0
            out[SENSOR_CONTACT_STEPS] = 1.0
    except Exception:
        pass
    return out
