"""Access to the real sub-stepped path a unit flew during the current tick.

Aircraft and missiles both integrate their motion in sub-steps beneath the
control tick, and both record the positions they actually swept. Anything that
needs to know where a unit was *part-way through* a tick -- missile guidance
re-aiming, closest-point-of-approach resolution -- should read that recorded
path rather than reconstruct a curve from the tick's two endpoints.

Reconstruction was the previous behaviour and it is systematically wrong for the
manoeuvre that matters most: interpolating between begin- and end-of-tick poses
assumes the unit travelled the short way between them, which is exactly what a
hard defensive break does not do. A jet pulling 5 g at 70 deg of bank sweeps an
arc whose midpoint sits over a hundred metres off the straight line, and the
proximity fuze is deciding at a 250 m lethal radius.
"""

from __future__ import annotations

#: Tolerance when matching a recorded path's timestamp against the sim clock.
#: ``elapsed_time_s`` is derived from a timedelta each call, so it is stable
#: within a tick; the tolerance only guards against float representation drift.
_STAMP_TOLERANCE_S = 1e-6


def unit_tick_path(unit, sim) -> list[tuple[float, float, float]] | None:
    """Return the unit's swept path for the CURRENT tick, or None.

    Returns None when the unit records no path, when the path is too short to
    interpolate, or when it is stale -- a unit skipped during this tick's
    dynamics stage still carries the previous tick's path, and using that would
    place it somewhere it no longer is.
    """
    path = getattr(unit, "tick_path", None)
    if not path or len(path) < 2:
        return None

    stamp = getattr(unit, "tick_path_time_s", None)
    if stamp is None:
        return None
    now = getattr(sim, "elapsed_time_s", None) if sim is not None else None
    if now is None:
        return None
    if abs(float(stamp) - float(now)) > _STAMP_TOLERANCE_S:
        return None

    return path


def pose_at_fraction(
    path: list[tuple[float, float, float]], fraction: float
) -> tuple[float, float, float]:
    """Sample a swept path at a fraction of the tick.

    Sub-steps partition the tick evenly, so path index maps linearly to time.
    Interpolation is linear *within* a segment, which is accurate because the
    segments are short -- the curvature the old endpoint reconstruction missed
    is carried by the vertices themselves.
    """
    f = max(0.0, min(1.0, float(fraction)))
    n = len(path) - 1
    if n <= 0:
        return path[0]

    scaled = f * n
    i = min(int(scaled), n - 1)
    local = scaled - i
    a, b = path[i], path[i + 1]
    return (
        a[0] + (b[0] - a[0]) * local,
        a[1] + (b[1] - a[1]) * local,
        a[2] + (b[2] - a[2]) * local,
    )
