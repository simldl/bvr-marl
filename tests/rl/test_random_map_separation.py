"""Random-map spawns must start inside detection range on the employment stages.

With a floor only, an early stage's config (400 km map, margin_frac 0.25, min_separation
50 km) produced a MEASURED median separation of 164 km -- p75 217, max 396 -- against a
Eurofighter ``radar_max_range_m`` of 185 km. A third of episodes began beyond detection
and the median one had to close ~164 km before a lock was physically possible, so
lock_rate and every derived shot opportunity collapsed to zero. The terms that reward
closing are potential-based and therefore policy-invariant, so the stage was asking for
the one behaviour its reward function could not pay for.
"""

from __future__ import annotations

import numpy as np
import pytest

from bvr_marl_core.rl.environment.scenarios.geometry_sampler import GeometrySampler

MAP_KM = 400.0
MARGIN = 0.25
MIN_SEP_M = 50_000.0
MAX_SEP_M = 100_000.0
EUROFIGHTER_RADAR_RANGE_KM = 185.0


def _separations(n: int, *, max_separation_m: float | None) -> np.ndarray:
    sampler = GeometrySampler(map_size_km=MAP_KM)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        geom = sampler.compute_random_map_positions(
            num_agents=1,
            num_opponents=1,
            margin_frac=MARGIN,
            min_separation_m=MIN_SEP_M,
            max_separation_m=max_separation_m,
            rng=rng,
        )
        a = geom["agent_positions"][0]
        o = geom["opponent_positions"][0]
        out.append(
            float(
                np.hypot((o.lat - a.lat) * sampler.km_per_deg, (o.lon - a.lon) * sampler.km_per_deg)
            )
        )
    return np.asarray(out)


def test_uncapped_spawns_routinely_start_beyond_radar_range():
    """Guards the premise -- if this stops being true the cap can be revisited."""
    seps = _separations(500, max_separation_m=None)

    assert np.median(seps) > EUROFIGHTER_RADAR_RANGE_KM / 2
    assert (seps > EUROFIGHTER_RADAR_RANGE_KM).mean() > 0.15


def test_capped_spawns_land_inside_radar_range():
    seps = _separations(500, max_separation_m=MAX_SEP_M)

    # The typical episode must start engageable, not in transit.
    assert np.median(seps) < MAX_SEP_M / 1000
    assert np.percentile(seps, 95) <= MAX_SEP_M / 1000
    # Comfortably inside detection range, which is the point of the cap.
    assert np.percentile(seps, 95) < EUROFIGHTER_RADAR_RANGE_KM


def test_cap_is_best_effort_not_a_hard_guarantee():
    """The resample loop gives up after max_resamples and accepts what it has.

    Documented rather than asserted away: a narrow band on a large box occasionally
    exhausts the retries, so a rare sample exceeds the cap. Callers must not assume a
    hard bound -- but it has to stay rare, or the band is mis-specified for the map.
    """
    seps = _separations(2000, max_separation_m=MAX_SEP_M)

    assert (seps > MAX_SEP_M / 1000).mean() < 0.01


def test_floor_is_still_respected():
    seps = _separations(500, max_separation_m=MAX_SEP_M)

    assert seps.min() >= MIN_SEP_M / 1000 - 1e-6


def test_inverted_band_is_rejected_rather_than_silently_ignored():
    sampler = GeometrySampler(map_size_km=MAP_KM)

    with pytest.raises(ValueError, match="max_separation_m"):
        sampler.compute_random_map_positions(
            num_agents=1,
            num_opponents=1,
            min_separation_m=100_000.0,
            max_separation_m=50_000.0,
        )
