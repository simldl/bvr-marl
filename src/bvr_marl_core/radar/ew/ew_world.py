"""EWWorld: global coordinator for electronic-warfare interactions.

Current scope is noise jamming modelled as *range denial* (see
radar/ew/noise_jammer.py). For a victim aircraft radar it reports, per enemy
jammer, the range inside which the radar burns through the jamming and recovers a
usable range measurement. Beyond that range the jammer's skin return stays
detectable on the correct bearing but its range is denied (a strobe). Missile
seekers are not jam-susceptible.

The previous SNR-degradation + DRFM false-target ("ghost") model has been removed.
"""

from bvr_marl_core.radar.core.utils import _effective_rcs
from bvr_marl_core.radar.ew.noise_jammer import burn_through_range_m


class EWWorld:
    def __init__(self, sim):
        self.sim = sim

    def collect_range_denial(self, victim_radar, t: float = 0.0) -> dict[int, float]:
        """Burn-through range (m) per enemy jammer threatening ``victim_radar``.

        Returns a dict ``{jammer_target_id: burn_through_range_m}``. A detection of
        one of these jammers is range-denied when its true range exceeds the
        corresponding burn-through range. Empty for missile seekers (not
        jam-susceptible) or when nothing is jamming.
        """
        if not getattr(victim_radar, "jam_susceptible", True):
            return {}

        victim_owner = getattr(victim_radar, "owner", None)
        if victim_owner is None or not hasattr(victim_owner, "position"):
            return {}
        victim_group = getattr(victim_owner, "group", None)
        victim_pos = victim_owner.position

        denial: dict[int, float] = {}
        for unit in self.sim.active_units.values():
            burn_km = float(getattr(unit, "noise_jammer_burn_through_km", 0.0))
            if burn_km <= 0.0:
                continue
            if getattr(unit, "group", None) == victim_group:
                continue
            uid = getattr(unit, "id", None)
            if uid is None:
                continue
            # Effective RCS the victim sees (same aspect model as detection), so a
            # bigger/less-stealthy jammer is burnt through farther.
            try:
                sigma_eff = float(_effective_rcs(unit, victim_pos))
            except Exception:
                sigma_eff = 1.0
            denial[uid] = burn_through_range_m(victim_radar, burn_km, sigma_eff)
        return denial
