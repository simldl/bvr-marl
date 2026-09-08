"""Composite kill-probability model (P_k pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bvr_marl_core.simulator.core.effectiveness.common import clamp01
from bvr_marl_core.simulator.core.effectiveness.fuze import FuzeModel
from bvr_marl_core.simulator.core.effectiveness.terminal_track import TerminalTrackQualityModel
from bvr_marl_core.simulator.core.effectiveness.vulnerability import VulnerabilityModel
from bvr_marl_core.simulator.core.effectiveness.warhead import WarheadModel


@dataclass
class KillProbabilityModel:
    """Factors the kill probability into conditional submodels:

        Pk = P_int * P_fuze * P_wh * P_vul * P_trk

    - ``P_int``  : valid intercept geometry was achieved. The substepper only
      calls ``on_hit`` at a within-fuse-radius closest approach, so this is 1.0
      by construction. Guidance reliability is not multiplied again after its
      outcome has already manifested in the achieved CPA.
    - ``P_fuze`` : :class:`FuzeModel`
    - ``P_wh``   : :class:`WarheadModel`
    - ``P_vul``  : :class:`VulnerabilityModel`
    - ``P_trk``  : :class:`TerminalTrackQualityModel`

    ``compute`` returns ``(Pk, components)`` where ``components`` is the flat
    per-term breakdown used by the terminal-event record.
    """

    fuze: FuzeModel = field(default_factory=FuzeModel)
    warhead: WarheadModel = field(default_factory=WarheadModel)
    vulnerability: VulnerabilityModel = field(default_factory=VulnerabilityModel)
    track: TerminalTrackQualityModel = field(default_factory=TerminalTrackQualityModel)

    def compute(
        self,
        missile: Any,
        target: Any | None = None,
        miss_distance_m: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        p_int = 1.0
        p_fuze = self.fuze.probability(missile, miss_distance_m)
        p_wh = self.warhead.probability(missile, miss_distance_m)
        p_vul = self.vulnerability.probability(missile, target, miss_distance_m)
        p_trk = self.track.probability(missile)

        pk = clamp01(p_int * p_fuze * p_wh * p_vul * p_trk)
        components = {
            "p_int": p_int,
            "p_fuze": p_fuze,
            "p_wh": p_wh,
            "p_vul": p_vul,
            "p_trk": p_trk,
        }
        return pk, components
