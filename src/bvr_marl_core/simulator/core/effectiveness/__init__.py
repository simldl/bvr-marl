"""Modular weapon-effectiveness submodels.

Terminal kill probability is factored into physically/tactically meaningful
conditional terms:

    Pk = P_int * P_fuze * P_wh * P_vul * P_trk

Each submodel is a small, pure-scalar, stateless object (no per-tick work, no
numpy allocation) — the whole pipeline runs once per detonation inside
``stochastic_on_hit``. Once the geometric CPA gate is achieved, ``P_int`` is
one; clean warhead effectiveness defaults to one, while fuze reliability,
miss-distance falloff, target vulnerability, and terminal-track quality remain
explicit factors.
"""

from bvr_marl_core.simulator.core.effectiveness.fuze import FuzeModel
from bvr_marl_core.simulator.core.effectiveness.kill_model import KillProbabilityModel
from bvr_marl_core.simulator.core.effectiveness.terminal_track import TerminalTrackQualityModel
from bvr_marl_core.simulator.core.effectiveness.vulnerability import (
    VulnerabilityClassParams,
    VulnerabilityModel,
)
from bvr_marl_core.simulator.core.effectiveness.warhead import WarheadModel

__all__ = [
    "FuzeModel",
    "WarheadModel",
    "VulnerabilityModel",
    "VulnerabilityClassParams",
    "TerminalTrackQualityModel",
    "KillProbabilityModel",
]
