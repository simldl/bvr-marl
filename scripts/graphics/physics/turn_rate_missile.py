"""Air-to-air missile turn rate vs Mach number graphic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure  # noqa: E402

from bvr_marl_core.physics.missiles import MissilePhysics  # noqa: E402
from bvr_marl_core.physics.physics import get_speed_of_sound  # noqa: E402


def plot_missile_turn_rate():
    """Plot missile instantaneous turn rate and its limits vs Mach number."""
    missile = MissilePhysics(MissilePhysics.Params(mass_kg=300.0, reference_area_m2=0.08))
    alt_m = 5_000.0
    mach = np.linspace(0.15, 5.0, 400)
    v_mps = get_speed_of_sound(alt_m) * mach

    turn_rate_ca = np.zeros_like(mach)
    turn_rate_n = np.zeros_like(mach)
    turn_rate_inst = np.zeros_like(mach)
    weight_n = missile.mass_kg * missile.g
    g = missile.g
    rho = missile.air.get_density(alt_m)
    for i, v in enumerate(v_mps):
        if v < 1e-3:
            continue
        q = 0.5 * rho * v**2
        n_ca = (q * missile.A_m2 * missile.get_ca_max(v, alt_m)) / weight_n
        if n_ca >= 1.0:
            turn_rate_ca[i] = math.degrees((g / v) * math.sqrt(max(0.0, n_ca**2 - 1.0)))
        turn_rate_n[i] = math.degrees((g / v) * math.sqrt(max(0.0, missile.n_max**2 - 1.0)))
        turn_rate_inst[i] = min(turn_rate_ca[i], turn_rate_n[i])

    fig, ax = paper_figure()
    ax.plot(mach, turn_rate_ca, color="green", linestyle="--", label=r"$\dot\chi(C_A^{\max})$")
    ax.plot(mach, turn_rate_n, color="red", linestyle="--", label=r"$\dot\chi(n_{\max}=30)$")
    ax.plot(mach, turn_rate_inst, color="blue", linewidth=1.6, label="Instantaneous")
    ax.set_xlabel("Mach number $M$")
    ax.set_ylabel(r"Turn rate $\dot{\chi}$ [deg/s]")
    ax.set_ylim([0, 40])
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_missile_turn_rate(), "turn_rate_missile")
