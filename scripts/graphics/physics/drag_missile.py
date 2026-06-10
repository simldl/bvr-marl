"""Air-to-air missile drag vs Mach number graphic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure  # noqa: E402

from bvr_marl_core.physics.missiles import MissilePhysics  # noqa: E402
from bvr_marl_core.physics.physics import get_speed_of_sound  # noqa: E402


def plot_missile_drag():
    """Plot missile drag components vs Mach number at 5000 m altitude."""
    missile = MissilePhysics(MissilePhysics.Params(mass_kg=300.0, reference_area_m2=0.08))
    alt_m = 5_000.0
    mach = np.linspace(0.05, 5.0, 500)
    v_mps = get_speed_of_sound(alt_m) * mach
    rho = missile.air.get_density(alt_m)

    d_parasite = np.zeros_like(mach)
    d_induced = np.zeros_like(mach)
    weight_n = missile.mass_kg * missile.g
    for i, v in enumerate(v_mps):
        q = 0.5 * rho * v**2
        d_parasite[i] = q * missile.A_m2 * missile.get_base_drag_cd(v, alt_m)
        cl = weight_n / (0.5 * rho * v**2 * missile.A_m2) if v > 1e-3 else 0.0
        d_induced[i] = q * missile.A_m2 * missile.K_ind * cl**2
    d_total = d_parasite + d_induced

    fig, ax = paper_figure()
    ax.plot(mach, d_parasite / 1e3, color="gray", label=r"$D_{\mathrm{parasite/wave}}$")
    ax.plot(mach, d_induced / 1e3, color="red", label=r"$D_{\mathrm{induced}}$")
    ax.plot(mach, d_total / 1e3, color="blue", linewidth=1.6, label=r"$D_{\mathrm{total}}$")
    ax.set_xlabel("Mach number $M$")
    ax.set_ylabel("Drag $D$ [kN]")
    ax.set_ylim([0, 20])
    ax.grid(True, which="both")
    ax.legend(loc="upper left")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_missile_drag(), "drag_missile")
