"""Air-to-Air Missile drag vs Mach number graphic."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from physics.missiles import MissilePhysics
from physics.physics import get_speed_of_sound


def plot_missile_drag() -> plt.Figure:
    """
    Plot missile drag components vs Mach number at 5000 m altitude.

    Uses the actual MissilePhysics implementation for accurate drag computations.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create missile physics model with generic missile parameters
    # Generic air-to-air missile: mass_kg=300.0, reference_area_m2=0.08
    # (Representative of medium-range missiles like AMRAAM family)
    missile = MissilePhysics(MissilePhysics.Params(
        mass_kg=300.0,
        reference_area_m2=0.08
    ))

    # Altitude for calculation
    alt_m = 5_000.0

    # Mach range
    mach = np.linspace(0.05, 5.0, 500)

    # Speed of sound at 5000 m
    a = get_speed_of_sound(alt_m)

    # Compute velocities
    V_mps = a * mach

    # Get air density at altitude
    rho = missile.air.get_density(alt_m)

    # Compute drag components
    D_parasite = np.zeros_like(mach)
    D_induced = np.zeros_like(mach)
    D_total = np.zeros_like(mach)

    W = missile.mass_kg * missile.g  # Weight in Newtons

    for i, v_mps in enumerate(V_mps):
        # Base drag coefficient
        cd0 = missile.get_base_drag_cd(v_mps, alt_m)

        # Parasite drag
        q = 0.5 * rho * v_mps**2  # Dynamic pressure
        D_parasite[i] = q * missile.A_m2 * cd0

        # Induced drag: CL = W / (0.5 * rho * V^2 * S)
        cl = W / (0.5 * rho * v_mps**2 * missile.A_m2) if v_mps > 1e-3 else 0.0
        D_induced[i] = q * missile.A_m2 * missile.K_ind * cl**2

        # Total drag
        D_total[i] = D_parasite[i] + D_induced[i]

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Convert to kN
    D_parasite_kN = D_parasite / 1000.0
    D_induced_kN = D_induced / 1000.0
    D_total_kN = D_total / 1000.0

    # Plot drag components
    ax.plot(mach, D_parasite_kN, color='gray', linewidth=2.0, label=r'$D_{\mathrm{parasite/wave}}$')
    ax.plot(mach, D_induced_kN, color='red', linewidth=2.0, label=r'$D_{\mathrm{induced}}$')
    ax.plot(mach, D_total_kN, color='blue', linewidth=2.0, label=r'$D_{\mathrm{total}}$')

    # Set labels (no title)
    ax.set_xlabel('Mach Number $M$', fontsize=28)
    ax.set_ylabel('Drag $D$ [kN]', fontsize=28)

    # Set y-axis limit
    ax.set_ylim([0, 20])

    # Grid
    ax.grid(True, which='both', alpha=0.3)

    # Legend
    ax.legend(fontsize=28, loc='upper left')

    # Tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = plot_missile_drag()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'graphics')
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, 'drag_missile.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {save_path}')
    plt.show()
