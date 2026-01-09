"""
SQI Heatmap visualization.
Shows missile shot quality heatmap as a function of range and closure rate.
Uses calculation function from aircrafts/core/nez.py:sqi()
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


def get_amraam_params():
    """Load AMRAAM parameters from codebase."""
    amraam_base_range_km = 150.0        # max_range_m: 150,000
    amraam_min_range_km = 1.5           # min_range_m: 1,500

    try:
        import inspect
        from missiles.fox3.amraam import AIM120_AMRAAM

        amraam_source = inspect.getsource(AIM120_AMRAAM.__init__)
        if '150_000' in amraam_source:
            amraam_base_range_km = 150.0
            print(f"[OK] Loaded AMRAAM from source: {amraam_base_range_km} km base range")
        elif '40_000' in amraam_source:
            amraam_base_range_km = 40.0
            print(f"[OK] Loaded AMRAAM from source: {amraam_base_range_km} km base range")
    except Exception as e:
        print(f"Using hardcoded AMRAAM values: {e}")

    return amraam_base_range_km, amraam_min_range_km


def compute_sqi(distance_normalized, closure_rate_norm, aspect_cos, rho_ratio=1.0):
    """
    Compute SQI using exact logistic function from aircrafts/core/nez.py

    Args:
        distance_normalized: Range factor (0=max range, 1=min range)
        closure_rate_norm: Closure rate / 400 m/s (clipped to [-1, 1])
        aspect_cos: Cosine of relative heading angle
        rho_ratio: Altitude density ratio (default 1.0 for sea level)

    Returns:
        SQI value [0, 1] where 0=poor and 1=excellent probability of kill
    """
    # SQI logistic model parameters from aircrafts/core/nez.py:sqi()
    a0 = -1.4      # baseline threshold
    a_d = 3.0      # distance/range factor (phi_d)
    a_Vc = 1.2     # closure rate factor (normalized by 400 m/s)
    a_th = 0.8     # aspect angle factor (cosine of relative heading)
    a_rho = 0.25   # altitude/density factor

    x = a0 + a_d * distance_normalized + a_Vc * closure_rate_norm + a_th * aspect_cos + a_rho * (rho_ratio - 1)
    return 1.0 / (1.0 + np.exp(-x))


def create_sqi_heatmap():
    """Create SQI heatmap plot."""
    amraam_base_range_km, _ = get_amraam_params()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    range_vals = np.linspace(0, amraam_base_range_km, 80)
    closure_vals = np.linspace(-200, 600, 80)

    X, Y = np.meshgrid(range_vals, closure_vals)
    Z = np.zeros_like(X)

    for i in range(len(closure_vals)):
        for j in range(len(range_vals)):
            distance_norm = 1.0 - np.clip(X[i, j] / amraam_base_range_km, 0, 1)
            closure_norm = np.clip(Y[i, j] / 400, -1, 1)
            Z[i, j] = compute_sqi(distance_norm, closure_norm, 0.0)

    im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
    contours = ax.contour(X, Y, Z, levels=[0.3, 0.55, 0.7, 0.9], colors='black', alpha=0.4, linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=24, fmt='%.2f')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('SQI Value', fontsize=24)
    cbar.ax.tick_params(labelsize=24)

    ax.axhline(400, color='white', linestyle=':', alpha=0.8, linewidth=2)
    ax.text(amraam_base_range_km * 0.05, 410, 'Optimal closure rate', fontsize=24, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ax.set_xlabel('Slant Range (km)', fontsize=28)
    ax.set_ylabel('Closure Rate (m/s)', fontsize=28)
    ax.set_xlim(0, amraam_base_range_km)
    ax.set_ylim(-200, 600)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = create_sqi_heatmap()

    # Save to graphics folder
    graphics_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'graphics')
    os.makedirs(graphics_dir, exist_ok=True)

    save_path = os.path.join(graphics_dir, 'sqi_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to {save_path}')

    plt.show()
