"""Shared matplotlib styling for single-column publication figures.

Every figure produced by the scripts in ``scripts/graphics`` is authored *at*
its final placement width (a single column, 3.25 in) rather than being drawn
large and shrunk down afterwards. Because matplotlib font sizes are absolute
(points), authoring at the target width means the size you set is the size on
the page: drop the PNG into a column at 3.25 in (100 %) and every piece of text
is at least :data:`MIN_READABLE_PT` (8 pt).

If a figure instead needs the full text width, render it at
``2 * COLUMN_WIDTH_IN`` and place it without scaling -- the fonts stay >= 8 pt.

Usage
-----
    from paper_style import paper_figure, save_paper_figure

    fig, ax = paper_figure()
    ax.plot(x, y)
    ax.set_xlabel("Mach number $M$")
    save_paper_figure(fig, "my_plot")

The scripts add this directory to ``sys.path`` so ``import paper_style`` works
when a script is run directly (e.g. ``python scripts/graphics/physics/...``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# IEEE / AIAA single-column text width. Figures are authored at this width so no
# post-hoc shrinking is needed to keep text readable.
COLUMN_WIDTH_IN = 3.25

# The smallest font size that must remain legible at COLUMN_WIDTH_IN.
MIN_READABLE_PT = 8.0

# Generated PNGs land here (scripts/graphics/output).
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# rcParams chosen so the *smallest* text (ticks, legend, inline annotations) is
# never below MIN_READABLE_PT when the figure is placed at COLUMN_WIDTH_IN.
_PAPER_RC = {
    # Match a typical AIAA/IEEE serif body font.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # Font sizes (all >= MIN_READABLE_PT).
    "font.size": 8,  # base size -> ax.text / annotations default to 8 pt
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    # Line / element weights tuned for a small physical size.
    "lines.linewidth": 1.3,
    "lines.markersize": 3.0,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.3,
    # Crisp raster output for print.
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "figure.constrained_layout.use": False,
}


def apply_paper_style() -> None:
    """Apply the single-column publication rcParams globally."""
    mpl.rcParams.update(_PAPER_RC)


def paper_figure(
    nrows: int = 1,
    ncols: int = 1,
    width_in: float = COLUMN_WIDTH_IN,
    height_in: float | None = None,
    row_height_in: float = 2.4,
    **kwargs,
):
    """Create a figure sized for single-column placement, paper style applied.

    Args:
        nrows: Number of subplot rows. Multi-panel figures are stacked
            vertically so each panel keeps the full column width.
        ncols: Number of subplot columns (usually 1 for single-column figures).
        width_in: Figure width in inches (defaults to one column, 3.25 in).
        height_in: Explicit figure height. If ``None``, it is derived from
            ``row_height_in * nrows``.
        row_height_in: Height per subplot row when ``height_in`` is not given.
        **kwargs: Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns:
        ``(fig, axes)`` exactly as :func:`matplotlib.pyplot.subplots` returns.
    """
    apply_paper_style()
    if height_in is None:
        height_in = row_height_in * nrows
    return plt.subplots(nrows, ncols, figsize=(width_in, height_in), **kwargs)


def save_paper_figure(fig, name: str, output_dir: Path | str = OUTPUT_DIR) -> Path:
    """Tidy layout and save ``fig`` as ``<output_dir>/<name>.png``.

    Args:
        fig: The matplotlib figure to save.
        name: File stem (without extension).
        output_dir: Destination directory (created if needed).

    Returns:
        The path the PNG was written to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {png_path}")
    return png_path
