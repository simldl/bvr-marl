import matplotlib.pyplot as plt

UNIBW_COLORS = {
    "dark_blue": "#0E2841",
    "light_gray": "#E8E8E8",
    "blue": "#156082",
    "orange": "#E97132",
    "green": "#196B24",
    "cyan": "#0F9ED5",
    "purple": "#A02B93",
    "light_green": "#4EA72E",
    "hyperlink_blue": "#467886",
    "visited_link": "#96607D",
}

RUN_COLOR_ORDER = [
    UNIBW_COLORS["blue"],
    UNIBW_COLORS["orange"],
    UNIBW_COLORS["green"],
    UNIBW_COLORS["cyan"],
    UNIBW_COLORS["purple"],
    UNIBW_COLORS["light_green"],
]


def set_unibw_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 4.2),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "sans-serif",
            "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": UNIBW_COLORS["dark_blue"],
            "axes.edgecolor": UNIBW_COLORS["dark_blue"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": UNIBW_COLORS["dark_blue"],
            "ytick.color": UNIBW_COLORS["dark_blue"],
            "lines.linewidth": 2.4,
            "lines.markersize": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": UNIBW_COLORS["light_gray"],
            "grid.alpha": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
