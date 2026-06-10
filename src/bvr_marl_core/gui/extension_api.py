"""
GUI extension API for bvr-marl-core.

Defines the contract that external extension packages must
implement to inject pages into the core GUI shell.

Extension packages register via the ``bvr_marl.gui_extensions`` entry-point
group in their ``pyproject.toml``:

    [project.entry-points."bvr_marl.gui_extensions"]
    my_extension = "my_package.gui_ext.registry:register_gui_pages"

The registered callable must return a list of ``PageSpec`` objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class PageSpec:
    """
    Specification for a single GUI page contributed by an extension package.

    Attributes
    ----------
    name:
        Display name shown in the sidebar navigation (e.g. ``"🏋 Training Ops"``).
    render:
        Zero-argument callable that renders the page when selected.
    section:
        Optional section heading used to group pages in the sidebar.
        Pages with the same section are visually grouped together.
    order:
        Sort key within a section (lower = earlier).  Core pages have
        ``order=0``; extensions default to ``order=100``.
    """

    name: str
    render: Callable[[], None]
    section: str = "Extensions"
    order: int = 100


# ---------------------------------------------------------------------------
# Extension callable type alias
# ---------------------------------------------------------------------------

# An extension entry-point must be a zero-argument callable that returns
# a list of PageSpec objects.
ExtensionFactory = Callable[[], list[PageSpec]]
