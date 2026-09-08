"""
GUI page registry — discovers and loads extension pages.

Scans the ``bvr_marl.gui_extensions`` entry-point group to find pages
contributed by installed extension packages.

Usage::

    from bvr_marl_core.gui.page_registry import load_extension_pages
    from bvr_marl_core.gui.extension_api import PageSpec

    extra_pages: list[PageSpec] = load_extension_pages()
"""

from __future__ import annotations

from importlib.metadata import entry_points

from bvr_marl_core.gui.extension_api import PageSpec

_ENTRY_POINT_GROUP = "bvr_marl.gui_extensions"


def load_extension_pages() -> list[PageSpec]:
    """
    Discover and load GUI pages from all installed extension packages.

    Each extension registers a callable under the
    ``bvr_marl.gui_extensions`` entry-point group.  That callable is
    invoked with no arguments and must return a ``list[PageSpec]``.

    Returns
    -------
    list[PageSpec]
        All pages contributed by installed extensions, sorted by
        (section, order, name).  Returns an empty list if no extensions
        are installed or if all extension loads fail.
    """
    pages: list[PageSpec] = []

    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except Exception:
        return pages

    for ep in eps:
        try:
            factory = ep.load()
            contributed = factory()
            if isinstance(contributed, list):
                pages.extend(contributed)
        except Exception as exc:
            # Gracefully degrade — a broken extension must not crash the app.
            print(f"[GUI] Failed to load extension '{ep.name}': {exc}")

    pages.sort(key=lambda p: (p.section, p.order, p.name))
    return pages
