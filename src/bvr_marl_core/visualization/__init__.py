"""Visualization module for air-to-air RL."""

import os
import sys
from collections.abc import Iterable
from importlib.util import find_spec

import matplotlib

_NON_INTERACTIVE_BACKENDS = {
    "agg",
    "cairo",
    "pdf",
    "pgf",
    "ps",
    "svg",
    "template",
}


def _host_has_display() -> bool:
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def is_interactive_matplotlib_backend(backend: str | None = None) -> bool:
    """Return whether a Matplotlib backend can drive a live GUI window."""
    name = (backend or matplotlib.get_backend()).lower()
    if "matplotlib_inline" in name:
        return False
    if name.startswith("module://"):
        return True
    backend_name = name.rsplit(".", 1)[-1]
    return backend_name not in _NON_INTERACTIVE_BACKENDS


def can_show_matplotlib_window(backend: str | None = None) -> bool:
    """Return whether Matplotlib can reasonably open a live window on this host."""
    return _host_has_display() and is_interactive_matplotlib_backend(backend)


def _interactive_backend_candidates() -> Iterable[str]:
    if os.name == "nt":
        return ("TkAgg", "QtAgg", "Qt5Agg")
    if sys.platform == "darwin":
        return ("MacOSX", "QtAgg", "TkAgg")
    return ("TkAgg", "QtAgg", "Qt5Agg", "GTK4Agg", "GTK3Agg", "WxAgg")


def _backend_dependencies_available(backend: str) -> bool:
    name = backend.lower()
    if name == "tkagg":
        return find_spec("tkinter") is not None
    if name in {"qtagg", "qt5agg"}:
        return any(
            find_spec(module_name) is not None
            for module_name in ("PyQt6", "PySide6", "PyQt5", "PySide2")
        )
    if name in {"gtk4agg", "gtk3agg"}:
        return find_spec("gi") is not None
    if name == "wxagg":
        return find_spec("wx") is not None
    return True


def _configure_matplotlib_backend() -> None:
    """Use a non-interactive backend when visualization runs on headless Unix hosts."""
    if os.environ.get("MPLBACKEND"):
        return

    if not _host_has_display():
        matplotlib.use("Agg")


def ensure_interactive_matplotlib_backend() -> str:
    """Prefer an interactive backend for live viewers on hosts with a display."""
    if not _host_has_display() or is_interactive_matplotlib_backend():
        return matplotlib.get_backend()

    for backend in _interactive_backend_candidates():
        if not _backend_dependencies_available(backend):
            continue
        try:
            matplotlib.use(backend, force=True)
        except (ImportError, ValueError, RuntimeError):
            continue
        if is_interactive_matplotlib_backend():
            break

    return matplotlib.get_backend()


_configure_matplotlib_backend()
