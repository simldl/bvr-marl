"""Authoritative declaration of the bvr_marl_core public API surface.

This module is the **single source of truth** for which ``bvr_marl_core``
modules extension packages are allowed to import.  Both core's own tooling and
every downstream extension repo should consume the contract from here rather
than maintaining their own copy.

A new extension repo wires up its import-boundary check in a few lines::

    import ast
    from bvr_marl_core.public_api import violation_for

    for node in ast.walk(tree):
        ...  # collect imported module names
        if (msg := violation_for(module)) is not None:
            report(msg)

The rules
---------
1. Only modules inside the ``bvr_marl_core`` namespace are governed.  Imports of
   third-party packages or the extension's own package are never violations.
2. Importing any module whose dotted path contains a private
   (underscore-prefixed) component is forbidden.
3. Otherwise the module must be one of :data:`PUBLIC_API_MODULES` (exact match),
   or a sub-module of one of them (the bare top-level package matches by exact
   name only, so it does not act as a wildcard over the whole tree).
"""

from __future__ import annotations

TOP_LEVEL = "bvr_marl_core"

#: The complete allowlist of modules extension packages may import from.
#:
#: ``bvr_marl_core`` (the bare top-level) exposes a curated flat convenience
#: surface via its ``__init__`` re-exports.  The heavyweight subsystems (RL,
#: visualization, analysis, tacview, gui) are gated behind explicit ``*.api`` /
#: ``*.extension_api`` modules so importing the contract never drags in torch,
#: ray, matplotlib or cartopy unless the consumer actually wants them.
PUBLIC_API_MODULES: tuple[str, ...] = (
    # The published contract itself (extensions read the allowlist from here)
    "bvr_marl_core.public_api",
    # Flat convenience API (Simulator, domain DTOs, interfaces, schema, …)
    "bvr_marl_core",
    # Lightweight stable subpackages
    "bvr_marl_core.domain",
    "bvr_marl_core.interfaces",
    "bvr_marl_core.schema",
    "bvr_marl_core.registry",
    "bvr_marl_core.tactical",
    "bvr_marl_core.simulator",
    "bvr_marl_core.physics.constraints",
    "bvr_marl_core.utils.config_loader",
    "bvr_marl_core.utils.simple_config",
    # Heavyweight subsystems behind explicit stable surfaces
    "bvr_marl_core.rl.api",
    "bvr_marl_core.visualization.api",
    "bvr_marl_core.analysis.api",
    "bvr_marl_core.tacview.api",
    "bvr_marl_core.gui.extension_api",
)


def is_in_namespace(module: str) -> bool:
    """Return True if *module* belongs to the ``bvr_marl_core`` namespace."""
    return module == TOP_LEVEL or module.startswith(TOP_LEVEL + ".")


def has_private_component(module: str) -> bool:
    """Return True if any dotted component of *module* starts with an underscore."""
    return any(part.startswith("_") for part in module.split("."))


def is_public_import(module: str) -> bool:
    """Return True if importing *module* is allowed by the public API contract.

    The bare top-level package matches only by exact name, so it never acts as a
    wildcard that would approve arbitrary internal sub-modules.
    """
    for approved in PUBLIC_API_MODULES:
        if module == approved:
            return True
        if approved != TOP_LEVEL and module.startswith(approved + "."):
            return True
    return False


def violation_for(module: str) -> str | None:
    """Return a human-readable violation message, or None if *module* is allowed.

    Modules outside the ``bvr_marl_core`` namespace are never violations, so a
    consumer can call this for *every* imported module without pre-filtering.
    """
    if not is_in_namespace(module):
        return None
    if has_private_component(module):
        return f"private-path import '{module}' (use the public API surface)"
    if not is_public_import(module):
        allowed = ", ".join(PUBLIC_API_MODULES)
        return f"non-approved import '{module}' (allowed: {allowed})"
    return None


__all__ = [
    "PUBLIC_API_MODULES",
    "TOP_LEVEL",
    "is_in_namespace",
    "has_private_component",
    "is_public_import",
    "violation_for",
]
