"""
Environment registry — discovers RL environments contributed by extension packages.

Core ships only the baseline environments (``BVRMultiAgentEnv``,
``SimplifiedMultiAgentEnv``).  Extension packages contribute specialised
environments — such as the line-objective training env — by registering a
factory under the ``bvr_marl.env_extensions`` entry-point group in their
``pyproject.toml``::

    [project.entry-points."bvr_marl.env_extensions"]
    my_extension = "my_package.rl.environment.registry:register_envs"

The registered callable takes no arguments and returns a ``dict[str, type]``
mapping a named environment kind (e.g. ``"line_objective"``) to its class.
Core resolves kinds by name without ever importing the extension package
directly, preserving the one-directional dependency boundary.
"""

from __future__ import annotations

from importlib.metadata import entry_points

_ENTRY_POINT_GROUP = "bvr_marl.env_extensions"


def load_extension_envs() -> dict[str, type]:
    """
    Discover environment classes from all installed extension packages.

    Returns
    -------
    dict[str, type]
        Mapping of environment kind to class, merged across all installed
        extensions.  Returns an empty dict if no extensions are installed or
        if all extension loads fail.
    """
    envs: dict[str, type] = {}

    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except Exception:
        return envs

    for ep in eps:
        try:
            factory = ep.load()
            contributed = factory()
            if isinstance(contributed, dict):
                envs.update(contributed)
        except Exception as exc:
            # Gracefully degrade — a broken extension must not crash callers.
            print(f"[env] Failed to load environment extension '{ep.name}': {exc}")

    return envs


def resolve_env_class(kind: str) -> type | None:
    """Return the env class registered under ``kind``, or ``None`` if absent."""
    return load_extension_envs().get(kind)
