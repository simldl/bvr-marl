"""Path resolution utilities for visualization."""

from pathlib import Path


def resolve_relative_path(path_str) -> "str | None":
    """
    Resolve a path string to an absolute path.

    Absolute paths are returned unchanged.
    Relative paths are resolved relative to the current working directory —
    this is intentional: the caller supplies a user-provided path string
    (e.g. from a CLI flag) so resolving against CWD is the correct behaviour.
    None is returned unchanged.
    """
    if path_str is None:
        return None
    p = Path(str(path_str).strip())
    if p.is_absolute():
        return str(p)
    # Explicit user input — resolve relative to CWD as the user intends.
    return str(Path(path_str).resolve())
