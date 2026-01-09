"""Path resolution utilities for visualization."""
from pathlib import Path


def resolve_relative_path(path_str):
    """
    Convert relative path starting with 'air_to_air_rl/' to absolute path.
    If path is already absolute or None, return as-is.

    Args:
        path_str: Path string (can be relative or absolute)

    Returns:
        Absolute path or None
    """
    if path_str is None:
        return None

    path_str = str(path_str).strip()

    # Check if it's a relative path starting with 'air_to_air_rl/'
    if path_str.startswith('air_to_air_rl/') or path_str.startswith('air_to_air_rl\\'):
        # Get the project root (three levels up from this file: utils -> visualization -> air_to_air_rl)
        project_root = Path(__file__).resolve().parents[2]
        # Remove 'air_to_air_rl/' prefix and resolve from project root
        relative_part = path_str.replace('air_to_air_rl/', '').replace('air_to_air_rl\\', '')
        return str(project_root / relative_part)

    # Check if it's already an absolute path
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return str(path_obj)

    # Otherwise, treat it as relative to the project root
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / path_str)
