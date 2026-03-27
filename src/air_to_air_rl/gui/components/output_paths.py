"""
Output Paths Display Component

Shows where different script types save their outputs.
"""

import os
from pathlib import Path

import streamlit as st


def get_output_paths():
    """Get the standard output paths for different script types."""

    from air_to_air_rl.core.paths import project_root as _project_root
    from air_to_air_rl.core.paths import rl_configs_root

    project_root = _project_root()

    paths = {
        "visualization": {
            "videos": project_root / "output",
            "reward_logs": project_root / "reward_logs",
            "screenshots": project_root / "screenshots",
        },
        "training": {
            "models": project_root / "models",
            "model_configs": project_root / "configs" / "training",
        },
        "tacview": {
            "tacview_files": project_root / "tacview_output",
            "scenarios": project_root / "tacview_scenarios",
        },
        "analysis": {
            "plots": project_root / "analysis_output",
            "reports": project_root / "reports",
        },
    }

    return paths


def display_output_paths(script_type="all"):
    """Display output paths for specified script type(s)."""

    paths = get_output_paths()

    if script_type == "all":
        st.subheader("📁 Output Paths")
        st.info("Here's where different scripts save their outputs:")

        for category, category_paths in paths.items():
            with st.expander(f"📂 {category.title()} Outputs"):
                for path_name, path_value in category_paths.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**{path_name.replace('_', ' ').title()}:**")
                    with col2:
                        st.code(str(path_value))
                        if path_value.exists():
                            st.success("✓ Directory exists")
                        else:
                            st.warning("⚠️ Directory will be created when needed")

    else:
        # Display specific category
        if script_type in paths:
            st.markdown(f"**📁 {script_type.title()} Output Paths:**")

            category_paths = paths[script_type]

            for path_name, path_value in category_paths.items():
                col1, col2, col3 = st.columns([2, 4, 1])

                with col1:
                    st.write(f"**{path_name.replace('_', ' ').title()}:**")

                with col2:
                    st.code(str(path_value), language=None)

                with col3:
                    if path_value.exists():
                        st.success("✓")
                    else:
                        st.warning("⚠️")


def display_compact_output_paths(script_type):
    """Display a compact version of output paths for a specific script type."""

    paths = get_output_paths()

    if script_type not in paths:
        return

    st.markdown("---")
    st.markdown(f"**📁 {script_type.title()} saves to:**")

    category_paths = paths[script_type]

    for path_name, path_value in category_paths.items():
        # Create a more compact display
        status_icon = "✓" if path_value.exists() else "📁"
        st.markdown(f"{status_icon} **{path_name.replace('_', ' ').title()}:** `{path_value}`")


def create_output_directory(path):
    """Create output directory if it doesn't exist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Failed to create directory {path}: {e}")
        return False


def get_latest_files(directory, pattern="*", limit=5):
    """Get the latest files in a directory matching a pattern."""
    try:
        directory = Path(directory)
        if not directory.exists():
            return []

        files = list(directory.glob(pattern))
        # Sort by modification time, newest first
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[:limit]

    except Exception:
        return []


def display_recent_outputs(script_type):
    """Display recent outputs for a specific script type."""

    paths = get_output_paths()

    if script_type not in paths:
        return

    st.markdown(f"**📄 Recent {script_type.title()} Outputs:**")

    category_paths = paths[script_type]

    found_files = False

    for path_name, directory in category_paths.items():
        if directory.exists():
            # Define patterns for different types
            patterns = {
                "videos": ["*.mp4", "*.avi", "*.gif"],
                "reward_logs": ["*.csv", "*.json"],
                "checkpoints": ["checkpoint*", "*.pkl"],
                "tacview_files": ["*.acmi", "*.txt.acmi"],
                "plots": ["*.png", "*.jpg", "*.pdf"],
            }

            pattern_list = patterns.get(path_name, ["*"])

            recent_files = []
            for pattern in pattern_list:
                recent_files.extend(get_latest_files(directory, pattern, limit=3))

            # Sort all files by modification time
            recent_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            recent_files = recent_files[:3]  # Keep only top 3

            if recent_files:
                found_files = True
                st.markdown(f"*{path_name.replace('_', ' ').title()}:*")
                for file in recent_files:
                    mod_time = file.stat().st_mtime
                    from datetime import datetime

                    time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
                    st.text(f"  • {file.name} ({time_str})")

    if not found_files:
        st.text(f"No recent {script_type} outputs found")
