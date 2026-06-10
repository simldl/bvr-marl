"""
Output Paths Display Component

Shows where different script types save their outputs.
"""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st


def get_output_paths():
    """Get the standard output paths for different script types."""

    from bvr_marl_core.utils.paths import (
        core_project_root,
        exported_plots_root,
        tacview_output_root,
    )

    project_root = core_project_root()

    paths = {
        "visualization": {
            "videos": project_root / "output" / "videos",
            "reward_logs": project_root / "output" / "reward_logs",
            "screenshots": project_root / "screenshots",
        },
        "training": {
            "models": project_root / "models",
            "model_configs": project_root / "configs" / "training",
        },
        "tacview": {
            "tacview_files": tacview_output_root(),
        },
        "analysis": {
            "plots": exported_plots_root(),
            "reports": project_root / "reports",
        },
    }

    return paths


def _folder_uri(path):
    """Return a file URI for a local folder path."""
    return Path(path).resolve().as_uri()


def _open_output_directory(path):
    """Open an output directory in the platform file browser."""
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        resolved = str(directory.resolve())
        if os.name == "nt":
            os.startfile(resolved)
        elif sys.platform == "darwin":
            subprocess.run(["open", resolved], check=True)
        else:
            subprocess.run(["xdg-open", resolved], check=True)
        st.success(f"Opened `{resolved}`.")
    except Exception as exc:
        st.error(f"Failed to open `{directory}`: {exc}")


def display_output_paths(script_type="all"):
    """Display output paths for specified script type(s)."""

    paths = get_output_paths()

    if script_type == "all":
        st.subheader("Output Paths")
        st.info("Here's where different scripts save their outputs:")

        for category, category_paths in paths.items():
            with st.expander(f"{category.title()} Outputs"):
                for path_name, path_value in category_paths.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**{path_name.replace('_', ' ').title()}:**")
                    with col2:
                        st.code(str(path_value))
                        if path_value.exists():
                            st.success("Directory exists")
                        else:
                            st.warning("Directory will be created when needed")

    else:
        # Display specific category
        if script_type in paths:
            st.markdown(f"**{script_type.title()} Output Paths:**")

            category_paths = paths[script_type]

            for path_name, path_value in category_paths.items():
                col1, col2, col3 = st.columns([2, 4, 1])

                with col1:
                    st.write(f"**{path_name.replace('_', ' ').title()}:**")

                with col2:
                    st.code(str(path_value), language=None)

                with col3:
                    if path_value.exists():
                        st.success("Available")
                    else:
                        st.warning("Pending")


def display_compact_output_paths(script_type):
    """Display a compact version of output paths for a specific script type."""

    paths = get_output_paths()

    if script_type not in paths:
        return

    st.markdown("---")
    st.markdown(f"**{script_type.title()} saves to:**")

    category_paths = paths[script_type]

    for path_name, path_value in category_paths.items():
        # Create a more compact display
        status_label = "Ready" if path_value.exists() else "Create on first use"
        resolved_path = Path(path_value).resolve()
        st.markdown(
            f"**{path_name.replace('_', ' ').title()}:** "
            f"[{resolved_path}]({_folder_uri(resolved_path)})  \n{status_label}"
        )
        if script_type == "tacview":
            if st.button(
                "Open Tacview Folder",
                key=f"open_{script_type}_{path_name}",
                use_container_width=True,
            ):
                _open_output_directory(path_value)


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

    st.markdown(f"**Recent {script_type.title()} Outputs:**")

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
