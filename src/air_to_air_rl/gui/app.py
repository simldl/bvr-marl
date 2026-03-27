"""
Air-to-Air RL Control Panel

A lightweight GUI for training, visualization, and analysis workflows.
"""

import streamlit as st

from air_to_air_rl.gui.components.analysis_interface import analysis_interface
from air_to_air_rl.gui.components.config_builder import config_builder
from air_to_air_rl.gui.components.tacview_generator import tacview_generator
from air_to_air_rl.gui.components.training_dashboard import training_dashboard
from air_to_air_rl.gui.components.visualization_config_builder import visualization_config_builder
from air_to_air_rl.gui.components.visualization_panel import visualization_panel
from air_to_air_rl.utils.config_loader import load_gui_config


def main():
    """Main GUI application."""
    gui_cfg = load_gui_config()
    display_cfg = gui_cfg.get("display", {})
    sidebar_state = "expanded" if display_cfg.get("sidebar_expanded", True) else "collapsed"

    st.set_page_config(
        page_title="BVR-MARL: Control Panel",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state=sidebar_state,
    )

    st.title("✈️ BVR-MARL: Control Panel")
    st.markdown("---")

    # Determine default startup tab from gui config
    _tab_map = {
        "training": "🚀 Training Dashboard",
        "visualization": "🎬 Visualization Panel",
        "tacview": "🎯 Tacview Generator",
        "tensorboard": "📈 Analysis Interface",
    }
    startup_tab = gui_cfg.get("startup_tab", "training")
    default_page = _tab_map.get(startup_tab, "🚀 Training Dashboard")

    pages = [
        "🚀 Training Dashboard",
        "🎬 Visualization Panel",
        "🎯 Tacview Generator",
        "📈 Analysis Interface",
        "⚙️ Training Config Builder",
        "📊 Visualization Config Builder",
    ]

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        pages,
        index=pages.index(default_page),
    )

    # Main content area
    if page == "🚀 Training Dashboard":
        training_dashboard()
    elif page == "⚙️ Training Config Builder":
        config_builder()
    elif page == "📊 Visualization Config Builder":
        visualization_config_builder()
    elif page == "🎬 Visualization Panel":
        visualization_panel()
    elif page == "📈 Analysis Interface":
        analysis_interface()
    elif page == "🎯 Tacview Generator":
        tacview_generator()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **BVR-MARL**
    GUI for BVR combat simulation
    """)


if __name__ == "__main__":
    main()
