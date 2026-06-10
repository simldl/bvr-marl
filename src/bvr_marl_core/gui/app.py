"""Streamlit entry point for the BVR-MARL command center."""

from __future__ import annotations

from collections import OrderedDict

import streamlit as st

from bvr_marl_core.gui.components.analysis_interface import analysis_interface
from bvr_marl_core.gui.components.config_builder import config_builder
from bvr_marl_core.gui.components.tacview_generator import tacview_generator
from bvr_marl_core.gui.components.training_dashboard import training_dashboard
from bvr_marl_core.gui.components.visualization_config_builder import (
    visualization_config_builder,
)
from bvr_marl_core.gui.components.visualization_panel import visualization_panel
from bvr_marl_core.gui.extension_api import PageSpec
from bvr_marl_core.gui.page_registry import load_extension_pages
from bvr_marl_core.gui.theme import (
    apply_professional_theme,
    render_sidebar_brand,
    render_workspace_banner,
)
from bvr_marl_core.utils.config_loader import load_gui_config

_CORE_PAGES: list[PageSpec] = [
    PageSpec(
        name="Training Dashboard",
        render=training_dashboard,
        section="Training & Configuration",
        order=0,
    ),
    PageSpec(
        name="Training Config Builder",
        render=config_builder,
        section="Training & Configuration",
        order=10,
    ),
    PageSpec(
        name="Visualization Config Builder",
        render=visualization_config_builder,
        section="Training & Configuration",
        order=20,
    ),
    PageSpec(name="Visualization Panel", render=visualization_panel, section="Analysis", order=0),
    PageSpec(name="Analysis Tools", render=analysis_interface, section="Analysis", order=10),
    PageSpec(name="Tacview Generator", render=tacview_generator, section="Analysis", order=20),
]

_STARTUP_PAGE_MAP = {
    "training": "Training Dashboard",
    "visualization": "Visualization Panel",
    "tacview": "Tacview Generator",
    "tensorboard": "Analysis Tools",
}

_SECTION_DESCRIPTIONS = {
    "Training & Configuration": (
        "Launch jobs, manage runtime operations, and maintain the training and "
        "visualization configurations that drive your workflows."
    ),
    "Analysis": (
        "Inspect runs, compare results, control visualizations, and review league "
        "and benchmark signals from one analysis workspace."
    ),
    "Behavior Trees": (
        "Browse tree structure, replay monitor logs, and launch live behavior-tree "
        "sessions with a cleaner operator-focused workflow."
    ),
}

_SECTION_ORDER = [
    "Training & Configuration",
    "Analysis",
    "Behavior Trees",
]


def _section_sort_key(section: str) -> tuple[int, str]:
    if section in _SECTION_ORDER:
        return (_SECTION_ORDER.index(section), section)
    return (len(_SECTION_ORDER), section)


def _page_sort_key(page: PageSpec) -> tuple[int, int, str, str]:
    section_rank, section_name = _section_sort_key(page.section)
    return (section_rank, page.order, section_name, page.name)


def _group_pages(pages: list[PageSpec]) -> OrderedDict[str, list[PageSpec]]:
    grouped: OrderedDict[str, list[PageSpec]] = OrderedDict()
    for page in sorted(pages, key=_page_sort_key):
        grouped.setdefault(page.section, []).append(page)
    return grouped


def _default_page_for_startup(startup_tab: str, pages: list[PageSpec]) -> PageSpec:
    default_name = _STARTUP_PAGE_MAP.get(startup_tab, "Training Dashboard")
    for page in pages:
        if page.name == default_name:
            return page
    return pages[0]


def _normalize_theme_mode(theme_mode: str | None) -> str:
    """Return a supported theme mode."""
    if theme_mode == "dark":
        return "dark"
    return "light"


def main() -> None:
    """Main GUI application."""
    gui_cfg = load_gui_config()
    display_cfg = gui_cfg.get("display", {})
    sidebar_state = "expanded" if display_cfg.get("sidebar_expanded", True) else "collapsed"

    st.set_page_config(
        page_title="BVR-MARL Command Center",
        layout="wide",
        initial_sidebar_state=sidebar_state,
    )

    default_theme_mode = _normalize_theme_mode(display_cfg.get("theme", "light"))
    if "gui_theme_mode" not in st.session_state:
        st.session_state.gui_theme_mode = default_theme_mode

    apply_professional_theme(st.session_state.gui_theme_mode)

    all_pages = _CORE_PAGES + load_extension_pages()
    if not all_pages:
        st.error("No GUI pages are available.")
        return

    sections = _group_pages(all_pages)
    section_names = list(sections)
    startup_page = _default_page_for_startup(gui_cfg.get("startup_tab", "training"), all_pages)

    if "gui_selected_section" not in st.session_state:
        st.session_state.gui_selected_section = startup_page.section
    if st.session_state.gui_selected_section not in section_names:
        st.session_state.gui_selected_section = section_names[0]

    render_sidebar_brand()
    st.sidebar.markdown("### Appearance")
    st.sidebar.radio(
        "Appearance",
        ["light", "dark"],
        key="gui_theme_mode",
        format_func=lambda value: value.title(),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("### Workspace")
    selected_section = st.sidebar.radio(
        "Workspace",
        section_names,
        index=section_names.index(st.session_state.gui_selected_section),
        key="gui_selected_section",
        label_visibility="collapsed",
    )
    st.sidebar.caption(_SECTION_DESCRIPTIONS.get(selected_section, "Additional tools"))

    render_workspace_banner(
        selected_section,
        _SECTION_DESCRIPTIONS.get(selected_section, "Additional tools"),
    )

    selected_section_pages = sections[selected_section]
    page_names = [page.name for page in selected_section_pages]
    page_state_key = f"gui_selected_page::{selected_section}"
    default_page_name = (
        startup_page.name
        if startup_page.section == selected_section and startup_page.name in page_names
        else page_names[0]
    )

    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = default_page_name
    if st.session_state[page_state_key] not in page_names:
        st.session_state[page_state_key] = page_names[0]

    selected_page_name = st.radio(
        "Section Pages",
        page_names,
        index=page_names.index(st.session_state[page_state_key]),
        key=page_state_key,
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_page = next(page for page in selected_section_pages if page.name == selected_page_name)
    selected_page.render()

    st.sidebar.markdown("---")
    st.sidebar.caption("Professional workflow tooling for training, analysis, and behavior review.")


if __name__ == "__main__":
    main()
