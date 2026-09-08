"""Shared presentation helpers for the Streamlit GUI shell."""

from __future__ import annotations

from html import escape

import streamlit as st

_THEME_TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "--gui-bg-top": "#f2f6f8",
        "--gui-bg-bottom": "#edf3f7",
        "--gui-app-background": (
            "radial-gradient(circle at top left, rgba(31, 90, 113, 0.10), transparent 28%), "
            "linear-gradient(180deg, #f2f6f8 0%, #ffffff 42%, #edf3f7 100%)"
        ),
        "--gui-surface": "rgba(255, 255, 255, 0.88)",
        "--gui-surface-strong": "#ffffff",
        "--gui-surface-muted": "rgba(246, 250, 252, 0.92)",
        "--gui-border": "rgba(15, 23, 42, 0.12)",
        "--gui-text": "#142330",
        "--gui-muted": "#5d6d79",
        "--gui-accent": "#1f5a71",
        "--gui-accent-strong": "#0f4154",
        "--gui-accent-soft": "rgba(31, 90, 113, 0.12)",
        "--gui-shadow": "0 18px 44px rgba(15, 23, 42, 0.08)",
        "--gui-sidebar-background": "linear-gradient(180deg, #f7fafc 0%, #edf4f7 100%)",
        "--gui-sidebar-border": "rgba(15, 23, 42, 0.08)",
        "--gui-sidebar-text": "#142330",
        "--gui-sidebar-muted": "#667784",
        "--gui-banner-background": (
            "linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(245, 249, 251, 0.98) 58%, "
            "rgba(231, 240, 245, 0.94) 100%)"
        ),
        "--gui-input-background": "rgba(255, 255, 255, 0.96)",
        "--gui-input-border": "rgba(20, 35, 48, 0.14)",
        "--gui-button-background": "#f7fbfd",
        "--gui-button-text": "#163142",
        "--gui-button-border": "rgba(20, 35, 48, 0.16)",
        "--gui-button-hover-background": "#eef5f8",
        "--gui-button-hover-border": "rgba(31, 90, 113, 0.24)",
        "--gui-primary-background": "linear-gradient(135deg, #1f5a71 0%, #0f4154 100%)",
        "--gui-primary-text": "#ffffff",
        "--gui-primary-hover-background": "linear-gradient(135deg, #174a5d 0%, #0b3443 100%)",
        "--gui-chip-background": "rgba(255, 255, 255, 0.92)",
        "--gui-chip-text": "#27404d",
        "--gui-chip-active-background": "rgba(31, 90, 113, 0.14)",
        "--gui-chip-active-text": "#0f4154",
        "--gui-disabled-background": "#edf2f5",
        "--gui-disabled-text": "#8a99a5",
    },
    "dark": {
        "--gui-bg-top": "#0c1721",
        "--gui-bg-bottom": "#122330",
        "--gui-app-background": (
            "radial-gradient(circle at top left, rgba(69, 131, 156, 0.18), transparent 28%), "
            "linear-gradient(180deg, #0c1721 0%, #0f1d29 45%, #122330 100%)"
        ),
        "--gui-surface": "rgba(17, 30, 40, 0.86)",
        "--gui-surface-strong": "#152735",
        "--gui-surface-muted": "rgba(22, 38, 50, 0.94)",
        "--gui-border": "rgba(206, 220, 230, 0.14)",
        "--gui-text": "#edf4f8",
        "--gui-muted": "#a9bcc7",
        "--gui-accent": "#66adc6",
        "--gui-accent-strong": "#8cc5db",
        "--gui-accent-soft": "rgba(102, 173, 198, 0.18)",
        "--gui-shadow": "0 22px 54px rgba(0, 0, 0, 0.28)",
        "--gui-sidebar-background": "linear-gradient(180deg, #08131c 0%, #0d1e2a 100%)",
        "--gui-sidebar-border": "rgba(255, 255, 255, 0.08)",
        "--gui-sidebar-text": "#f3f8fb",
        "--gui-sidebar-muted": "rgba(243, 248, 251, 0.70)",
        "--gui-banner-background": (
            "linear-gradient(135deg, rgba(21, 39, 53, 0.96) 0%, rgba(18, 34, 46, 0.98) 58%, "
            "rgba(14, 28, 39, 0.96) 100%)"
        ),
        "--gui-input-background": "rgba(20, 35, 48, 0.96)",
        "--gui-input-border": "rgba(216, 228, 236, 0.16)",
        "--gui-button-background": "#182b39",
        "--gui-button-text": "#edf4f8",
        "--gui-button-border": "rgba(216, 228, 236, 0.18)",
        "--gui-button-hover-background": "#1d3344",
        "--gui-button-hover-border": "rgba(140, 197, 219, 0.32)",
        "--gui-primary-background": "linear-gradient(135deg, #4a8ea7 0%, #2c6981 100%)",
        "--gui-primary-text": "#f9fdff",
        "--gui-primary-hover-background": "linear-gradient(135deg, #5ca3be 0%, #32738d 100%)",
        "--gui-chip-background": "rgba(21, 39, 53, 0.96)",
        "--gui-chip-text": "#e4edf2",
        "--gui-chip-active-background": "rgba(102, 173, 198, 0.20)",
        "--gui-chip-active-text": "#f5fbff",
        "--gui-disabled-background": "#253746",
        "--gui-disabled-text": "#8ca0ab",
    },
}


def normalize_theme_mode(theme_mode: str | None) -> str:
    """Return a supported theme mode."""
    if theme_mode == "dark":
        return "dark"
    return "light"


def _css_variables(theme_mode: str) -> str:
    """Convert the selected token set into CSS variables."""
    tokens = _THEME_TOKENS[normalize_theme_mode(theme_mode)]
    return "\n".join(f"            {name}: {value};" for name, value in tokens.items())


def apply_professional_theme(theme_mode: str = "light") -> None:
    """Inject a restrained visual theme for the control center."""
    css_variables = _css_variables(theme_mode)

    st.markdown(
        f"""
        <style>
        :root {{
{css_variables}
        }}

        .stApp {{
            background: var(--gui-app-background);
            color: var(--gui-text);
            font-family: Aptos, "Segoe UI", "Helvetica Neue", sans-serif;
        }}

        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: var(--gui-sidebar-background);
            border-right: 1px solid var(--gui-sidebar-border);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--gui-sidebar-text);
        }}

        .gui-sidebar-brand {{
            padding: 0.15rem 0 1rem;
        }}

        .gui-sidebar-brand p {{
            margin: 0;
            color: var(--gui-sidebar-muted);
        }}

        .gui-sidebar-brand .gui-sidebar-overline {{
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }}

        .gui-sidebar-brand h2 {{
            margin: 0.35rem 0 0.45rem;
            font-size: 1.45rem;
            line-height: 1.1;
            color: var(--gui-sidebar-text);
        }}

        .gui-shell-banner {{
            padding: 1.45rem 1.55rem;
            margin-bottom: 1rem;
            border: 1px solid var(--gui-border);
            border-radius: 24px;
            background: var(--gui-banner-background);
            box-shadow: var(--gui-shadow);
        }}

        .gui-shell-banner .gui-shell-overline {{
            margin: 0 0 0.45rem;
            color: var(--gui-accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }}

        .gui-shell-banner h1 {{
            margin: 0;
            color: var(--gui-text);
            font-size: 2.05rem;
            line-height: 1.08;
        }}

        .gui-shell-banner .gui-shell-description {{
            margin: 0.55rem 0 0;
            max-width: 54rem;
            color: var(--gui-muted);
            font-size: 1rem;
            line-height: 1.5;
        }}

        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        div[data-testid="stForm"],
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid var(--gui-border);
            border-radius: 18px;
            background: var(--gui-surface);
            box-shadow: var(--gui-shadow);
        }}

        div[data-testid="stMetric"] {{
            padding: 0.85rem 1rem;
        }}

        div[data-testid="stExpander"] {{
            background: var(--gui-surface-muted);
        }}

        .stButton > button,
        [data-testid="stDownloadButton"] > button {{
            min-height: 2.85rem;
            padding: 0.62rem 1.08rem;
            border-radius: 14px;
            border: 1px solid var(--gui-button-border) !important;
            background: var(--gui-button-background);
            color: var(--gui-button-text) !important;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
        }}

        .stButton > button *,
        [data-testid="stDownloadButton"] > button * {{
            color: inherit !important;
            fill: currentColor !important;
        }}

        .stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover {{
            background: var(--gui-button-hover-background);
            border-color: var(--gui-button-hover-border) !important;
            transform: translateY(-1px);
        }}

        .stButton > button[kind="primary"],
        [data-testid="stDownloadButton"] > button[kind="primary"] {{
            border: 1px solid transparent !important;
            background: var(--gui-primary-background);
            color: var(--gui-primary-text) !important;
        }}

        .stButton > button[kind="primary"]:hover,
        [data-testid="stDownloadButton"] > button[kind="primary"]:hover {{
            background: var(--gui-primary-hover-background);
        }}

        .stButton > button:disabled,
        [data-testid="stDownloadButton"] > button:disabled {{
            background: var(--gui-disabled-background) !important;
            color: var(--gui-disabled-text) !important;
            border-color: transparent !important;
            opacity: 1;
            box-shadow: none;
        }}

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stDateInput input,
        .stTimeInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {{
            border-radius: 14px;
            background: var(--gui-input-background);
            border-color: var(--gui-input-border);
            color: var(--gui-text) !important;
        }}

        .stSelectbox label,
        .stMultiSelect label,
        .stTextInput label,
        .stNumberInput label,
        .stTextArea label,
        .stDateInput label,
        .stTimeInput label {{
            color: var(--gui-text) !important;
            font-weight: 600;
        }}

        .stTextInput input::placeholder,
        .stNumberInput input::placeholder,
        .stTextArea textarea::placeholder {{
            color: var(--gui-muted);
            opacity: 0.88;
        }}

        .stSelectbox [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] * {{
            color: var(--gui-text) !important;
        }}

        div[role="listbox"] {{
            background: var(--gui-surface-strong);
            border: 1px solid var(--gui-border);
            border-radius: 14px;
            box-shadow: var(--gui-shadow);
        }}

        div[role="option"] {{
            color: var(--gui-text) !important;
            background: var(--gui-surface-strong);
        }}

        div[role="option"][aria-selected="true"] {{
            background: var(--gui-chip-active-background);
        }}

        [data-baseweb="tab-list"] {{
            gap: 0.45rem;
        }}

        [data-baseweb="tab"] {{
            border-radius: 14px;
            padding: 0.52rem 1rem;
            border: 1px solid var(--gui-button-border);
            background: var(--gui-chip-background);
            color: var(--gui-chip-text);
            font-weight: 600;
        }}

        [data-baseweb="tab"][aria-selected="true"] {{
            color: var(--gui-chip-active-text);
            background: var(--gui-chip-active-background);
            border-color: var(--gui-button-hover-border);
        }}

        div[role="radiogroup"] {{
            gap: 0.55rem;
        }}

        div[role="radiogroup"] > label {{
            background: var(--gui-chip-background);
            border: 1px solid var(--gui-button-border);
            border-radius: 14px;
            padding: 0.60rem 0.95rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }}

        div[role="radiogroup"] > label * {{
            color: var(--gui-chip-text) !important;
        }}

        div[role="radiogroup"] > label:has(input:checked) {{
            background: var(--gui-chip-active-background);
            border-color: var(--gui-button-hover-border);
        }}

        div[role="radiogroup"] > label:has(input:checked) * {{
            color: var(--gui-chip-active-text) !important;
            font-weight: 600;
        }}

        [data-testid="stCheckbox"] label,
        [data-testid="stToggle"] label {{
            color: var(--gui-text) !important;
        }}

        .stMarkdown,
        .stCaption,
        .stCodeBlock,
        .stAlert,
        .stTabs,
        .stRadio {{
            color: var(--gui-text);
        }}

        hr {{
            border-top: 1px solid var(--gui-border);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Render the persistent sidebar brand block."""
    st.sidebar.markdown(
        """
        <div class="gui-sidebar-brand">
            <p class="gui-sidebar-overline">BVR-MARL</p>
            <h2>Command Center</h2>
            <p>Training, analysis, and behavior tooling in one streamlined workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_banner(title: str, description: str) -> None:
    """Render a shared banner for the selected workspace."""
    st.markdown(
        f"""
        <section class="gui-shell-banner">
            <p class="gui-shell-overline">Control Center</p>
            <h1>{escape(title)}</h1>
            <p class="gui-shell-description">{escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
