"""Theme tokens and shared Streamlit styling for the flood dashboard."""

from copy import deepcopy
import html

import streamlit as st


FONT_STACK = '"Inter", "Segoe UI", Arial, sans-serif'

SEVERITY_COLORS = {
    "normal": "#15803D",
    "watch": "#A16207",
    "warning": "#C2410C",
    "emergency": "#B91C1C",
}

LIGHT = {
    "name": "light",
    "background": "#F4F7FB",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "border": "#E2E8F0",
    "text": "#0F1B2D",
    "muted": "#5B6B84",
    "accent": "#0369A1",
    "button_text": "#FFFFFF",
    "shadow": "rgba(15, 27, 45, 0.06)",
    "focus": "rgba(3, 105, 161, 0.24)",
}

DARK = {
    "name": "dark",
    "background": "#0C1524",
    "surface": "#142235",
    "surface_alt": "#1B2B40",
    "border": "#34445A",
    "text": "#F3F6FB",
    "muted": "#BDCADB",
    "accent": "#7DD3FC",
    "button_text": "#0C1524",
    "shadow": "rgba(0, 0, 0, 0.24)",
    "focus": "rgba(125, 211, 252, 0.30)",
}


def hex_to_rgb(value: str) -> list[int]:
    """Convert a six-digit hex color into RGB values for PyDeck."""
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex color, got {value!r}")
    return [int(value[index:index + 2], 16) for index in (0, 2, 4)]


def rgba(value: str, alpha: float) -> str:
    """Return a CSS rgba() value from a six-digit hex color."""
    red, green, blue = hex_to_rgb(value)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def theme_tokens(dark_mode: bool = False) -> dict:
    """Return a fresh palette with severity colors and theme-aware surfaces."""
    tokens = deepcopy(DARK if dark_mode else LIGHT)
    tokens["severity"] = {
        "normal": {"label": "Normal", "icon": "✓", "color": SEVERITY_COLORS["normal"]},
        "watch": {"label": "Watch", "icon": "i", "color": SEVERITY_COLORS["watch"]},
        "warning": {"label": "Warning", "icon": "!", "color": SEVERITY_COLORS["warning"]},
        "emergency": {"label": "Emergency", "icon": "!", "color": SEVERITY_COLORS["emergency"]},
    }
    for level in tokens["severity"].values():
        level["border"] = level["color"]
        level["bg"] = rgba(level["color"], 0.14 if dark_mode else 0.08)
        level["rgb"] = hex_to_rgb(level["color"])
    return tokens


def theme_css(tokens: dict) -> str:
    """Build the responsive Streamlit styles from the active token set."""
    severity = tokens["severity"]
    return f"""
    <style>
    :root {{
        --page: {tokens["background"]};
        --surface: {tokens["surface"]};
        --surface-alt: {tokens["surface_alt"]};
        --border: {tokens["border"]};
        --text: {tokens["text"]};
        --muted: {tokens["muted"]};
        --accent: {tokens["accent"]};
        --shadow: {tokens["shadow"]};
        --focus: {tokens["focus"]};
        --normal: {severity["normal"]["color"]};
        --watch: {severity["watch"]["color"]};
        --warning: {severity["warning"]["color"]};
        --emergency: {severity["emergency"]["color"]};
        color-scheme: {"dark" if tokens["name"] == "dark" else "light"};
    }}

    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background: var(--page) !important;
        color: var(--text) !important;
        font-family: {FONT_STACK} !important;
        font-size: 15px;
    }}

    .main .block-container {{
        max-width: 1560px;
        padding: 1.5rem clamp(1rem, 3vw, 2.5rem) 2.5rem;
    }}

    #MainMenu, footer, header {{ visibility: hidden !important; }}
    [data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        gap: 1rem;
    }}

    .command-card {{
        box-sizing: border-box;
        min-height: 150px;
        height: 100%;
        padding: 20px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--surface);
        box-shadow: 0 2px 8px var(--shadow);
    }}
    .kpi-label {{
        margin-bottom: 12px;
        color: var(--muted);
        font-size: 13px;
        font-weight: 500;
    }}
    .kpi-value {{
        color: var(--text);
        font-size: 30px;
        font-weight: 700;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }}
    .kpi-meta {{
        margin-top: 12px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }}
    .severity-banner {{
        margin: 18px 0 20px;
        padding: 18px 22px;
        border: 1px solid var(--border);
        border-left: 6px solid var(--status-color);
        border-radius: 12px;
        background: var(--status-tint);
        box-shadow: 0 2px 8px var(--shadow);
    }}
    .severity-tag {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        min-height: 32px;
        padding: 4px 11px;
        border-radius: 6px;
        background: var(--status-color);
        color: #FFFFFF;
        font-size: 13px;
        font-weight: 700;
    }}
    .severity-icon {{
        display: inline-grid;
        width: 20px;
        height: 20px;
        place-items: center;
        border: 1px solid currentColor;
        border-radius: 50%;
        font-size: 13px;
        line-height: 1;
    }}
    .severity-title {{
        margin: 10px 0 6px;
        color: var(--text);
        font-size: 15px;
        font-weight: 700;
        line-height: 1.4;
    }}
    .severity-body {{
        color: var(--text);
        font-size: 15px;
        line-height: 1.5;
    }}
    .status-badge {{
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 4px 9px;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--surface);
        color: var(--text);
        font-size: 13px;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }}
    .stButton > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {{
        min-height: 44px;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        background: var(--accent) !important;
        color: {tokens["button_text"]} !important;
        font-family: {FONT_STACK} !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: filter 150ms ease, transform 150ms ease;
    }}
    .stButton > button:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {{
        filter: brightness(0.92);
        color: {tokens["button_text"]} !important;
    }}
    .stButton > button:active,
    [data-testid="stBaseButton-secondary"]:active,
    [data-testid="stBaseButton-primary"]:active {{
        transform: translateY(1px);
    }}
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    label, .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--muted) !important;
        font-family: {FONT_STACK} !important;
        font-size: 13px !important;
    }}
    [data-baseweb="radio"] label, [data-baseweb="select"] *,
    [data-baseweb="input"] input, [data-testid="stNumberInput"] input {{
        color: var(--text) !important;
        font-family: {FONT_STACK} !important;
    }}
    [data-baseweb="radio"] [role="radio"] > div:first-child {{
        border-color: var(--accent) !important;
    }}
    [data-baseweb="radio"] [role="radio"][aria-checked="true"] > div:first-child {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }}
    [data-baseweb="slider"] [role="slider"] {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--focus) !important;
    }}
    [data-baseweb="slider"] [data-testid="stTickBar"] {{
        background: var(--accent) !important;
    }}
    [data-baseweb="checkbox"] input:checked + div,
    [data-baseweb="switch"] input:checked + div {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }}
    input:focus, button:focus-visible, [role="slider"]:focus-visible,
    [role="radio"]:focus-visible {{
        outline: 3px solid var(--focus) !important;
        outline-offset: 2px;
    }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] > div {{
        min-height: 44px;
        border-color: var(--border) !important;
        background: var(--surface) !important;
    }}
    .stTabs [role="tablist"] {{ gap: 8px; border-bottom-color: var(--border); }}
    .stTabs [role="tab"] {{
        min-height: 44px;
        background: var(--surface);
        color: var(--muted);
        border-radius: 8px 8px 0 0;
        font-family: {FONT_STACK};
        font-size: 15px;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background: var(--surface);
        color: var(--text);
        border-bottom-color: var(--accent);
    }}
    [data-testid="stMetricValue"] {{
        color: var(--text);
        font-size: 30px;
        font-variant-numeric: tabular-nums;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{
        color: var(--muted);
        font-size: 13px;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }}
    a {{ color: var(--accent) !important; }}
    hr {{ border-color: var(--border); }}

    @media (max-width: 1050px) {{
        .st-key-kpi-grid [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap;
            gap: 12px !important;
        }}
        .st-key-kpi-grid [data-testid="stHorizontalBlock"] > div {{
            flex: 1 1 calc(50% - 12px) !important;
            min-width: min(100%, 260px);
        }}
    }}
    @media (max-width: 640px) {{
        .main .block-container {{ padding: 1rem 0.85rem 2rem; }}
        .st-key-kpi-grid [data-testid="stHorizontalBlock"] > div {{
            flex-basis: 100% !important;
            min-width: 0;
        }}
        .command-card {{ min-height: 136px; padding: 16px; }}
        .kpi-value {{ font-size: 28px; }}
        .severity-banner {{ padding: 16px; }}
        [data-testid="stHorizontalBlock"] {{ gap: 12px !important; }}
        .stTabs [role="tablist"] {{ overflow-x: auto; }}
    }}
    </style>
    """


SITE_CSS = """
<style>
    #MainMenu, footer, header {
        visibility: hidden !important;
    }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .main .block-container {
        max-width: 1480px;
        padding: 1.1rem clamp(1rem, 3vw, 2.8rem) 3rem;
    }
    .site-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.9rem 1.1rem;
        border: 1px solid #D9D0BF;
        border-radius: 12px;
        background: #F3EEE3;
        color: #172C36;
    }
    .site-brand {
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
        font-family: Georgia, "Times New Roman", serif;
    }
    .site-brand-mark {
        color: #237D83;
        font-size: 1.25rem;
    }
    .site-brand-name {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.035em;
    }
    .site-brand-caption, .site-nav-note {
        color: #69777A;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .site-nav-note {
        text-align: right;
    }
    .st-key-site-nav-links {
        margin: 0.55rem 0 1.4rem;
    }
    .st-key-site-nav-links [data-testid="stBaseButton-secondary"] {
        border-color: transparent !important;
        background: transparent !important;
        color: #D7DFDF !important;
    }
    .st-key-site-nav-links [data-testid="stBaseButton-secondary"]:hover {
        border-color: #667983 !important;
        background: #1A2B39 !important;
        color: #FFFFFF !important;
    }
    .st-key-site-nav-links [data-testid="stBaseButton-primary"] {
        border-color: #D8C8A6 !important;
        background: #F3EEE3 !important;
        color: #172C36 !important;
    }
    .st-key-site-nav-links button {
        min-height: 42px;
        border-radius: 7px !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.02em;
    }
    .site-nav-rule {
        height: 1px;
        margin: 0 0 1.35rem;
        background: linear-gradient(90deg, #C6A15B, #314C59 35%, transparent);
    }
    .site-page-intro {
        margin: 1.6rem 0 1.4rem;
    }
    .site-eyebrow {
        margin: 0 0 0.7rem;
        color: #75C7C4;
        font-size: 0.71rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .site-page-intro h1 {
        margin: 0;
        color: #F3EEE3;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.1rem, 5vw, 3.8rem);
        font-weight: 500;
        letter-spacing: -0.045em;
        line-height: 1.02;
    }
    .site-page-intro p {
        max-width: 720px;
        margin: 0.85rem 0 0;
        color: #B7C3C7;
        font-size: 1rem;
        line-height: 1.65;
    }
    .river-hero {
        display: grid;
        grid-template-columns: 0.92fr 1.08fr;
        min-height: 370px;
        overflow: hidden;
        border: 1px solid #D8D0C2;
        border-radius: 14px;
        background: #F3EEE3;
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.2);
    }
    .river-hero-copy {
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: clamp(1.7rem, 4vw, 3.7rem);
        color: #172C36;
    }
    .river-hero-copy .site-eyebrow {
        color: #527377;
    }
    .river-hero-copy h1 {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.8rem, 5.8vw, 5.4rem);
        font-weight: 500;
        letter-spacing: -0.06em;
        line-height: 0.96;
    }
    .river-hero-copy p {
        max-width: 430px;
        margin: 1.25rem 0 0;
        color: #536568;
        font-size: 1rem;
        line-height: 1.65;
    }
    .river-hero-status {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.6rem;
        margin-top: 1.6rem;
        color: #536568;
        font-size: 0.76rem;
    }
    .river-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--status-color);
        box-shadow: 0 0 0 4px color-mix(in srgb, var(--status-color) 14%, transparent);
        animation: status-breathe 2.4s ease-in-out infinite;
    }
    .river-hero-art {
        position: relative;
        min-height: 370px;
        overflow: hidden;
        background: #122B39;
    }
    .river-hero-art svg {
        display: block;
        width: 100%;
        height: 100%;
        min-height: 370px;
    }
    .river-art-label {
        position: absolute;
        right: 1.3rem;
        bottom: 1rem;
        color: rgba(243, 238, 227, 0.74);
        font-size: 0.67rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .river-current {
        stroke-dasharray: 4 12;
        animation: river-current 13s linear infinite;
    }
    .river-current-secondary {
        stroke-dasharray: 2 17;
        animation: river-current 19s linear infinite;
    }
    .river-beacon {
        transform-box: fill-box;
        transform-origin: center;
        animation: beacon-breathe 3s ease-in-out infinite;
    }
    .site-section-kicker {
        margin: 1.6rem 0 0.4rem;
        color: #75C7C4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    [data-testid="stPopover"] button {
        min-height: 42px;
        border-color: #D9D0BF !important;
        border-radius: 7px !important;
        background: #F3EEE3 !important;
        color: #172C36 !important;
    }
    [data-testid="stPopoverBody"] {
        border: 1px solid #B8D3DF;
        background: #EAF3F6;
        color: #172C36;
    }
    [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"],
    [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p {
        color: #172C36 !important;
    }
    [data-testid="stPopoverBody"] .stButton > button,
    [data-testid="stPopoverBody"] [data-testid="stBaseButton-secondary"] {
        min-height: 46px;
        border: 1px solid #AFCBD8 !important;
        border-radius: 7px !important;
        background: #D7EAF1 !important;
        color: #172C36 !important;
        box-shadow: none !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        transform: none !important;
    }
    [data-testid="stPopoverBody"] .stButton > button:hover,
    [data-testid="stPopoverBody"] [data-testid="stBaseButton-secondary"]:hover {
        border-color: #80B3C6 !important;
        background: #BDDCE8 !important;
        color: #102832 !important;
        box-shadow: 0 2px 7px rgba(34, 89, 107, 0.16) !important;
        transform: none !important;
    }
    [data-testid="stPopoverBody"] hr {
        border-color: #B8D3DF !important;
    }
    @keyframes river-current {
        to { stroke-dashoffset: -224; }
    }
    @keyframes status-breathe {
        50% { box-shadow: 0 0 0 7px rgba(63, 155, 120, 0.05); }
    }
    @keyframes beacon-breathe {
        50% { opacity: 0.45; transform: scale(0.82); }
    }
    @media (max-width: 760px) {
        .site-nav {
            align-items: flex-start;
            flex-direction: column;
        }
        .site-nav-note {
            text-align: left;
        }
        .st-key-site-nav-links [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        .river-hero {
            grid-template-columns: 1fr;
        }
        .river-hero-art, .river-hero-art svg {
            min-height: 270px;
        }
        .river-hero-copy h1 {
            font-size: clamp(2.7rem, 12vw, 4.3rem);
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
        }
    }
</style>
"""


def render_site_navigation(active_page: str) -> None:
    """Render the shared, sidebar-free navigation for all app pages."""
    st.markdown(SITE_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="site-nav">
            <div class="site-brand">
                <span class="site-brand-mark">≈</span>
                <span class="site-brand-name">Tamakoshi</span>
                <span class="site-brand-caption">River watch</span>
            </div>
            <div class="site-nav-note">Glacier · River · Valley &nbsp; / &nbsp; Dolakha, Nepal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    main_script = st.session_state.get("_tamakoshi_main_script", "streamlit_app.py")
    pages = (
        ("home", "Command Center", main_script),
        ("predictor", "Risk Predictor", "pages/2_Data_and_Risk_Predictor.py"),
        ("history", "Live & History", "pages/3_Live_Data_and_History.py"),
    )
    with st.container(key="site-nav-links"):
        columns = st.columns([1.25, 1.25, 1.25, 0.9], gap="small")
        with columns[0]:
            if st.button(
                "Command Center",
                type="primary" if active_page == "home" else "secondary",
                width="stretch",
                key="site_nav_home",
            ):
                st.switch_page(main_script)
        with columns[1]:
            if st.button(
                "Risk Predictor",
                type="primary" if active_page == "predictor" else "secondary",
                width="stretch",
                key="site_nav_predictor",
            ):
                st.switch_page("pages/2_Data_and_Risk_Predictor.py")
        with columns[2]:
            if st.button(
                "Live & History",
                type="primary" if active_page == "history" else "secondary",
                width="stretch",
                key="site_nav_history",
            ):
                st.switch_page("pages/3_Live_Data_and_History.py")
        with columns[3]:
            with st.popover("Contents", width="stretch"):
                st.caption("EXPLORE THE COMMAND CENTER")
                for page_key, label, target in pages:
                    if st.button(
                        label,
                        key=f"site_contents_{page_key}",
                        width="stretch",
                    ):
                        st.switch_page(target)
                st.divider()
                if st.button("Refresh telemetry", key="site_refresh_data", width="stretch"):
                    st.cache_data.clear()
                    st.rerun()

    st.markdown('<div class="site-nav-rule"></div>', unsafe_allow_html=True)


def render_page_intro(eyebrow: str, title: str, description: str) -> None:
    """Render the editorial heading used on non-dashboard pages."""
    st.markdown(
        f"""
        <section class="site-page-intro">
            <div class="site-eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_river_hero(status: str, synced_at: str) -> None:
    """Render an animated, Rive-inspired river illustration without remote assets."""
    safe_status = html.escape(status)
    safe_synced = html.escape(synced_at)
    status_color = "#3F9B78" if status.lower().startswith("live") else "#B58237"
    st.markdown(
        f"""
        <section class="river-hero">
            <div class="river-hero-copy">
                <div class="site-eyebrow">EARLY WARNING · TAMAKOSHI BASIN</div>
                <h1>Read the river.<br>Prepare the valley.</h1>
                <p>
                    From Tsho Rolpa's glacial headwaters to the lower valley,
                    one clear view of today's signals and tomorrow's risk.
                </p>
                <div class="river-hero-status">
                    <span class="river-status-dot" style="--status-color:{status_color}"></span>
                    <strong>{safe_status}</strong>
                    <span>·</span>
                    <span>Synced {safe_synced}</span>
                </div>
            </div>
            <div class="river-hero-art" aria-label="Animated river corridor illustration">
                <svg viewBox="0 0 760 480" role="img" aria-label="River flowing from the Himalayan mountains">
                    <defs>
                        <linearGradient id="river-sky" x1="0" x2="1" y1="0" y2="1">
                            <stop offset="0" stop-color="#1D4351"/>
                            <stop offset="1" stop-color="#102633"/>
                        </linearGradient>
                        <linearGradient id="river-water" x1="0" x2="1">
                            <stop offset="0" stop-color="#55B9AE"/>
                            <stop offset="0.55" stop-color="#5FCDD0"/>
                            <stop offset="1" stop-color="#D1B778"/>
                        </linearGradient>
                        <pattern id="topo-lines" width="72" height="48" patternUnits="userSpaceOnUse">
                            <path d="M-12 30 C8 7 28 7 45 26 S76 44 89 17" fill="none"
                                  stroke="#F3EEE3" stroke-opacity=".09" stroke-width="1"/>
                        </pattern>
                    </defs>
                    <rect width="760" height="480" fill="url(#river-sky)"/>
                    <rect width="760" height="480" fill="url(#topo-lines)"/>
                    <circle cx="595" cy="96" r="44" fill="#D8BF82" fill-opacity=".12"/>
                    <circle cx="595" cy="96" r="6" fill="#E8D19A" class="river-beacon"/>
                    <path d="M0 238 120 86 205 194 316 52 438 204 535 94 678 226 760 156V480H0Z"
                          fill="#263F4B"/>
                    <path d="M0 277 128 145 225 245 335 100 451 255 558 146 688 270 760 220V480H0Z"
                          fill="#1B3541"/>
                    <path d="M0 345 C116 292 160 374 267 316 S419 285 495 354
                             S632 402 760 315V480H0Z" fill="#142E3B"/>
                    <path d="M92 124 120 86 147 119 130 111 120 98 110 112Z"
                          fill="#E9E0CD" fill-opacity=".75"/>
                    <path d="M291 83 316 52 345 88 324 76 315 65 305 80Z"
                          fill="#E9E0CD" fill-opacity=".82"/>
                    <path d="M514 121 535 94 561 123 542 114 535 105 525 118Z"
                          fill="#E9E0CD" fill-opacity=".7"/>
                    <path d="M40 405 C152 356 210 439 317 382 S479 344 553 396 S662 420 738 375"
                          fill="none" stroke="url(#river-water)" stroke-width="22"
                          stroke-linecap="round" opacity=".8"/>
                    <path d="M40 405 C152 356 210 439 317 382 S479 344 553 396 S662 420 738 375"
                          fill="none" stroke="#E4F2E9" stroke-width="3"
                          stroke-linecap="round" class="river-current"/>
                    <path d="M20 430 C141 378 218 457 327 401 S482 367 569 415 S676 438 750 397"
                          fill="none" stroke="#76D5D0" stroke-opacity=".7" stroke-width="2"
                          class="river-current-secondary"/>
                    <g fill="#F3EEE3" font-family="Arial, sans-serif" font-size="12" letter-spacing="2">
                        <text x="36" y="46">TASHI LAPCHA</text>
                        <text x="36" y="66" fill-opacity=".58">4,580 M · HEADWATERS</text>
                        <text x="573" y="454">DOWNSTREAM →</text>
                    </g>
                    <circle cx="316" cy="382" r="7" fill="#F3EEE3" stroke="#57C7C1" stroke-width="4"/>
                    <circle cx="553" cy="396" r="6" fill="#F3EEE3" stroke="#D3B978" stroke-width="4"/>
                </svg>
                <div class="river-art-label">A living corridor · Dolakha, Nepal</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
