"""Theme tokens and shared Streamlit styling for the flood dashboard."""

from copy import deepcopy


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
        font-size: 18px;
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
        color: #FFFFFF !important;
        font-family: {FONT_STACK} !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: filter 150ms ease, transform 150ms ease;
    }}
    .stButton > button:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {{
        filter: brightness(0.92);
        color: #FFFFFF !important;
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
        color: var(--muted);
        border-radius: 8px 8px 0 0;
        font-family: {FONT_STACK};
        font-size: 15px;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        color: var(--text);
        border-bottom-color: var(--accent);
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
