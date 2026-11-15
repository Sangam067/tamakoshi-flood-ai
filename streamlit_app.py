"""
Tamakoshi Himalayan Flood AI | Early Warning Command Center
============================================================
Life-Safety Early Warning Dashboard & Hydrological Decision Support System
Catchment: Tamakoshi River Corridor, Dolakha District, Nepal (4,580m -> 1,250m ASL)
Theme: Light Theme (Black, White, Red, Green & Life-Safety Accents)
"""

import sys
import types
import os
import json
from datetime import datetime, timezone, timedelta

# -----------------------------------------------------------------------------
# 1. Critical ML Compatibility Shim: scikit-learn _loss
# -----------------------------------------------------------------------------
try:
    import _loss  # noqa: F401
except ModuleNotFoundError:
    import sklearn._loss._loss as _cy_loss
    import sklearn._loss.loss as _py_loss

    _loss_shim = types.ModuleType("_loss")
    _loss_shim.__dict__.update(_py_loss.__dict__)
    _loss_shim.__dict__.update(_cy_loss.__dict__)
    sys.modules["_loss"] = _loss_shim

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Ensure local imports resolve
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.tamakoshi_api import (
    TAMAKOSHI_STATIONS,
    load_model_bundle,
    route_live_basin_telemetry,
    compute_forecast_from_inputs,
)

# -----------------------------------------------------------------------------
# 2. Design System Tokens: Clean Light Theme (Black, White, Red, Green)
# -----------------------------------------------------------------------------
PALETTE = {
    # Surfaces & Canvas (Crisp Light Theme)
    "bg_main": "#F8FAFC",        # Soft light neutral canvas
    "bg_surface": "#FFFFFF",     # Crisp pure white card surface
    "bg_surface_alt": "#F1F5F9", # Elevated container / header / tabs
    "border_subtle": "#E2E8F0",  # Clean light border divider
    "border_focus": "#0F172A",   # High-contrast black/slate focus

    # Typography (Black / Slate Grays)
    "text_primary": "#0F172A",   # Slate 900 (Deep near-black, high legibility)
    "text_secondary": "#475569", # Slate 600 (Informational)
    "text_muted": "#64748B",     # Slate 500 (Metadata, timestamps)

    # Primary Functional Accent (Black / Dark Slate)
    "accent_primary": "#0F172A", # High-contrast primary interactive tone
    "accent_subtle": "#334155",  # Secondary dark slate

    # Strict Life-Safety Signaling Scale (Strictly Red & Green, with Amber/Orange for intermediate watch/warning)
    "emergency": {
        "tier": "EMERGENCY",
        "icon": "🚨",
        "color": "#DC2626",      # Vivid High-Contrast Red (Danger Line 6.69 m³/s)
        "bg": "rgba(220, 38, 38, 0.08)",
        "border": "#DC2626",
    },
    "warning": {
        "tier": "WARNING",
        "icon": "⚠️",
        "color": "#EA580C",      # High-Contrast Orange (Pre-Breach Warning)
        "bg": "rgba(234, 88, 12, 0.08)",
        "border": "#EA580C",
    },
    "watch": {
        "tier": "WATCH",
        "icon": "🟡",
        "color": "#D97706",      # High-Contrast Amber (Watch Buffer 6.02 m³/s)
        "bg": "rgba(217, 119, 6, 0.08)",
        "border": "#D97706",
    },
    "normal": {
        "tier": "NORMAL",
        "icon": "✅",
        "color": "#16A34A",      # Vivid High-Contrast Green (Safe Baseline)
        "bg": "rgba(22, 163, 74, 0.08)",
        "border": "#16A34A",
    },
}

# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Global Light CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tamakoshi Himalayan Flood AI | Early Warning Command Center",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    /* CSS Variables bound strictly to PALETTE */
    :root {{
        --bg-main: {PALETTE["bg_main"]};
        --bg-surface: {PALETTE["bg_surface"]};
        --bg-surface-alt: {PALETTE["bg_surface_alt"]};
        --border-subtle: {PALETTE["border_subtle"]};
        --border-focus: {PALETTE["border_focus"]};
        --text-primary: {PALETTE["text_primary"]};
        --text-secondary: {PALETTE["text_secondary"]};
        --text-muted: {PALETTE["text_muted"]};
        --accent-primary: {PALETTE["accent_primary"]};
    }}

    /* Global canvas */
    html, body, .stApp {{
        background: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }}

    .main .block-container {{
        padding: 1rem 1.5rem 2rem 1.5rem !important;
        max-width: 1560px !important;
    }}

    /* Hide standard menu & footer */
    #MainMenu, footer, header {{
        visibility: hidden !important;
    }}

    /* Sidebar styled strictly as a filter panel in light theme */
    section[data-testid="stSidebar"] {{
        background: #FFFFFF !important;
        border-right: 1px solid var(--border-subtle) !important;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding: 1.2rem 1rem !important;
    }}

    /* 8px Grid & Consistent Cards */
    .command-card {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .command-card:hover {{
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.07);
    }}

    .kpi-label {{
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 4px;
    }}
    .kpi-value {{
        font-size: 1.80rem;
        font-weight: 800;
        line-height: 1.2;
        color: var(--text-primary);
    }}
    .kpi-meta {{
        font-size: 0.80rem;
        font-weight: 500;
        margin-top: 6px;
        color: var(--text-muted);
    }}

    /* Action Banner */
    .severity-banner {{
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border: 1px solid;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .severity-title {{
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }}
    .severity-body {{
        font-size: 0.92rem;
        line-height: 1.55;
        color: var(--text-primary);
    }}
    .action-list {{
        margin: 8px 0 0 0;
        padding-left: 20px;
        font-size: 0.88rem;
        color: #1E293B;
        line-height: 1.6;
    }}

    /* Tab styling */
    .stTabs [role="tablist"] {{
        gap: 8px;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 4px;
    }}
    .stTabs [role="tab"] {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        padding: 10px 16px;
        font-weight: 600;
        font-size: 0.88rem;
        min-height: 44px;
        transition: all 0.15s ease;
    }}
    .stTabs [role="tab"]:hover {{
        color: var(--text-primary);
        border-color: #CBD5E1;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background: var(--bg-surface-alt);
        border-color: #CBD5E1;
        border-top: 3px solid var(--accent-primary) !important;
        color: var(--text-primary) !important;
        font-weight: 700;
    }}

    /* Streamlit controls minimum tap target 44px */
    .stButton > button {{
        min-height: 44px;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #CBD5E1;
        background: #FFFFFF;
        color: var(--text-primary);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: all 0.15s ease-in-out;
    }}
    .stButton > button:hover {{
        border-color: var(--accent-primary);
        color: var(--accent-primary);
        background: var(--bg-surface-alt);
    }}
    .stSelectbox > div, .stSlider > div, .stNumberInput > div {{
        border-radius: 8px;
    }}

    /* Header Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Safe Model Loading & Core Data Logic
# -----------------------------------------------------------------------------
try:
    bundle = load_model_bundle()
    DANGER_THRESHOLD = float(bundle["threshold"])          # 6.69 m³/s
    YELLOW_THRESHOLD = DANGER_THRESHOLD * 0.90             # 6.02 m³/s
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Critical Model Asset Error: Unable to load 'tamakoshi_ultimate_model.joblib'. Details: {e}")
    st.stop()


def classify_alert_severity(pred_flow: float, flood_prob: float) -> dict:
    """Calculates single authoritative life-safety severity level and operational protocols."""
    if pred_flow >= DANGER_THRESHOLD or flood_prob >= 60.0:
        base = PALETTE["emergency"]
        return {
            **base,
            "headline": f"{base['icon']} EMERGENCY: CRITICAL OVERTOPPING BREACH IMMINENT",
            "summary": (
                f"Predicted 24-hr discharge reaches <b>{pred_flow:.2f} m³/s</b>, exceeding the "
                f"95th-percentile safety threshold of <b>{DANGER_THRESHOLD:.2f} m³/s</b> by "
                f"<b>{(pred_flow - DANGER_THRESHOLD):+.2f} m³/s</b> with <b>{flood_prob:.1f}%</b> inundation likelihood."
            ),
            "protocol_header": "🚨 Mandated Emergency Life-Safety Actions (Level 4 - RED):",
            "protocols": [
                "<b>Immediate Siren Alarms:</b> Broadcast VHF/SMS emergency evacuation to Singati Bazaar, Gongar, Jagat, and Lamabagar.",
                "<b>Dam Radial Spillway Gates:</b> Actuate Upper Tamakoshi (456 MW) spillways immediately for reservoir drawdown buffering.",
                "<b>Tactical Rescue Deployment:</b> Mobilize Armed Police Force (APF) water rescue and Dolakha DDMC emergency teams.",
                "<b>Emergency Channel Placeholder:</b> [PLACEHOLDER: Dolakha DEOC VHF Ch 4 / Hotline: +977-49-421111]",
            ],
        }
    elif pred_flow >= (DANGER_THRESHOLD - 0.35) or flood_prob >= 50.0:
        base = PALETTE["warning"]
        return {
            **base,
            "headline": f"{base['icon']} WARNING: SEVERE SURGE ACCELERATION — PRE-BREACH WATCH",
            "summary": (
                f"Predicted discharge reaches <b>{pred_flow:.2f} m³/s</b> with rapid acceleration. "
                f"Channel clearance is only <b>{(DANGER_THRESHOLD - pred_flow):.2f} m³/s</b> with <b>{flood_prob:.1f}%</b> breach probability."
            ),
            "protocol_header": "⚠️ High-Alert Standby Actions (Level 3 - WARNING):",
            "protocols": [
                "<b>Pre-Warning Sirens:</b> Sound advisory sirens in low-lying riverside settlement zones along Singati corridor.",
                "<b>Continuous Telemetry Polling:</b> Switch upstream radar gauges at Tsho Rolpa (4,580m) to high-frequency 10-minute cycles.",
                "<b>Grid Coordination:</b> Coordinate with Nepal Electricity Authority (NEA) Load Dispatch for contingency load-shedding.",
                "<b>Emergency Channel Placeholder:</b> [PLACEHOLDER: APF Disaster Response Base Singati / Hotline: +977-49-421999]",
            ],
        }
    elif pred_flow >= YELLOW_THRESHOLD or flood_prob >= 35.0:
        base = PALETTE["watch"]
        return {
            **base,
            "headline": f"{base['icon']} WATCH: MONSOON SURGE IN UPPER CATCHMENT — BUFFER WATCH",
            "summary": (
                f"Predicted discharge reaches <b>{pred_flow:.2f} m³/s</b>, entering the 10% safety buffer zone "
                f"(<b>{YELLOW_THRESHOLD:.2f} – {DANGER_THRESHOLD:.2f} m³/s</b>). AI inundation likelihood: <b>{flood_prob:.1f}%</b>."
            ),
            "protocol_header": "🟡 Active Hydrological Watch Actions (Level 2 - WATCH):",
            "protocols": [
                "<b>Floodplain Clearance:</b> Order riverbank sand/gravel excavation crews to immediately evacuate the gorge floodplain.",
                "<b>Glacial Siphon Surveillance:</b> Monitor Tsho Rolpa moraine lake siphon discharge and positive temperature anomalies.",
                "<b>DDMC Advisory:</b> Notify Dolakha disaster coordinators and transport police along Singati-Charikot highway.",
                "<b>Emergency Channel Placeholder:</b> [PLACEHOLDER: Upper Tamakoshi Hydropower Control Room / Hotline: +977-49-420000]",
            ],
        }
    else:
        base = PALETTE["normal"]
        return {
            **base,
            "headline": f"{base['icon']} NORMAL: SAFE HYDROLOGICAL EQUILIBRIUM",
            "summary": (
                f"Discharge is safely contained within natural gorge banks at <b>{pred_flow:.2f} m³/s</b> "
                f"(Clearance to Danger Line: <b>{(DANGER_THRESHOLD - pred_flow):.2f} m³/s</b>). Flood probability: <b>{flood_prob:.1f}%</b>."
            ),
            "protocol_header": "✅ Standard Sentinel Surveillance (Level 1 - GREEN):",
            "protocols": [
                "<b>Nominal Hydro Dispatch:</b> Maintain scheduled 456 MW generation dispatch at Upper Tamakoshi Plant.",
                "<b>Automated Monitoring:</b> Autonomous GloFAS satellite precipitation and acoustic telemetry polling every 30 minutes.",
                "<b>Inquiry Placeholder:</b> [PLACEHOLDER: DHM Flood Forecasting Information Desk / Toll-Free: 1155]",
            ],
        }


# Calibrated Historical Scenarios for Stress Testing
SCENARIOS = {
    "🔴 Scenario 3: Cloudburst + Glacial Melt (Red Alert)": (
        6.60, 5.55, 1.15, 4.35, 9.80, 58.0, 42.0, 135.0, 230.0, 3, 48, 24.0, 15.5, 0.0, 22.5, 8
    ),
    "🟡 Scenario 2: Rising Monsoon Surge (Yellow Watch)": (
        5.85, 5.20, 0.85, 3.70, 8.40, 28.0, 20.0, 68.0, 135.0, 7, 41, 22.0, 14.0, 0.0, 19.5, 7
    ),
    "🟢 Scenario 1: Dry Winter Baseflow (Normal)": (
        0.35, 0.36, 0.05, 0.22, 0.55, 0.0, 0.0, 0.5, 2.0, 0, 22, 14.0, 2.5, 0.0, 14.5, 2
    ),
    "⚙️ Custom Simulation (Manual Sliders)": None,
}

# -----------------------------------------------------------------------------
# 5. Sidebar: FILTERS & PARAMETERS ONLY
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="padding: 10px 4px 16px 4px; border-bottom: 1px solid {PALETTE['border_subtle']}; margin-bottom: 14px;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: {PALETTE['text_secondary']}; text-transform: uppercase;">
            Early Warning Command Center
        </div>
        <div style="font-size: 1.30rem; font-weight: 800; color: {PALETTE['text_primary']}; margin-top: 2px;">
            🏔️ Tamakoshi Flood AI
        </div>
        <div style="font-size: 0.80rem; color: {PALETTE['text_muted']}; margin-top: 2px;">
            Dolakha Watershed, Nepal
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Operational Mode Filter
st.sidebar.markdown("<div class='kpi-label'>Operating Telemetry Mode</div>", unsafe_allow_html=True)
op_mode = st.sidebar.radio(
    "Select Operating Mode:",
    ["📡 Operational Live Telemetry", "🎛️ Simulation / Stress Testing"],
    index=0,
    label_visibility="collapsed",
    key="filter_op_mode",
)

if "📡" in op_mode:
    # Live Mode Filters
    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='kpi-label'>Telemetry Lookback Window</div>", unsafe_allow_html=True)
    past_days_filter = st.sidebar.slider(
        "Lookback Days",
        min_value=3,
        max_value=14,
        value=7,
        help="Historical lookback window for GloFAS river discharge and basin precipitation.",
        key="filter_lookback_days",
    )

    refresh_clicked = st.sidebar.button("🔄 Refresh Live Telemetry", use_container_width=True, key="btn_refresh_live")
    if refresh_clicked:
        st.cache_data.clear()
        st.sidebar.success("Telemetry cache refreshed.")

    # Fetch live telemetry
    with st.spinner("Connecting to Open-Meteo GloFAS & Basin Weather APIs..."):
        try:
            live_payload = route_live_basin_telemetry(past_days=past_days_filter)
        except Exception as e:
            st.sidebar.warning(f"Telemetry API error, using verified baseline: {e}")
            live_payload = route_live_basin_telemetry(past_days=7)

    active_inputs = live_payload["inputs"]
    data_source_label = live_payload["status_label"]
    data_synced_at = live_payload["synced_at"]
    live_history_df = live_payload["history_df"]
    is_live_active = True

else:
    # Simulation Mode Filters
    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='kpi-label'>Calibrated Watershed Scenario</div>", unsafe_allow_html=True)
    scenario_choice = st.sidebar.selectbox(
        "Select Scenario Preset:",
        list(SCENARIOS.keys()),
        index=0,
        label_visibility="collapsed",
        key="filter_scenario_preset",
    )
    sim_defaults = SCENARIOS[scenario_choice] or (
        6.60, 5.55, 1.15, 4.35, 9.80, 58.0, 42.0, 135.0, 230.0, 3, 48, 24.0, 15.5, 0.0, 22.5, 8
    )

    with st.sidebar.expander("🌊 Station Discharge Sensors (m³/s)", expanded=True):
        sim_q_mid = st.number_input("Singati Target Flow (m³/s)", value=float(sim_defaults[0]), step=0.1, key="sim_q_mid")
        sim_q_yesterday = st.number_input("Singati Flow Yesterday (m³/s)", value=float(sim_defaults[1]), step=0.1, key="sim_q_yesterday")
        sim_q_glacier = st.number_input("Tsho Rolpa Glacier Outflow (m³/s)", value=float(sim_defaults[2]), step=0.05, key="sim_q_glacier")
        sim_q_upper = st.number_input("Gongar Dam Intake Flow (m³/s)", value=float(sim_defaults[3]), step=0.1, key="sim_q_upper")
        sim_q_lower = st.number_input("Lower Confluence Flow (m³/s)", value=float(sim_defaults[4]), step=0.1, key="sim_q_lower")

    with st.sidebar.expander("🌧️ Monsoon Cloudburst & Rain Dynamics", expanded=False):
        sim_rain_mid = st.number_input("Valley Rain Today (mm)", value=float(sim_defaults[5]), step=1.0, key="sim_rain_mid")
        sim_rain_glacier = st.number_input("Glacier Rain Today (mm)", value=float(sim_defaults[6]), step=1.0, key="sim_rain_glacier")
        sim_rain_3d = st.number_input("3-Day Cumulative Rain (mm)", value=float(sim_defaults[7]), step=5.0, key="sim_rain_3d")
        sim_rain_7d = st.number_input("7-Day Cumulative Rain (mm)", value=float(sim_defaults[8]), step=5.0, key="sim_rain_7d")
        sim_rain_hours = st.slider("Rain Duration Hours (Cloudburst = low)", 1, 24, int(sim_defaults[9]), key="sim_rain_hours")

    with st.sidebar.expander("🏔️ Vadose Soil & Glacial Thermodynamics", expanded=False):
        sim_soil_sat = st.slider("Deep Soil Saturation (%)", 10, 50, int(sim_defaults[10]), key="sim_soil_sat")
        sim_tmax = st.number_input("Valley Max Temp (°C)", value=float(sim_defaults[11]), step=0.5, key="sim_tmax")
        sim_tmin = st.number_input("Valley Min Temp (°C)", value=float(sim_defaults[12]), step=0.5, key="sim_tmin")
        sim_snow = st.number_input("Glacier Snow Today (cm)", value=float(sim_defaults[13]), step=0.5, key="sim_snow")
        sim_solar = st.number_input("Solar Radiation (MJ/m²)", value=float(sim_defaults[14]), step=0.5, key="sim_solar")
        sim_month = st.slider("Month of Year", 1, 12, int(sim_defaults[15]), key="sim_month")

    active_inputs = {
        "q_mid": sim_q_mid, "q_yesterday": sim_q_yesterday, "q_glacier": sim_q_glacier,
        "q_upper": sim_q_upper, "q_lower": sim_q_lower, "rain_mid": sim_rain_mid,
        "rain_glacier": sim_rain_glacier, "rain_3d": sim_rain_3d, "rain_7d": sim_rain_7d,
        "rain_hours": sim_rain_hours, "soil_sat_pct": sim_soil_sat, "tmax": sim_tmax,
        "tmin": sim_tmin, "snow_glacier": sim_snow, "solar_rad": sim_solar, "month": sim_month,
    }
    data_source_label = f"MANUAL SIMULATION ({scenario_choice})"
    npt_now = datetime.now(timezone(timedelta(hours=5, minutes=45))).strftime("%Y-%m-%d %H:%M NPT")
    data_synced_at = f"{npt_now} (Interactive)"
    live_history_df = pd.DataFrame([
        {"Date": "T-3d", "Singati (m³/s)": sim_q_yesterday * 0.92, "Tsho Rolpa (m³/s)": sim_q_glacier * 0.90, "Gongar (m³/s)": sim_q_upper * 0.91, "Lower Tamakoshi (m³/s)": sim_q_lower * 0.92, "Rain (mm)": sim_rain_3d * 0.25},
        {"Date": "T-2d", "Singati (m³/s)": sim_q_yesterday * 0.95, "Tsho Rolpa (m³/s)": sim_q_glacier * 0.94, "Gongar (m³/s)": sim_q_upper * 0.95, "Lower Tamakoshi (m³/s)": sim_q_lower * 0.95, "Rain (mm)": sim_rain_3d * 0.35},
        {"Date": "Yesterday", "Singati (m³/s)": sim_q_yesterday, "Tsho Rolpa (m³/s)": sim_q_glacier * 0.98, "Gongar (m³/s)": sim_q_upper, "Lower Tamakoshi (m³/s)": sim_q_lower, "Rain (mm)": sim_rain_3d * 0.40},
        {"Date": "Today", "Singati (m³/s)": sim_q_mid, "Tsho Rolpa (m³/s)": sim_q_glacier, "Gongar (m³/s)": sim_q_upper, "Lower Tamakoshi (m³/s)": sim_q_lower, "Rain (mm)": sim_rain_mid},
    ])
    is_live_active = False

# -----------------------------------------------------------------------------
# 6. Core ML Prediction Engine (Exact 58 Features & 6 Models)
# -----------------------------------------------------------------------------
forecast = compute_forecast_from_inputs(bundle, active_inputs)

pred_flow = forecast["pred_flow"]
delta_pred = forecast["delta_pred"]
low_flow = forecast["low_flow"]
high_flow = forecast["high_flow"]
flood_prob = forecast["flood_prob"]
individual_preds = forecast["individual_preds"]

# Determine single authoritative severity profile
severity = classify_alert_severity(pred_flow, flood_prob)

# -----------------------------------------------------------------------------
# 7. MAIN VIEW HIERARCHY: Top Command Header & Data Lineage
# -----------------------------------------------------------------------------
top_c1, top_c2 = st.columns([8, 4])
with top_c1:
    st.markdown(
        f"""
        <div style="margin-bottom: 4px;">
            <span style="font-size: 1.45rem; font-weight: 800; color: {PALETTE['text_primary']};">
                🏔️ Tamakoshi River Flood Early Warning Command Center
            </span>
        </div>
        <div style="font-size: 0.82rem; color: {PALETTE['text_secondary']};">
            Dolakha Watershed • Tsho Rolpa Glacier (4,580m) to Singati Gorge (1,250m) • Early Warning Telemetry
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_c2:
    st.markdown(
        f"""
        <div style="text-align: right; font-size: 0.80rem; color: {PALETTE['text_secondary']}; padding-top: 4px;">
            <div><b>Data Source:</b> <span style="color: {PALETTE['text_primary']}; font-weight: 600;">{data_source_label}</span></div>
            <div><b>Last Synced:</b> <span style="color: {PALETTE['text_muted']};">{data_synced_at}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. HIERARCHY STEP 1: Emergency Action & Severity Status Banner
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="severity-banner" style="background: {severity['bg']}; border-color: {severity['border']};">
        <div class="severity-title" style="color: {severity['color']};">
            {severity['headline']}
        </div>
        <div class="severity-body">
            {severity['summary']}
        </div>
        <div style="margin-top: 10px; font-weight: 700; font-size: 0.88rem; color: {severity['color']};">
            {severity['protocol_header']}
        </div>
        <ul class="action-list">
            {''.join(f"<li>{item}</li>" for item in severity['protocols'])}
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 9. HIERARCHY STEP 2: 4 Primary KPI Cards
# -----------------------------------------------------------------------------
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    delta_color = PALETTE["emergency"]["color"] if delta_pred > 0 else PALETTE["normal"]["color"]
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">Forecast Tomorrow Flow</div>
            <div class="kpi-value" style="color: {severity['color']};">
                {pred_flow:.2f} <span style="font-size: 1rem; color: {PALETTE['text_secondary']};">m³/s</span>
            </div>
            <div class="kpi-meta" style="color: {delta_color}; font-weight: 600;">
                {delta_pred:+.2f} m³/s vs today ({active_inputs['q_mid']:.2f} m³/s)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c2:
    ci_half = (high_flow - low_flow) / 2.0
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">80% Confidence Interval</div>
            <div class="kpi-value" style="color: {PALETTE['text_primary']};">
                {low_flow:.2f} – {high_flow:.2f} <span style="font-size: 1rem; color: {PALETTE['text_secondary']};">m³/s</span>
            </div>
            <div class="kpi-meta" style="color: {PALETTE['text_secondary']};">
                Quantile 10%–90% spread: ±{ci_half:.2f} m³/s
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c3:
    freeboard = DANGER_THRESHOLD - pred_flow
    freeboard_label = "Freeboard Clearance" if freeboard >= 0 else "Overtopping Depth"
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">Channel Safety Margin</div>
            <div class="kpi-value" style="color: {severity['color']};">
                {freeboard:+.2f} <span style="font-size: 1rem; color: {PALETTE['text_secondary']};">m³/s</span>
            </div>
            <div class="kpi-meta" style="color: {PALETTE['text_secondary']};">
                Danger: {DANGER_THRESHOLD:.2f} m³/s • Buffer: {YELLOW_THRESHOLD:.2f} m³/s
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c4:
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">AI Breach Probability</div>
            <div class="kpi-value" style="color: {severity['color']};">
                {flood_prob:.1f}<span style="font-size: 1.1rem; color: {PALETTE['text_secondary']};">%</span>
            </div>
            <div class="kpi-meta" style="color: {severity['color']}; font-weight: 700;">
                {severity['icon']} Tier: {severity['tier']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. HIERARCHY STEP 3: Core Visual Tier (Map + Hydrograph)
# -----------------------------------------------------------------------------
col_hydro, col_map = st.columns([7, 5])

with col_hydro:
    st.markdown("<div class='kpi-label'>24-Hour Singati River Hydrograph & Surge Cone</div>", unsafe_allow_html=True)
    timeline_hours = np.linspace(-24, 24, 49)
    past_curve = np.interp(timeline_hours[timeline_hours <= 0], [-24, 0], [active_inputs["q_yesterday"], active_inputs["q_mid"]])
    future_curve = np.interp(timeline_hours[timeline_hours > 0], [0, 24], [active_inputs["q_mid"], pred_flow])
    future_hours = timeline_hours[timeline_hours >= 0]
    ci_factor = future_hours / 24.0
    ci_upper = pred_flow + (high_flow - pred_flow) * ci_factor
    ci_lower = np.maximum(0.0, pred_flow - (pred_flow - low_flow) * ci_factor)

    fig_hydro = go.Figure()
    # 80% CI Cone
    fig_hydro.add_trace(
        go.Scatter(
            x=np.concatenate([future_hours, future_hours[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(15, 23, 42, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="80% Prediction Interval",
        )
    )
    # Sensor History (Solid Black)
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours <= 0],
            y=past_curve,
            mode="lines+markers",
            name="Telemetry History (T-24h to T0)",
            line=dict(color="#0F172A", width=3),
            marker=dict(size=4, color="#0F172A"),
        )
    )
    # AI 24h Forecast (Green for Normal, Red for Alert)
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours >= 0],
            y=future_curve,
            mode="lines+markers",
            name="AI Forecast (T0 to T+24h)",
            line=dict(color=severity["color"], width=3.5, dash="dash"),
            marker=dict(size=6, symbol="diamond", color=severity["color"]),
        )
    )
    # Safety Threshold Lines (Red & Amber)
    fig_hydro.add_hline(
        y=DANGER_THRESHOLD,
        line_dash="dot",
        line_color=PALETTE["emergency"]["color"],
        line_width=2,
        annotation_text=f"Red Danger ({DANGER_THRESHOLD:.2f} m³/s)",
        annotation_position="top right",
        annotation_font=dict(color=PALETTE["emergency"]["color"], size=10),
    )
    fig_hydro.add_hline(
        y=YELLOW_THRESHOLD,
        line_dash="dashdot",
        line_color=PALETTE["watch"]["color"],
        line_width=1.5,
        annotation_text=f"Yellow Watch ({YELLOW_THRESHOLD:.2f} m³/s)",
        annotation_position="bottom right",
        annotation_font=dict(color=PALETTE["watch"]["color"], size=10),
    )
    fig_hydro.update_layout(
        template="plotly_white",
        paper_bgcolor=PALETTE["bg_surface"],
        plot_bgcolor=PALETTE["bg_surface"],
        xaxis=dict(title="Lead Time (Hours from Present T0)", gridcolor=PALETTE["border_subtle"], zeroline=False),
        yaxis=dict(title="Discharge at Singati Gorge (m³/s)", gridcolor=PALETTE["border_subtle"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color=PALETTE["text_primary"])),
        margin=dict(l=24, r=18, t=28, b=24),
        height=370,
    )
    st.plotly_chart(fig_hydro, use_container_width=True)

with col_map:
    st.markdown("<div class='kpi-label'>3D Tamakoshi Corridor Telemetry Map</div>", unsafe_allow_html=True)

    def station_severity_color(val: float, is_target: bool = False):
        check_val = pred_flow if is_target else val
        if check_val >= DANGER_THRESHOLD:
            return [220, 38, 38, 240]   # Vivid Red
        elif check_val >= YELLOW_THRESHOLD:
            return [217, 119, 6, 230]   # Amber Watch
        else:
            return [22, 163, 74, 220]   # Vivid Green

    map_stations_data = [
        {"name": "1. Tsho Rolpa Glacier (4,580m)", "lat": 27.8617, "lon": 86.4769, "flow": active_inputs["q_glacier"], "color": station_severity_color(active_inputs["q_glacier"]), "height": max(300, active_inputs["q_glacier"] * 1600 + 400)},
        {"name": "2. Gongar Dam Intake (1,990m)", "lat": 27.8442, "lon": 86.2205, "flow": active_inputs["q_upper"], "color": station_severity_color(active_inputs["q_upper"]), "height": max(300, active_inputs["q_upper"] * 1200 + 400)},
        {"name": "3. Singati Gorge Target (1,250m)", "lat": 27.7314, "lon": 86.1628, "flow": pred_flow, "color": station_severity_color(pred_flow, is_target=True), "height": max(300, pred_flow * 1400 + 500)},
        {"name": "4. Lower Confluence (880m)", "lat": 27.5250, "lon": 86.0750, "flow": active_inputs["q_lower"], "color": station_severity_color(active_inputs["q_lower"]), "height": max(300, active_inputs["q_lower"] * 800 + 300)},
    ]
    df_map_stations = pd.DataFrame(map_stations_data)

    corridor_arcs = pd.DataFrame([
        {"from_lon": 86.4769, "from_lat": 27.8617, "to_lon": 86.2205, "to_lat": 27.8442, "color": station_severity_color(active_inputs["q_glacier"])},
        {"from_lon": 86.2205, "from_lat": 27.8442, "to_lon": 86.1628, "to_lat": 27.7314, "color": station_severity_color(active_inputs["q_upper"])},
        {"from_lon": 86.1628, "from_lat": 27.7314, "to_lon": 86.0750, "to_lat": 27.5250, "color": station_severity_color(pred_flow, is_target=True)},
    ])

    map_view = pdk.ViewState(latitude=27.72, longitude=86.24, zoom=9.8, pitch=52, bearing=-18)
    col_layer = pdk.Layer(
        "ColumnLayer", data=df_map_stations, get_position=["lon", "lat"], get_elevation="height",
        elevation_scale=1, radius=520, get_fill_color="color", pickable=True, auto_highlight=True,
    )
    arc_layer = pdk.Layer(
        "ArcLayer", data=corridor_arcs, get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"], get_source_color="color", get_target_color="color",
        get_width=4.5, pickable=True,
    )

    deck_chart = pdk.Deck(
        layers=[arc_layer, col_layer],
        initial_view_state=map_view,
        map_style="carto-positron",  # Clean, reliable light theme cartographic style
        tooltip={
            "html": (
                "<div style='background:#FFFFFF; padding:8px 12px; border-radius:6px; border:1px solid #CBD5E1; color:#0F172A; box-shadow:0 2px 6px rgba(0,0,0,0.1); font-family:sans-serif;'>"
                "<b>{name}</b><br>Discharge: <b>{flow} m³/s</b></div>"
            ),
            "style": {"color": "#0F172A"},
        },
    )
    st.pydeck_chart(deck_chart)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 11. HIERARCHY STEP 4: Deep-Dive Diagnostic Tabs (One Job Per Tab)
# -----------------------------------------------------------------------------
tab_gorge, tab_council, tab_gauges, tab_table, tab_method = st.tabs([
    "🌊 3D River Gorge Twin",
    "🧠 Council of 6 AI Models",
    "⚡ Compound Physics Gauges",
    "📋 Corridor Station Data",
    "🔬 Methodology & Contacts",
])

# ----------------- Tab 1: 3D River Gorge Twin -----------------
with tab_gorge:
    st.markdown("<div class='kpi-label'>Interactive 3D Singati River Gorge Digital Twin (Three.js WebGL)</div>", unsafe_allow_html=True)
    st.caption("Inspect dynamic water surface elevation, canyon terrain walls, and safety threshold planes in full 3D.")

    water_height_ratio = min(1.8, max(0.1, pred_flow / DANGER_THRESHOLD))
    freeboard_val = DANGER_THRESHOLD - pred_flow
    freeboard_text = f"Clearance: +{freeboard_val:.2f} m³/s (Safe)" if freeboard_val >= 0 else f"OVERTOPPING: {freeboard_val:+.2f} m³/s"

    three_js_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; overflow: hidden; background: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
            #hud {{
                position: absolute; top: 12px; left: 14px;
                background: rgba(255, 255, 255, 0.95); border: 1px solid #CBD5E1;
                border-radius: 8px; padding: 10px 14px; color: #0F172A; font-size: 0.80rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08); pointer-events: none;
            }}
            #hud b {{ color: {severity['color']}; }}
            #hint {{
                position: absolute; bottom: 10px; right: 14px;
                background: rgba(255, 255, 255, 0.90); border: 1px solid #E2E8F0;
                border-radius: 6px; padding: 6px 10px; color: #475569; font-size: 0.72rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05); pointer-events: none;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="hud">
            <div style="font-weight:700; color:{PALETTE['text_primary']}; margin-bottom:4px;">Singati Canyon Hydraulic Stage</div>
            <div>Forecast: <b>{pred_flow:.2f} m³/s</b> ({severity['tier']})</div>
            <div>{freeboard_text}</div>
            <div style="color:#64748B; font-size:0.72rem; margin-top:2px;">Danger Line: {DANGER_THRESHOLD:.2f} m³/s • Buffer: {YELLOW_THRESHOLD:.2f} m³/s</div>
        </div>
        <div id="hint">🖱️ Left-Drag: Rotate | Right-Drag: Pan | Scroll: Zoom</div>
        <script>
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xF8FAFC);
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / 480, 0.1, 1000);
            camera.position.set(24, 18, 32);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, 480);
            document.body.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 1.5, 0);

            // Lighting (Bright Neutral Daylight)
            scene.add(new THREE.AmbientLight(0xE2E8F0, 0.85));
            const sun = new THREE.DirectionalLight(0xFFFFFF, 1.2);
            sun.position.set(25, 35, 20);
            scene.add(sun);
            const alertLight = new THREE.PointLight('{severity["color"]}', 1.5, 40);
            alertLight.position.set(0, 8, 0);
            scene.add(alertLight);

            // Canyon Mesh (Natural Mountain Stone)
            const canyonGeo = new THREE.PlaneGeometry(30, 56, 50, 50);
            canyonGeo.rotateX(-Math.PI / 2);
            const pos = canyonGeo.attributes.position;
            for (let i = 0; i < pos.count; i++) {{
                let x = pos.getX(i);
                let z = pos.getZ(i);
                let y = Math.pow(Math.abs(x) / 3.8, 2.0) - 1.2 + Math.sin(x * 0.4 + z * 0.2) * 0.6;
                pos.setY(i, y);
            }}
            canyonGeo.computeVertexNormals();
            const canyonMat = new THREE.MeshStandardMaterial({{ color: 0x94A3B8, roughness: 0.85, flatShading: true }});
            scene.add(new THREE.Mesh(canyonGeo, canyonMat));

            // Dynamic Water Mesh (Green for normal, Red for danger)
            const waterGeo = new THREE.PlaneGeometry(13, 56, 30, 30);
            waterGeo.rotateX(-Math.PI / 2);
            const waterMat = new THREE.MeshStandardMaterial({{
                color: '{severity["color"]}', roughness: 0.2, metalness: 0.6, transparent: true, opacity: 0.88
            }});
            const water = new THREE.Mesh(waterGeo, waterMat);
            water.position.y = -0.9 + {water_height_ratio} * 2.8;
            scene.add(water);

            // Danger Threshold Wireframe (Red)
            const dangerGeo = new THREE.PlaneGeometry(16, 56);
            dangerGeo.rotateX(-Math.PI / 2);
            const dangerPlane = new THREE.Mesh(dangerGeo, new THREE.MeshBasicMaterial({{ color: 0xDC2626, wireframe: true, transparent: true, opacity: 0.45 }}));
            dangerPlane.position.y = 1.9;
            scene.add(dangerPlane);

            let clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                let t = clock.getElapsedTime();
                let wp = waterGeo.attributes.position;
                for (let i = 0; i < wp.count; i++) {{
                    let u = wp.getX(i);
                    let v = wp.getZ(i);
                    wp.setY(i, Math.sin(v * 0.5 + t * 4.0) * 0.10 + Math.cos(u * 0.8 + t * 2.5) * 0.06);
                }}
                waterGeo.attributes.position.needsUpdate = true;
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    components.html(three_js_html, height=490)

# ----------------- Tab 2: Council of 6 AI Models -----------------
with tab_council:
    st.markdown("<div class='kpi-label'>Council of 6 Base Estimators vs. RidgeCV Meta-Learner</div>", unsafe_allow_html=True)
    st.caption("Comparison of base estimators with the stacked meta-regressor resolving individual model deltas.")

    council_data = dict(individual_preds)
    council_data["RidgeCV Meta-Learner"] = pred_flow

    m_names = list(council_data.keys())
    m_flows = [council_data[m] for m in m_names]
    m_colors = ["#1E293B" if m != "RidgeCV Meta-Learner" else severity["color"] for m in m_names]

    fig_council = go.Figure(
        go.Bar(
            x=m_flows,
            y=m_names,
            orientation="h",
            marker_color=m_colors,
            text=[f"{v:.3f} m³/s" for v in m_flows],
            textposition="auto",
        )
    )
    fig_council.add_vline(x=DANGER_THRESHOLD, line_dash="dot", line_color=PALETTE["emergency"]["color"], line_width=2)
    fig_council.add_vline(x=YELLOW_THRESHOLD, line_dash="dashdot", line_color=PALETTE["watch"]["color"], line_width=1.5)
    fig_council.update_layout(
        template="plotly_white",
        paper_bgcolor=PALETTE["bg_surface"],
        plot_bgcolor=PALETTE["bg_surface"],
        xaxis=dict(title="Estimated Discharge Tomorrow (m³/s)", gridcolor=PALETTE["border_subtle"]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=20, b=24),
        height=340,
    )
    st.plotly_chart(fig_council, use_container_width=True)

    base_only_flows = [council_data[m] for m in m_names if m != "RidgeCV Meta-Learner"]
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Council Range Spread", f"{max(base_only_flows) - min(base_only_flows):.3f} m³/s", "Max - Min across 6 base models")
    with sc2:
        st.metric("Unweighted Council Mean", f"{np.mean(base_only_flows):.3f} m³/s", "Simple average")
    with sc3:
        st.metric("Meta-Learner Correction", f"{pred_flow - np.mean(base_only_flows):+.3f} m³/s", "RidgeCV cross-validated delta")

# ----------------- Tab 3: Compound Physics Gauges -----------------
with tab_gauges:
    st.markdown("<div class='kpi-label'>Coupled Glacial-Monsoon Physical Forcing Gauges</div>", unsafe_allow_html=True)
    st.caption("Physical environmental metrics driving non-linear overland runoff and moraine thermal melt.")

    g1, g2, g3, g4 = st.columns(4)
    rain_intensity = active_inputs["rain_mid"] / (active_inputs["rain_hours"] + 0.5)
    glacier_temp = active_inputs["tmax"] - 6.0

    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=rain_intensity, number={"suffix": " mm/h"},
            title={"text": "Cloudburst Rain Intensity", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 40], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": "#0F172A"},
                "steps": [
                    {"range": [0, 10], "color": "rgba(22, 163, 74, 0.18)"},
                    {"range": [10, 20], "color": "rgba(217, 119, 6, 0.18)"},
                    {"range": [20, 40], "color": "rgba(220, 38, 38, 0.22)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 18.0},
            },
        ))
        fig_g1.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g1, use_container_width=True)

    with g2:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=active_inputs["rain_7d"], number={"suffix": " mm"},
            title={"text": "Antecedent 7-Day Rain", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 300], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": "#0F172A"},
                "steps": [
                    {"range": [0, 80], "color": "rgba(22, 163, 74, 0.18)"},
                    {"range": [80, 150], "color": "rgba(217, 119, 6, 0.18)"},
                    {"range": [150, 300], "color": "rgba(220, 38, 38, 0.22)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 140.0},
            },
        ))
        fig_g2.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        fig_g3 = go.Figure(go.Indicator(
            mode="gauge+number", value=glacier_temp, number={"suffix": " °C"},
            title={"text": "Tsho Rolpa (4,580m) Temp", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 30], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": "#0F172A"},
                "steps": [
                    {"range": [0, 8], "color": "rgba(22, 163, 74, 0.18)"},
                    {"range": [8, 16], "color": "rgba(217, 119, 6, 0.18)"},
                    {"range": [16, 30], "color": "rgba(220, 38, 38, 0.22)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 15.0},
            },
        ))
        fig_g3.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g3, use_container_width=True)

    with g4:
        fig_g4 = go.Figure(go.Indicator(
            mode="gauge+number", value=active_inputs["soil_sat_pct"], number={"suffix": "%"},
            title={"text": "Vadose Deep Soil Saturation", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 60], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": "#0F172A"},
                "steps": [
                    {"range": [0, 25], "color": "rgba(22, 163, 74, 0.18)"},
                    {"range": [25, 40], "color": "rgba(217, 119, 6, 0.18)"},
                    {"range": [40, 60], "color": "rgba(220, 38, 38, 0.22)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 42.0},
            },
        ))
        fig_g4.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g4, use_container_width=True)

# ----------------- Tab 4: Corridor Station Data -----------------
with tab_table:
    st.markdown("<div class='kpi-label'>Multi-Station Historical Discharge & Sensor Vectors</div>", unsafe_allow_html=True)
    st.caption("Active data fed into the 58-feature hydrological inference engine.")

    st.dataframe(live_history_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='kpi-label' style='margin-top:14px;'>Current Corridor Station Input Snapshot</div>", unsafe_allow_html=True)
    df_active_snapshot = pd.DataFrame([{
        "Singati Target (m³/s)": active_inputs["q_mid"],
        "Singati Yesterday (m³/s)": active_inputs["q_yesterday"],
        "Tsho Rolpa Glacier (m³/s)": active_inputs["q_glacier"],
        "Gongar Dam Intake (m³/s)": active_inputs["q_upper"],
        "Lower Confluence (m³/s)": active_inputs["q_lower"],
        "Valley Rain (mm)": active_inputs["rain_mid"],
        "3-Day Rain (mm)": active_inputs["rain_3d"],
        "7-Day Rain (mm)": active_inputs["rain_7d"],
        "Soil Saturation (%)": active_inputs["soil_sat_pct"],
        "Glacier Temp (°C)": glacier_temp,
    }])
    st.dataframe(df_active_snapshot, use_container_width=True, hide_index=True)

# ----------------- Tab 5: Methodology & Emergency Contacts -----------------
with tab_method:
    st.markdown("<div class='kpi-label'>Life-Safety System Architecture & Model Limitations</div>", unsafe_allow_html=True)

    meth_col1, meth_col2 = st.columns(2)
    with meth_col1:
        st.markdown(
            f"""
            <div class="command-card" style="font-size: 0.88rem; line-height: 1.6;">
                <b style="color: {PALETTE['text_primary']};">🔬 58 Hydrological Features Pipeline:</b>
                <ul style="margin-top: 6px; padding-left: 18px; color: {PALETTE['text_secondary']};">
                    <li><b>Cascading Hydrodynamics:</b> Station flow gradients, velocity ratios, and upstream sums.</li>
                    <li><b>Temporal Acceleration:</b> 1st and 2nd order differencing plus percent surge change.</li>
                    <li><b>Vadose Runoff Physics:</b> Coupled non-linear infiltration $R = P_{{basin}} \\times (\\text{{Soil}}_{{deep}})^2$.</li>
                    <li><b>Glacial Melt Thermodynamics:</b> Lapse-rate corrected 4,500m positive degree-days.</li>
                    <li><b>Seasonality:</b> Dual-harmonic Fourier intra-annual curves.</li>
                </ul>
                <b style="color: {PALETTE['emergency']['color']};">⚠️ Known Model Uncertainty & Limitations:</b>
                <ul style="margin-top: 6px; padding-left: 18px; color: {PALETTE['text_secondary']};">
                    <li>Ensemble trained on 42,540 station-days. High uncertainty during unmodeled landslide dam outbursts (LLDOF) or sudden glacial moraine wall collapse.</li>
                    <li>Confidence bounds represent 80% coverage; extreme outliers beyond training envelope may exceed intervals.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with meth_col2:
        st.markdown(
            f"""
            <div class="command-card" style="font-size: 0.88rem; line-height: 1.6;">
                <b style="color: {PALETTE['emergency']['color']};">📞 Official Disaster Response Contacts (Dolakha):</b>
                <div style="margin-top: 8px;">
                    <div><b>Dolakha District Emergency Ops (DEOC):</b></div>
                    <code style="color:{PALETTE['text_primary']}; background:#F1F5F9; padding:2px 6px; border-radius:4px;">[PLACEHOLDER: Charikot DEOC VHF Ch 4 / +977-49-421111]</code>
                </div>
                <div style="margin-top: 8px;">
                    <div><b>Armed Police Force Disaster Unit (Singati):</b></div>
                    <code style="color:{PALETTE['text_primary']}; background:#F1F5F9; padding:2px 6px; border-radius:4px;">[PLACEHOLDER: APF Disaster Base / +977-49-421999]</code>
                </div>
                <div style="margin-top: 8px;">
                    <div><b>Upper Tamakoshi Hydropower Control:</b></div>
                    <code style="color:{PALETTE['text_primary']}; background:#F1F5F9; padding:2px 6px; border-radius:4px;">[PLACEHOLDER: Gongar Dam Control Room / +977-49-420000]</code>
                </div>
                <div style="margin-top: 8px;">
                    <div><b>National Flood Forecasting Toll-Free:</b></div>
                    <code style="color:{PALETTE['text_primary']}; background:#F1F5F9; padding:2px 6px; border-radius:4px;">[PLACEHOLDER: DHM Emergency Desk / Toll-Free 1155]</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
