import sys
import types
import os

# ---------------------------------------------------------
# Critical ML Compatibility Shim: _loss
# ---------------------------------------------------------
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
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from services.tamakoshi_api import (
    TAMAKOSHI_STATIONS,
    load_model_bundle,
    route_live_basin_telemetry,
    compute_forecast_from_inputs,
)

# ---------------------------------------------------------
# Page Configuration & Dark Slate Hydrology Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tamakoshi Himalayan Flood AI | Early Warning Command Center",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Load Pre-trained Multi-Model Stacking Ensemble
# ---------------------------------------------------------
bundle = load_model_bundle()
base_models = bundle["base_models"]
model_names = bundle["model_names"]
meta_learner = bundle["meta_learner"]
q_low_model = bundle["q_low_model"]
q_high_model = bundle["q_high_model"]
flood_clf = bundle["flood_clf"]
feature_cols = bundle["features"]
DANGER_THRESHOLD = float(bundle["threshold"])
YELLOW_THRESHOLD = DANGER_THRESHOLD * 0.90  # 6.02 m³/s

# ---------------------------------------------------------
# Custom Styling: Premium Hydrology Command Center Theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-main: #07111f;
        --bg-panel: rgba(12, 19, 31, 0.85);
        --bg-panel-strong: rgba(15, 23, 42, 0.96);
        --card-border: rgba(148, 163, 184, 0.18);
        --accent-cyan: #67e8f9;
        --accent-blue: #38bdf8;
        --accent-emerald: #2CE5B5;
        --accent-amber: #fbbf24;
        --accent-red: #f87171;
        --text-primary: #f8fafc;
        --text-secondary: #b8c3d6;
        --text-muted: #8ea3bf;
    }

    html, body {
        background: var(--bg-main);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(44, 229, 181, 0.08), transparent 22%),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.09), transparent 20%),
            linear-gradient(180deg, #07111f 0%, #0b1728 100%);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1540px;
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 13, 22, 0.94);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    .stSidebar .block-container {
        padding-top: 0.75rem;
    }

    [data-testid="stSidebarNav"] {
        background: rgba(15, 23, 42, 0.5);
    }

    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebarNav"] span {
        color: var(--text-secondary) !important;
    }

    .stSelectbox > div,
    .stNumberInput > div,
    .stSlider > div,
    .stCheckbox > label,
    .stRadio > div {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
    }

    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        color: var(--text-primary) !important;
        background: transparent !important;
    }

    .hero-container {
        position: relative;
        overflow: hidden;
        margin-bottom: 1.4rem;
        padding: 1.9rem 2rem 1.6rem 2rem;
        border-radius: 22px;
        border: 1px solid rgba(103, 232, 249, 0.28);
        background:
            linear-gradient(135deg, rgba(8, 15, 25, 0.72), rgba(7, 17, 31, 0.80)),
            url('https://images.unsplash.com/photo-1544735716-392fe2489ffa?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat;
        box-shadow: 0 24px 60px rgba(2, 6, 23, 0.6);
    }

    .hero-container::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(56, 189, 248, 0.08), rgba(44, 229, 181, 0.05), transparent 70%);
        pointer-events: none;
    }

    .hero-title,
    .hero-subtitle,
    .badge-pill-container {
        position: relative;
        z-index: 1;
    }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 0.65rem 0;
        font-size: clamp(1.7rem, 2.5vw, 2.6rem);
        font-weight: 800;
        letter-spacing: -0.05em;
        color: var(--text-primary);
        text-shadow: 0 4px 24px rgba(2, 6, 23, 0.8);
    }

    .hero-subtitle {
        max-width: 980px;
        font-size: 1.0rem;
        line-height: 1.7;
        color: rgba(226, 232, 240, 0.92);
        text-shadow: 0 2px 8px rgba(2, 6, 23, 0.75);
        margin-bottom: 1.15rem;
    }

    .badge-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.54rem 0.88rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #e2e8f0;
        backdrop-filter: blur(10px);
    }

    .badge-pill.cyan { border-color: rgba(103, 232, 249, 0.45); color: #67e8f9; }
    .badge-pill.emerald { border-color: rgba(44, 229, 181, 0.45); color: #2CE5B5; }
    .badge-pill.amber { border-color: rgba(251, 191, 36, 0.45); color: #fbbf24; }
    .badge-pill.red { border-color: rgba(248, 113, 113, 0.45); color: #f87171; }

    /* Home Page Big Navigation Cards */
    .home-card-box {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(12, 19, 31, 0.90));
        border: 1px solid rgba(103, 232, 249, 0.25);
        border-radius: 22px;
        padding: 1.8rem 1.6rem 1.5rem 1.6rem;
        box-shadow: 0 20px 45px rgba(2, 6, 23, 0.38);
        backdrop-filter: blur(16px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 0.8rem;
        min-height: 380px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .home-card-box:hover {
        transform: translateY(-4px);
        border-color: rgba(44, 229, 181, 0.60);
        box-shadow: 0 26px 60px rgba(44, 229, 181, 0.18);
    }

    .home-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }

    .home-card-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }

    .home-card-subtitle {
        font-size: 0.86rem;
        color: #67e8f9;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .home-card-desc {
        font-size: 0.92rem;
        line-height: 1.65;
        color: #b8c3d6;
        margin-bottom: 1.2rem;
    }

    .home-card-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 1.2rem;
    }

    .home-card-tag {
        font-size: 0.72rem;
        padding: 0.35rem 0.65rem;
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #94a3b8;
        font-weight: 600;
    }

    /* Top Switch Bar */
    .switch-bar-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        padding: 0.85rem 1.2rem;
        margin-bottom: 1.2rem;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.94), rgba(8, 17, 31, 0.90));
        border: 1px solid rgba(103, 232, 249, 0.24);
        box-shadow: 0 12px 28px rgba(2, 6, 23, 0.25);
    }

    .switch-bar-text {
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    .switch-bar-text b {
        color: #67e8f9;
    }

    .glass-card {
        position: relative;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.72));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 18px 40px rgba(2, 6, 23, 0.28);
        backdrop-filter: blur(14px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(103, 232, 249, 0.42);
        box-shadow: 0 24px 50px rgba(14, 165, 233, 0.10);
    }

    .glass-card .label {
        margin-bottom: 0.45rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .glass-card .value {
        font-size: clamp(1.4rem, 2vw, 1.95rem);
        font-weight: 800;
        line-height: 1.2;
        color: var(--text-primary);
    }

    .glass-card .delta {
        margin-top: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .action-banner {
        border-radius: 18px;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1.4rem;
        border: 1px solid;
        box-shadow: 0 18px 45px rgba(2, 6, 23, 0.26);
        backdrop-filter: blur(14px);
    }

    .action-banner.red {
        background: linear-gradient(180deg, rgba(127, 29, 29, 0.28), rgba(69, 10, 10, 0.13));
        border-color: rgba(248, 113, 113, 0.5);
    }

    .action-banner.yellow {
        background: linear-gradient(180deg, rgba(120, 53, 15, 0.28), rgba(91, 40, 10, 0.15));
        border-color: rgba(251, 191, 36, 0.5);
    }

    .action-banner.green {
        background: linear-gradient(180deg, rgba(6, 78, 59, 0.22), rgba(6, 78, 59, 0.12));
        border-color: rgba(52, 211, 153, 0.5);
    }

    .banner-title {
        margin-bottom: 0.65rem;
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .banner-text {
        color: var(--text-primary);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .protocol-list {
        margin-top: 0.85rem;
        padding-left: 1.15rem;
        color: rgba(226, 232, 240, 0.9);
        line-height: 1.65;
    }

    .protocol-list li {
        margin-bottom: 0.15rem;
    }

    h3 {
        margin-top: 1.6rem !important;
        margin-bottom: 0.25rem !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em !important;
    }

    .stCaption {
        color: var(--text-secondary) !important;
        margin-bottom: 0.75rem !important;
    }

    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }

    .stTabs [role="tab"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px 12px 0 0;
        color: var(--text-secondary);
        padding: 0.55rem 0.95rem;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, rgba(44, 229, 181, 0.17), rgba(15, 23, 42, 0.8));
        color: var(--text-primary);
        border-color: rgba(44, 229, 181, 0.45);
    }

    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.48rem 0.85rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        backdrop-filter: blur(10px);
    }

    .source-chip.live {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(44, 229, 181, 0.5);
        color: #2CE5B5;
    }

    .source-chip.fallback {
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid rgba(251, 191, 36, 0.5);
        color: #FBBF24;
    }

    .stApp > div > div {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Navigation Routing State
# ---------------------------------------------------------
PAGE_HOME = "home"
PAGE_SIM = "simulation"
PAGE_LIVE = "live_telemetry"

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGE_HOME

# ---------------------------------------------------------
# Sidebar: Navigation Router & Dynamic Controls
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='
        padding: 1rem 0.9rem 0.85rem 0.9rem;
        margin: 0.2rem 0 1.1rem 0;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(8, 13, 22, 0.90));
        border: 1px solid rgba(44, 229, 181, 0.28);
        box-shadow: 0 14px 28px rgba(2, 6, 23, 0.32);
        text-align: center;
    '>
        <div style='font-size: 0.70rem; letter-spacing: 0.14em; text-transform: uppercase; color: #2CE5B5; font-weight: 700; margin-bottom: 0.3rem;'>Himalayan Basin AI</div>
        <h2 style='margin: 0; font-size: 1.35rem; color: #F8FAFC; line-height: 1.2;'>🏔️ Tamakoshi AI</h2>
        <div style='margin-top: 0.3rem; font-size: 0.78rem; color: #9fb4cf;'>Dolakha Early Warning System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGE_OPTIONS = {
    PAGE_HOME: "🏠 Home Command Center",
    PAGE_SIM: "🎛️ Manual Simulation (Input & Predict)",
    PAGE_LIVE: "📡 Real-Time Live Telemetry (Auto Fetch)",
}
PAGE_KEYS = list(PAGE_OPTIONS.keys())
PAGE_LABELS = list(PAGE_OPTIONS.values())

curr_idx = PAGE_KEYS.index(st.session_state.current_page) if st.session_state.current_page in PAGE_KEYS else 0

selected_label = st.sidebar.radio(
    "🧭 Select Tamakoshi Mode:",
    PAGE_LABELS,
    index=curr_idx,
    key="global_nav_radio",
)
target_key = PAGE_KEYS[PAGE_LABELS.index(selected_label)]
if target_key != st.session_state.current_page:
    st.session_state.current_page = target_key
    st.rerun()

st.sidebar.markdown("<hr style='border-color: rgba(148, 163, 184, 0.15); margin: 0.9rem 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Calibrated Watershed Scenarios (Tamakoshi Model)
# ---------------------------------------------------------
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
    "⚙️ Custom Simulation": None,
}

# Sidebar inputs specific to Manual Simulation Mode
if st.session_state.current_page == PAGE_SIM:
    st.sidebar.markdown("### 🎛️ Simulation Controls & Scenarios")
    choice = st.sidebar.selectbox(
        "Select Calibrated Watershed Scenario:",
        list(SCENARIOS.keys()),
        index=0,
    )
    defaults = SCENARIOS[choice] or (
        6.60, 5.55, 1.15, 4.35, 9.80, 58.0, 42.0, 135.0, 230.0, 3, 48, 24.0, 15.5, 0.0, 22.5, 8
    )

    with st.sidebar.expander("🌊 1. Station Discharge Sensors (m³/s)", expanded=True):
        q_mid = st.number_input("Singati Target Flow Today (m³/s)", value=float(defaults[0]), step=0.1)
        q_yesterday = st.number_input("Singati Flow Yesterday (m³/s)", value=float(defaults[1]), step=0.1)
        q_glacier = st.number_input("Tsho Rolpa Glacier Outflow (m³/s)", value=float(defaults[2]), step=0.05)
        q_upper = st.number_input("Upper Tamakoshi / Gongar Flow (m³/s)", value=float(defaults[3]), step=0.1)
        q_lower = st.number_input("Lower Tamakoshi Confluence (m³/s)", value=float(defaults[4]), step=0.1)

    with st.sidebar.expander("🌧️ 2. Monsoon Cloudburst & Rain Dynamics", expanded=False):
        rain_mid = st.number_input("Valley Rainfall Today (mm)", value=float(defaults[5]), step=1.0)
        rain_glacier = st.number_input("Glacier Headwaters Rain Today (mm)", value=float(defaults[6]), step=1.0)
        rain_3d = st.number_input("3-Day Basin Rain (mm)", value=float(defaults[7]), step=5.0)
        rain_7d = st.number_input("7-Day Basin Rain (mm)", value=float(defaults[8]), step=5.0)
        rain_hours = st.slider("Rain Duration (Hours — lower = cloudburst)", 0, 24, int(defaults[9]))

    with st.sidebar.expander("🏔️ 3. Soil Saturation & Glacial Melt Physics", expanded=False):
        soil_sat_pct = st.slider("Deep Soil Saturation (%)", 10, 50, int(defaults[10]))
        tmax = st.number_input("Valley Max Temp (°C)", value=float(defaults[11]), step=0.5)
        tmin = st.number_input("Valley Min Temp (°C)", value=float(defaults[12]), step=0.5)
        snow_glacier = st.number_input("Glacier Snowfall Today (cm)", value=float(defaults[13]), step=0.5)
        solar_rad = st.number_input("Solar Radiation (MJ/m²)", value=float(defaults[14]), step=0.5)
        month = st.slider("Month of Year (1=Jan, 8=Aug)", 1, 12, int(defaults[15]))

elif st.session_state.current_page == PAGE_LIVE:
    st.sidebar.markdown("### 📡 Live Feed Settings")
    past_days_live = st.sidebar.slider("📅 Historical Lookback (days)", 3, 14, 7, help="Days of past GloFAS discharge to fetch.")
    st.sidebar.markdown(
        """
        <div style='font-size: 0.78rem; color: #8ea3bf; line-height: 1.6; padding: 0.4rem 0.2rem;'>
            <b style='color: #67e8f9;'>Live Sources:</b><br>
            • <code>flood-api.open-meteo.com</code> (GloFAS 4 stations)<br>
            • <code>api.open-meteo.com</code> (Singati basin weather)<br><br>
            <b style='color: #2CE5B5;'>Failover:</b> Verified Nov 10–16, 2026 baseline if API unreachable.
        </div>
        """,
        unsafe_allow_html=True,
    )
    defaults = SCENARIOS["🔴 Scenario 3: Cloudburst + Glacial Melt (Red Alert)"]
    q_mid, q_yesterday, q_glacier, q_upper, q_lower = defaults[0:5]
    rain_mid, rain_glacier, rain_3d, rain_7d, rain_hours = defaults[5:10]
    soil_sat_pct, tmax, tmin, snow_glacier, solar_rad, month = defaults[10:16]

else:  # Home page
    st.sidebar.markdown("### 💡 Quick Guide")
    st.sidebar.info(
        "**Tamakoshi River Flood AI Command Center**\n\n"
        "• **Card 1: Manual Simulation** allows you to input custom sensor data (or select calibrated scenarios) and immediately view the 3D digital twin and AI predictions.\n\n"
        "• **Card 2: Real-Time Live Telemetry** automatically fetches live GloFAS river discharge and Open-Meteo weather for all 4 Tamakoshi stations to generate live early warnings."
    )
    defaults = SCENARIOS["🔴 Scenario 3: Cloudburst + Glacial Melt (Red Alert)"]
    q_mid, q_yesterday, q_glacier, q_upper, q_lower = defaults[0:5]
    rain_mid, rain_glacier, rain_3d, rain_7d, rain_hours = defaults[5:10]
    soil_sat_pct, tmax, tmin, snow_glacier, solar_rad, month = defaults[10:16]

# ---------------------------------------------------------
# Exact 58-Feature Engineering & Inference for Simulation Mode
# ---------------------------------------------------------
sim_input_dict = {
    "q_mid": q_mid, "q_yesterday": q_yesterday, "q_glacier": q_glacier,
    "q_upper": q_upper, "q_lower": q_lower, "rain_mid": rain_mid,
    "rain_glacier": rain_glacier, "rain_3d": rain_3d, "rain_7d": rain_7d,
    "rain_hours": rain_hours, "soil_sat_pct": soil_sat_pct, "tmax": tmax,
    "tmin": tmin, "snow_glacier": snow_glacier, "solar_rad": solar_rad, "month": month,
}
sim_forecast = compute_forecast_from_inputs(bundle, sim_input_dict)

pred_flow = sim_forecast["pred_flow"]
delta_pred = sim_forecast["delta_pred"]
low_flow = sim_forecast["low_flow"]
high_flow = sim_forecast["high_flow"]
flood_prob = sim_forecast["flood_prob"]
sim_alert_level = sim_forecast["alert_level"]
sim_individual_preds = sim_forecast["individual_preds"]

is_red = sim_alert_level == "RED_ALERT"
is_yellow = sim_alert_level == "YELLOW_WATCH"
is_green = sim_alert_level == "GREEN_NORMAL"

if is_red:
    alert_tier = "RED ALERT"
    alert_color = "#EF4444"
    alert_icon = "🚨"
elif is_yellow:
    alert_tier = "YELLOW WATCH"
    alert_color = "#F59E0B"
    alert_icon = "⚠️"
else:
    alert_tier = "GREEN NORMAL"
    alert_color = "#10B981"
    alert_icon = "✅"


# =============================================================================
# VIEW 1: HOME PAGE (2 Interactive Mode Cards)
# =============================================================================
if st.session_state.current_page == PAGE_HOME:
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">
                <span>🏔️ TAMAKOSHI HIMALAYAN FLOOD AI COMMAND CENTER</span>
            </div>
            <div class="hero-subtitle">
                Compound Glacial-Monsoon Hydrological Intelligence & Early Warning System for the Dolakha Catchment, Nepal.
                Monitoring the high-altitude gradient from Tsho Rolpa Glacier (4,580m) past the Upper Tamakoshi Dam (1,990m) down to Singati Gorge (1,250m).
            </div>
            <div class="badge-pill-container">
                <span class="badge-pill cyan">🏔️ Dolakha Corridor: 4 Cascading Stations</span>
                <span class="badge-pill cyan">⚡ Upper Tamakoshi Dam: 456 MW Spillway</span>
                <span class="badge-pill cyan">🎯 Forecast Target: Singati Gorge (1,250m ASL)</span>
                <span class="badge-pill emerald">🧠 6-Model AI Stacking Ensemble (42,540 Station-Days)</span>
                <span class="badge-pill red">🚨 Danger Threshold: 6.69 m³/s (95th Percentile)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧭 Select Exploration Mode")
    st.caption("Choose between the manual parameter simulation engine or the automated real-time API telemetry pipeline.")

    col_card1, col_card2 = st.columns(2)

    with col_card1:
        st.markdown(
            f"""
            <div class="home-card-box">
                <div>
                    <div class="home-card-header">
                        <span class="badge-pill cyan" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                            🎛️ PARAMETER INPUT & SIMULATION
                        </span>
                        <span style="font-size: 1.8rem;">🎛️</span>
                    </div>
                    <div class="home-card-title">1. Manual Simulation & What-If Engine</div>
                    <div class="home-card-subtitle">
                        Input Custom Sensor Data & Show Immediate AI Prediction
                    </div>
                    <div class="home-card-desc">
                        Directly adjust river station discharge sensors, monsoon cloudburst rainfall, vadose soil saturation, 
                        and glacial melt temperature anomalies. Run calibrated flood scenarios and inspect the 
                        <b>interactive 3D Singati gorge fluid digital twin</b>, 80% confidence hydrographs, and emergency evacuation protocols.
                    </div>
                </div>
                <div>
                    <div class="home-card-tags">
                        <span class="home-card-tag">🌊 3D Gorge Fluid Twin</span>
                        <span class="home-card-tag">🎛️ 15 Sensor Sliders</span>
                        <span class="home-card-tag">🧠 6-Model Stacking Ensemble</span>
                        <span class="home-card-tag">🚨 3 Calibrated Scenarios</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🎛️ Open Manual Simulation Mode ➔", key="btn_home_to_sim", use_container_width=True):
            st.session_state.current_page = PAGE_SIM
            st.rerun()

    with col_card2:
        st.markdown(
            """
            <div class="home-card-box">
                <div>
                    <div class="home-card-header">
                        <span class="badge-pill emerald" style="font-size: 0.72rem; padding: 0.35rem 0.75rem;">
                            📡 REAL-TIME LIVE DATA FEED
                        </span>
                        <span style="font-size: 1.8rem;">📡</span>
                    </div>
                    <div class="home-card-title">2. Real-Time Live Telemetry & 24h Prediction</div>
                    <div class="home-card-subtitle" style="color: #2CE5B5;">
                        Automated Data Fetching & Live Result Display
                    </div>
                    <div class="home-card-desc">
                        Pulls live river discharge for all 4 Tamakoshi stations directly from <b>Open-Meteo GloFAS Flood API</b> 
                        and basin weather (rain, soil, temperature, solar flux) from Open-Meteo Weather API. 
                        Engineers the 58-feature physics pipeline and automatically computes a <b>live 24-hour flood forecast</b> with verified failover.
                    </div>
                </div>
                <div>
                    <div class="home-card-tags">
                        <span class="home-card-tag">🌐 GloFAS River API (4 Stations)</span>
                        <span class="home-card-tag">🛰️ Basin Weather Telemetry</span>
                        <span class="home-card-tag">📈 Live 7-Day Hydrograph</span>
                        <span class="home-card-tag">⚡ Automated AI Inference</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📡 Launch Real-Time Live Telemetry ➔", key="btn_home_to_live", use_container_width=True):
            st.session_state.current_page = PAGE_LIVE
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Platform Overview KPIs
    st.markdown("### 📊 Tamakoshi Corridor Hydrological Architecture")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Forecast Tomorrow Flow (Sim)</div>
                <div class="value" style="color: {alert_color};">{pred_flow:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: {alert_color};">Status: {alert_tier}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="label">Tsho Rolpa Headwaters</div>
                <div class="value">4,580 <span style="font-size:1.1rem; color:#94A3B8;">m ASL</span></div>
                <div class="delta" style="color: #2CE5B5;">Glacial Sentinel Station</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Singati Gorge Danger Limit</div>
                <div class="value">{DANGER_THRESHOLD:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #F59E0B;">Watch Trigger: {YELLOW_THRESHOLD:.2f} m³/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            """
            <div class="glass-card">
                <div class="label">Stacking AI Ensemble</div>
                <div class="value">6 <span style="font-size:1.1rem; color:#94A3B8;">Base Models</span></div>
                <div class="delta" style="color: #38bdf8;">RidgeCV Meta-Regressor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Station Corridor Summary Table
    st.markdown("### 🏔️ Tamakoshi River Cascading Corridor Stations")
    st.caption("Cascading telemetry stations along the Tamakoshi river from alpine glacial source down to lower confluence.")

    df_tam_corridor = pd.DataFrame([
        {"Station Name": "1. Tsho Rolpa Glacier Lake", "Elevation (m)": 4580, "Coordinates": "27.8617°N, 86.4769°E", "Role": "Glacial Lake Outflow Headwaters", "Hazard Type": "GLOF (Glacial Lake Outburst Flood)"},
        {"Station Name": "2. Gongar / Upper Tamakoshi Dam", "Elevation (m)": 1990, "Coordinates": "27.8442°N, 86.2205°E", "Role": "456 MW Dam & Radial Spillway Intake", "Hazard Type": "Reservoir Surcharge & Penstock Hydro-peak"},
        {"Station Name": "3. Singati Gorge (Target)", "Elevation (m)": 1250, "Coordinates": "27.7314°N, 86.1628°E", "Role": "Bottleneck Canyon & Primary Settlement", "Hazard Type": "Rapid Gorge Inundation & Bank Breach"},
        {"Station Name": "4. Lower Tamakoshi Confluence", "Elevation (m)": 880, "Coordinates": "27.5250°N, 86.0750°E", "Role": "Downstream Drainage & Sun Koshi Confluence", "Hazard Type": "Basin Discharge & Flood Surge Ingress"},
    ])
    st.dataframe(df_tam_corridor, use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 2: MANUAL SIMULATION (Input Data & Show Results)
# =============================================================================
elif st.session_state.current_page == PAGE_SIM:
    # Switch Bar
    st.markdown(
        """
        <div class="switch-bar-container">
            <div class="switch-bar-text">
                📍 You are viewing <b>Page 1: Manual Simulation (Input Data & Predict)</b>. Want autonomous live API data fetching?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sb_c1, sb_c2 = st.columns([1, 1])
    with sb_c1:
        if st.button("🏠 Back to Home", key="btn_sim_back_home", use_container_width=True):
            st.session_state.current_page = PAGE_HOME
            st.rerun()
    with sb_c2:
        if st.button("📡 Switch to Real-Time Live Telemetry ➔", key="btn_sim_to_live", use_container_width=True):
            st.session_state.current_page = PAGE_LIVE
            st.rerun()

    # Hero Banner
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">
                <span>🎛️ TAMAKOSHI WATERSHED SIMULATION & WHAT-IF ENGINE</span>
            </div>
            <div class="hero-subtitle">
                Interactive parameter simulation mode. Modify station flows, cloudburst intensities, and vadose soil moisture
                via the sidebar to evaluate hydraulic responses, confidence intervals, and 3D digital twin water elevation.
            </div>
            <div class="badge-pill-container">
                <span class="badge-pill cyan">🎛️ Scenario: {choice}</span>
                <span class="badge-pill cyan">🎯 Target: Singati Gorge (1,250m)</span>
                <span class="badge-pill cyan">🧠 6-Model AI Ensemble Active</span>
                <span class="badge-pill {'red' if is_red else 'amber' if is_yellow else 'emerald'}">
                    {alert_icon} Status: {alert_tier} ({pred_flow:.2f} m³/s)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Emergency Action & Decision Support Banner
    if is_red:
        st.markdown(
            f"""
            <div class="action-banner red">
                <div class="banner-title" style="color: #FCA5A5;">
                    🚨 RED ALERT — CRITICAL FLOOD & OVERTOPPING BREACH IMMINENT
                </div>
                <div class="banner-text">
                    Predicted 24-hour discharge reaches <b>{pred_flow:.2f} m³/s</b> (Surge: {delta_pred:+.2f} m³/s), exceeding the 95th-percentile riverbank safety limit of <b>{DANGER_THRESHOLD:.2f} m³/s</b> by <b>{(pred_flow - DANGER_THRESHOLD):+.2f} m³/s</b>. AI Inundation Likelihood: <b>{flood_prob:.1f}%</b>.
                </div>
                <ul class="protocol-list">
                    <li><b>Immediate Evacuation Dispatch:</b> Broadcast automated CAP/SMS alarms to riverside settlements across Singati Bazaar, Gongar, Jagat, and Lamabagar.</li>
                    <li><b>Upper Tamakoshi Dam Control:</b> Actuate radial spillway gates to regulate buffer reservoir drawdown and mitigate hydro-peak downstream.</li>
                    <li><b>Tactical First-Responders:</b> Mobilize Dolakha District Disaster Management Committee (DDMC) and APF water rescue units along Singati-Charikot corridor.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif is_yellow:
        st.markdown(
            f"""
            <div class="action-banner yellow">
                <div class="banner-title" style="color: #FCD34D;">
                    ⚠️ YELLOW WATCH — RAPID MONSOON SURGE IN UPPER CATCHMENT
                </div>
                <div class="banner-text">
                    Predicted tomorrow flow reaches <b>{pred_flow:.2f} m³/s</b>, entering the 10% safety buffer zone (<b>{YELLOW_THRESHOLD:.2f} – {DANGER_THRESHOLD:.2f} m³/s</b>). Compound rainfall and glacial runoff require active watch.
                </div>
                <ul class="protocol-list">
                    <li><b>DDMC Alert Standby:</b> Notify Dolakha disaster coordinators and riverbank construction crews to clear low-elevation floodplains.</li>
                    <li><b>High-Frequency Telemetry:</b> Switch upstream acoustic river sensors at Tsho Rolpa and Gongar to 15-minute Doppler radar polling.</li>
                    <li><b>Hydropower Grid Coordination:</b> Alert Nepal Electricity Authority (NEA) load dispatch center for potential reservoir spill operations.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="action-banner green">
                <div class="banner-title" style="color: #6EE7B7;">
                    ✅ GREEN NORMAL — SAFE HYDROLOGICAL EQUILIBRIUM
                </div>
                <div class="banner-text">
                    Watershed discharge is safely contained within natural gorge banks at <b>{pred_flow:.2f} m³/s</b> (Clearance to Danger Line: <b>{(DANGER_THRESHOLD - pred_flow):.2f} m³/s</b>). AI Breach Probability is <b>{flood_prob:.1f}%</b>.
                </div>
                <ul class="protocol-list">
                    <li><b>Operational Dispatch:</b> Maintain regular 456 MW power generation dispatch at Upper Tamakoshi Hydropower Project.</li>
                    <li><b>Continuous Sentinel Surveillance:</b> Routine autonomous model inference on incoming satellite precipitation and acoustic sensors.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Glassmorphism Live KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Forecast Tomorrow Flow</div>
                <div class="value" style="color: {alert_color};">{pred_flow:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: {'#EF4444' if delta_pred > 0 else '#10B981'};">
                    {delta_pred:+.2f} m³/s vs today ({q_mid:.2f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        ci_range = high_flow - low_flow
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">80% Confidence Interval</div>
                <div class="value">{low_flow:.2f} – {high_flow:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #38BDF8;">Uncertainty band: ±{ci_range / 2.0:.2f} m³/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Danger / Watch Threshold</div>
                <div class="value">{DANGER_THRESHOLD:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #F59E0B;">Yellow Buffer: {YELLOW_THRESHOLD:.2f} m³/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">AI Flood Probability</div>
                <div class="value" style="color: {alert_color};">{flood_prob:.1f}<span style="font-size:1.2rem; color:#94A3B8;">%</span></div>
                <div class="delta" style="color: {alert_color};">Status: {alert_tier}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 3D Himalayan Terrain & Watershed Map
    st.markdown("### 🗺️ 3D Himalayan Terrain & Cascading Telemetry Map")
    st.caption(
        "Interactive 3D Deck.gl terrain view centered on Dolakha, Nepal (Pitch: 55°, Bearing: -20°). "
        "Extruded 3D columns and river corridor arcs dynamically scale with station discharge."
    )

    def get_sim_station_color(flow_val, is_target=False):
        effective = pred_flow if is_target else flow_val
        if effective >= DANGER_THRESHOLD:
            return [239, 68, 68, 240]
        elif effective >= YELLOW_THRESHOLD:
            return [245, 158, 11, 230]
        else:
            return [14, 165, 233, 215]

    sim_stations_info = [
        {
            "id": "tsho_rolpa",
            "name": "Tsho Rolpa Glacial Lake",
            "elevation_m": 4580,
            "lat": 27.8617,
            "lon": 86.4769,
            "flow": q_glacier,
            "role": "Glacial Lake Outflow Headwaters",
            "status": "GLOF Risk Sector" if q_glacier > 0.8 else "Nominal Outflow",
            "color": get_sim_station_color(q_glacier),
            "height": max(300, q_glacier * 1800 + 400),
        },
        {
            "id": "gongar_dam",
            "name": "Upper Tamakoshi Dam (Gongar)",
            "elevation_m": 1990,
            "lat": 27.8442,
            "lon": 86.2205,
            "flow": q_upper,
            "role": "456 MW Dam & Spillway Intake",
            "status": "Spillway Active" if q_upper > 4.0 else "Controlled Reservoir",
            "color": get_sim_station_color(q_upper),
            "height": max(300, q_upper * 1200 + 400),
        },
        {
            "id": "singati_gorge",
            "name": "Singati Target Station",
            "elevation_m": 1250,
            "lat": 27.7314,
            "lon": 86.1628,
            "flow": pred_flow,
            "role": "Narrow Gorge & Floodplain Inundation Hub",
            "status": f"AI Forecast Target: {alert_tier}",
            "color": get_sim_station_color(pred_flow, is_target=True),
            "height": max(300, pred_flow * 1400 + 600),
        },
        {
            "id": "lower_tamakoshi",
            "name": "Lower Tamakoshi Confluence",
            "elevation_m": 880,
            "lat": 27.5250,
            "lon": 86.0750,
            "flow": q_lower,
            "role": "Downstream Drainage & Confluence",
            "status": "Surge Recipient" if q_lower > 7.0 else "Clear Drainage",
            "color": get_sim_station_color(q_lower),
            "height": max(300, q_lower * 800 + 300),
        },
    ]

    df_sim_stations = pd.DataFrame(sim_stations_info)

    sim_corridor_arcs = [
        {"from_lon": 86.4769, "from_lat": 27.8617, "to_lon": 86.2205, "to_lat": 27.8442, "flow": q_glacier, "name": "Glacier to Dam"},
        {"from_lon": 86.2205, "from_lat": 27.8442, "to_lon": 86.1628, "to_lat": 27.7314, "flow": q_upper, "name": "Dam to Singati"},
        {"from_lon": 86.1628, "from_lat": 27.7314, "to_lon": 86.0750, "to_lat": 27.5250, "flow": pred_flow, "name": "Singati to Lower Catchment"},
    ]
    df_sim_arcs = pd.DataFrame(sim_corridor_arcs)

    def sim_arc_color(val):
        if val >= DANGER_THRESHOLD:
            return [239, 68, 68, 220]
        elif val >= YELLOW_THRESHOLD:
            return [245, 158, 11, 200]
        else:
            return [56, 189, 248, 180]

    df_sim_arcs["color"] = df_sim_arcs["flow"].apply(sim_arc_color)

    sim_view_state = pdk.ViewState(
        latitude=27.72,
        longitude=86.24,
        zoom=10.0,
        pitch=52,
        bearing=-18,
    )

    sim_column_layer = pdk.Layer(
        "ColumnLayer",
        data=df_sim_stations,
        get_position=["lon", "lat"],
        get_elevation="height",
        elevation_scale=1,
        radius=500,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    sim_arc_layer = pdk.Layer(
        "ArcLayer",
        data=df_sim_arcs,
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_source_color="color",
        get_target_color="color",
        get_width=5.0,
        pickable=True,
    )

    sim_deck = pdk.Deck(
        layers=[sim_arc_layer, sim_column_layer],
        initial_view_state=sim_view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={
            "html": """
            <div style="font-family: -apple-system, sans-serif; padding: 10px 12px; background: rgba(8, 15, 28, 0.95); border: 1px solid rgba(103, 232, 249, 0.4); border-radius: 12px; color: #fff; min-width: 200px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
                <div style="font-weight: 800; font-size: 1.05rem; color: #67e8f9; margin-bottom: 4px;">{name}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px;">Elevation: <b>{elevation_m}m ASL</b></div>
                <div style="font-size: 0.85rem; margin-bottom: 4px;">Discharge Flow: <b style="color:#38BDF8;">{flow} m³/s</b></div>
                <div style="font-size: 0.78rem; color: #E2E8F0;">{role}</div>
                <div style="margin-top: 6px; font-size: 0.75rem; font-weight:700; color: #FCD34D;">{status}</div>
            </div>
            """,
            "style": {"color": "white"},
        },
    )
    st.pydeck_chart(sim_deck)

    # Interactive 3D Singati River Gorge Digital Twin
    st.markdown("### 🌊 Interactive 3D Singati River Gorge & Spillway Digital Twin")
    st.caption(
        "Cinematic 3D WebGL Digital Twin of the Singati Canyon and Upper Tamakoshi Spillway. "
        "Rotate, pan, and zoom to inspect canyon walls, dynamic water elevation, and glowing warning planes."
    )

    tab_3d, tab_plotly = st.tabs(["⚡ 3D WebGL Digital Twin (Three.js)", "📊 3D Plotly Surface & Breach Contour"])

    with tab_3d:
        freeboard = DANGER_THRESHOLD - pred_flow
        if freeboard < 0:
            breach_badge_html = f"<span style='color: #EF4444; font-weight: 800;'>⚠️ OVERTOPPING (+{abs(freeboard):.2f} m³/s)</span>"
        elif freeboard < 0.67:
            breach_badge_html = "<span style='color: #F59E0B; font-weight: 800;'>🟡 YELLOW BUFFER ZONE</span>"
        else:
            breach_badge_html = "<span style='color: #10B981; font-weight: 800;'>✅ SAFE CHANNEL STORAGE</span>"

        three_js_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    margin: 0;
                    overflow: hidden;
                    background: #070D18;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                }}
                #canvas-container {{
                    width: 100vw;
                    height: 520px;
                    position: relative;
                }}
                #hud {{
                    position: absolute;
                    top: 14px;
                    left: 16px;
                    background: rgba(10, 18, 32, 0.85);
                    border: 1px solid rgba(103, 232, 249, 0.3);
                    border-radius: 12px;
                    padding: 10px 14px;
                    color: #E2E8F0;
                    font-size: 0.8rem;
                    pointer-events: none;
                    backdrop-filter: blur(8px);
                    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
                }}
                #hud .title {{
                    font-weight: 800;
                    font-size: 0.92rem;
                    color: #67E8F9;
                    margin-bottom: 4px;
                }}
                #controls-hint {{
                    position: absolute;
                    bottom: 12px;
                    right: 16px;
                    background: rgba(10, 18, 32, 0.7);
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    border-radius: 8px;
                    padding: 6px 10px;
                    color: #94A3B8;
                    font-size: 0.72rem;
                    pointer-events: none;
                }}
            </style>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        </head>
        <body>
            <div id="canvas-container">
                <div id="hud">
                    <div class="title">🏔️ Singati Gorge Digital Twin — Live Hydrometric Stage</div>
                    <div>Forecast Discharge: <b style="color:{alert_color};">{pred_flow:.2f} m³/s</b> ({alert_tier})</div>
                    <div>Gorge Channel Freeboard: {breach_badge_html}</div>
                    <div>Red Danger Limit: <b style="color:#EF4444;">{DANGER_THRESHOLD:.2f} m³/s</b> | Watch Buffer: <b style="color:#F59E0B;">{YELLOW_THRESHOLD:.2f} m³/s</b></div>
                </div>
                <div id="controls-hint">🖱️ Left-Click + Drag: Rotate | Right-Click: Pan | Scroll: Zoom</div>
            </div>

            <script>
                const container = document.getElementById('canvas-container');
                const width = container.clientWidth;
                const height = 520;

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x070D18);
                scene.fog = new THREE.FogExp2(0x070D18, 0.015);

                const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                camera.position.set(24, 18, 34);

                const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                container.appendChild(renderer.domElement);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.06;
                controls.maxPolarAngle = Math.PI / 2 - 0.02;
                controls.target.set(0, 2, 0);

                const ambientLight = new THREE.AmbientLight(0x38BDF8, 0.45);
                scene.add(ambientLight);

                const dirLight = new THREE.DirectionalLight(0xFFFFFF, 1.1);
                dirLight.position.set(30, 45, 20);
                dirLight.castShadow = true;
                scene.add(dirLight);

                const alertLight = new THREE.PointLight('{alert_color}', 2.2, 50);
                alertLight.position.set(0, 10, 0);
                scene.add(alertLight);

                const canyonLength = 60;
                const canyonWidth = 32;
                const segX = 70;
                const segZ = 70;

                const terrainGeo = new THREE.PlaneGeometry(canyonWidth, canyonLength, segX, segZ);
                terrainGeo.rotateX(-Math.PI / 2);

                const pos = terrainGeo.attributes.position;
                for (let i = 0; i < pos.count; i++) {{
                    const x = pos.getX(i);
                    const z = pos.getZ(i);
                    const distFromCenter = Math.abs(x);
                    let y = Math.pow(distFromCenter / 4.2, 2.1) - 1.5;
                    y += Math.sin(x * 0.45 + z * 0.25) * 0.85;
                    y += Math.cos(z * 0.35) * 0.65;
                    pos.setY(i, y);
                }}
                terrainGeo.computeVertexNormals();

                const terrainMat = new THREE.MeshStandardMaterial({{
                    color: 0x1E293B,
                    roughness: 0.85,
                    metalness: 0.15,
                    flatShading: true,
                }});
                const terrainMesh = new THREE.Mesh(terrainGeo, terrainMat);
                terrainMesh.receiveShadow = true;
                scene.add(terrainMesh);

                const waterFlow = {pred_flow};
                const dangerFlow = {DANGER_THRESHOLD};
                const waterY = -1.2 + (waterFlow / dangerFlow) * 3.8;

                const waterGeo = new THREE.PlaneGeometry(14, canyonLength, 40, 40);
                waterGeo.rotateX(-Math.PI / 2);

                const waterColor = { '0xEF4444' if is_red else '0xF59E0B' if is_yellow else '0x0284C7' };
                const waterMat = new THREE.MeshStandardMaterial({{
                    color: waterColor,
                    roughness: 0.15,
                    metalness: 0.85,
                    transparent: true,
                    opacity: 0.88,
                }});
                const waterMesh = new THREE.Mesh(waterGeo, waterMat);
                waterMesh.position.y = waterY;
                scene.add(waterMesh);

                const yellowPlaneGeo = new THREE.PlaneGeometry(16, canyonLength);
                yellowPlaneGeo.rotateX(-Math.PI / 2);
                const yellowPlaneMat = new THREE.MeshBasicMaterial({{
                    color: 0xF59E0B,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.35,
                }});
                const yellowPlane = new THREE.Mesh(yellowPlaneGeo, yellowPlaneMat);
                yellowPlane.position.y = -1.2 + ({YELLOW_THRESHOLD} / dangerFlow) * 3.8;
                scene.add(yellowPlane);

                const dangerPlaneGeo = new THREE.PlaneGeometry(18, canyonLength);
                dangerPlaneGeo.rotateX(-Math.PI / 2);
                const dangerPlaneMat = new THREE.MeshBasicMaterial({{
                    color: 0xEF4444,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.45,
                }});
                const dangerPlane = new THREE.Mesh(dangerPlaneGeo, dangerPlaneMat);
                dangerPlane.position.y = 2.6;
                scene.add(dangerPlane);

                const damGeo = new THREE.BoxGeometry(16, 8, 4);
                const damMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.6 }});
                const dam = new THREE.Mesh(damGeo, damMat);
                dam.position.set(0, 2.5, -canyonLength / 2 + 3);
                scene.add(dam);

                let clock = new THREE.Clock();
                function animate() {{
                    requestAnimationFrame(animate);
                    const t = clock.getElapsedTime();
                    const wPos = waterGeo.attributes.position;
                    for (let i = 0; i < wPos.count; i++) {{
                        const u = wPos.getX(i);
                        const v = wPos.getZ(i);
                        const wave = Math.sin(v * 0.45 + t * 4.5) * 0.12 + Math.cos(u * 0.7 + t * 3.0) * 0.08;
                        wPos.setY(i, wave);
                    }}
                    waterGeo.computeVertexNormals();
                    waterGeo.attributes.position.needsUpdate = true;
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();

                window.addEventListener('resize', () => {{
                    const newW = container.clientWidth;
                    camera.aspect = newW / height;
                    camera.updateProjectionMatrix();
                    renderer.setSize(newW, height);
                }});
            </script>
        </body>
        </html>
        """
        components.html(three_js_html, height=540)

    with tab_plotly:
        x_grid = np.linspace(-15, 15, 50)
        y_grid = np.linspace(0, 60, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.power(np.abs(X) / 3.8, 1.9) + np.sin(X * 0.3 + Y * 0.2) * 0.5 - 1.0

        fig_3d = go.Figure(
            data=[
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale="Earth",
                    showscale=False,
                    opacity=0.88,
                    name="Canyon Terrain Walls",
                ),
                go.Surface(
                    x=X, y=Y,
                    z=np.full_like(Z, -0.8 + (pred_flow / DANGER_THRESHOLD) * 3.4),
                    colorscale=[[0, "#0284C7"], [1, alert_color]],
                    showscale=False,
                    opacity=0.75,
                    name=f"Forecast Flow ({pred_flow:.2f} m³/s)",
                ),
            ]
        )
        fig_3d.update_layout(
            title=f"Singati Gorge Cross-Section Surface & Water Stage: {alert_tier}",
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            scene=dict(
                xaxis=dict(title="Gorge Width (m)", backgroundcolor="#0B1120", gridcolor="#1E293B"),
                yaxis=dict(title="Canyon Length (m)", backgroundcolor="#0B1120", gridcolor="#1E293B"),
                zaxis=dict(title="River Stage / Elevation (m)", backgroundcolor="#0B1120", gridcolor="#1E293B"),
                camera=dict(eye=dict(x=1.6, y=1.5, z=1.3)),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=520,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Telemetry Charts & Explainable AI
    st.markdown("### 📈 Telemetry Forecast Hydrograph & Model Explainability")
    h_col1, h_col2 = st.columns([6, 5])

    with h_col1:
        st.markdown("#### 24-Hour Singati River Hydrograph & Surge Cone")
        timeline_hours = np.linspace(-24, 24, 49)
        past_curve = np.interp(timeline_hours[timeline_hours <= 0], [-24, 0], [q_yesterday, q_mid])
        future_curve = np.interp(timeline_hours[timeline_hours > 0], [0, 24], [q_mid, pred_flow])
        full_hydrograph = np.concatenate([past_curve, future_curve])
        future_hours = timeline_hours[timeline_hours >= 0]
        ci_factor = future_hours / 24.0
        ci_upper = pred_flow + (high_flow - pred_flow) * ci_factor
        ci_lower = np.maximum(0.0, pred_flow - (pred_flow - low_flow) * ci_factor)

        fig_hydro = go.Figure()
        fig_hydro.add_trace(
            go.Scatter(
                x=np.concatenate([future_hours, future_hours[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="80% Confidence Cone",
            )
        )
        fig_hydro.add_trace(
            go.Scatter(
                x=timeline_hours[timeline_hours <= 0],
                y=past_curve,
                mode="lines+markers",
                name="Historical Telemetry (T-24h to T0)",
                line=dict(color="#38BDF8", width=3.5),
                marker=dict(size=5, color="#67E8F9"),
            )
        )
        fig_hydro.add_trace(
            go.Scatter(
                x=timeline_hours[timeline_hours >= 0],
                y=future_curve,
                mode="lines+markers",
                name="AI 24h Forecast (T0 to T+24h)",
                line=dict(color=alert_color, width=4, dash="dash"),
                marker=dict(size=6, color=alert_color),
            )
        )
        fig_hydro.add_hline(
            y=DANGER_THRESHOLD,
            line_dash="dot",
            line_color="#EF4444",
            line_width=2.5,
            annotation_text=f"Red Danger: {DANGER_THRESHOLD:.2f} m³/s",
            annotation_position="top right",
        )
        fig_hydro.add_hline(
            y=YELLOW_THRESHOLD,
            line_dash="dashdot",
            line_color="#F59E0B",
            line_width=2.0,
            annotation_text=f"Yellow Watch: {YELLOW_THRESHOLD:.2f} m³/s",
            annotation_position="bottom right",
        )
        fig_hydro.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            xaxis=dict(title="Lead Time (Hours from Present T0)", gridcolor="#1E293B"),
            yaxis=dict(title="Singati Discharge (m³/s)", gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=20, t=30, b=30),
            height=380,
        )
        st.plotly_chart(fig_hydro, use_container_width=True)

    with h_col2:
        st.markdown("#### Council of 6 AI Models vs. Stacked Meta-Ensemble")
        council_flows = dict(sim_individual_preds)
        council_flows["Stacked Ensemble"] = pred_flow

        models_list = list(council_flows.keys())
        flows_list = [council_flows[m] for m in models_list]
        colors_list = ["#38BDF8" if m != "Stacked Ensemble" else alert_color for m in models_list]

        fig_council = go.Figure(
            go.Bar(
                x=flows_list,
                y=models_list,
                orientation="h",
                marker_color=colors_list,
                text=[f"{f:.2f} m³/s" for f in flows_list],
                textposition="auto",
            )
        )
        fig_council.add_vline(x=DANGER_THRESHOLD, line_dash="dot", line_color="#EF4444", line_width=2)
        fig_council.add_vline(x=YELLOW_THRESHOLD, line_dash="dashdot", line_color="#F59E0B", line_width=1.5)
        fig_council.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            xaxis=dict(title="Discharge Flow Tomorrow (m³/s)", gridcolor="#1E293B"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=30, r=20, t=30, b=30),
            height=380,
        )
        st.plotly_chart(fig_council, use_container_width=True)

    # Compound Himalayan Flood Physics Gauges
    st.markdown("#### ⚡ Compound Himalayan Flood Physics Gauges")
    g1, g2, g3, g4 = st.columns(4)

    rain_intensity_mid = rain_mid / (rain_hours + 0.5)
    glacier_tmax = tmax - 6.0

    with g1:
        fig_g1 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rain_intensity_mid,
                number={"suffix": " mm/h"},
                title={"text": "Cloudburst Intensity<br><span style='font-size:0.75rem; color:#94A3B8;'>Rain / Duration Ratio</span>"},
                gauge={
                    "axis": {"range": [0, 40]},
                    "bar": {"color": "#38BDF8"},
                    "steps": [
                        {"range": [0, 10], "color": "rgba(16, 185, 129, 0.3)"},
                        {"range": [10, 20], "color": "rgba(245, 158, 11, 0.3)"},
                        {"range": [20, 40], "color": "rgba(239, 68, 68, 0.4)"},
                    ],
                    "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 18.0},
                },
            )
        )
        fig_g1.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=220, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_g1, use_container_width=True)

    with g2:
        fig_g2 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rain_7d,
                number={"suffix": " mm"},
                title={"text": "Antecedent 7-Day Rain<br><span style='font-size:0.75rem; color:#94A3B8;'>Cumulative Catchment Pre-charge</span>"},
                gauge={
                    "axis": {"range": [0, 300]},
                    "bar": {"color": "#67E8F9"},
                    "steps": [
                        {"range": [0, 80], "color": "rgba(16, 185, 129, 0.3)"},
                        {"range": [80, 150], "color": "rgba(245, 158, 11, 0.3)"},
                        {"range": [150, 300], "color": "rgba(239, 68, 68, 0.4)"},
                    ],
                    "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 140.0},
                },
            )
        )
        fig_g2.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=220, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        fig_g3 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=glacier_tmax,
                number={"suffix": " °C"},
                title={"text": "Tsho Rolpa 4500m Temp<br><span style='font-size:0.75rem; color:#94A3B8;'>Glacial Ablation Index</span>"},
                gauge={
                    "axis": {"range": [0, 30]},
                    "bar": {"color": "#F59E0B"},
                    "steps": [
                        {"range": [0, 8], "color": "rgba(16, 185, 129, 0.3)"},
                        {"range": [8, 16], "color": "rgba(245, 158, 11, 0.3)"},
                        {"range": [16, 30], "color": "rgba(239, 68, 68, 0.4)"},
                    ],
                    "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 15.0},
                },
            )
        )
        fig_g3.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=220, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_g3, use_container_width=True)

    with g4:
        fig_g4 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=soil_sat_pct,
                number={"suffix": "%"},
                title={"text": "Deep Soil Saturation<br><span style='font-size:0.75rem; color:#94A3B8;'>Vadose Zone Infiltration</span>"},
                gauge={
                    "axis": {"range": [0, 60]},
                    "bar": {"color": "#10B981"},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(16, 185, 129, 0.3)"},
                        {"range": [25, 40], "color": "rgba(245, 158, 11, 0.3)"},
                        {"range": [40, 60], "color": "rgba(239, 68, 68, 0.4)"},
                    ],
                    "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 42.0},
                },
            )
        )
        fig_g4.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=220, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_g4, use_container_width=True)

    # Technical Architecture & Methodology Expander
    with st.expander("🔬 Veritas AI Hackathon: Architecture & Hydrological Physics Methodology", expanded=False):
        st.markdown(
            """
            ### 🏔️ Tamakoshi Glacial-Monsoon Early Warning System Architecture
            
            #### 1. Compound Glacial-Monsoon Hydrological Hazards
            The Tamakoshi River in Dolakha, Nepal, represents one of the steepest Himalayan hydrologic catchments on Earth.
            It spans from the **Tsho Rolpa Moraine Dam (4,580m)** down through the **Upper Tamakoshi 456 MW Dam (1,990m)**
            to the narrow **Singati Gorge (1,250m)**. Flood hazards are compound:
            - **Monsoon Cloudbursts:** Convective localized storm cells delivering >50mm in under 3 hours.
            - **Glacial Melt Thermal Surges:** High solar irradiance combined with positive temperature anomaly at 4,500m ASL.
            - **Vadose Soil Saturation Excess:** When antecedent 7-day rainfall exceeds 100mm, soil moisture saturates (>40%), causing near-total overland runoff.

            #### 2. The 58 Engineered Hydrological Features
            - **Upstream-Downstream Hydrodynamics:** Cascading discharge gradients (`q_basin_gradient`), upstream discharge sums (`q_upstream_sum`), and cross-station velocity ratios.
            - **Temporal Differencing & Acceleration:** First-order differences (`q_diff1`), second-order (`q_diff2`), percentage surge acceleration (`q_accel`, `q_pct_change1`).
            - **Rolling Statistics:** Multi-day means, maximums, and standard deviations spanning 3, 7, 14, and 30-day temporal windows.
            - **Coupled Runoff Physics:** Non-linear vadose interaction formula $R = P_{\\text{basin}} \\times (\\text{Soil}_{\\text{deep}})^2$.
            - **Positive Degree-Day Thermodynamics:** Glacier temperature differential ($T_{\\text{glacier}} = T_{\\text{valley}} - 6.0^\\circ\\text{C}$) coupled with solar radiation flux.
            - **Harmonic Seasonality:** Dual Fourier harmonic terms capturing intra-annual monsoon seasonality.

            #### 3. Council of 6 Base Estimators + RidgeCV Stacking Meta-Learner
            - **Base Estimators:** `XGBoost`, `LightGBM`, `CatBoost`, `HistGradientBoosting`, `ExtraTrees`, and `Deep Residual MLP`.
            - **Stacked Meta-Learner:** Cross-validated `RidgeCV` meta-regressor weighting base model deltas to eliminate single-model bias.
            - **Quantile Regression (80% Confidence):** Gradient Boosting regressors fitted at $\\alpha = 0.10$ and $\\alpha = 0.90$ to bound prediction uncertainty.
            - **Logistic Calibrated Flood Classifier:** Direct binary probability estimating riverbank breach risk.
            - **Threshold Calibration:** Statistically calibrated 95th-percentile flood discharge limit of **6.69 m³/s**, with a **6.02 m³/s (10% buffer)** Yellow Watch trigger.
            """
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    if st.button("📡 Switch to Real-Time Live Telemetry ➔", key="btn_sim_bottom_to_live", use_container_width=True):
        st.session_state.current_page = PAGE_LIVE
        st.rerun()


# =============================================================================
# VIEW 3: REAL-TIME LIVE TELEMETRY (Auto Fetch & Show Results)
# =============================================================================
elif st.session_state.current_page == PAGE_LIVE:
    # Switch Bar
    st.markdown(
        """
        <div class="switch-bar-container">
            <div class="switch-bar-text">
                📍 You are viewing <b>Page 2: Real-Time Live Telemetry (Auto Fetch)</b>. Want to test manual sensor simulations?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    lb_c1, lb_c2 = st.columns([1, 1])
    with lb_c1:
        if st.button("🏠 Back to Home", key="btn_live_back_home", use_container_width=True):
            st.session_state.current_page = PAGE_HOME
            st.rerun()
    with lb_c2:
        if st.button("🎛️ Switch to Manual Simulation ➔", key="btn_live_to_sim", use_container_width=True):
            st.session_state.current_page = PAGE_SIM
            st.rerun()

    # Route live telemetry from Open-Meteo GloFAS & Weather APIs
    past_days = past_days_live if "past_days_live" in locals() else 7
    live_telemetry = route_live_basin_telemetry(past_days=past_days)

    live_source_mode = live_telemetry["source_mode"]
    live_source_label = live_telemetry["status_label"]
    live_synced_at = live_telemetry["synced_at"]
    live_history_df = live_telemetry["history_df"]
    live_inp = live_telemetry["inputs"]

    live_fc = compute_forecast_from_inputs(bundle, live_inp)

    l_pred_flow = live_fc["pred_flow"]
    l_delta_pred = live_fc["delta_pred"]
    l_low_flow = live_fc["low_flow"]
    l_high_flow = live_fc["high_flow"]
    l_flood_prob = live_fc["flood_prob"]
    l_alert_level = live_fc["alert_level"]
    l_individual_preds = live_fc["individual_preds"]

    ALERT_STYLES = {
        "RED_ALERT": {"color": "#EF4444", "icon": "🚨", "label": "RED ALERT", "banner": "red"},
        "YELLOW_WATCH": {"color": "#F59E0B", "icon": "⚠️", "label": "YELLOW WATCH", "banner": "yellow"},
        "GREEN_NORMAL": {"color": "#10B981", "icon": "✅", "label": "GREEN NORMAL", "banner": "green"},
    }
    l_sty = ALERT_STYLES[l_alert_level]

    chip_class = "live" if live_source_mode == "LIVE_API" else "fallback"
    chip_icon = "🟢" if live_source_mode == "LIVE_API" else "🟡"

    # Hero Banner
    st.markdown(
        f"""
        <div class="hero-container">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 0.65rem;">
                <div class="hero-title" style="margin-bottom: 0;">
                    <span>📡 REAL-TIME LIVE TELEMETRY & 24H PREDICTION</span>
                </div>
                <div class="source-chip {chip_class}">
                    {chip_icon} {live_source_label}
                </div>
            </div>
            <div class="hero-subtitle">
                Autonomous real-time ingestion from Open-Meteo GloFAS river discharge (Tsho Rolpa, Gongar, Singati, Lower Tamakoshi)
                and basin weather. Automatically executes the 58-feature ML pipeline and 6-model ensemble.
            </div>
            <div class="badge-pill-container">
                <span class="badge-pill cyan">🛰️ 4 Corridor Stations Live</span>
                <span class="badge-pill emerald">🕐 Synced: {live_synced_at}</span>
                <span class="badge-pill {'red' if l_alert_level == 'RED_ALERT' else 'amber' if l_alert_level == 'YELLOW_WATCH' else 'emerald'}">
                    {l_sty['icon']} Live Status: {l_sty['label']} ({l_pred_flow:.2f} m³/s)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Banner based on Live API Data
    if l_alert_level == "RED_ALERT":
        st.markdown(
            f"""
            <div class="action-banner red">
                <div class="banner-title" style="color: #FCA5A5;">
                    🚨 RED ALERT — LIVE FLOOD SURGE DETECTED IN DOLAKHA
                </div>
                <div class="banner-text">
                    Predicted 24-hour discharge reaches <b>{l_pred_flow:.2f} m³/s</b>, exceeding the danger threshold of <b>{DANGER_THRESHOLD:.2f} m³/s</b>.
                    AI Breach Probability: <b>{l_flood_prob:.1f}%</b>.
                </div>
                <ul class="protocol-list">
                    <li>Immediate siren alarms across Singati Bazaar and riverside settlements.</li>
                    <li>Actuate Upper Tamakoshi dam radial spillway gates for reservoir drawdown.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif l_alert_level == "YELLOW_WATCH":
        st.markdown(
            f"""
            <div class="action-banner yellow">
                <div class="banner-title" style="color: #FCD34D;">
                    ⚠️ YELLOW WATCH — MONSOON DISCHARGE ELEVATED
                </div>
                <div class="banner-text">
                    Predicted discharge reaches <b>{l_pred_flow:.2f} m³/s</b>, entering the 10% safety buffer zone.
                </div>
                <ul class="protocol-list">
                    <li>Switch upstream acoustic sensors to 15-minute polling.</li>
                    <li>Alert Dolakha disaster coordinators and riverbank personnel.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="action-banner green">
                <div class="banner-title" style="color: #6EE7B7;">
                    ✅ GREEN NORMAL — TAMAKOSHI RIVER WITHIN SAFE CHANNEL
                </div>
                <div class="banner-text">
                    Live watershed discharge safely contained at <b>{l_pred_flow:.2f} m³/s</b> (Clearance: <b>{(DANGER_THRESHOLD - l_pred_flow):.2f} m³/s</b>). AI Flood Probability: <b>{l_flood_prob:.1f}%</b>.
                </div>
                <ul class="protocol-list">
                    <li>Maintain nominal 456 MW power generation dispatch at Upper Tamakoshi dam.</li>
                    <li>Autonomous telemetry sentinel polling every 30 minutes.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Live KPI Cards
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Live Forecast Tomorrow</div>
                <div class="value" style="color: {l_sty['color']};">{l_pred_flow:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: {'#EF4444' if l_delta_pred > 0 else '#10B981'};">
                    {l_delta_pred:+.2f} m³/s vs current ({live_inp['q_mid']:.2f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lc2:
        l_ci_range = l_high_flow - l_low_flow
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">80% Confidence Interval</div>
                <div class="value">{l_low_flow:.2f} – {l_high_flow:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #38BDF8;">Uncertainty: ±{l_ci_range / 2.0:.2f} m³/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lc3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Danger / Watch Threshold</div>
                <div class="value">{DANGER_THRESHOLD:.2f} <span style="font-size:1.1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #F59E0B;">Watch Trigger: {YELLOW_THRESHOLD:.2f} m³/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lc4:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">AI Flood Probability</div>
                <div class="value" style="color: {l_sty['color']};">{l_flood_prob:.1f}<span style="font-size:1.2rem; color:#94A3B8;">%</span></div>
                <div class="delta" style="color: {l_sty['color']};">Status: {l_sty['label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Tabs for Live Telemetry Breakdown
    ltab1, ltab2, ltab3, ltab4 = st.tabs([
        "📈 Hydrograph & Surge Forecast",
        "🧠 Council of 6 AI Models",
        "🗺️ 3D Telemetry Map",
        "📋 Raw Telemetry Data & Inputs",
    ])

    with ltab1:
        st.markdown("### 📈 24-Hour Singati River Hydrograph & Confidence Cone")
        l_timeline_hours = np.linspace(-24, 24, 49)
        l_past_curve = np.interp(l_timeline_hours[l_timeline_hours <= 0], [-24, 0], [live_inp["q_yesterday"], live_inp["q_mid"]])
        l_future_curve = np.interp(l_timeline_hours[l_timeline_hours > 0], [0, 24], [live_inp["q_mid"], l_pred_flow])
        l_future_hours = l_timeline_hours[l_timeline_hours >= 0]
        l_ci_factor = l_future_hours / 24.0
        l_ci_upper = l_pred_flow + (l_high_flow - l_pred_flow) * l_ci_factor
        l_ci_lower = np.maximum(0.0, l_pred_flow - (l_pred_flow - l_low_flow) * l_ci_factor)

        fig_l_hydro = go.Figure()
        fig_l_hydro.add_trace(
            go.Scatter(
                x=np.concatenate([l_future_hours, l_future_hours[::-1]]),
                y=np.concatenate([l_ci_upper, l_ci_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(44, 229, 181, 0.14)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="80% Confidence Cone",
            )
        )
        fig_l_hydro.add_trace(
            go.Scatter(
                x=l_timeline_hours[l_timeline_hours <= 0],
                y=l_past_curve,
                mode="lines+markers",
                name="Live Telemetry History (T-24h to T0)",
                line=dict(color="#38BDF8", width=3.5),
                marker=dict(size=5, color="#67E8F9"),
            )
        )
        fig_l_hydro.add_trace(
            go.Scatter(
                x=l_timeline_hours[l_timeline_hours >= 0],
                y=l_future_curve,
                mode="lines+markers",
                name="AI 24h Forecast (T0 to T+24h)",
                line=dict(color=l_sty["color"], width=4, dash="dash"),
                marker=dict(size=6, color=l_sty["color"]),
            )
        )
        fig_l_hydro.add_hline(
            y=DANGER_THRESHOLD, line_dash="dot", line_color="#EF4444", line_width=2.5,
            annotation_text=f"Red Danger: {DANGER_THRESHOLD:.2f} m³/s", annotation_position="top right"
        )
        fig_l_hydro.add_hline(
            y=YELLOW_THRESHOLD, line_dash="dashdot", line_color="#F59E0B", line_width=2.0,
            annotation_text=f"Yellow Watch: {YELLOW_THRESHOLD:.2f} m³/s", annotation_position="bottom right"
        )
        fig_l_hydro.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            xaxis=dict(title="Lead Time (Hours from Present T0)", gridcolor="#1E293B"),
            yaxis=dict(title="Singati Discharge (m³/s)", gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=20, t=30, b=30),
            height=380,
        )
        st.plotly_chart(fig_l_hydro, use_container_width=True)

        st.markdown("### 📊 Multi-Station Discharge History (Past Week)")
        fig_l_hist = go.Figure()
        for col, color, dash in [
            ("Tsho Rolpa (m³/s)", "#A78BFA", "dot"),
            ("Gongar (m³/s)", "#67E8F9", "dash"),
            ("Singati (m³/s)", "#38BDF8", "solid"),
            ("Lower Tamakoshi (m³/s)", "#2CE5B5", "dashdot"),
        ]:
            fig_l_hist.add_trace(
                go.Scatter(
                    x=live_history_df["Date"],
                    y=live_history_df[col],
                    mode="lines+markers",
                    name=col,
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(size=5),
                )
            )
        fig_l_hist.add_hline(
            y=DANGER_THRESHOLD, line_dash="dot", line_color="#EF4444", line_width=2,
            annotation_text=f"Red Danger Line ({DANGER_THRESHOLD:.2f})", annotation_position="top right"
        )
        fig_l_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            xaxis=dict(title="Date", gridcolor="#1E293B"),
            yaxis=dict(title="Discharge (m³/s)", gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=20, t=30, b=30),
            height=340,
        )
        st.plotly_chart(fig_l_hist, use_container_width=True)

    with ltab2:
        st.markdown("### 🧠 Council of 6 AI Models vs. Stacked Meta-Ensemble")
        l_council_flows = dict(l_individual_preds)
        l_council_flows["Stacked Ensemble"] = l_pred_flow

        l_models_list = list(l_council_flows.keys())
        l_flows_list = [l_council_flows[m] for m in l_models_list]
        l_colors_list = ["#38BDF8" if m != "Stacked Ensemble" else l_sty["color"] for m in l_models_list]

        fig_l_council = go.Figure(
            go.Bar(
                x=l_flows_list,
                y=l_models_list,
                orientation="h",
                marker_color=l_colors_list,
                text=[f"{f:.3f} m³/s" for f in l_flows_list],
                textposition="auto",
            )
        )
        fig_l_council.add_vline(x=DANGER_THRESHOLD, line_dash="dot", line_color="#EF4444", line_width=2)
        fig_l_council.add_vline(x=YELLOW_THRESHOLD, line_dash="dashdot", line_color="#F59E0B", line_width=1.5)
        fig_l_council.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            xaxis=dict(title="Discharge Flow Tomorrow (m³/s)", gridcolor="#1E293B"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=30, r=20, t=30, b=30),
            height=380,
        )
        st.plotly_chart(fig_l_council, use_container_width=True)

    with ltab3:
        st.markdown("### 🗺️ 3D Tamakoshi Corridor Telemetry Map")
        st.caption("Real-time station columns along the Tamakoshi river corridor.")

        def l_station_color(flow_val):
            if flow_val >= DANGER_THRESHOLD:
                return [239, 68, 68, 240]
            elif flow_val >= YELLOW_THRESHOLD:
                return [245, 158, 11, 230]
            else:
                return [44, 229, 181, 220]

        l_flow_map = {
            "tsho_rolpa": live_inp["q_glacier"],
            "gongar": live_inp["q_upper"],
            "singati": l_pred_flow,
            "lower_tamakoshi": live_inp["q_lower"],
        }

        l_map_data = []
        for s in TAMAKOSHI_STATIONS:
            f = l_flow_map.get(s["id"], 0.5)
            l_map_data.append({
                "name": s["name"],
                "lat": s["lat"],
                "lon": s["lon"],
                "flow": round(f, 3),
                "role": s["role"],
                "color": l_station_color(f),
                "height": max(300, f * 1400 + 400),
            })
        df_l_map = pd.DataFrame(l_map_data)

        l_corridor_arcs = pd.DataFrame([
            {"from_lon": 86.4769, "from_lat": 27.8617, "to_lon": 86.2205, "to_lat": 27.8442, "color": l_station_color(live_inp["q_glacier"])},
            {"from_lon": 86.2205, "from_lat": 27.8442, "to_lon": 86.1628, "to_lat": 27.7314, "color": l_station_color(live_inp["q_upper"])},
            {"from_lon": 86.1628, "from_lat": 27.7314, "to_lon": 86.0750, "to_lat": 27.5250, "color": l_station_color(l_pred_flow)},
        ])

        l_view = pdk.ViewState(latitude=27.72, longitude=86.24, zoom=10.0, pitch=52, bearing=-18)
        l_col_layer = pdk.Layer(
            "ColumnLayer", data=df_l_map, get_position=["lon", "lat"], get_elevation="height",
            elevation_scale=1, radius=500, get_fill_color="color", pickable=True, auto_highlight=True,
        )
        l_arc_layer = pdk.Layer(
            "ArcLayer", data=l_corridor_arcs, get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"], get_source_color="color", get_target_color="color",
            get_width=5, pickable=True,
        )
        l_deck = pdk.Deck(
            layers=[l_arc_layer, l_col_layer], initial_view_state=l_view,
            map_style="mapbox://styles/mapbox/dark-v11",
            tooltip={"html": "<div style='color:#fff; padding:6px 10px; background:#0F172A;'><b>{name}</b><br>Discharge: {flow} m³/s</div>"},
        )
        st.pydeck_chart(l_deck)

    with ltab4:
        st.markdown("### 📋 Raw Telemetry Data & Inputs")
        st.caption(f"Source: **{live_source_label}** • Synced at {live_synced_at}")
        st.markdown("#### 📊 Station Discharge & Weather History")
        st.dataframe(live_history_df, use_container_width=True, hide_index=True)

        st.markdown("#### 🎛️ Live Sensor Parameters Fed to AI Engine")
        l_inp_df = pd.DataFrame([{
            "Singati Today (m³/s)": live_inp["q_mid"],
            "Singati Yesterday (m³/s)": live_inp["q_yesterday"],
            "Tsho Rolpa (m³/s)": live_inp["q_glacier"],
            "Gongar Dam (m³/s)": live_inp["q_upper"],
            "Lower Confluence (m³/s)": live_inp["q_lower"],
            "Valley Rain Today (mm)": live_inp["rain_mid"],
            "3-Day Rain (mm)": live_inp["rain_3d"],
            "7-Day Rain (mm)": live_inp["rain_7d"],
            "Vadose Soil (%)": live_inp["soil_sat_pct"],
            "Max Temp (°C)": live_inp["tmax"],
            "Min Temp (°C)": live_inp["tmin"],
            "Solar (MJ/m²)": live_inp["solar_rad"],
            "Month": live_inp["month"],
        }])
        st.dataframe(l_inp_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    if st.button("🎛️ Switch to Manual Simulation ➔", key="btn_live_bottom_to_sim", use_container_width=True):
        st.session_state.current_page = PAGE_SIM
        st.rerun()
