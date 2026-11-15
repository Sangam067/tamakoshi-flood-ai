import sys
import types

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
@st.cache_resource
def load_bundle():
    return joblib.load("tamakoshi_ultimate_model.joblib")

bundle = load_bundle()
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
        --accent-amber: #fbbf24;
        --accent-red: #f87171;
        --accent-emerald: #34d399;
        --text-primary: #f8fafc;
        --text-secondary: #b8c3d6;
        --text-muted: #8ea3bf;
    }

    html, body {
        background: var(--bg-main);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.10), transparent 24%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.08), transparent 20%),
            linear-gradient(180deg, #07111f 0%, #0b1728 100%);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 13, 22, 0.92);
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
        margin-bottom: 1.5rem;
        padding: 2rem 2rem 1.6rem 2rem;
        border-radius: 22px;
        border: 1px solid rgba(103, 232, 249, 0.25);
        background:
            linear-gradient(135deg, rgba(8, 15, 25, 0.68), rgba(7, 17, 31, 0.75)),
            url('https://images.unsplash.com/photo-1544735716-392fe2489ffa?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat;
        box-shadow: 0 24px 60px rgba(2, 6, 23, 0.6);
    }

    .hero-container::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(56, 189, 248, 0.08), rgba(14, 165, 233, 0.04), transparent 70%);
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
        font-size: clamp(1.8rem, 2.7vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -0.05em;
        color: var(--text-primary);
        text-shadow: 0 4px 24px rgba(2, 6, 23, 0.8);
    }

    .hero-subtitle {
        max-width: 980px;
        font-size: 1.04rem;
        line-height: 1.7;
        color: rgba(226, 232, 240, 0.9);
        text-shadow: 0 2px 8px rgba(2, 6, 23, 0.75);
        margin-bottom: 1.2rem;
    }

    .badge-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.58rem 0.9rem;
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
    .badge-pill.amber { border-color: rgba(251, 191, 36, 0.45); color: #fbbf24; }
    .badge-pill.emerald { border-color: rgba(52, 211, 153, 0.45); color: #34d399; }
    .badge-pill.red { border-color: rgba(248, 113, 113, 0.45); color: #f87171; }

    .glass-card {
        position: relative;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.72));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1.2rem 1.25rem;
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
        margin-bottom: 0.5rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .glass-card .value {
        font-size: clamp(1.5rem, 2vw, 2rem);
        font-weight: 800;
        line-height: 1.2;
        color: var(--text-primary);
    }

    .glass-card .delta {
        margin-top: 0.45rem;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .action-banner {
        border-radius: 18px;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1.5rem;
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
        margin-bottom: 0.7rem;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .banner-text {
        color: var(--text-primary);
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .protocol-list {
        margin-top: 0.9rem;
        padding-left: 1.15rem;
        color: rgba(226, 232, 240, 0.9);
        line-height: 1.7;
    }

    .protocol-list li {
        margin-bottom: 0.15rem;
    }

    h3 {
        margin-top: 1.8rem !important;
        margin-bottom: 0.25rem !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em !important;
    }

    .stCaption {
        color: var(--text-secondary) !important;
        margin-bottom: 0.8rem !important;
    }

    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }

    .stTabs [role="tab"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px 12px 0 0;
        color: var(--text-secondary);
        padding: 0.6rem 1rem;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.17), rgba(15, 23, 42, 0.8));
        color: var(--text-primary);
        border-color: rgba(103, 232, 249, 0.4);
    }

    .stApp > div > div {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar: Scenarios & Sensor Sliders
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='
        padding: 1rem 0.9rem 0.8rem 0.9rem;
        margin: 0.2rem 0 1.2rem 0;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(8, 13, 22, 0.88));
        border: 1px solid rgba(103, 232, 249, 0.22);
        box-shadow: 0 14px 28px rgba(2, 6, 23, 0.28);
        text-align: center;
    '>
        <div style='font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #67e8f9; font-weight: 700; margin-bottom: 0.35rem;'>Hydrology AI</div>
        <h2 style='margin: 0; font-size: 1.45rem; color: #F8FAFC; line-height: 1.2;'>🏔️ Tamakoshi AI</h2>
        <div style='margin-top: 0.35rem; font-size: 0.8rem; color: #9fb4cf;'>Himalayan Basin Early Warning</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

# ---------------------------------------------------------
# Exact Hydrological Feature Engineering (58 Features)
# DO NOT MODIFY FORMULAS OR FEATURE NAMES
# ---------------------------------------------------------
q_diff1 = q_mid - q_yesterday
q_diff2 = q_diff1 * 1.5
q_diff3 = q_diff1 * 2.0
q_accel = q_diff1 * 0.25
q_pct_change1 = np.clip(q_diff1 / max(0.1, q_yesterday), -2, 5)

q_mean_3d = (q_mid + q_yesterday) / 2.0
q_max_3d = max(q_mid, q_yesterday)
q_std_3d = abs(q_diff1) * 0.5
q_mean_7d = q_mean_3d * 0.92
q_max_7d = q_max_3d
q_std_7d = abs(q_diff1) * 0.6
q_mean_14d = q_mean_3d * 0.88
q_max_14d = q_max_3d
q_std_14d = abs(q_diff1) * 0.7
q_mean_30d = q_mean_3d * 0.82
q_max_30d = q_max_3d
q_std_30d = abs(q_diff1) * 0.8
q_ratio_to_14d = q_mid / (q_mean_14d + 0.1)

q_glacier_diff1 = q_glacier * 0.15 if q_diff1 > 0 else 0.0
q_upper_diff1 = q_upper * 0.15 if q_diff1 > 0 else 0.0
q_upstream_sum = q_glacier + q_upper
q_upstream_ratio = q_upstream_sum / (q_mid + 0.1)
q_basin_gradient = q_lower - q_upper

basin_precip = (rain_mid + rain_glacier) / 2.0
basin_rain = basin_precip
rain_intensity_mid = rain_mid / (rain_hours + 0.5)
rain_intensity_upper = rain_glacier / (rain_hours + 0.5)
basin_rain_14d = rain_7d * 1.6
basin_rain_30d = rain_7d * 2.8
antecedent_precip_idx = rain_3d * 0.65

soil_frac = soil_sat_pct / 100.0
soil_top = soil_frac
soil_mid = soil_frac
soil_deep = soil_frac
soil_top_diff3 = 0.04 if rain_3d > 40 else -0.01
runoff_potential = basin_precip * (soil_deep ** 2)

glacier_tmax = tmax - 6.0
glacier_tmin = tmin - 6.0
glacier_snow_7d = snow_glacier * 2.0
glacier_snow_14d = snow_glacier * 3.5

pos_glacier_t = max(0.0, glacier_tmax)
glacial_melt_energy = pos_glacier_t * (solar_rad / 10.0)
glacial_melt_3d = glacial_melt_energy * 3.0
glacial_melt_7d = glacial_melt_energy * 6.5
rain_on_snow = rain_glacier if glacier_snow_14d > 0.5 else 0.0
water_balance_7d = rain_7d - (max(1.0, tmax * 0.18) * 7.0)

doy = int(month) * 30 - 15
sin_doy1 = np.sin(2 * np.pi * doy / 365.25)
cos_doy1 = np.cos(2 * np.pi * doy / 365.25)
sin_doy2 = np.sin(4 * np.pi * doy / 365.25)
cos_doy2 = np.cos(4 * np.pi * doy / 365.25)

row = pd.DataFrame([{
    "q_main": q_mid, "q_glacier": q_glacier, "q_upper": q_upper, "q_lower": q_lower,
    "q_diff1": q_diff1, "q_diff2": q_diff2, "q_diff3": q_diff3, "q_accel": q_accel,
    "q_pct_change1": q_pct_change1, "q_mean_3d": q_mean_3d, "q_max_3d": q_max_3d,
    "q_std_3d": q_std_3d, "q_mean_7d": q_mean_7d, "q_max_7d": q_max_7d,
    "q_std_7d": q_std_7d, "q_mean_14d": q_mean_14d, "q_max_14d": q_max_14d,
    "q_std_14d": q_std_14d, "q_mean_30d": q_mean_30d, "q_max_30d": q_max_30d,
    "q_std_30d": q_std_30d, "q_ratio_to_14d": q_ratio_to_14d,
    "q_glacier_diff1": q_glacier_diff1, "q_upper_diff1": q_upper_diff1,
    "q_upstream_sum": q_upstream_sum, "q_upstream_ratio": q_upstream_ratio,
    "q_basin_gradient": q_basin_gradient, "basin_precip": basin_precip,
    "basin_rain": basin_rain, "rain_intensity_mid": rain_intensity_mid,
    "rain_intensity_upper": rain_intensity_upper, "basin_rain_3d": rain_3d,
    "basin_rain_7d": rain_7d, "basin_rain_14d": basin_rain_14d,
    "basin_rain_30d": basin_rain_30d, "antecedent_precip_idx": antecedent_precip_idx,
    "soil_top": soil_top, "soil_mid": soil_mid, "soil_deep": soil_deep,
    "soil_top_diff3": soil_top_diff3, "runoff_potential": runoff_potential,
    "glacier_tmax": glacier_tmax, "glacier_tmin": glacier_tmin,
    "valley_tmax": tmax, "valley_tmin": tmin, "glacier_solar": solar_rad,
    "glacier_snow": snow_glacier, "glacier_snow_7d": glacier_snow_7d,
    "glacier_snow_14d": glacier_snow_14d, "glacial_melt_energy": glacial_melt_energy,
    "glacial_melt_3d": glacial_melt_3d, "glacial_melt_7d": glacial_melt_7d,
    "rain_on_snow": rain_on_snow, "water_balance_7d": water_balance_7d,
    "sin_doy1": sin_doy1, "cos_doy1": cos_doy1, "sin_doy2": sin_doy2, "cos_doy2": cos_doy2,
}])[feature_cols]

# ---------------------------------------------------------
# Multi-Model Stacking Ensemble Inference
# ---------------------------------------------------------
base_deltas_dict = {name: float(base_models[name].predict(row)[0]) for name in model_names}
base_deltas = np.array([[base_deltas_dict[name] for name in model_names]])
delta_pred = float(meta_learner.predict(base_deltas)[0])
pred_flow = max(0.0, q_mid + delta_pred)
low_flow = max(0.0, q_mid + float(q_low_model.predict(row)[0]))
high_flow = max(pred_flow, q_mid + float(q_high_model.predict(row)[0]))
flood_prob = float(flood_clf.predict_proba(row)[0, 1]) * 100.0

# ---------------------------------------------------------
# Alert Status Categorization
# ---------------------------------------------------------
is_red = pred_flow >= DANGER_THRESHOLD or flood_prob >= 60.0
is_yellow = not is_red and (pred_flow >= YELLOW_THRESHOLD or flood_prob >= 35.0)
is_green = not is_red and not is_yellow

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

# ---------------------------------------------------------
# Hero Banner & Telemetry Header
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">
            <span>🏔️ TAMAKOSHI HIMALAYAN EARLY WARNING COMMAND CENTER</span>
        </div>
        <div class="hero-subtitle">
            Autonomous 3D Glacial-Monsoon Flood Digital Twin & Cascading Hydro-Telemetry Intelligence for the Dolakha Watershed, Nepal.
            Engineered for Veritas AI Hackathon (Track 4.3: AI + Climate).
        </div>
        <div class="badge-pill-container">
            <span class="badge-pill cyan">🛰️ Dolakha Telemetry: 4 Cascading Stations</span>
            <span class="badge-pill cyan">🏔️ Tsho Rolpa Headwaters: 4,580m ASL</span>
            <span class="badge-pill cyan">⚡ Upper Tamakoshi Dam: 456 MW Penstock</span>
            <span class="badge-pill cyan">🎯 Forecast Target: Singati Gorge (1,250m)</span>
            <span class="badge-pill cyan">🧠 6-Model AI Ensemble: 42,540 Station-Days</span>
            <span class="badge-pill {'red' if is_red else 'amber' if is_yellow else 'emerald'}">
                {alert_icon} Status: {alert_tier} ({pred_flow:.2f} m³/s)
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Emergency Action & Decision Support Banner
# ---------------------------------------------------------
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
                <li><b>Immediate Evacuation Dispatch:</b> Broadcast automated CAP/SMS emergency alarms to riverside settlements across Singati Bazaar, Gongar, Jagat, and Lamabagar.</li>
                <li><b>Upper Tamakoshi Dam Control:</b> Actuate radial spillway gates to regulate buffer reservoir drawdown and mitigate hydro-peak downstream.</li>
                <li><b>Tactical First-Responders:</b> Mobilize Dolakha District Disaster Management Committee (DDMC) and Armed Police Force (APF) water rescue units along the Singati-Charikot transport corridor.</li>
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

# ---------------------------------------------------------
# Glassmorphism Live KPI Cards
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Visual Section 1: 3D Himalayan Terrain & Watershed Map
# ---------------------------------------------------------
st.markdown("### 🗺️ 3D Himalayan Terrain & Cascading Telemetry Map")
st.caption(
    "Interactive 3D Deck.gl terrain view centered on Dolakha, Nepal (Pitch: 55°, Bearing: -20°). "
    "Extruded 3D columns and river corridor arcs dynamically scale with real-time station discharge and upstream flood surge."
)

def get_station_color(flow_val, is_target=False):
    effective_flow = pred_flow if is_target else flow_val
    if effective_flow >= DANGER_THRESHOLD:
        return [239, 68, 68, 240]  # Red
    elif effective_flow >= YELLOW_THRESHOLD:
        return [245, 158, 11, 230]  # Amber
    else:
        return [14, 165, 233, 215]  # Glacial Cyan

stations_info = [
    {
        "id": "tsho_rolpa",
        "name": "Tsho Rolpa Glacial Lake",
        "elevation_m": 4580,
        "lat": 27.8617,
        "lon": 86.4769,
        "flow": q_glacier,
        "role": "Glacial Lake Outflow Headwaters",
        "status": "GLOF Risk Sector" if q_glacier > 0.8 else "Nominal Outflow",
        "color": get_station_color(q_glacier),
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
        "color": get_station_color(q_upper),
        "height": max(300, q_upper * 1200 + 400),
    },
    {
        "id": "singati_target",
        "name": "Singati Gorge (Forecast Target)",
        "elevation_m": 1250,
        "lat": 27.7314,
        "lon": 86.1628,
        "flow": pred_flow,
        "role": "Target Hydro Station & Settlement",
        "status": alert_tier,
        "color": get_station_color(pred_flow, is_target=True),
        "height": max(400, pred_flow * 1400 + 500),
    },
    {
        "id": "lower_tamakoshi",
        "name": "Lower Tamakoshi Confluence",
        "elevation_m": 850,
        "lat": 27.5250,
        "lon": 86.0750,
        "flow": q_lower,
        "role": "Downstream Valley Confluence",
        "status": "Surge Recipient" if q_lower > 8.0 else "Normal Channel",
        "color": get_station_color(q_lower),
        "height": max(300, q_lower * 800 + 400),
    },
]

df_stations = pd.DataFrame(stations_info)

# 3D Extruded Columns Layer
column_layer = pdk.Layer(
    "ColumnLayer",
    data=df_stations,
    get_position=["lon", "lat"],
    get_elevation="height",
    elevation_scale=1,
    radius=1400,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

# Text Layer Hovering over Columns
text_layer = pdk.Layer(
    "TextLayer",
    data=df_stations,
    get_position=["lon", "lat"],
    get_text="name",
    get_size=13,
    get_color=[248, 250, 252, 255],
    get_angle=0,
    get_text_anchor="'middle'",
    get_alignment_baseline="'bottom'",
    pixel_offset=[0, -22],
)

# Glowing 3D River Corridor Path
river_corridor_coords = [
    [86.4769, 27.8617],
    [86.3800, 27.8550],
    [86.2800, 27.8500],
    [86.2205, 27.8442],
    [86.1950, 27.7900],
    [86.1628, 27.7314],
    [86.1200, 27.6300],
    [86.0750, 27.5250],
]

river_path_color = [239, 68, 68, 240] if is_red else [245, 158, 11, 230] if is_yellow else [56, 189, 248, 230]

path_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": river_corridor_coords, "name": "Tamakoshi Glacial-Fluvial Corridor"}],
    get_path="path",
    get_color=river_path_color,
    width_min_pixels=4,
    width_max_pixels=8,
    pickable=True,
)

# Glowing Downstream Surge Propagation Arcs
arc_routes = [
    {
        "source": [86.4769, 27.8617],
        "target": [86.2205, 27.8442],
        "source_color": [14, 165, 233, 220],
        "target_color": get_station_color(q_upper),
    },
    {
        "source": [86.2205, 27.8442],
        "target": [86.1628, 27.7314],
        "source_color": get_station_color(q_upper),
        "target_color": get_station_color(pred_flow, is_target=True),
    },
    {
        "source": [86.1628, 27.7314],
        "target": [86.0750, 27.5250],
        "source_color": get_station_color(pred_flow, is_target=True),
        "target_color": get_station_color(q_lower),
    },
]

arc_layer = pdk.Layer(
    "ArcLayer",
    data=pd.DataFrame(arc_routes),
    get_source_position="source",
    get_target_position="target",
    get_source_color="source_color",
    get_target_color="target_color",
    get_width=4,
    get_tilt=15,
)

view_state = pdk.ViewState(
    latitude=27.72,
    longitude=86.25,
    zoom=9.6,
    pitch=55,
    bearing=-20,
)

deck = pdk.Deck(
    layers=[path_layer, arc_layer, column_layer, text_layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    tooltip={
        "html": """
        <div style="font-family: sans-serif; padding: 4px;">
            <b style="font-size: 1.05rem; color: #38BDF8;">{name}</b><br/>
            <span style="color: #94A3B8;">Role:</span> {role}<br/>
            <span style="color: #94A3B8;">Altitude:</span> {elevation_m} m ASL<br/>
            <span style="color: #94A3B8;">Discharge:</span> <b style="color: #FFF;">{flow:.2f} m³/s</b><br/>
            <span style="color: #94A3B8;">Station Status:</span> <b>{status}</b>
        </div>
        """,
        "style": {
            "backgroundColor": "#0F172A",
            "color": "#F8FAFC",
            "border": "1px solid rgba(56, 189, 248, 0.4)",
            "borderRadius": "8px",
        },
    },
)

st.pydeck_chart(deck, use_container_width=True)

# Telemetry Station Quick Status Table
tcols = st.columns(4)
for i, s in enumerate(stations_info):
    with tcols[i]:
        tag_color = "#EF4444" if s["color"][0] > 200 and s["color"][1] < 100 else "#F59E0B" if s["color"][1] > 100 and s["color"][2] < 100 else "#38BDF8"
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px;">
                <div style="font-size: 0.78rem; font-weight: 600; color: #94A3B8;">{s['elevation_m']}m ASL</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F1F5F9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{s['name']}</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {tag_color}; margin-top: 4px;">
                    {s['flow']:.2f} <span style="font-size: 0.8rem; color: #94A3B8;">m³/s</span>
                </div>
                <div style="font-size: 0.75rem; color: #CBD5E1; margin-top: 2px;">{s['status']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Visual Section 2: Interactive 3D River Gorge Digital Twin
# ---------------------------------------------------------
st.markdown("### 🌊 Interactive 3D Singati River Gorge & Spillway Digital Twin")
st.caption(
    "Cinematic 3D WebGL Digital Twin of the Singati Canyon and Upper Tamakoshi Spillway. "
    "Users can rotate, pan, and zoom in 3D space to inspect the valley terrain walls, dynamic water surface elevation, "
    "and glowing 3D warning threshold planes (6.02 m³/s Yellow Watch & 6.69 m³/s Red Danger Overtopping)."
)

tab_3d, tab_plotly = st.tabs(["⚡ 3D WebGL Digital Twin (Three.js)", "📊 3D Plotly Surface & Breach Contour"])

with tab_3d:
    freeboard = DANGER_THRESHOLD - pred_flow
    freeboard_str = f"{freeboard:+.2f} m³/s"
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
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                overflow: hidden;
                background-color: #0B1120;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            #container {{
                width: 100%;
                height: 520px;
                position: relative;
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid rgba(56, 189, 248, 0.25);
            }}
            .hud-overlay {{
                position: absolute;
                top: 14px;
                left: 16px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 10px;
                padding: 10px 16px;
                color: #F8FAFC;
                pointer-events: none;
                z-index: 10;
            }}
            .hud-title {{
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #38BDF8;
            }}
            .hud-stat {{
                font-size: 1.15rem;
                font-weight: 800;
                margin-top: 2px;
            }}
            .hud-controls {{
                position: absolute;
                bottom: 14px;
                left: 16px;
                display: flex;
                gap: 8px;
                z-index: 10;
            }}
            .hud-btn {{
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: #E2E8F0;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 0.76rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .hud-btn:hover {{
                background: #38BDF8;
                color: #0B1120;
                border-color: #38BDF8;
            }}
            .legend-overlay {{
                position: absolute;
                top: 14px;
                right: 16px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 0.78rem;
                color: #CBD5E1;
                pointer-events: none;
                z-index: 10;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 4px;
            }}
            .legend-dot {{
                width: 10px;
                height: 10px;
                border-radius: 2px;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="container">
            <div class="hud-overlay">
                <div class="hud-title">Singati Canyon Digital Twin</div>
                <div class="hud-stat">Stage Flow: <span style="color: {alert_color};">{pred_flow:.2f} m³/s</span></div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 3px;">
                    Breach Status: {breach_badge_html}
                </div>
            </div>
            
            <div class="legend-overlay">
                <div class="legend-item">
                    <div class="legend-dot" style="background: #EF4444; box-shadow: 0 0 6px #EF4444;"></div>
                    <span>6.69 m³/s — Red Danger Overtopping</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #F59E0B; box-shadow: 0 0 6px #F59E0B;"></div>
                    <span>6.02 m³/s — Yellow Watch Buffer</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #06B6D4;"></div>
                    <span>River Surface ({pred_flow:.2f} m³/s)</span>
                </div>
            </div>

            <div class="hud-controls">
                <button class="hud-btn" onclick="setCameraView('iso')">🎥 Canyon Isometric</button>
                <button class="hud-btn" onclick="setCameraView('spillway')">⚡ Upstream Dam</button>
                <button class="hud-btn" onclick="setCameraView('cross')">📐 Bank Cross-Section</button>
                <button class="hud-btn" id="orbit-btn" onclick="toggleAutoOrbit()">🔄 Auto-Rotate: ON</button>
            </div>
        </div>

        <script>
            const container = document.getElementById('container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0B1120);
            scene.fog = new THREE.FogExp2(0x0B1120, 0.012);

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(32, 22, 36);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.6;
            controls.target.set(0, 4, 0);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0x1E293B, 1.2);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0x93C5FD, 1.4);
            dirLight.position.set(20, 40, 20);
            dirLight.castShadow = true;
            scene.add(dirLight);

            const blueFill = new THREE.DirectionalLight(0x38BDF8, 0.6);
            blueFill.position.set(-25, 20, -25);
            scene.add(blueFill);

            // Red/Yellow Alarm Lights if in flood state
            const isRedAlert = {str(is_red).lower()};
            const isYellowWatch = {str(is_yellow).lower()};
            let alarmLight = null;
            if (isRedAlert || isYellowWatch) {{
                alarmLight = new THREE.PointLight(isRedAlert ? 0xEF4444 : 0xF59E0B, 3.5, 60);
                alarmLight.position.set(0, 10, -20);
                scene.add(alarmLight);
            }}

            // -------------------------------------------------------------
            // Procedural Himalayan Singati Canyon Terrain Walls
            // -------------------------------------------------------------
            const canyonWidth = 60;
            const canyonLength = 80;
            const xSeg = 70;
            const zSeg = 70;
            const terrainGeom = new THREE.PlaneGeometry(canyonWidth, canyonLength, xSeg, zSeg);
            terrainGeom.rotateX(-Math.PI / 2);

            const pos = terrainGeom.attributes.position;
            for (let i = 0; i < pos.count; i++) {{
                const x = pos.getX(i);
                const z = pos.getZ(i);
                
                // U-to-V-shaped canyon floor with steep Himalayan gorge side-cliffs
                const distFromCenter = Math.abs(x);
                let y = 0;
                if (distFromCenter < 5.0) {{
                    y = (distFromCenter / 5.0) * 1.5; // river channel bed
                }} else {{
                    const cliff = distFromCenter - 5.0;
                    y = 1.5 + Math.pow(cliff * 0.55, 1.7) + Math.sin(x * 0.4) * 0.8 + Math.cos(z * 0.2) * 0.6;
                }}
                
                // Rocky irregularity
                y += Math.sin(x * 0.9 + z * 0.4) * 0.4 + Math.sin(z * 0.8) * 0.3;
                pos.setY(i, y);
            }}
            terrainGeom.computeVertexNormals();

            const terrainMat = new THREE.MeshStandardMaterial({{
                color: 0x1E293B,
                roughness: 0.88,
                metalness: 0.15,
                flatShading: true
            }});
            const terrainMesh = new THREE.Mesh(terrainGeom, terrainMat);
            terrainMesh.receiveShadow = true;
            scene.add(terrainMesh);

            // Wireframe contour accents on canyon walls
            const wireMat = new THREE.MeshBasicMaterial({{
                color: 0x334155,
                wireframe: true,
                transparent: true,
                opacity: 0.25
            }});
            const wireMesh = new THREE.Mesh(terrainGeom, wireMat);
            scene.add(wireMesh);

            // -------------------------------------------------------------
            // Concrete Dam & Spillway Structure Upstream (Z = -34)
            // -------------------------------------------------------------
            const damGroup = new THREE.Group();
            const damMat = new THREE.MeshStandardMaterial({{
                color: 0x334155,
                roughness: 0.6,
                metalness: 0.2
            }});
            const damWallGeom = new THREE.BoxGeometry(32, 14, 4);
            const damWall = new THREE.Mesh(damWallGeom, damMat);
            damWall.position.set(0, 7, -35);
            damGroup.add(damWall);

            // Spillway Piers
            for (let px = -10; px <= 10; px += 5) {{
                const pierGeom = new THREE.BoxGeometry(1.2, 16, 6);
                const pier = new THREE.Mesh(pierGeom, damMat);
                pier.position.set(px, 8, -35);
                damGroup.add(pier);

                // Dam Crest Beacon Lights
                const beaconGeom = new THREE.SphereGeometry(0.35, 8, 8);
                const beaconMat = new THREE.MeshBasicMaterial({{
                    color: isRedAlert ? 0xEF4444 : isYellowWatch ? 0xF59E0B : 0x10B981
                }});
                const beacon = new THREE.Mesh(beaconGeom, beaconMat);
                beacon.position.set(px, 16.5, -35);
                damGroup.add(beacon);
            }}
            scene.add(damGroup);

            // -------------------------------------------------------------
            // Warning Threshold Planes: 6.02 m³/s (Yellow) & 6.69 m³/s (Red)
            // -------------------------------------------------------------
            // Mapping discharge to 3D Y coordinates:
            // 0.35 m³/s -> Y = 1.2
            // 6.02 m³/s -> Y = 5.4 (Yellow Watch Plane)
            // 6.69 m³/s -> Y = 6.6 (Red Danger Line)
            // 7.50 m³/s -> Y = 7.8 (Overtopping)
            
            const yellowPlaneY = 5.4;
            const redPlaneY = 6.6;

            // 1. Yellow Watch Plane (6.02 m³/s)
            const yellowPlaneGeom = new THREE.PlaneGeometry(22, 70);
            yellowPlaneGeom.rotateX(-Math.PI / 2);
            const yellowPlaneMat = new THREE.MeshBasicMaterial({{
                color: 0xF59E0B,
                transparent: true,
                opacity: 0.22,
                side: THREE.DoubleSide
            }});
            const yellowPlane = new THREE.Mesh(yellowPlaneGeom, yellowPlaneMat);
            yellowPlane.position.set(0, yellowPlaneY, -2);
            scene.add(yellowPlane);

            const yellowEdges = new THREE.LineSegments(
                new THREE.EdgesGeometry(yellowPlaneGeom),
                new THREE.LineBasicMaterial({{ color: 0xF59E0B, transparent: true, opacity: 0.7 }})
            );
            yellowEdges.position.set(0, yellowPlaneY, -2);
            scene.add(yellowEdges);

            // 2. Red Danger Plane (6.69 m³/s)
            const redPlaneGeom = new THREE.PlaneGeometry(26, 70);
            redPlaneGeom.rotateX(-Math.PI / 2);
            const redPlaneMat = new THREE.MeshBasicMaterial({{
                color: 0xEF4444,
                transparent: true,
                opacity: 0.28,
                side: THREE.DoubleSide
            }});
            const redPlane = new THREE.Mesh(redPlaneGeom, redPlaneMat);
            redPlane.position.set(0, redPlaneY, -2);
            scene.add(redPlane);

            const redEdges = new THREE.LineSegments(
                new THREE.EdgesGeometry(redPlaneGeom),
                new THREE.LineBasicMaterial({{ color: 0xEF4444, transparent: true, opacity: 0.85 }})
            );
            redEdges.position.set(0, redPlaneY, -2);
            scene.add(redEdges);

            // -------------------------------------------------------------
            // Dynamic 3D Water Surface Mesh (Rises with pred_flow)
            // -------------------------------------------------------------
            const predFlow = {pred_flow:.2f};
            
            // Calculate dynamic water height:
            let waterTargetY = 1.0;
            if (predFlow <= 6.02) {{
                waterTargetY = 1.0 + (predFlow / 6.02) * (yellowPlaneY - 1.0);
            }} else if (predFlow <= 6.69) {{
                waterTargetY = yellowPlaneY + ((predFlow - 6.02) / (6.69 - 6.02)) * (redPlaneY - yellowPlaneY);
            }} else {{
                waterTargetY = redPlaneY + ((predFlow - 6.69) / 2.0) * 1.6;
            }}

            const waterWidth = 28;
            const waterLength = 76;
            const waterGeom = new THREE.PlaneGeometry(waterWidth, waterLength, 40, 50);
            waterGeom.rotateX(-Math.PI / 2);

            let waterColor = 0x06B6D4; // Glacial cyan
            if (isRedAlert) {{
                waterColor = 0xEF4444; // Violent red flood surge
            }} else if (isYellowWatch) {{
                waterColor = 0xF59E0B; // Silted monsoon surge
            }}

            const waterMat = new THREE.MeshPhysicalMaterial({{
                color: waterColor,
                transparent: true,
                opacity: 0.82,
                roughness: 0.15,
                metalness: 0.1,
                transmission: 0.4,
                ior: 1.33
            }});
            const waterMesh = new THREE.Mesh(waterGeom, waterMat);
            waterMesh.position.set(0, waterTargetY, -2);
            scene.add(waterMesh);

            // Water Wave Animation
            const origWaterPositions = waterGeom.attributes.position.clone();

            // -------------------------------------------------------------
            // Camera View Switcher
            // -------------------------------------------------------------
            window.setCameraView = function(view) {{
                if (view === 'iso') {{
                    camera.position.set(32, 22, 36);
                    controls.target.set(0, 4, 0);
                }} else if (view === 'spillway') {{
                    camera.position.set(0, 16, -48);
                    controls.target.set(0, 4, -10);
                }} else if (view === 'cross') {{
                    camera.position.set(38, 7, 0);
                    controls.target.set(0, 6, 0);
                }}
                controls.update();
            }};

            window.toggleAutoOrbit = function() {{
                controls.autoRotate = !controls.autoRotate;
                document.getElementById('orbit-btn').innerText = controls.autoRotate ? '🔄 Auto-Rotate: ON' : '⏸️ Auto-Rotate: OFF';
            }};

            // -------------------------------------------------------------
            // Animation Loop
            // -------------------------------------------------------------
            let clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const t = clock.getElapsedTime();
                
                // Animate water waves
                const wPos = waterGeom.attributes.position;
                const orig = origWaterPositions;
                const waveSpeed = isRedAlert ? 4.5 : isYellowWatch ? 3.0 : 1.8;
                const waveAmp = isRedAlert ? 0.35 : isYellowWatch ? 0.22 : 0.12;

                for (let i = 0; i < wPos.count; i++) {{
                    const x = orig.getX(i);
                    const z = orig.getZ(i);
                    const wave = Math.sin(x * 0.4 + t * waveSpeed) * waveAmp + Math.cos(z * 0.3 + t * waveSpeed * 0.8) * (waveAmp * 0.7);
                    wPos.setY(i, orig.getY(i) + wave);
                }}
                waterGeom.computeVertexNormals();
                wPos.needsUpdate = true;

                // Pulsing edge warning in Red/Yellow alert
                if (alarmLight) {{
                    alarmLight.intensity = 2.5 + Math.sin(t * 5.0) * 1.5;
                }}

                controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            // Responsive Resize
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    components.html(three_js_html, height=540)

with tab_plotly:
    # High-precision 3D Plotly Surface & Breach Contour
    x_range = np.linspace(-20, 20, 50)
    y_range = np.linspace(-30, 30, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Canyon bedrock elevation
    Z_terrain = 1.0 + 0.03 * (X**2) + 0.0004 * (X**4) + 0.5 * np.sin(X * 0.3) * np.cos(Y * 0.1)
    
    # Scaled river water level
    water_z = 1.0 + (pred_flow / 6.69) * 5.6
    Z_water = np.full_like(Z_terrain, water_z)
    
    # Horizontal Warning Planes
    Z_yellow = np.full_like(Z_terrain, 1.0 + (YELLOW_THRESHOLD / 6.69) * 5.6)
    Z_red = np.full_like(Z_terrain, 1.0 + (DANGER_THRESHOLD / 6.69) * 5.6)

    fig_3d = go.Figure()
    
    # Valley Terrain Surface
    fig_3d.add_trace(
        go.Surface(
            x=X, y=Y, z=Z_terrain,
            colorscale="Earth",
            showscale=False,
            name="Canyon Bedrock",
            opacity=0.92,
        )
    )
    
    # Water Surface
    water_color_scale = "Reds" if is_red else "YlOrBr" if is_yellow else "Teal"
    fig_3d.add_trace(
        go.Surface(
            x=X, y=Y, z=Z_water,
            colorscale=water_color_scale,
            showscale=False,
            name=f"Tomorrow Flow ({pred_flow:.2f} m³/s)",
            opacity=0.75,
        )
    )
    
    # Warning Threshold Planes
    fig_3d.add_trace(
        go.Surface(
            x=X, y=Y, z=Z_yellow,
            colorscale=[[0, "rgba(245, 158, 11, 0.4)"], [1, "rgba(245, 158, 11, 0.4)"]],
            showscale=False,
            name=f"Yellow Watch ({YELLOW_THRESHOLD:.2f} m³/s)",
            opacity=0.35,
        )
    )
    
    fig_3d.add_trace(
        go.Surface(
            x=X, y=Y, z=Z_red,
            colorscale=[[0, "rgba(239, 68, 68, 0.5)"], [1, "rgba(239, 68, 68, 0.5)"]],
            showscale=False,
            name=f"Red Danger ({DANGER_THRESHOLD:.2f} m³/s)",
            opacity=0.45,
        )
    )

    fig_3d.update_layout(
        title=f"Singati Gorge Cross-Section Surface & Overtopping Plane (Target Flow: {pred_flow:.2f} m³/s)",
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

# ---------------------------------------------------------
# Visual Section 3: Telemetry Charts & Explainable AI
# ---------------------------------------------------------
st.markdown("### 📈 Telemetry Forecast Hydrograph & Model Explainability")

h_col1, h_col2 = st.columns([6, 5])

with h_col1:
    st.markdown("#### 24-Hour Singati River Hydrograph & Surge Cone")
    
    # 24-Hour Hydrograph (Yesterday -> Today -> Tomorrow)
    timeline_hours = np.linspace(-24, 24, 49)
    # Piecewise interpolation from q_yesterday (-24) to q_mid (0) to pred_flow (+24)
    past_curve = np.interp(timeline_hours[timeline_hours <= 0], [-24, 0], [q_yesterday, q_mid])
    future_curve = np.interp(timeline_hours[timeline_hours > 0], [0, 24], [q_mid, pred_flow])
    full_hydrograph = np.concatenate([past_curve, future_curve])
    
    # Confidence interval cone spreading from Today (0 spread) to Tomorrow
    future_hours = timeline_hours[timeline_hours >= 0]
    ci_factor = future_hours / 24.0
    cone_low = q_mid + (low_flow - q_mid) * ci_factor
    cone_high = q_mid + (high_flow - q_mid) * ci_factor

    fig_hydro = go.Figure()

    # 80% Confidence Interval Shaded Cone
    fig_hydro.add_trace(
        go.Scatter(
            x=future_hours,
            y=cone_high,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_hydro.add_trace(
        go.Scatter(
            x=future_hours,
            y=cone_low,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.18)" if not is_red else "rgba(239, 68, 68, 0.22)",
            name="80% Quantile Confidence Cone",
            hoverinfo="skip",
        )
    )

    # Danger Threshold Horizontal Line
    fig_hydro.add_hline(
        y=DANGER_THRESHOLD,
        line_dash="dash",
        line_color="#EF4444",
        line_width=2,
        annotation_text=f"Red Danger Threshold: {DANGER_THRESHOLD:.2f} m³/s",
        annotation_position="top left",
        annotation_font_color="#EF4444",
    )

    # Yellow Watch Threshold Horizontal Line
    fig_hydro.add_hline(
        y=YELLOW_THRESHOLD,
        line_dash="dot",
        line_color="#F59E0B",
        line_width=1.5,
        annotation_text=f"Yellow Watch Buffer: {YELLOW_THRESHOLD:.2f} m³/s",
        annotation_position="bottom left",
        annotation_font_color="#F59E0B",
    )

    # Historical Flow (Yesterday -> Today)
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours <= 0],
            y=past_curve,
            mode="lines+markers",
            name="Measured Telemetry (Past 24h)",
            line=dict(color="#94A3B8", width=3),
            marker=dict(size=6),
        )
    )

    # Predicted Forecast Flow (Today -> Tomorrow)
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours >= 0],
            y=future_curve,
            mode="lines+markers",
            name="AI Forecast Trajectory (+24h)",
            line=dict(color=alert_color, width=4),
            marker=dict(size=[8] + [0] * (len(future_curve) - 2) + [10], color=alert_color),
        )
    )

    # Key Data Markers
    fig_hydro.add_annotation(
        x=-24, y=q_yesterday,
        text=f"Yesterday: {q_yesterday:.2f}",
        showarrow=True, arrowhead=2, arrowcolor="#94A3B8",
        font=dict(color="#94A3B8", size=10),
    )
    fig_hydro.add_annotation(
        x=0, y=q_mid,
        text=f"Today: {q_mid:.2f}",
        showarrow=True, arrowhead=2, arrowcolor="#38BDF8",
        font=dict(color="#38BDF8", size=11, family="monospace"),
    )
    fig_hydro.add_annotation(
        x=24, y=pred_flow,
        text=f"Tomorrow Peak: {pred_flow:.2f} ({delta_pred:+.2f})",
        showarrow=True, arrowhead=2, arrowcolor=alert_color,
        font=dict(color=alert_color, size=11, family="monospace"),
    )

    fig_hydro.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis=dict(
            title="Forecast Timeline (Hours relative to Today T=0)",
            gridcolor="#1E293B",
            zerolinecolor="#334155",
            tickvals=[-24, -12, 0, 12, 24],
            ticktext=["-24h (Yesterday)", "-12h", "T=0 (Today)", "+12h", "+24h (Tomorrow)"],
        ),
        yaxis=dict(
            title="River Discharge (m³/s)",
            gridcolor="#1E293B",
            zerolinecolor="#334155",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=30, b=40),
        height=380,
    )
    st.plotly_chart(fig_hydro, use_container_width=True)

with h_col2:
    st.markdown("#### Council of 6 AI Models vs. Stacked Meta-Ensemble")
    
    # Calculate each model's full predicted discharge (q_mid + delta)
    model_flow_preds = {name: max(0.0, q_mid + base_deltas_dict[name]) for name in model_names}
    
    # Chart items sorted
    names_display = list(model_names) + ["🏆 Meta-Ensemble"]
    flows_display = [model_flow_preds[name] for name in model_names] + [pred_flow]
    deltas_display = [base_deltas_dict[name] for name in model_names] + [delta_pred]

    bar_colors = []
    for f in flows_display:
        if f >= DANGER_THRESHOLD:
            bar_colors.append("#EF4444")
        elif f >= YELLOW_THRESHOLD:
            bar_colors.append("#F59E0B")
        else:
            bar_colors.append("#06B6D4")

    # Distinct highlight for the Meta-Ensemble bar
    bar_colors[-1] = "#38BDF8" if flows_display[-1] < YELLOW_THRESHOLD else alert_color

    fig_models = go.Figure()
    
    fig_models.add_trace(
        go.Bar(
            y=names_display,
            x=flows_display,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="#F8FAFC", width=[1] * len(model_names) + [2.5]),
            ),
            text=[f"{f:.2f} m³/s ({d:+.2f})" for f, d in zip(flows_display, deltas_display)],
            textposition="auto",
            hoverinfo="text",
            hovertext=[
                f"<b>{name}</b><br/>Tomorrow Flow: {f:.2f} m³/s<br/>Predicted Surge: {d:+.2f} m³/s"
                for name, f, d in zip(names_display, flows_display, deltas_display)
            ],
        )
    )

    # Danger Threshold Vertical Reference Line
    fig_models.add_vline(
        x=DANGER_THRESHOLD,
        line_dash="dash",
        line_color="#EF4444",
        line_width=2,
        annotation_text="Danger (6.69)",
        annotation_position="top right",
        annotation_font_color="#EF4444",
    )
    
    # Yellow Watch Reference Line
    fig_models.add_vline(
        x=YELLOW_THRESHOLD,
        line_dash="dot",
        line_color="#F59E0B",
        line_width=1.5,
        annotation_text="Watch (6.02)",
        annotation_position="bottom right",
        annotation_font_color="#F59E0B",
    )

    # Today's baseline line
    fig_models.add_vline(
        x=q_mid,
        line_dash="solid",
        line_color="#64748B",
        line_width=1.2,
        annotation_text=f"Today ({q_mid:.2f})",
        annotation_position="top left",
        annotation_font_color="#94A3B8",
    )

    fig_models.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis=dict(
            title="Predicted Tomorrow Discharge (m³/s)",
            gridcolor="#1E293B",
            zerolinecolor="#334155",
        ),
        yaxis=dict(
            gridcolor="#1E293B",
        ),
        margin=dict(l=20, r=20, t=30, b=40),
        height=380,
    )
    st.plotly_chart(fig_models, use_container_width=True)

# ---------------------------------------------------------
# Compound Flood Physics Gauges
# ---------------------------------------------------------
st.markdown("#### ⚡ Compound Himalayan Flood Physics Gauges")
st.caption(
    "Coupled thermodynamics, high-altitude precipitation, and vadose-zone moisture physics driving river surge."
)

g1, g2, g3, g4 = st.columns(4)

with g1:
    fig_g1 = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=runoff_potential,
            title={"text": "Runoff Potential Index<br><span style='font-size:0.75rem; color:#94A3B8;'>Precip × (Soil Saturation)²</span>"},
            gauge={
                "axis": {"range": [0, 20]},
                "bar": {"color": "#38BDF8"},
                "steps": [
                    {"range": [0, 6], "color": "rgba(16, 185, 129, 0.3)"},
                    {"range": [6, 12], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [12, 20], "color": "rgba(239, 68, 68, 0.4)"},
                ],
                "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 11.0},
            },
        )
    )
    fig_g1.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=220, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_g1, use_container_width=True)

with g2:
    fig_g2 = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=glacial_melt_energy,
            title={"text": "Glacial Melt Energy<br><span style='font-size:0.75rem; color:#94A3B8;'>Thermal Flux × Solar Rad</span>"},
            gauge={
                "axis": {"range": [0, 50]},
                "bar": {"color": "#06B6D4"},
                "steps": [
                    {"range": [0, 15], "color": "rgba(16, 185, 129, 0.3)"},
                    {"range": [15, 30], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [30, 50], "color": "rgba(239, 68, 68, 0.4)"},
                ],
                "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 30.0},
            },
        )
    )
    fig_g2.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=220, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_g2, use_container_width=True)

with g3:
    fig_g3 = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rain_intensity_mid,
            title={"text": "Cloudburst Intensity<br><span style='font-size:0.75rem; color:#94A3B8;'>Peak Rainfall (mm/hr)</span>"},
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

# ---------------------------------------------------------
# Technical Architecture & Methodology Expander
# ---------------------------------------------------------
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
        - **Harmonic Seasonality:** Dual Fourier harmonic terms ($\\\\sin, \\\\cos$) capturing intra-annual monsoon seasonality.

        #### 3. Council of 6 Base Estimators + RidgeCV Stacking Meta-Learner
        - **Base Estimators:** `XGBoost`, `LightGBM`, `CatBoost`, `HistGradientBoosting`, `ExtraTrees`, and `Deep Residual MLP`.
        - **Stacked Meta-Learner:** Cross-validated `RidgeCV` meta-regressor weighting base model deltas to eliminate single-model bias.
        - **Quantile Regression (80% Confidence):** Gradient Boosting regressors fitted at $\\alpha = 0.10$ and $\\alpha = 0.90$ to bound prediction uncertainty.
        - **Logistic Calibrated Flood Classifier:** Direct binary probability estimating riverbank breach risk.
        - **Threshold Calibration:** Statistically calibrated 95th-percentile flood discharge limit of **6.69 m³/s**, with a **6.02 m³/s (10% buffer)** Yellow Watch trigger.
        """
    )
