"""
📡 Live Tamakoshi Telemetry & 24-Hour Flood Prediction
------------------------------------------------------
Autonomous live-feed page backed by the services/tamakoshi_api.py
API Routing & Inference Engine. Pulls GloFAS + Open-Meteo weather
telemetry for all 4 Tamakoshi corridor stations, engineers the full
58-feature hydro-physics pipeline, and runs the 6-model stacking
meta-ensemble to produce a 24-hour Singati Gorge flood forecast.
"""

import sys
import os
import types

# _loss shim must run before any joblib / sklearn import
try:
    import _loss  # noqa: F401
except ModuleNotFoundError:
    import sklearn._loss._loss as _cy_loss
    import sklearn._loss.loss as _py_loss

    _loss_shim = types.ModuleType("_loss")
    _loss_shim.__dict__.update(_py_loss.__dict__)
    _loss_shim.__dict__.update(_cy_loss.__dict__)
    sys.modules["_loss"] = _loss_shim

# Ensure project root is on the import path so `services` resolves
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from services.tamakoshi_api import (
    TAMAKOSHI_STATIONS,
    compute_forecast_from_inputs,
    load_model_bundle,
    route_live_basin_telemetry,
)

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="📡 Live Tamakoshi Forecast | Flood AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Dark-Slate Hydrology Command Center Theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-main: #07111f;
        --bg-panel: rgba(12, 19, 31, 0.85);
        --accent-cyan: #67e8f9;
        --accent-blue: #38bdf8;
        --accent-emerald: #2CE5B5;
        --accent-amber: #fbbf24;
        --accent-red: #f87171;
        --text-primary: #f8fafc;
        --text-secondary: #b8c3d6;
        --text-muted: #8ea3bf;
    }
    html, body { background: var(--bg-main); }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(44, 229, 181, 0.08), transparent 22%),
            radial-gradient(circle at bottom right, rgba(56, 189, 248, 0.07), transparent 22%),
            linear-gradient(180deg, #07111f 0%, #0b1728 100%);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1540px; }
    section[data-testid="stSidebar"] {
        background: rgba(8, 13, 22, 0.94);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }
    .glass-card {
        position: relative;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.74));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 18px 40px rgba(2, 6, 23, 0.32);
        backdrop-filter: blur(14px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(44, 229, 181, 0.50);
        box-shadow: 0 24px 50px rgba(44, 229, 181, 0.10);
    }
    .glass-card .label {
        margin-bottom: 0.45rem; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted);
    }
    .glass-card .value {
        font-size: clamp(1.4rem, 2vw, 1.95rem); font-weight: 800;
        line-height: 1.2; color: var(--text-primary);
    }
    .glass-card .delta { margin-top: 0.4rem; font-size: 0.78rem; font-weight: 600; }
    .action-banner {
        border-radius: 18px; padding: 1.3rem 1.35rem; margin-bottom: 1.3rem;
        border: 1px solid; box-shadow: 0 18px 45px rgba(2, 6, 23, 0.26);
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
    .banner-title { margin-bottom: 0.65rem; font-size: 1.18rem; font-weight: 800; letter-spacing: -0.02em; }
    .banner-text { color: var(--text-primary); font-size: 0.94rem; line-height: 1.6; }
    .protocol-list { margin-top: 0.85rem; padding-left: 1.1rem; color: rgba(226, 232, 240, 0.9); line-height: 1.65; }
    .protocol-list li { margin-bottom: 0.12rem; }
    h3 { margin-top: 1.6rem !important; margin-bottom: 0.25rem !important; color: var(--text-primary) !important; }
    .stCaption { color: var(--text-secondary) !important; }
    .stTabs [role="tab"] {
        background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px 12px 0 0; color: var(--text-secondary); padding: 0.55rem 0.9rem;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, rgba(44, 229, 181, 0.17), rgba(15, 23, 42, 0.8));
        color: var(--text-primary); border-color: rgba(44, 229, 181, 0.45);
    }
    .source-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 0.5rem 0.85rem; border-radius: 999px;
        font-size: 0.74rem; font-weight: 700; letter-spacing: 0.02em;
        backdrop-filter: blur(10px);
    }
    .source-chip.live {
        background: rgba(16, 185, 129, 0.18); border: 1px solid rgba(44, 229, 181, 0.5); color: #2CE5B5;
    }
    .source-chip.fallback {
        background: rgba(245, 158, 11, 0.18); border: 1px solid rgba(251, 191, 36, 0.5); color: #FBBF24;
    }
    .stApp > div > div { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Branding & Controls
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='
        padding: 1rem 0.9rem 0.85rem 0.9rem; margin: 0.2rem 0 1.1rem 0;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.97), rgba(8, 13, 22, 0.90));
        border: 1px solid rgba(44, 229, 181, 0.28);
        box-shadow: 0 14px 28px rgba(2, 6, 23, 0.32);
        text-align: center;
    '>
        <div style='font-size: 0.70rem; letter-spacing: 0.14em; text-transform: uppercase; color: #2CE5B5; font-weight: 700; margin-bottom: 0.3rem;'>Live Telemetry Engine</div>
        <h2 style='margin: 0; font-size: 1.35rem; color: #F8FAFC; line-height: 1.2;'>📡 Tamakoshi Forecast</h2>
        <div style='margin-top: 0.3rem; font-size: 0.78rem; color: #9fb4cf;'>GloFAS + Weather API → AI Ensemble</div>
    </div>
    """,
    unsafe_allow_html=True,
)
past_days = st.sidebar.slider("📅 Historical Lookback (days)", 3, 14, 7, help="Number of past days to pull from the Open-Meteo APIs.")
st.sidebar.markdown("<hr style='border-color: rgba(148, 163, 184, 0.12); margin: 0.8rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style='font-size: 0.78rem; color: #8ea3bf; line-height: 1.6; padding: 0.4rem 0.2rem;'>
        <b style='color: #67e8f9;'>API Routes:</b><br>
        • <code>flood-api.open-meteo.com</code> → GloFAS river discharge for 4 stations<br>
        • <code>api.open-meteo.com</code> → Basin weather (rain, temp, soil, solar)<br><br>
        <b style='color: #67e8f9;'>Failover:</b> Verified Nov 10–16, 2026 Tamakoshi snapshot when API is unreachable.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Model Bundle & Route Telemetry
# ---------------------------------------------------------
bundle = load_model_bundle()
telemetry = route_live_basin_telemetry(past_days=past_days)

source_mode = telemetry["source_mode"]
source_label = telemetry["status_label"]
synced_at = telemetry["synced_at"]
history_df = telemetry["history_df"]
inp = telemetry["inputs"]

fc = compute_forecast_from_inputs(bundle, inp)

pred_flow = fc["pred_flow"]
delta_pred = fc["delta_pred"]
low_flow = fc["low_flow"]
high_flow = fc["high_flow"]
flood_prob = fc["flood_prob"]
danger_threshold = fc["danger_threshold"]
yellow_trigger = fc["yellow_trigger"]
alert_level = fc["alert_level"]
individual_preds = fc["individual_preds"]

# Alert styling helpers
ALERT_STYLES = {
    "RED_ALERT": {"color": "#EF4444", "icon": "🚨", "label": "RED ALERT", "banner": "red"},
    "YELLOW_WATCH": {"color": "#F59E0B", "icon": "⚠️", "label": "YELLOW WATCH", "banner": "yellow"},
    "GREEN_NORMAL": {"color": "#10B981", "icon": "✅", "label": "GREEN NORMAL", "banner": "green"},
}
sty = ALERT_STYLES[alert_level]

# ---------------------------------------------------------
# Hero Header
# ---------------------------------------------------------
chip_class = "live" if source_mode == "LIVE_API" else "fallback"
chip_icon = "🟢" if source_mode == "LIVE_API" else "🟡"

st.markdown(
    f"""
    <div style="
        position: relative; overflow: hidden; margin-bottom: 1.4rem;
        padding: 1.8rem 2rem 1.5rem 2rem; border-radius: 22px;
        border: 1px solid rgba(44, 229, 181, 0.28);
        background:
            linear-gradient(135deg, rgba(8, 15, 25, 0.72), rgba(7, 17, 31, 0.80)),
            url('https://images.unsplash.com/photo-1544735716-392fe2489ffa?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat;
        box-shadow: 0 24px 60px rgba(2, 6, 23, 0.6);
    ">
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 0.65rem;">
                <div style="display: flex; align-items: center; gap: 12px; font-size: clamp(1.6rem, 2.4vw, 2.4rem); font-weight: 800; color: #F8FAFC; letter-spacing: -0.04em;">
                    <span>📡 LIVE TAMAKOSHI TELEMETRY & 24-HOUR FLOOD PREDICTION</span>
                </div>
                <div class="source-chip {chip_class}">
                    {chip_icon} {source_label}
                </div>
            </div>
            <div style="max-width: 960px; font-size: 0.98rem; line-height: 1.7; color: rgba(226, 232, 240, 0.90); margin-bottom: 1rem;">
                Autonomous inference pipeline: Open-Meteo GloFAS river discharge (4 Tamakoshi corridor stations) + basin weather
                (rain, soil, temperature, solar) → 58-feature hydrological physics → 6-model stacking meta-ensemble → Singati Gorge
                24-hour flood forecast with 80% confidence bounds.
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 0.55rem;">
                <span style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.48rem 0.8rem; border-radius: 999px; font-size: 0.73rem; font-weight: 700; background: rgba(15, 23, 42, 0.68); border: 1px solid rgba(44, 229, 181, 0.40); color: #2CE5B5; backdrop-filter: blur(10px);">
                    🛰️ 4 Cascading Stations: Tsho Rolpa → Gongar → Singati → Lower Tamakoshi
                </span>
                <span style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.48rem 0.8rem; border-radius: 999px; font-size: 0.73rem; font-weight: 700; background: rgba(15, 23, 42, 0.68); border: 1px solid rgba(56, 189, 248, 0.40); color: #38BDF8; backdrop-filter: blur(10px);">
                    🧠 6-Model AI Ensemble: XGBoost · LightGBM · CatBoost · HistGBM · ExtraTrees · DeepMLP + RidgeCV Meta
                </span>
                <span style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.48rem 0.8rem; border-radius: 999px; font-size: 0.73rem; font-weight: 700; background: rgba(15, 23, 42, 0.68); border: 1px solid {sty['color']}55; color: {sty['color']}; backdrop-filter: blur(10px);">
                    {sty['icon']} Status: {sty['label']} ({pred_flow:.2f} m³/s)
                </span>
                <span style="display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.48rem 0.8rem; border-radius: 999px; font-size: 0.73rem; font-weight: 700; background: rgba(15, 23, 42, 0.68); border: 1px solid rgba(148, 163, 184, 0.25); color: #94A3B8; backdrop-filter: blur(10px);">
                    🕐 Synced: {synced_at}
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Emergency Action Banner
# ---------------------------------------------------------
if alert_level == "RED_ALERT":
    st.markdown(
        f"""
        <div class="action-banner red">
            <div class="banner-title" style="color: #FCA5A5;">
                🚨 RED ALERT — CRITICAL FLOOD & OVERTOPPING BREACH IMMINENT
            </div>
            <div class="banner-text">
                AI-predicted 24-hour Singati discharge: <b>{pred_flow:.2f} m³/s</b>
                (Surge: {delta_pred:+.2f} m³/s), exceeding the 95th-percentile safety limit of
                <b>{danger_threshold:.2f} m³/s</b> by <b>{(pred_flow - danger_threshold):+.2f} m³/s</b>.
                Inundation likelihood: <b>{flood_prob:.1f}%</b>.
            </div>
            <ul class="protocol-list">
                <li><b>Immediate Evacuation:</b> CAP/SMS alarms to Singati Bazaar, Gongar, Jagat, Lamabagar.</li>
                <li><b>Dam Control:</b> Actuate Upper Tamakoshi radial spillway gates.</li>
                <li><b>First Responders:</b> Mobilize Dolakha DDMC & APF water rescue units.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif alert_level == "YELLOW_WATCH":
    st.markdown(
        f"""
        <div class="action-banner yellow">
            <div class="banner-title" style="color: #FCD34D;">
                ⚠️ YELLOW WATCH — RAPID MONSOON SURGE IN UPPER CATCHMENT
            </div>
            <div class="banner-text">
                Predicted tomorrow flow: <b>{pred_flow:.2f} m³/s</b>, entering the 10% buffer zone
                (<b>{yellow_trigger:.2f} – {danger_threshold:.2f} m³/s</b>).
                Flood probability: <b>{flood_prob:.1f}%</b>.
            </div>
            <ul class="protocol-list">
                <li><b>DDMC Standby:</b> Notify Dolakha disaster coordinators.</li>
                <li><b>High-Frequency Telemetry:</b> Switch upstream sensors to 15-min polling.</li>
                <li><b>Hydropower Coordination:</b> Alert NEA for potential reservoir spill.</li>
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
                Discharge safely contained at <b>{pred_flow:.2f} m³/s</b>
                (Clearance: <b>{(danger_threshold - pred_flow):.2f} m³/s</b>).
                AI breach probability: <b>{flood_prob:.1f}%</b>.
            </div>
            <ul class="protocol-list">
                <li><b>Operational:</b> Maintain 456 MW generation at Upper Tamakoshi.</li>
                <li><b>Sentinel:</b> Routine autonomous model inference active.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Glassmorphism KPI Cards
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="label">Forecast Tomorrow Flow</div>
            <div class="value" style="color: {sty['color']};">{pred_flow:.2f} <span style="font-size:1rem; color:#94A3B8;">m³/s</span></div>
            <div class="delta" style="color: {'#EF4444' if delta_pred > 0 else '#10B981'};">
                {delta_pred:+.2f} m³/s vs today ({inp['q_mid']:.2f})
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
            <div class="value">{low_flow:.2f} – {high_flow:.2f} <span style="font-size:1rem; color:#94A3B8;">m³/s</span></div>
            <div class="delta" style="color: #38BDF8;">Uncertainty: ±{ci_range / 2.0:.2f} m³/s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="label">Red / Yellow Thresholds</div>
            <div class="value">{danger_threshold:.2f} <span style="font-size:1rem; color:#94A3B8;">m³/s</span></div>
            <div class="delta" style="color: #F59E0B;">Yellow Buffer: {yellow_trigger:.2f} m³/s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="label">AI Flood Probability</div>
            <div class="value" style="color: {sty['color']};">{flood_prob:.1f}<span style="font-size:1.1rem; color:#94A3B8;">%</span></div>
            <div class="delta" style="color: {sty['color']};">Status: {sty['label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabbed Sections
# ---------------------------------------------------------
tab_hydro, tab_council, tab_map, tab_data = st.tabs([
    "📈 Hydrograph & Surge Forecast",
    "🧠 Council of 6 AI Models",
    "🗺️ 3D Telemetry Map",
    "📋 Raw Telemetry Data & Inputs",
])

# ---------------------------------------------------------
# Tab 1: Hydrograph & Surge Forecast
# ---------------------------------------------------------
with tab_hydro:
    st.markdown("### 📈 24-Hour Singati River Hydrograph & Confidence Cone")
    st.caption("Live sensor history (past days) stitched to the AI 24-hour forecast with 80% quantile regression confidence bounds.")

    q_mid = inp["q_mid"]
    q_yesterday = inp["q_yesterday"]

    # Build timeline
    timeline_hours = np.linspace(-24, 24, 49)
    past_curve = np.interp(timeline_hours[timeline_hours <= 0], [-24, 0], [q_yesterday, q_mid])
    future_curve = np.interp(timeline_hours[timeline_hours > 0], [0, 24], [q_mid, pred_flow])
    future_hours = timeline_hours[timeline_hours >= 0]
    ci_factor = future_hours / 24.0
    ci_upper = pred_flow + (high_flow - pred_flow) * ci_factor
    ci_lower = np.maximum(0.0, pred_flow - (pred_flow - low_flow) * ci_factor)

    fig_hydro = go.Figure()
    # Confidence cone
    fig_hydro.add_trace(
        go.Scatter(
            x=np.concatenate([future_hours, future_hours[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(44, 229, 181, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="80% Confidence Cone",
        )
    )
    # Past telemetry
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours <= 0],
            y=past_curve,
            mode="lines+markers",
            name="Sensor History (T-24h to T0)",
            line=dict(color="#38BDF8", width=3.5),
            marker=dict(size=5, color="#67E8F9"),
        )
    )
    # AI forecast
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours >= 0],
            y=future_curve,
            mode="lines+markers",
            name="AI 24h Forecast (T0→T+24h)",
            line=dict(color=sty["color"], width=4, dash="dash"),
            marker=dict(size=6, color=sty["color"]),
        )
    )
    # Threshold lines
    fig_hydro.add_hline(
        y=danger_threshold, line_dash="dot", line_color="#EF4444", line_width=2.5,
        annotation_text=f"Red Danger: {danger_threshold:.2f} m³/s", annotation_position="top right",
    )
    fig_hydro.add_hline(
        y=yellow_trigger, line_dash="dashdot", line_color="#F59E0B", line_width=2,
        annotation_text=f"Yellow Watch: {yellow_trigger:.2f} m³/s", annotation_position="bottom right",
    )
    fig_hydro.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        xaxis=dict(title="Lead Time (Hours from Present T0)", gridcolor="#1E293B"),
        yaxis=dict(title="Singati Discharge (m³/s)", gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=30, b=30),
        height=420,
    )
    st.plotly_chart(fig_hydro, use_container_width=True)

    # Multi-day station discharge history chart
    st.markdown("### 📊 Multi-Station Discharge History (Past Week)")
    st.caption("GloFAS river discharge telemetry across all 4 Tamakoshi corridor stations.")

    fig_history = go.Figure()
    for col, color, dash in [
        ("Tsho Rolpa (m³/s)", "#A78BFA", "dot"),
        ("Gongar (m³/s)", "#67E8F9", "dash"),
        ("Singati (m³/s)", "#38BDF8", "solid"),
        ("Lower Tamakoshi (m³/s)", "#2CE5B5", "dashdot"),
    ]:
        fig_history.add_trace(
            go.Scatter(
                x=history_df["Date"],
                y=history_df[col],
                mode="lines+markers",
                name=col,
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=5),
            )
        )
    fig_history.add_hline(
        y=danger_threshold, line_dash="dot", line_color="#EF4444", line_width=2,
        annotation_text=f"Red Danger Line ({danger_threshold:.2f})", annotation_position="top right",
    )
    fig_history.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        xaxis=dict(title="Date", gridcolor="#1E293B"),
        yaxis=dict(title="Discharge (m³/s)", gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=30, b=30),
        height=350,
    )
    st.plotly_chart(fig_history, use_container_width=True)

    # Compound Physics Gauges
    st.markdown("### ⚡ Compound Himalayan Flood Physics Gauges")
    g1, g2, g3, g4 = st.columns(4)

    rain_intensity = inp["rain_mid"] / (inp["rain_hours"] + 0.5)
    glacier_tmax = inp["tmax"] - 6.0

    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=rain_intensity, number={"suffix": " mm/h"},
            title={"text": "Cloudburst Intensity<br><span style='font-size:0.72rem; color:#94A3B8;'>Rain / Duration</span>"},
            gauge={"axis": {"range": [0, 40]}, "bar": {"color": "#38BDF8"},
                   "steps": [{"range": [0, 10], "color": "rgba(16,185,129,0.3)"}, {"range": [10, 20], "color": "rgba(245,158,11,0.3)"}, {"range": [20, 40], "color": "rgba(239,68,68,0.4)"}],
                   "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 18.0}},
        ))
        fig_g1.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=210, margin=dict(l=18, r=18, t=28, b=8))
        st.plotly_chart(fig_g1, use_container_width=True)

    with g2:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=inp["rain_7d"], number={"suffix": " mm"},
            title={"text": "7-Day Antecedent Rain<br><span style='font-size:0.72rem; color:#94A3B8;'>Basin Pre-charge</span>"},
            gauge={"axis": {"range": [0, 300]}, "bar": {"color": "#67E8F9"},
                   "steps": [{"range": [0, 80], "color": "rgba(16,185,129,0.3)"}, {"range": [80, 150], "color": "rgba(245,158,11,0.3)"}, {"range": [150, 300], "color": "rgba(239,68,68,0.4)"}],
                   "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 140.0}},
        ))
        fig_g2.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=210, margin=dict(l=18, r=18, t=28, b=8))
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        fig_g3 = go.Figure(go.Indicator(
            mode="gauge+number", value=glacier_tmax, number={"suffix": " °C"},
            title={"text": "Tsho Rolpa 4500m Temp<br><span style='font-size:0.72rem; color:#94A3B8;'>Glacial Ablation</span>"},
            gauge={"axis": {"range": [0, 30]}, "bar": {"color": "#F59E0B"},
                   "steps": [{"range": [0, 8], "color": "rgba(16,185,129,0.3)"}, {"range": [8, 16], "color": "rgba(245,158,11,0.3)"}, {"range": [16, 30], "color": "rgba(239,68,68,0.4)"}],
                   "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 15.0}},
        ))
        fig_g3.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=210, margin=dict(l=18, r=18, t=28, b=8))
        st.plotly_chart(fig_g3, use_container_width=True)

    with g4:
        fig_g4 = go.Figure(go.Indicator(
            mode="gauge+number", value=inp["soil_sat_pct"], number={"suffix": "%"},
            title={"text": "Deep Soil Saturation<br><span style='font-size:0.72rem; color:#94A3B8;'>Vadose Zone</span>"},
            gauge={"axis": {"range": [0, 60]}, "bar": {"color": "#10B981"},
                   "steps": [{"range": [0, 25], "color": "rgba(16,185,129,0.3)"}, {"range": [25, 40], "color": "rgba(245,158,11,0.3)"}, {"range": [40, 60], "color": "rgba(239,68,68,0.4)"}],
                   "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 42.0}},
        ))
        fig_g4.update_layout(template="plotly_dark", paper_bgcolor="#0B1120", height=210, margin=dict(l=18, r=18, t=28, b=8))
        st.plotly_chart(fig_g4, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Council of 6 AI Models
# ---------------------------------------------------------
with tab_council:
    st.markdown("### 🧠 Council of 6 AI Models vs. Stacked Meta-Ensemble")
    st.caption(
        "Individual base-model predictions alongside the cross-validated RidgeCV stacked meta-learner. "
        "The ensemble resolves single-model bias by optimally weighting each council member."
    )

    council_flows = dict(individual_preds)
    council_flows["Stacked Ensemble"] = pred_flow

    models_list = list(council_flows.keys())
    flows_list = [council_flows[m] for m in models_list]
    colors_list = ["#38BDF8" if m != "Stacked Ensemble" else sty["color"] for m in models_list]

    fig_council = go.Figure(
        go.Bar(
            x=flows_list,
            y=models_list,
            orientation="h",
            marker_color=colors_list,
            text=[f"{f:.3f} m³/s" for f in flows_list],
            textposition="auto",
        )
    )
    fig_council.add_vline(x=danger_threshold, line_dash="dot", line_color="#EF4444", line_width=2.5,
                          annotation_text=f"Red Danger ({danger_threshold:.2f})", annotation_position="top right")
    fig_council.add_vline(x=yellow_trigger, line_dash="dashdot", line_color="#F59E0B", line_width=2,
                          annotation_text=f"Yellow Watch ({yellow_trigger:.2f})", annotation_position="bottom right")
    fig_council.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        xaxis=dict(title="Discharge Flow Tomorrow (m³/s)", gridcolor="#1E293B"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=20, t=30, b=30),
        height=400,
    )
    st.plotly_chart(fig_council, use_container_width=True)

    # Model Spread KPIs
    sp1, sp2, sp3 = st.columns(3)
    base_flows = [v for k, v in council_flows.items() if k != "Stacked Ensemble"]
    with sp1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Council Spread</div>
                <div class="value">{max(base_flows) - min(base_flows):.3f} <span style="font-size:1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #38BDF8;">Max − Min across 6 base models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sp2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Council Mean</div>
                <div class="value">{np.mean(base_flows):.3f} <span style="font-size:1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #67E8F9;">Unweighted average of 6 models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sp3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="label">Meta-Ensemble Advantage</div>
                <div class="value">{pred_flow - np.mean(base_flows):+.3f} <span style="font-size:1rem; color:#94A3B8;">m³/s</span></div>
                <div class="delta" style="color: #A78BFA;">RidgeCV correction vs. naive mean</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Tab 3: 3D Telemetry Map
# ---------------------------------------------------------
with tab_map:
    st.markdown("### 🗺️ 3D Tamakoshi Corridor Telemetry Map")
    st.caption("Interactive PyDeck 3D map centered on Dolakha, Nepal. Column heights scale with station discharge.")

    def station_color(flow_val, threshold=danger_threshold, yellow=yellow_trigger):
        if flow_val >= threshold:
            return [239, 68, 68, 240]
        elif flow_val >= yellow:
            return [245, 158, 11, 230]
        else:
            return [44, 229, 181, 220]

    flow_map = {
        "tsho_rolpa": inp["q_glacier"],
        "gongar": inp["q_upper"],
        "singati": pred_flow,
        "lower_tamakoshi": inp["q_lower"],
    }
    role_map = {
        "tsho_rolpa": "Glacial Lake Outflow Headwaters",
        "gongar": "456 MW Dam & Spillway Intake",
        "singati": f"AI Forecast Target: {sty['label']}",
        "lower_tamakoshi": "Downstream Drainage & Confluence",
    }

    map_data = []
    for s in TAMAKOSHI_STATIONS:
        f = flow_map.get(s["id"], 0.5)
        map_data.append({
            "name": s["name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "flow": round(f, 3),
            "role": role_map.get(s["id"], s["role"]),
            "color": station_color(f if s["id"] != "singati" else pred_flow),
            "height": max(300, f * 1400 + 400),
        })

    map_df = pd.DataFrame(map_data)

    corridor_arcs = pd.DataFrame([
        {"from_lon": 86.4769, "from_lat": 27.8617, "to_lon": 86.2205, "to_lat": 27.8442,
         "color": station_color(inp["q_glacier"]), "name": "Glacier → Dam"},
        {"from_lon": 86.2205, "from_lat": 27.8442, "to_lon": 86.1628, "to_lat": 27.7314,
         "color": station_color(inp["q_upper"]), "name": "Dam → Singati"},
        {"from_lon": 86.1628, "from_lat": 27.7314, "to_lon": 86.0750, "to_lat": 27.5250,
         "color": station_color(pred_flow), "name": "Singati → Lower"},
    ])

    view_state = pdk.ViewState(latitude=27.72, longitude=86.24, zoom=10.0, pitch=52, bearing=-18)

    column_layer = pdk.Layer(
        "ColumnLayer", data=map_df,
        get_position=["lon", "lat"], get_elevation="height", elevation_scale=1,
        radius=500, get_fill_color="color", pickable=True, auto_highlight=True,
    )
    arc_layer = pdk.Layer(
        "ArcLayer", data=corridor_arcs,
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_source_color="color", get_target_color="color", get_width=5, pickable=True,
    )

    deck = pdk.Deck(
        layers=[arc_layer, column_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={
            "html": """
            <div style="font-family: -apple-system, sans-serif; padding: 10px 12px; background: rgba(8,15,28,0.95); border: 1px solid rgba(44,229,181,0.4); border-radius: 12px; color: #fff; min-width: 200px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
                <div style="font-weight: 800; font-size: 1.02rem; color: #2CE5B5; margin-bottom: 4px;">{name}</div>
                <div style="font-size: 0.82rem; margin-bottom: 4px;">Discharge: <b style="color:#38BDF8;">{flow} m³/s</b></div>
                <div style="font-size: 0.76rem; color: #CBD5E1;">{role}</div>
            </div>
            """,
            "style": {"color": "white"},
        },
    )
    st.pydeck_chart(deck)

    # Legend
    st.markdown(
        f"""
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #94A3B8;">
                <div style="width: 12px; height: 12px; border-radius: 3px; background: #2CE5B5;"></div>
                <span>Normal (&lt; {yellow_trigger:.2f})</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #94A3B8;">
                <div style="width: 12px; height: 12px; border-radius: 3px; background: #F59E0B;"></div>
                <span>Yellow ({yellow_trigger:.2f} – {danger_threshold:.2f})</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #94A3B8;">
                <div style="width: 12px; height: 12px; border-radius: 3px; background: #EF4444;"></div>
                <span>Red Danger (≥ {danger_threshold:.2f})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Tab 4: Raw Telemetry Data & Sensor Inputs
# ---------------------------------------------------------
with tab_data:
    st.markdown("### 📋 Raw API Telemetry Data")
    st.caption(f"Data source: **{source_label}** • Synced at {synced_at}")

    st.markdown("#### 📊 Multi-Day Station Discharge & Weather History")
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    st.markdown("#### 🎛️ Sensor Inputs Fed to 58-Feature Engine")
    inp_display = pd.DataFrame([{
        "Singati Today (m³/s)": inp["q_mid"],
        "Singati Yesterday (m³/s)": inp["q_yesterday"],
        "Tsho Rolpa (m³/s)": inp["q_glacier"],
        "Gongar (m³/s)": inp["q_upper"],
        "Lower (m³/s)": inp["q_lower"],
        "Rain Today (mm)": inp["rain_mid"],
        "Rain 3d (mm)": inp["rain_3d"],
        "Rain 7d (mm)": inp["rain_7d"],
        "Soil (%)": inp["soil_sat_pct"],
        "Tmax (°C)": inp["tmax"],
        "Tmin (°C)": inp["tmin"],
        "Snow (cm)": inp["snow_glacier"],
        "Solar (MJ/m²)": inp["solar_rad"],
        "Month": inp["month"],
    }])
    st.dataframe(inp_display, use_container_width=True, hide_index=True)

    st.markdown("#### 🔬 AI Forecast Output Summary")
    out_display = pd.DataFrame([{
        "Pred Flow (m³/s)": pred_flow,
        "Delta (m³/s)": delta_pred,
        "Low 10% (m³/s)": low_flow,
        "High 90% (m³/s)": high_flow,
        "Flood Prob (%)": flood_prob,
        "Alert Level": sty["label"],
        "Danger Line": danger_threshold,
        "Yellow Line": yellow_trigger,
        "Data Source": source_mode,
    }])
    st.dataframe(out_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 2.5rem; padding: 1.2rem 1.4rem; border-radius: 16px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(8, 13, 22, 0.80));
        border: 1px solid rgba(148, 163, 184, 0.14); text-align: center;">
        <div style="font-size: 0.82rem; color: #8ea3bf; line-height: 1.6;">
            <b style="color: #2CE5B5;">📡 Live Tamakoshi Telemetry Engine</b> — Powered by Open-Meteo GloFAS & Weather API<br>
            6-Model AI Stacking Ensemble (XGBoost · LightGBM · CatBoost · HistGBM · ExtraTrees · Deep Residual MLP + RidgeCV Meta-Learner)<br>
            <span style="color: #67e8f9;">Veritas AI Hackathon — Track 4.3: AI + Climate Resilience</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
