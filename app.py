"""
Tamakoshi Himalayan Flood AI | Early Warning Command Center
============================================================
Life-Safety Early Warning Dashboard & Hydrological Decision Support System
Catchment: Tamakoshi River Corridor, Dolakha District, Nepal (4,580m -> 1,250m ASL)
Theme: Light Command Theme (Black, White, Red, Green & Precision Life-Safety Accents)
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

from ui_theme import hex_to_rgb, rgba, theme_css, theme_tokens

# Ensure local imports resolve
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.tamakoshi_api import (
    load_model_bundle,
    route_live_basin_telemetry,
    compute_forecast_from_inputs,
)

# -----------------------------------------------------------------------------
# 2. Shared theme tokens and page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tamakoshi River Flood Early Warning Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state.setdefault("dark_theme", False)
PALETTE = theme_tokens(bool(st.session_state["dark_theme"]))
PALETTE.update({
    "bg_main": PALETTE["background"],
    "bg_surface": PALETTE["surface"],
    "bg_surface_alt": PALETTE["surface_alt"],
    "border_subtle": PALETTE["border"],
    "text_primary": PALETTE["text"],
    "text_secondary": PALETTE["muted"],
    "text_muted": PALETTE["muted"],
    "accent_primary": PALETTE["accent"],
})
for level_name, level_tokens in PALETTE["severity"].items():
    PALETTE[level_name] = {
        **level_tokens,
        "tier": level_tokens["label"],
        "level_num": {"normal": "1", "watch": "2", "warning": "3", "emergency": "4"}[level_name],
    }
st.markdown(theme_css(PALETTE), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Safe Model Loading & Core Data Logic
# -----------------------------------------------------------------------------
try:
    bundle = load_model_bundle()
    DANGER_THRESHOLD = float(bundle["threshold"])
    YELLOW_THRESHOLD = DANGER_THRESHOLD * 0.90
except Exception as e:
    st.error(f"Critical Model Asset Error: Unable to load 'tamakoshi_ultimate_model.joblib'. Details: {e}")
    st.stop()


def classify_alert_severity(pred_flow: float, flood_prob: float) -> dict:
    """Return a plain-language status while preserving the existing model thresholds."""
    if pred_flow >= DANGER_THRESHOLD or flood_prob >= 60.0:
        base = PALETTE["emergency"]
        headline = "Flood danger may be present within the next 24 hours"
        action = "Follow local emergency instructions and move away from the river to safer ground."
    elif pred_flow >= (DANGER_THRESHOLD - 0.35) or flood_prob >= 50.0:
        base = PALETTE["warning"]
        headline = "Flood risk may increase within the next 24 hours"
        action = "Follow official local advisories and avoid riverbanks if conditions worsen."
    elif pred_flow >= YELLOW_THRESHOLD or flood_prob >= 35.0:
        base = PALETTE["watch"]
        headline = "River conditions need closer attention"
        action = "Check official local updates and stay prepared to follow their advice."
    else:
        base = PALETTE["normal"]
        headline = "No flood risk expected in the next 24 hours"
        action = "Continue to follow updates from local authorities."

    return {
        **base,
        "headline": headline,
        "action": action,
    }


# Calibrated Historical Scenarios for Stress Testing
SCENARIOS = {
    "Scenario 3: Cloudburst and Glacial Siphon Breach (Emergency Alert)": (
        6.60, 5.55, 1.15, 4.35, 9.80, 58.0, 42.0, 135.0, 230.0, 3, 48, 24.0, 15.5, 0.0, 22.5, 8
    ),
    "Scenario 2: Active Monsoon Surge (Yellow Watch)": (
        5.85, 5.20, 0.85, 3.70, 8.40, 28.0, 20.0, 68.0, 135.0, 7, 41, 22.0, 14.0, 0.0, 19.5, 7
    ),
    "Scenario 1: Dry Winter Baseflow (Normal Baseline)": (
        0.35, 0.36, 0.05, 0.22, 0.55, 0.0, 0.0, 0.5, 2.0, 0, 22, 14.0, 2.5, 0.0, 14.5, 2
    ),
    "Custom Simulation (Manual Calibration)": None,
}

# -----------------------------------------------------------------------------
# 5. Sidebar: FILTERS & PARAMETERS ONLY
# -----------------------------------------------------------------------------
# Operational Mode Filter
st.sidebar.markdown("<div class='kpi-label'>Telemetry operating mode</div>", unsafe_allow_html=True)
op_mode = st.sidebar.radio(
    "Select Operating Mode:",
    ["Operational Live Telemetry", "Simulation & Stress Testing"],
    index=0,
    label_visibility="collapsed",
    key="filter_op_mode",
)

if op_mode == "Operational Live Telemetry":
    # Live Mode Filters
    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='kpi-label'>Telemetry lookback window</div>", unsafe_allow_html=True)
    past_days_filter = st.sidebar.slider(
        "Lookback Days",
        min_value=3,
        max_value=14,
        value=7,
        help="Historical lookback window for GloFAS river discharge and basin precipitation.",
        key="filter_lookback_days",
    )

    refresh_clicked = st.sidebar.button("Refresh telemetry", use_container_width=True, key="btn_refresh_live")
    if refresh_clicked:
        st.cache_data.clear()
        st.sidebar.caption("Refresh requested. Loading the latest available telemetry.")

    # Fetch live telemetry
    with st.spinner("Connecting to Open-Meteo GloFAS & Basin Weather APIs..."):
        try:
            live_payload = route_live_basin_telemetry(past_days=past_days_filter)
        except Exception as e:
            st.error(f"Telemetry data unavailable: {e}")
            st.stop()

    active_inputs = live_payload["inputs"]
    if live_payload["source_mode"] == "FALLBACK_SNAPSHOT":
        data_source_label = "Live feed unavailable · showing baseline snapshot"
    else:
        data_source_label = live_payload["status_label"]
    data_synced_at = live_payload["synced_at"]
    live_history_df = live_payload["history_df"]
    is_live_active = True

else:
    # Simulation Mode Filters
    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='kpi-label'>Calibrated watershed scenario</div>", unsafe_allow_html=True)
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

    with st.sidebar.expander("Station Discharge Sensors (m³/s)", expanded=True):
        sim_q_mid = st.number_input("Singati Target Flow (m³/s)", value=float(sim_defaults[0]), step=0.1, key="sim_q_mid")
        sim_q_yesterday = st.number_input("Singati Flow Yesterday (m³/s)", value=float(sim_defaults[1]), step=0.1, key="sim_q_yesterday")
        sim_q_glacier = st.number_input("Tsho Rolpa Glacier Outflow (m³/s)", value=float(sim_defaults[2]), step=0.05, key="sim_q_glacier")
        sim_q_upper = st.number_input("Gongar Dam Intake Flow (m³/s)", value=float(sim_defaults[3]), step=0.1, key="sim_q_upper")
        sim_q_lower = st.number_input("Lower Confluence Flow (m³/s)", value=float(sim_defaults[4]), step=0.1, key="sim_q_lower")

    with st.sidebar.expander("Precipitation & Cloudburst Dynamics", expanded=False):
        sim_rain_mid = st.number_input("Valley Rain Today (mm)", value=float(sim_defaults[5]), step=1.0, key="sim_rain_mid")
        sim_rain_glacier = st.number_input("Glacier Rain Today (mm)", value=float(sim_defaults[6]), step=1.0, key="sim_rain_glacier")
        sim_rain_3d = st.number_input("3-Day Cumulative Rain (mm)", value=float(sim_defaults[7]), step=5.0, key="sim_rain_3d")
        sim_rain_7d = st.number_input("7-Day Cumulative Rain (mm)", value=float(sim_defaults[8]), step=5.0, key="sim_rain_7d")
        sim_rain_hours = st.slider("Rain duration (hours)", 0, 24, int(sim_defaults[9]), key="sim_rain_hours")

    with st.sidebar.expander("Soil Moisture & Thermal Melt Dynamics", expanded=False):
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
top_c1, top_c2, top_c3 = st.columns([7, 3, 2], vertical_alignment="center")
with top_c1:
    st.markdown(
        f"""
        <div style="margin-bottom: 6px; color: {PALETTE['text_primary']}; font-size: 30px; font-weight: 700; line-height: 1.2;">
                Tamakoshi River Flood Early Warning Command Center
        </div>
            <div style="font-size: 15px; color: {PALETTE['text_secondary']};">
            Dolakha Watershed, Nepal · Tsho Rolpa to Singati Gorge
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_c2:
    st.markdown(
        f"""
        <div style="font-size: 13px; color: {PALETTE['text_secondary']}; line-height: 1.6;">
            <div><strong>Source</strong> <span class="status-badge">{data_source_label}</span></div>
            <div style="margin-top: 4px;"><strong>Last synced</strong> {data_synced_at}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_c3:
    st.toggle(
        "Dark theme",
        key="dark_theme",
        help="Switch between light and dark display themes.",
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. HIERARCHY STEP 1: Emergency Action & Severity Status Banner
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="severity-banner" style="--status-color: {severity['color']}; --status-tint: {severity['bg']};">
        <div class="severity-tag">
            <span class="severity-icon" aria-hidden="true">{severity['icon']}</span>
            <span>{severity['tier']}</span>
        </div>
        <div class="severity-title">{severity['headline']}</div>
        <div class="severity-body">{severity['action']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 9. HIERARCHY STEP 2: 4 Primary KPI Cards
# -----------------------------------------------------------------------------
kpi_grid = st.container(key="kpi-grid")
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = kpi_grid.columns(4)

with kpi_c1:
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">Tomorrow's river flow</div>
            <div class="kpi-value">
                {pred_flow:.2f} <span style="font-size: 13px; font-weight: 500; color: {PALETTE['text_secondary']};">m³/s</span>
            </div>
            <div class="kpi-meta">Change from today's reading: {delta_pred:+.2f} m³/s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c2:
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">Forecast range</div>
            <div class="kpi-value">
                {low_flow:.2f}–{high_flow:.2f} <span style="font-size: 13px; font-weight: 500; color: {PALETTE['text_secondary']};">m³/s</span>
            </div>
            <div class="kpi-meta">Model's 80% prediction interval</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c3:
    freeboard = DANGER_THRESHOLD - pred_flow
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">Margin to model threshold</div>
            <div class="kpi-value">
                {freeboard:+.2f} <span style="font-size: 13px; font-weight: 500; color: {PALETTE['text_secondary']};">m³/s</span>
            </div>
            <div class="kpi-meta">Threshold not yet independently validated</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c4:
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">Estimated flood probability</div>
            <div class="kpi-value">
                {flood_prob:.1f}<span style="font-size: 13px; font-weight: 500; color: {PALETTE['text_secondary']};">%</span>
            </div>
            <div class="kpi-meta">Model estimate, not an official alert</div>
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
    st.markdown("<div class='kpi-label'>24-Hour Predictive Hydrograph & Confidence Envelope</div>", unsafe_allow_html=True)
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
            fillcolor=rgba(PALETTE["accent"], 0.14),
            line=dict(color=rgba(PALETTE["accent"], 0)),
            hoverinfo="skip",
            name="80% Prediction Interval",
        )
    )
    # Sensor history uses neutral text color; the forecast remains the UI accent.
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours <= 0],
            y=past_curve,
            mode="lines+markers",
            name="Telemetry History (T-24h to T0)",
            line=dict(color=PALETTE["text_secondary"], width=2),
            marker=dict(size=4, color=PALETTE["text_secondary"]),
        )
    )
    # The forecast line is always the UI accent, separate from alert colors.
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours >= 0],
            y=future_curve,
            mode="lines+markers",
            name="AI Forecast (T0 to T+24h)",
            line=dict(color=PALETTE["accent"], width=3, dash="solid"),
            marker=dict(size=6, symbol="circle", color=PALETTE["accent"]),
        )
    )
    # Safety Threshold Lines (Red & Amber)
    fig_hydro.add_hline(
        y=DANGER_THRESHOLD,
        line_dash="dot",
        line_color=PALETTE["emergency"]["color"],
        line_width=2,
        annotation_text=f"Model threshold (unverified) · {DANGER_THRESHOLD:.2f} m³/s",
        annotation_position="top right",
        annotation_font=dict(color=PALETTE["text_primary"], size=13),
        annotation_bgcolor=PALETTE["emergency"]["bg"],
    )
    fig_hydro.add_hline(
        y=YELLOW_THRESHOLD,
        line_dash="dashdot",
        line_color=PALETTE["watch"]["color"],
        line_width=1.5,
        annotation_text=f"Model watch line (unverified) · {YELLOW_THRESHOLD:.2f} m³/s",
        annotation_position="bottom right",
        annotation_font=dict(color=PALETTE["text_primary"], size=13),
        annotation_bgcolor=PALETTE["watch"]["bg"],
    )
    fig_hydro.update_layout(
        template="plotly_white",
        paper_bgcolor=PALETTE["bg_surface"],
        plot_bgcolor=PALETTE["bg_surface"],
        font=dict(family="Inter, Arial, sans-serif", color=PALETTE["text_primary"], size=13),
        xaxis=dict(title="Hours from now", gridcolor=PALETTE["border_subtle"], zeroline=False, linecolor=PALETTE["border_subtle"]),
        yaxis=dict(title="River flow at Singati (m³/s)", gridcolor=PALETTE["border_subtle"], zeroline=False, linecolor=PALETTE["border_subtle"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color=PALETTE["text_primary"])),
        margin=dict(l=24, r=18, t=28, b=24),
        height=370,
    )
    st.plotly_chart(fig_hydro, use_container_width=True, config={"responsive": True, "displaylogo": False})

with col_map:
    st.markdown("<div class='kpi-label'>River corridor monitoring stations</div>", unsafe_allow_html=True)

    def station_severity(val: float, is_target: bool = False) -> dict:
        if is_target:
            return severity
        check_val = val
        if check_val >= DANGER_THRESHOLD:
            level_name = "emergency"
        elif check_val >= (DANGER_THRESHOLD - 0.35):
            level_name = "warning"
        elif check_val >= YELLOW_THRESHOLD:
            level_name = "watch"
        else:
            level_name = "normal"
        return PALETTE[level_name]

    map_stations_data = [
        {"name": "Tsho Rolpa Glacier", "label": "1 · Tsho Rolpa", "lat": 27.8617, "lon": 86.4769, "flow": active_inputs["q_glacier"], "status": station_severity(active_inputs["q_glacier"]), "height": max(300, active_inputs["q_glacier"] * 1600 + 400)},
        {"name": "Gongar Dam Intake", "label": "2 · Gongar", "lat": 27.8442, "lon": 86.2205, "flow": active_inputs["q_upper"], "status": station_severity(active_inputs["q_upper"]), "height": max(300, active_inputs["q_upper"] * 1200 + 400)},
        {"name": "Singati Gorge Target", "label": "3 · Singati forecast", "lat": 27.7314, "lon": 86.1628, "flow": pred_flow, "status": station_severity(pred_flow, is_target=True), "height": max(300, pred_flow * 1400 + 500)},
        {"name": "Lower Confluence", "label": "4 · Lower Confluence", "lat": 27.5250, "lon": 86.0750, "flow": active_inputs["q_lower"], "status": station_severity(active_inputs["q_lower"]), "height": max(300, active_inputs["q_lower"] * 800 + 300)},
    ]
    for station in map_stations_data:
        station["color"] = station["status"]["rgb"] + [230]
        station["label_color"] = hex_to_rgb(PALETTE["text"])
        station["map_label"] = f"{station['status']['icon']}  {station['label']} · {station['status']['label']}"
    df_map_stations = pd.DataFrame([
        {key: value for key, value in station.items() if key != "status"}
        for station in map_stations_data
    ])

    corridor_arcs = pd.DataFrame([
        {"from_lon": 86.4769, "from_lat": 27.8617, "to_lon": 86.2205, "to_lat": 27.8442, "color": map_stations_data[0]["color"]},
        {"from_lon": 86.2205, "from_lat": 27.8442, "to_lon": 86.1628, "to_lat": 27.7314, "color": map_stations_data[1]["color"]},
        {"from_lon": 86.1628, "from_lat": 27.7314, "to_lon": 86.0750, "to_lat": 27.5250, "color": map_stations_data[2]["color"]},
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
    labels_layer = pdk.Layer(
        "TextLayer",
        data=df_map_stations,
        get_position=["lon", "lat"],
        get_text="map_label",
        get_color="label_color",
        get_size=14,
        get_pixel_offset=[0, -18],
        get_alignment_baseline="bottom",
        pickable=True,
    )

    deck_chart = pdk.Deck(
        layers=[arc_layer, col_layer, labels_layer],
        initial_view_state=map_view,
        map_style="carto-positron",
        tooltip={
            "html": (
                f"<div style='background:{PALETTE['surface']}; padding:8px 12px; border-radius:8px; border:1px solid {PALETTE['border']}; color:{PALETTE['text']}; box-shadow:0 2px 6px {PALETTE['shadow']}; font-family:Inter, sans-serif;'>"
                "<b>{name}</b><br>Discharge: <b>{flow} m³/s</b></div>"
            ),
            "style": {"color": PALETTE["text"]},
        },
    )
    st.pydeck_chart(deck_chart)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 11. HIERARCHY STEP 4: Deep-Dive Diagnostic Tabs (No Emojis)
# -----------------------------------------------------------------------------
tab_gorge, tab_council, tab_gauges, tab_table, tab_method = st.tabs([
    "3D corridor view",
    "Multi-Model Ensemble",
    "Hydrological Stress Indicators",
    "Corridor Sensor Telemetry",
    "System design",
], key="detail-tabs")

# ----------------- Tab 1: 3D River Gorge Model -----------------
with tab_gorge:
    st.markdown("<div class='kpi-label'>Illustrative 3D Singati river corridor</div>", unsafe_allow_html=True)
    st.caption("Rotate and zoom the conceptual terrain view. It is not a surveyed hydraulic model.")

    water_height_ratio = min(1.8, max(0.1, pred_flow / DANGER_THRESHOLD))

    three_js_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; overflow: hidden; background: {PALETTE['bg_main']}; font-family: Inter, sans-serif; }}
            #hud {{
                position: absolute; top: 12px; left: 14px;
                background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
                border-radius: 12px; padding: 12px 16px; color: {PALETTE['text']}; font-size: 15px;
                box-shadow: 0 4px 12px {PALETTE['shadow']}; pointer-events: none;
            }}
            #hud b {{ color: {severity['color']}; }}
            #hint {{
                position: absolute; bottom: 10px; right: 14px;
                background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
                border-radius: 8px; padding: 8px 12px; color: {PALETTE['text_secondary']}; font-size: 13px;
                box-shadow: 0 2px 6px {PALETTE['shadow']}; pointer-events: none;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="hud">
            <div style="font-weight:700; color:{PALETTE['text_primary']}; margin-bottom:4px;">Illustrative river corridor model</div>
            <div style="color:{PALETTE['text_secondary']}; font-size:13px; margin-top:4px;">Conceptual terrain only; not a surveyed flood model.</div>
        </div>
        <div id="hint">Interaction: Left Click to Rotate | Right Click to Pan | Scroll to Zoom</div>
        <script>
            const scene = new THREE.Scene();
            scene.background = new THREE.Color('{PALETTE["bg_main"]}');
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
            scene.add(new THREE.AmbientLight('{PALETTE["border"]}', 0.85));
            const sun = new THREE.DirectionalLight('{PALETTE["surface"]}', 1.2);
            sun.position.set(25, 35, 20);
            scene.add(sun);
            const alertLight = new THREE.PointLight('{PALETTE["accent"]}', 1.5, 40);
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
            const canyonMat = new THREE.MeshStandardMaterial({{ color: '{PALETTE["muted"]}', roughness: 0.85, flatShading: true }});
            scene.add(new THREE.Mesh(canyonGeo, canyonMat));

            // Dynamic water surface uses the neutral UI accent; severity remains in the status label.
            const waterGeo = new THREE.PlaneGeometry(13, 56, 30, 30);
            waterGeo.rotateX(-Math.PI / 2);
            const waterMat = new THREE.MeshStandardMaterial({{
                color: '{PALETTE["accent"]}', roughness: 0.2, metalness: 0.6, transparent: true, opacity: 0.88
            }});
            const water = new THREE.Mesh(waterGeo, waterMat);
            water.position.y = -0.9 + {water_height_ratio} * 2.8;
            scene.add(water);

            // Model threshold wireframe; the value is unverified.
            const dangerGeo = new THREE.PlaneGeometry(16, 56);
            dangerGeo.rotateX(-Math.PI / 2);
            const dangerPlane = new THREE.Mesh(dangerGeo, new THREE.MeshBasicMaterial({{ color: '{PALETTE["emergency"]["color"]}', wireframe: true, transparent: true, opacity: 0.45 }}));
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

# ----------------- Tab 2: Multi-Model Ensemble -----------------
with tab_council:
    st.markdown("<div class='kpi-label'>Council of 6 Base Estimators vs. RidgeCV Meta-Learner</div>", unsafe_allow_html=True)
    st.caption("Comparison of base estimators with the stacked meta-regressor resolving individual model deltas.")

    council_data = dict(individual_preds)
    council_data["RidgeCV Meta-Learner"] = pred_flow

    m_names = list(council_data.keys())
    m_flows = [council_data[m] for m in m_names]
    m_colors = [PALETTE["text_secondary"] if m != "RidgeCV Meta-Learner" else severity["color"] for m in m_names]

    fig_council = go.Figure(
        go.Bar(
            x=m_flows,
            y=m_names,
            orientation="h",
            marker_color=m_colors,
            text=[f"{v:.3f} m³/s" for v in m_flows],
            textposition="auto",
            textfont=dict(color=PALETTE["text_primary"], size=13),
        )
    )
    fig_council.add_vline(x=DANGER_THRESHOLD, line_dash="dot", line_color=PALETTE["emergency"]["color"], line_width=2)
    fig_council.add_vline(x=YELLOW_THRESHOLD, line_dash="dashdot", line_color=PALETTE["watch"]["color"], line_width=1.5)
    fig_council.update_layout(
        template="plotly_white",
        paper_bgcolor=PALETTE["bg_surface"],
        plot_bgcolor=PALETTE["bg_surface"],
        font=dict(family="Inter, Arial, sans-serif", color=PALETTE["text_primary"], size=13),
        xaxis=dict(title="Estimated Discharge Tomorrow (m³/s)", gridcolor=PALETTE["border_subtle"]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=20, b=24),
        height=340,
    )
    st.plotly_chart(fig_council, use_container_width=True, config={"responsive": True, "displaylogo": False})

    base_only_flows = [council_data[m] for m in m_names if m != "RidgeCV Meta-Learner"]
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Council Range Spread", f"{max(base_only_flows) - min(base_only_flows):.3f} m³/s", "Max - Min across 6 base models")
    with sc2:
        st.metric("Unweighted Council Mean", f"{np.mean(base_only_flows):.3f} m³/s", "Simple average")
    with sc3:
        st.metric("Meta-Learner Correction", f"{pred_flow - np.mean(base_only_flows):+.3f} m³/s", "RidgeCV cross-validated delta")

# ----------------- Tab 3: Hydrological Stress Indicators -----------------
with tab_gauges:
    st.markdown("<div class='kpi-label'>Coupled Glacial-Monsoon Physical Forcing Gauges</div>", unsafe_allow_html=True)
    st.caption("Physical environmental metrics driving non-linear overland runoff and moraine thermal melt.")

    g1, g2, g3, g4 = st.columns(4)
    rain_intensity = active_inputs["rain_mid"] / (active_inputs["rain_hours"] + 0.5)
    glacier_temp = active_inputs["tmax"] - 6.0

    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=rain_intensity, number={"suffix": " mm/h", "font": {"color": PALETTE["text_primary"]}},
            title={"text": "Cloudburst Rain Intensity", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 40], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent"]},
            },
        ))
        fig_g1.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], font=dict(family="Inter, Arial, sans-serif", color=PALETTE["text_primary"], size=13), height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g1, use_container_width=True, config={"responsive": True, "displaylogo": False})

    with g2:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=active_inputs["rain_7d"], number={"suffix": " mm", "font": {"color": PALETTE["text_primary"]}},
            title={"text": "Antecedent 7-Day Rain", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 300], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent"]},
            },
        ))
        fig_g2.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], font=dict(family="Inter, Arial, sans-serif", color=PALETTE["text_primary"], size=13), height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g2, use_container_width=True, config={"responsive": True, "displaylogo": False})

    with g3:
        fig_g3 = go.Figure(go.Indicator(
            mode="gauge+number", value=glacier_temp, number={"suffix": " °C", "font": {"color": PALETTE["text_primary"]}},
            title={"text": "Tsho Rolpa (4,580m) Temp", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 30], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent"]},
            },
        ))
        fig_g3.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], font=dict(family="Inter, Arial, sans-serif", color=PALETTE["text_primary"], size=13), height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g3, use_container_width=True, config={"responsive": True, "displaylogo": False})

    with g4:
        fig_g4 = go.Figure(go.Indicator(
            mode="gauge+number", value=active_inputs["soil_sat_pct"], number={"suffix": "%", "font": {"color": PALETTE["text_primary"]}},
            title={"text": "Vadose Deep Soil Saturation", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 60], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent"]},
            },
        ))
        fig_g4.update_layout(template="plotly_white", paper_bgcolor=PALETTE["bg_surface"], font=dict(family="Inter, Arial, sans-serif", color=PALETTE["text_primary"], size=13), height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g4, use_container_width=True, config={"responsive": True, "displaylogo": False})

# ----------------- Tab 4: Corridor Station Telemetry -----------------
with tab_table:
    st.markdown("<div class='kpi-label'>Recent corridor readings</div>", unsafe_allow_html=True)

    st.dataframe(live_history_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='kpi-label' style='margin-top:14px;'>Additional model inputs</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{
        "3-day rain (mm)": active_inputs["rain_3d"],
        "7-day rain (mm)": active_inputs["rain_7d"],
        "Soil saturation (%)": active_inputs["soil_sat_pct"],
        "Glacier temperature estimate (°C)": glacier_temp,
    }]), use_container_width=True, hide_index=True)

# ----------------- Tab 5: System Architecture & Contacts -----------------
with tab_method:
    st.markdown("<div class='kpi-label'>System design and model limitations</div>", unsafe_allow_html=True)
    st.info(
        "Admin note: Model thresholds and forecast outputs have not been independently validated "
        "by local authorities and are not official warnings. Confirm operational decisions with "
        "the responsible authorities."
    )

    with st.expander("Technical model details", expanded=False, key="technical-details"):
        st.markdown(
            """
            **Hydrological pipeline**

            - Multi-station discharge gradients and upstream flow relationships
            - Temporal flow changes, rainfall accumulation, and soil moisture
            - Glacier temperature and melt indicators
            - Seasonal features and a stacked multi-model forecast

            **Model limits**

            - Prediction intervals reflect the model's training distribution; rare events may fall outside it.
            - The corridor visualization is illustrative and is not a surveyed hydraulic model.
            - No emergency contact directory is configured. Add only contacts verified with local authorities.
            """
        )
