"""
Tamakoshi Himalayan Flood AI | Early Warning Command Center
============================================================
Life-Safety Early Warning Dashboard & Hydrological Decision Support System
Catchment: Tamakoshi River Corridor, Dolakha District, Nepal (4,580m -> 1,250m ASL)
"""

import sys
import types
import os
import json

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
from ui_theme import render_river_hero, render_site_navigation

# -----------------------------------------------------------------------------
# 2. Design System Tokens: Modern Hydrological Command Palette
# -----------------------------------------------------------------------------
PALETTE = {
    # Surfaces & Canvas
    "bg_main": "#0B1120",        # Deep midnight slate base
    "bg_surface": "#131D31",     # Primary card surface
    "bg_surface_alt": "#1A2642", # Elevated container / header
    "border_subtle": "#23324E",  # Subtle structural dividers
    "border_focus": "#38BDF8",   # Primary interactive focus

    # Typography
    "text_primary": "#F8FAFC",   # Slate 50 (High contrast)
    "text_secondary": "#94A3B8", # Slate 400 (Informational)
    "text_muted": "#64748B",     # Slate 500 (Metadata, timestamps)

    # Functional Accents
    "accent_primary": "#38BDF8", # Sky / Cyan (Interactive, telemetry ray)
    "accent_teal": "#2DD4BF",    # Teal (Sensors, water accents)
    "accent_indigo": "#818CF8",  # Indigo (Model ensembles)

    # Life-Safety Calibrated Severity Scale
    "emergency": {
        "tier": "CRITICAL DANGER",
        "level_num": "4",
        "color": "#EF4444",      # Red (Danger Line 6.69 m³/s)
        "bg": "rgba(239, 68, 68, 0.12)",
        "glow": "rgba(239, 68, 68, 0.25)",
        "border": "#EF4444",
    },
    "warning": {
        "tier": "FLOOD WARNING",
        "level_num": "3",
        "color": "#F97316",      # Orange (Pre-Breach Warning)
        "bg": "rgba(249, 115, 22, 0.12)",
        "glow": "rgba(249, 115, 22, 0.25)",
        "border": "#F97316",
    },
    "watch": {
        "tier": "HYDRO WATCH",
        "level_num": "2",
        "color": "#F59E0B",      # Amber (Watch Buffer 6.02 m³/s)
        "bg": "rgba(245, 158, 11, 0.12)",
        "glow": "rgba(245, 158, 11, 0.25)",
        "border": "#F59E0B",
    },
    "normal": {
        "tier": "NORMAL BASEFLOW",
        "level_num": "1",
        "color": "#10B981",      # Emerald (Safe Baseline)
        "bg": "rgba(16, 185, 129, 0.12)",
        "glow": "rgba(16, 185, 129, 0.25)",
        "border": "#10B981",
    },
}

# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Modern Dark CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tamakoshi River Flood Early Warning Command Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.session_state["_tamakoshi_main_script"] = os.path.basename(__file__)

CUSTOM_CSS = f"""
<style>
    /* Global Reset & Base Canvas */
    html, body, .stApp {{
        background: {PALETTE["bg_main"]} !important;
        color: {PALETTE["text_primary"]} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }}

    .main .block-container {{
        padding: 1.2rem 1.8rem 2.2rem 1.8rem !important;
        max-width: 1560px !important;
    }}

    /* Hide standard menu & footer */
    #MainMenu, footer, header {{
        visibility: hidden !important;
    }}

    /* Modern Command Center Cards */
    .command-card {{
        background: {PALETTE["bg_surface"]};
        border: 1px solid {PALETTE["border_subtle"]};
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }}
    .command-card:hover {{
        border-color: {PALETTE["border_focus"]};
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.12);
        transform: translateY(-1px);
    }}

    .kpi-label {{
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {PALETTE["text_secondary"]};
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 2.10rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
        color: {PALETTE["text_primary"]};
    }}
    .kpi-meta {{
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 8px;
        color: {PALETTE["text_muted"]};
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Status Dot with pulse */
    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }}

    /* Action Banner */
    .severity-banner {{
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
        border: 1px solid {PALETTE["border_subtle"]};
        border-left: 6px solid;
        background: {PALETTE["bg_surface"]};
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28);
    }}
    .severity-tag {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }}
    .severity-title {{
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 6px;
    }}
    .severity-body {{
        font-size: 0.94rem;
        line-height: 1.6;
        color: {PALETTE["text_secondary"]};
    }}
    .action-list {{
        margin: 12px 0 0 0;
        padding-left: 20px;
        font-size: 0.90rem;
        color: {PALETTE["text_primary"]};
        line-height: 1.7;
    }}

    /* Modern Tabs */
    .stTabs [role="tablist"] {{
        gap: 8px;
        border-bottom: 1px solid {PALETTE["border_subtle"]};
        padding-bottom: 4px;
    }}
    .stTabs [role="tab"] {{
        background: {PALETTE["bg_surface"]};
        border: 1px solid {PALETTE["border_subtle"]};
        border-radius: 8px 8px 0 0;
        color: {PALETTE["text_secondary"]};
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.88rem;
        min-height: 44px;
        transition: all 0.15s ease;
    }}
    .stTabs [role="tab"]:hover {{
        color: {PALETTE["text_primary"]};
        border-color: {PALETTE["border_focus"]};
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background: {PALETTE["bg_surface_alt"]} !important;
        border-color: {PALETTE["border_focus"]} !important;
        border-top: 3px solid {PALETTE["accent_primary"]} !important;
        color: {PALETTE["text_primary"]} !important;
        font-weight: 700 !important;
    }}

    /* High-Impact Tactical Buttons */
    .stButton > button {{
        min-height: 44px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
        border: 1px solid {PALETTE["border_focus"]};
        background: linear-gradient(180deg, #1A2642 0%, #131D31 100%);
        color: {PALETTE["text_primary"]};
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        cursor: pointer;
    }}
    .stButton > button:hover {{
        background: linear-gradient(180deg, #24355A 0%, #1A2642 100%);
        border-color: #7DD3FC;
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }}

    /* Sleek Expanders */
    .streamlit-expanderHeader {{
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: {PALETTE["text_primary"]} !important;
        background: {PALETTE["bg_surface"]} !important;
        border: 1px solid {PALETTE["border_subtle"]} !important;
        border-radius: 8px !important;
    }}

    /* Live Stream Pill Badge */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid {PALETTE["border_subtle"]};
        background: {PALETTE["bg_surface"]};
        color: {PALETTE["text_secondary"]};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_site_navigation("home")

# -----------------------------------------------------------------------------
# 4. Safe Model Loading & Core Data Logic
# -----------------------------------------------------------------------------
try:
    bundle = load_model_bundle()
    DANGER_THRESHOLD = float(bundle["threshold"])          # 6.69 m³/s
    YELLOW_THRESHOLD = DANGER_THRESHOLD * 0.90             # 6.02 m³/s
    model_loaded = True
except Exception as e:
    st.error(f"Critical Model Asset Error: Unable to load 'tamakoshi_ultimate_model.joblib'. Details: {e}")
    st.stop()


def classify_alert_severity(pred_flow: float, flood_prob: float) -> dict:
    """Calculates single authoritative life-safety severity level and operational protocols."""
    if pred_flow >= DANGER_THRESHOLD or flood_prob >= 60.0:
        base = PALETTE["emergency"]
        return {
            **base,
            "headline": "CRITICAL OVERTOPPING BREACH IMMINENT",
            "summary": (
                f"Predicted 24-hour discharge reaches <b>{pred_flow:.2f} m³/s</b>, exceeding the "
                f"95th-percentile safety threshold of <b>{DANGER_THRESHOLD:.2f} m³/s</b> by "
                f"<b>{(pred_flow - DANGER_THRESHOLD):+.2f} m³/s</b> with <b>{flood_prob:.1f}%</b> inundation likelihood."
            ),
            "protocol_header": "MANDATED EMERGENCY ACTIONS (LEVEL 4 - CRITICAL DANGER):",
            "protocols": [
                "<b>Immediate Evacuation Sirens:</b> Broadcast VHF/SMS emergency evacuation orders to Singati Bazaar, Gongar, Jagat, and Lamabagar.",
                "<b>Dam Radial Spillway Gates:</b> Actuate Upper Tamakoshi (456 MW) spillways immediately for reservoir drawdown buffering.",
                "<b>Tactical Rescue Deployment:</b> Mobilize Armed Police Force (APF) water rescue and Dolakha DDMC emergency teams.",
                "<b>Emergency Communications Desk:</b> [PLACEHOLDER: Dolakha DEOC VHF Ch 4 / Hotline: +977-49-421111]",
            ],
        }
    elif pred_flow >= (DANGER_THRESHOLD - 0.35) or flood_prob >= 50.0:
        base = PALETTE["warning"]
        return {
            **base,
            "headline": "SEVERE SURGE ACCELERATION — PRE-BREACH WARNING",
            "summary": (
                f"Predicted discharge reaches <b>{pred_flow:.2f} m³/s</b> with rapid acceleration. "
                f"Channel clearance is only <b>{(DANGER_THRESHOLD - pred_flow):.2f} m³/s</b> with <b>{flood_prob:.1f}%</b> breach probability."
            ),
            "protocol_header": "HIGH-ALERT STANDBY ACTIONS (LEVEL 3 - WARNING):",
            "protocols": [
                "<b>Pre-Warning Sirens:</b> Sound advisory sirens in low-lying riverside settlement zones along Singati corridor.",
                "<b>Continuous Telemetry Polling:</b> Switch upstream radar gauges at Tsho Rolpa (4,580m) to high-frequency 10-minute cycles.",
                "<b>Grid Coordination:</b> Coordinate with Nepal Electricity Authority (NEA) Load Dispatch for contingency load-shedding.",
                "<b>Emergency Communications Desk:</b> [PLACEHOLDER: APF Disaster Response Base Singati / Hotline: +977-49-421999]",
            ],
        }
    elif pred_flow >= YELLOW_THRESHOLD or flood_prob >= 35.0:
        base = PALETTE["watch"]
        return {
            **base,
            "headline": "MONSOON SURGE IN UPPER CATCHMENT — BUFFER WATCH",
            "summary": (
                f"Predicted discharge reaches <b>{pred_flow:.2f} m³/s</b>, entering the 10% safety buffer zone "
                f"(<b>{YELLOW_THRESHOLD:.2f} – {DANGER_THRESHOLD:.2f} m³/s</b>). AI inundation likelihood: <b>{flood_prob:.1f}%</b>."
            ),
            "protocol_header": "ACTIVE HYDROLOGICAL WATCH ACTIONS (LEVEL 2 - WATCH):",
            "protocols": [
                "<b>Floodplain Clearance:</b> Order riverbank sand and gravel excavation crews to immediately evacuate the gorge floodplain.",
                "<b>Glacial Siphon Surveillance:</b> Monitor Tsho Rolpa moraine lake siphon discharge and positive temperature anomalies.",
                "<b>DDMC Advisory:</b> Notify Dolakha disaster coordinators and transport police along Singati-Charikot highway.",
                "<b>Emergency Communications Desk:</b> [PLACEHOLDER: Upper Tamakoshi Hydropower Control Room / Hotline: +977-49-420000]",
            ],
        }
    else:
        base = PALETTE["normal"]
        return {
            **base,
            "headline": "SAFE HYDROLOGICAL EQUILIBRIUM",
            "summary": (
                f"Discharge is safely contained within natural gorge banks at <b>{pred_flow:.2f} m³/s</b> "
                f"(Clearance to Danger Line: <b>{(DANGER_THRESHOLD - pred_flow):.2f} m³/s</b>). Flood probability: <b>{flood_prob:.1f}%</b>."
            ),
            "protocol_header": "STANDARD SENTINEL SURVEILLANCE (LEVEL 1 - NORMAL):",
            "protocols": [
                "<b>Nominal Hydro Dispatch:</b> Maintain scheduled 456 MW generation dispatch at Upper Tamakoshi Plant.",
                "<b>Automated Monitoring:</b> Autonomous GloFAS satellite precipitation and acoustic telemetry polling every 30 minutes.",
                "<b>Information Desk:</b> [PLACEHOLDER: DHM Flood Forecasting Information Desk / Toll-Free: 1155]",
            ],
        }


# 5. Live basin telemetry
with st.spinner("Connecting to the Tamakoshi telemetry services..."):
    live_payload = route_live_basin_telemetry(past_days=14)

active_inputs = live_payload["inputs"]
data_source_label = live_payload["status_label"]
data_synced_at = live_payload["synced_at"]
live_history_df = live_payload["history_df"]

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
# 7. MAIN VIEW HIERARCHY: Editorial River Header
# -----------------------------------------------------------------------------
render_river_hero(data_source_label, data_synced_at)
st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. HIERARCHY STEP 1: Emergency Action & Severity Status Banner
# -----------------------------------------------------------------------------
freeboard = DANGER_THRESHOLD - pred_flow
freeboard_text = f"+{freeboard:.2f} m³/s Safe Clearance" if freeboard >= 0 else f"OVERTOPPING BY {abs(freeboard):.2f} m³/s"
freeboard_color = PALETTE["normal"]["color"] if freeboard >= 0 else PALETTE["emergency"]["color"]

st.markdown(
    f"""
    <div class="severity-banner" style="border-left-color: {severity['border']}; background: linear-gradient(90deg, {severity['bg']} 0%, {PALETTE['bg_surface']} 40%);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div class="severity-tag" style="background: {severity['bg']}; color: {severity['color']}; border: 1px solid {severity['border']}; box-shadow: 0 0 12px {severity['glow']};">
                <span class="status-dot" style="background: {severity['color']}; box-shadow: 0 0 8px {severity['color']};"></span>
                LEVEL {severity['level_num']} ALERT: {severity['tier']}
            </div>
            <div style="font-size: 0.78rem; font-weight: 700; color: {PALETTE['text_secondary']}; text-transform: uppercase; letter-spacing: 0.05em;">
                Threshold: <span style="color:{PALETTE['emergency']['color']}; font-weight:800;">{DANGER_THRESHOLD:.2f} m³/s Danger Line</span>
            </div>
        </div>
        <div class="severity-title" style="color: {severity['color']};">
            {severity['headline']}
        </div>
        <div class="severity-body">
            {severity['summary']}
        </div>
        <div style="margin-top: 14px; font-weight: 700; font-size: 0.88rem; color: {PALETTE['text_primary']}; letter-spacing: 0.03em;">
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
            <div class="kpi-label">Forecast Tomorrow Flow (Singati)</div>
            <div class="kpi-value" style="color: {severity['color']};">
                {pred_flow:.2f} <span style="font-size: 1.15rem; font-weight: 600; color: {PALETTE['text_muted']};">m³/s</span>
            </div>
            <div class="kpi-meta" style="color: {delta_color}; font-weight: 600;">
                <span class="status-dot" style="background: {delta_color}; box-shadow: 0 0 6px {delta_color};"></span>
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
            <div class="kpi-value" style="color: {PALETTE['accent_primary']};">
                {low_flow:.2f} – {high_flow:.2f} <span style="font-size: 1.15rem; font-weight: 600; color: {PALETTE['text_muted']};">m³/s</span>
            </div>
            <div class="kpi-meta" style="color: {PALETTE['text_secondary']};">
                <span class="status-dot" style="background: {PALETTE['accent_primary']};"></span>
                Quantile spread: ±{ci_half:.2f} m³/s (10%–90%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c3:
    st.markdown(
        f"""
        <div class="command-card">
            <div class="kpi-label">Channel Safety Clearance</div>
            <div class="kpi-value" style="color: {freeboard_color};">
                {freeboard:+.2f} <span style="font-size: 1.15rem; font-weight: 600; color: {PALETTE['text_muted']};">m³/s</span>
            </div>
            <div class="kpi-meta" style="color: {PALETTE['text_muted']};">
                <span class="status-dot" style="background: {freeboard_color};"></span>
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
                {flood_prob:.1f}<span style="font-size: 1.25rem; font-weight: 600; color: {PALETTE['text_muted']};">%</span>
            </div>
            <div class="kpi-meta" style="color: {severity['color']}; font-weight: 700;">
                <span class="status-dot" style="background: {severity['color']}; box-shadow:0 0 6px {severity['color']};"></span>
                Threat Level: {severity['tier']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. HIERARCHY STEP 3: Core Visual Tier (Hydrograph + 3D Basin Corridor Map)
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
            fillcolor="rgba(56, 189, 248, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="80% Prediction Interval",
        )
    )
    # Sensor History
    fig_hydro.add_trace(
        go.Scatter(
            x=timeline_hours[timeline_hours <= 0],
            y=past_curve,
            mode="lines+markers",
            name="Recorded Discharge (T-24h to T0)",
            line=dict(color=PALETTE["accent_primary"], width=3),
            marker=dict(size=4, color=PALETTE["accent_primary"]),
        )
    )
    # AI 24h Forecast Ray
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
    # Safety Threshold Lines (Red Danger & Yellow Watch)
    fig_hydro.add_hline(
        y=DANGER_THRESHOLD,
        line_dash="dot",
        line_color=PALETTE["emergency"]["color"],
        line_width=2,
        annotation_text=f"Red Danger Line ({DANGER_THRESHOLD:.2f} m³/s)",
        annotation_position="top right",
        annotation_font=dict(color=PALETTE["emergency"]["color"], size=10),
    )
    fig_hydro.add_hline(
        y=YELLOW_THRESHOLD,
        line_dash="dashdot",
        line_color=PALETTE["watch"]["color"],
        line_width=1.5,
        annotation_text=f"Yellow Watch Buffer ({YELLOW_THRESHOLD:.2f} m³/s)",
        annotation_position="bottom right",
        annotation_font=dict(color=PALETTE["watch"]["color"], size=10),
    )
    fig_hydro.update_layout(
        template="plotly_dark",
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
    st.markdown("<div class='kpi-label'>3D River Corridor & Gauging Network</div>", unsafe_allow_html=True)

    def station_severity_color(val: float, is_target: bool = False):
        check_val = pred_flow if is_target else val
        if check_val >= DANGER_THRESHOLD:
            return [239, 68, 68, 240]   # Red
        elif check_val >= YELLOW_THRESHOLD:
            return [245, 158, 11, 230]  # Amber
        else:
            return [16, 185, 129, 220]  # Emerald

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
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={
            "html": (
                "<div style='background:#131D31; padding:8px 12px; border-radius:8px; border:1px solid #23324E; color:#F8FAFC; box-shadow:0 4px 12px rgba(0,0,0,0.5); font-family:sans-serif;'>"
                "<b style='color:#38BDF8;'>{name}</b><br>Discharge: <b>{flow} m³/s</b></div>"
            ),
            "style": {"color": "#F8FAFC"},
        },
    )
    st.pydeck_chart(deck_chart)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 11. HIERARCHY STEP 4: Deep-Dive Diagnostic Tabs
# -----------------------------------------------------------------------------
tab_gorge, tab_council, tab_gauges, tab_table, tab_method = st.tabs([
    "3D River Gorge Digital Twin",
    "Multi-Model Ensemble",
    "Hydrological Stress Indicators",
    "Corridor Sensor Telemetry",
    "System Architecture & Contacts",
])

# ----------------- Tab 1: 3D River Gorge Digital Twin -----------------
with tab_gorge:
    st.markdown("<div class='kpi-label'>Interactive 3D Singati River Gorge Hydraulic Simulation (Three.js WebGL)</div>", unsafe_allow_html=True)
    st.caption("Physically calibrated dynamic water surface with downstream wave velocity, canyon cliff topography, staff gauge, and danger plane.")

    stage_m = round(min(8.5, max(0.5, 1.2 + (pred_flow / DANGER_THRESHOLD) * 3.8)), 2)
    velocity_ms = round(min(7.2, max(0.8, 1.2 + (pred_flow / 3.0))), 1)

    # Dynamic Water Color: Glacial Turquoise in safe flow, Amber-Silt in watch, Turbulent Crimson in flood!
    if pred_flow >= DANGER_THRESHOLD:
        water_hex_color = "#C53030"
        water_specular = "#EF4444"
        beacon_color = "#EF4444"
    elif pred_flow >= YELLOW_THRESHOLD:
        water_hex_color = "#B45309"
        water_specular = "#F59E0B"
        beacon_color = "#F59E0B"
    else:
        water_hex_color = "#0284C7"
        water_specular = "#38BDF8"
        beacon_color = "#10B981"

    three_js_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                overflow: hidden;
                background: #0B1120;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                user-select: none;
            }}
            #canvas-container {{
                width: 100vw;
                height: 520px;
                position: relative;
            }}
            #hud-top {{
                position: absolute;
                top: 14px;
                left: 16px;
                background: rgba(19, 29, 49, 0.88);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(56, 189, 248, 0.25);
                border-radius: 10px;
                padding: 12px 18px;
                color: #F8FAFC;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
                pointer-events: none;
                min-width: 280px;
            }}
            .hud-tag {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: {severity['color']};
                margin-bottom: 4px;
            }}
            .hud-dot {{
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: {severity['color']};
                box-shadow: 0 0 8px {severity['color']};
            }}
            .hud-title {{
                font-size: 0.95rem;
                font-weight: 800;
                color: #F8FAFC;
                letter-spacing: -0.01em;
                margin-bottom: 8px;
            }}
            .hud-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 6px 14px;
                font-size: 0.80rem;
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                padding-top: 8px;
            }}
            .hud-label {{
                color: #94A3B8;
            }}
            .hud-val {{
                font-weight: 700;
                color: #F8FAFC;
                text-align: right;
            }}
            #camera-bar {{
                position: absolute;
                bottom: 14px;
                left: 16px;
                display: flex;
                gap: 8px;
            }}
            .cam-btn {{
                background: rgba(19, 29, 49, 0.85);
                backdrop-filter: blur(6px);
                border: 1px solid rgba(56, 189, 248, 0.25);
                color: #F8FAFC;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 0.75rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s ease;
            }}
            .cam-btn:hover {{
                background: #1A2642;
                border-color: #38BDF8;
                color: #38BDF8;
                box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
            }}
            #hint-bar {{
                position: absolute;
                bottom: 14px;
                right: 16px;
                background: rgba(19, 29, 49, 0.75);
                backdrop-filter: blur(6px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
                padding: 6px 12px;
                color: #94A3B8;
                font-size: 0.74rem;
                pointer-events: none;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container"></div>
        <div id="hud-top">
            <div class="hud-tag"><span class="hud-dot"></span>{severity['tier']}</div>
            <div class="hud-title">Singati River Gorge Hydraulic Stage</div>
            <div class="hud-grid">
                <div class="hud-label">Discharge:</div>
                <div class="hud-val" style="color:{severity['color']};">{pred_flow:.2f} m³/s</div>
                <div class="hud-label">River Stage:</div>
                <div class="hud-val">+{stage_m:.2f} m</div>
                <div class="hud-label">Current Velocity:</div>
                <div class="hud-val">~{velocity_ms:.1f} m/s</div>
                <div class="hud-label">Danger Margin:</div>
                <div class="hud-val" style="color:{freeboard_color};">{freeboard:+.2f} m³/s</div>
            </div>
        </div>
        <div id="camera-bar">
            <button class="cam-btn" onclick="setCamera('side')">Side Bank View</button>
            <button class="cam-btn" onclick="setCamera('aerial')">Aerial Drone</button>
            <button class="cam-btn" onclick="setCamera('gorge')">Downstream Gorge</button>
            <button class="cam-btn" onclick="toggleDangerPlane()">Toggle Danger Plane</button>
        </div>
        <div id="hint-bar">Left Drag: Rotate | Right Drag: Pan | Scroll: Zoom</div>

        <script>
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0B1120);
            scene.fog = new THREE.FogExp2(0x0B1120, 0.018);

            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / 520, 0.1, 1000);
            camera.position.set(24, 16, 28);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(window.innerWidth, 520);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2.05;
            controls.minDistance = 8;
            controls.maxDistance = 80;
            controls.target.set(0, 0.5, 0);

            // Lighting: Ambient Sky + Warm Directional Sun + Alert Beacon
            const ambient = new THREE.AmbientLight(0x334155, 1.2);
            scene.add(ambient);

            const sun = new THREE.DirectionalLight(0xFFFFFF, 1.6);
            sun.position.set(25, 40, 20);
            sun.castShadow = true;
            scene.add(sun);

            const riverGlow = new THREE.PointLight('{water_specular}', 1.8, 35);
            riverGlow.position.set(0, 3, 0);
            scene.add(riverGlow);

            // 1. Realistic Canyon Cliff Mesh
            const canyonGeo = new THREE.PlaneGeometry(38, 68, 80, 80);
            canyonGeo.rotateX(-Math.PI / 2);
            const pos = canyonGeo.attributes.position;
            for (let i = 0; i < pos.count; i++) {{
                let x = pos.getX(i);
                let z = pos.getZ(i);
                let dist = Math.abs(x);
                let canyonSlope = Math.pow(dist / 4.2, 2.2) - 1.8;
                let ledge = Math.sin(dist * 0.9) * 0.35;
                let noise1 = Math.sin(x * 0.4 + z * 0.25) * 0.7;
                let noise2 = Math.cos(x * 0.8 - z * 0.5) * 0.3;
                let y = canyonSlope + ledge + noise1 + noise2;
                if (dist < 3.2) {{
                    y = -2.2 + (dist / 3.2) * 0.5 + Math.sin(z * 0.6) * 0.12;
                }}
                pos.setY(i, y);
            }}
            canyonGeo.computeVertexNormals();
            const canyonMat = new THREE.MeshStandardMaterial({{
                color: 0x1E293B, roughness: 0.88, metalness: 0.15, flatShading: true
            }});
            const canyonMesh = new THREE.Mesh(canyonGeo, canyonMat);
            canyonMesh.receiveShadow = true;
            scene.add(canyonMesh);

            // 2. Shoreline Boulders
            const rockMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.9, flatShading: true }});
            for (let r = -24; r <= 24; r += 7) {{
                const rockGeo = new THREE.DodecahedronGeometry(0.8 + Math.random() * 0.6, 1);
                const rockL = new THREE.Mesh(rockGeo, rockMat);
                rockL.position.set(-3.5 - Math.random() * 0.8, -0.6 + Math.random() * 0.4, r);
                rockL.scale.set(1, 0.7 + Math.random() * 0.5, 1);
                scene.add(rockL);

                const rockR = new THREE.Mesh(rockGeo, rockMat);
                rockR.position.set(3.5 + Math.random() * 0.8, -0.6 + Math.random() * 0.4, r + 3);
                rockR.scale.set(1, 0.7 + Math.random() * 0.5, 1);
                scene.add(rockR);
            }}

            // 3. Dynamic Water Surface (Physical Elevation Calibrated to Flow)
            const predFlow = {pred_flow};
            const dangerThreshold = {DANGER_THRESHOLD};
            const waterHeight = -1.8 + Math.min(3.8, (predFlow / dangerThreshold) * 2.8);

            const waterGeo = new THREE.PlaneGeometry(14.5, 68, 64, 64);
            waterGeo.rotateX(-Math.PI / 2);
            const waterMat = new THREE.MeshStandardMaterial({{
                color: '{water_hex_color}',
                roughness: 0.12,
                metalness: 0.35,
                transparent: true,
                opacity: 0.88
            }});
            const waterMesh = new THREE.Mesh(waterGeo, waterMat);
            waterMesh.position.y = waterHeight;
            scene.add(waterMesh);

            // 4. Vertical River Staff Gauge (Measuring Post)
            const staffGeo = new THREE.BoxGeometry(0.3, 5.0, 0.3);
            const staffMat = new THREE.MeshStandardMaterial({{ color: 0xE2E8F0, roughness: 0.5 }});
            const staff = new THREE.Mesh(staffGeo, staffMat);
            staff.position.set(-3.8, 0.7, 0);
            scene.add(staff);

            // Depth markings on staff gauge
            for (let mk = -1.5; mk <= 2.8; mk += 0.8) {{
                const ringGeo = new THREE.BoxGeometry(0.38, 0.08, 0.38);
                const ringMat = new THREE.MeshBasicMaterial({{ color: (mk >= 1.0 ? 0xEF4444 : 0x0F172A) }});
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.position.set(-3.8, mk, 0);
                scene.add(ring);
            }}

            // 5. Horizontal Safety Threshold Planes
            const dangerGeo = new THREE.PlaneGeometry(16, 68);
            dangerGeo.rotateX(-Math.PI / 2);
            const dangerPlane = new THREE.Mesh(
                dangerGeo,
                new THREE.MeshBasicMaterial({{ color: 0xEF4444, wireframe: true, transparent: true, opacity: 0.38 }})
            );
            dangerPlane.position.y = 1.0; // 6.69 m³/s Danger Line
            scene.add(dangerPlane);

            const watchPlane = new THREE.Mesh(
                dangerGeo,
                new THREE.MeshBasicMaterial({{ color: 0xF59E0B, wireframe: true, transparent: true, opacity: 0.22 }})
            );
            watchPlane.position.y = 0.72; // 6.02 m³/s Watch Buffer
            scene.add(watchPlane);

            // 6. Floating Telemetry Sensor Buoy
            const buoyGroup = new THREE.Group();
            const buoyBody = new THREE.Mesh(
                new THREE.CylinderGeometry(0.55, 0.35, 0.4, 16),
                new THREE.MeshStandardMaterial({{ color: 0xFACC15, roughness: 0.3 }})
            );
            buoyGroup.add(buoyBody);

            const antenna = new THREE.Mesh(
                new THREE.CylinderGeometry(0.04, 0.04, 0.8, 8),
                new THREE.MeshStandardMaterial({{ color: 0x0F172A }})
            );
            antenna.position.y = 0.5;
            buoyGroup.add(antenna);

            const beacon = new THREE.Mesh(
                new THREE.SphereGeometry(0.12, 12, 12),
                new THREE.MeshBasicMaterial({{ color: '{beacon_color}' }})
            );
            beacon.position.y = 0.92;
            buoyGroup.add(beacon);

            buoyGroup.position.set(0.8, waterHeight + 0.15, 2.0);
            scene.add(buoyGroup);

            // Camera Angle Presets
            window.setCamera = function(type) {{
                if (type === 'side') {{
                    camera.position.set(16, 8, 2);
                    controls.target.set(0, waterHeight, 0);
                }} else if (type === 'aerial') {{
                    camera.position.set(0, 38, 8);
                    controls.target.set(0, 0, 0);
                }} else if (type === 'gorge') {{
                    camera.position.set(0, 3.5, 26);
                    controls.target.set(0, waterHeight, -15);
                }}
                controls.update();
            }};

            let showPlane = true;
            window.toggleDangerPlane = function() {{
                showPlane = !showPlane;
                dangerPlane.visible = showPlane;
                watchPlane.visible = showPlane;
            }};

            // Animation Loop (Multi-harmonic Gerstner waves with flow velocity linked to Q)
            const clock = new THREE.Clock();
            const waveSpeed = 0.8 + (predFlow / 2.8);
            const waveAmp = 0.05 + Math.min(0.22, (predFlow / 16.0));

            function animate() {{
                requestAnimationFrame(animate);
                const t = clock.getElapsedTime() * waveSpeed;

                // Wave vertex displacement
                const wp = waterGeo.attributes.position;
                for (let i = 0; i < wp.count; i++) {{
                    let u = wp.getX(i);
                    let v = wp.getZ(i);
                    let flowZ = v - t * 3.5;
                    let w1 = Math.sin(flowZ * 0.4 + u * 0.3) * waveAmp;
                    let w2 = Math.cos(flowZ * 0.85 - u * 0.6) * (waveAmp * 0.5);
                    let ripple = Math.sin(u * 2.0 + flowZ * 1.6) * 0.02;
                    wp.setY(i, w1 + w2 + ripple);
                }}
                waterGeo.attributes.position.needsUpdate = true;
                waterGeo.computeVertexNormals();

                // Buoy physics bobbing
                const buoyWave = Math.sin((buoyGroup.position.z - t * 3.5) * 0.4 + buoyGroup.position.x * 0.3) * waveAmp;
                buoyGroup.position.y = waterHeight + 0.15 + buoyWave;
                buoyGroup.rotation.z = Math.sin(t * 2.0) * 0.08;
                buoyGroup.rotation.x = Math.cos(t * 1.8) * 0.08;

                // Beacon pulsing
                beacon.scale.setScalar(1.0 + Math.sin(t * 6.0) * 0.3);

                controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            // Resize Handler
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / 520;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, 520);
            }});
        </script>
    </body>
    </html>
    """
    components.html(three_js_html, height=530)

# ----------------- Tab 2: Multi-Model Ensemble -----------------
with tab_council:
    st.markdown("<div class='kpi-label'>Council of 6 Base Estimators vs. RidgeCV Meta-Learner Consensus</div>", unsafe_allow_html=True)
    st.caption("Comparison of base models with stacked out-of-fold cross-validated meta-regression.")

    council_data = dict(individual_preds)
    council_data["RidgeCV Meta-Learner"] = pred_flow

    m_names = list(council_data.keys())
    m_flows = [council_data[m] for m in m_names]
    m_colors = [PALETTE["accent_primary"] if m != "RidgeCV Meta-Learner" else severity["color"] for m in m_names]

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
        template="plotly_dark",
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
        st.metric("Council Range Spread", f"{max(base_only_flows) - min(base_only_flows):.3f} m³/s", "Spread across 6 base estimators")
    with sc2:
        st.metric("Unweighted Council Mean", f"{np.mean(base_only_flows):.3f} m³/s", "Arithmetic mean")
    with sc3:
        st.metric("Meta-Learner Consensus", f"{pred_flow - np.mean(base_only_flows):+.3f} m³/s", "RidgeCV cross-validated delta")

# ----------------- Tab 3: Hydrological Stress Indicators -----------------
with tab_gauges:
    st.markdown("<div class='kpi-label'>Coupled Glacial-Monsoon Physical Forcing Gauges</div>", unsafe_allow_html=True)
    st.caption("Physical environmental metrics driving overland infiltration and glacial degree-day melt.")

    g1, g2, g3, g4 = st.columns(4)
    rain_intensity = active_inputs["rain_mid"] / (active_inputs["rain_hours"] + 0.5)
    glacier_temp = active_inputs["tmax"] - 6.0

    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=rain_intensity, number={"suffix": " mm/h"},
            title={"text": "Cloudburst Rain Intensity", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 40], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent_primary"]},
                "steps": [
                    {"range": [0, 10], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [10, 20], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [20, 40], "color": "rgba(239, 68, 68, 0.35)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 18.0},
            },
        ))
        fig_g1.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g1, use_container_width=True)

    with g2:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=active_inputs["rain_7d"], number={"suffix": " mm"},
            title={"text": "Antecedent 7-Day Rain", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 300], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent_primary"]},
                "steps": [
                    {"range": [0, 80], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [80, 150], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [150, 300], "color": "rgba(239, 68, 68, 0.35)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 140.0},
            },
        ))
        fig_g2.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g2, use_container_width=True)

    with g3:
        fig_g3 = go.Figure(go.Indicator(
            mode="gauge+number", value=glacier_temp, number={"suffix": " °C"},
            title={"text": "Tsho Rolpa (4,580m) Temp", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 30], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent_primary"]},
                "steps": [
                    {"range": [0, 8], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [8, 16], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [16, 30], "color": "rgba(239, 68, 68, 0.35)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 15.0},
            },
        ))
        fig_g3.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g3, use_container_width=True)

    with g4:
        fig_g4 = go.Figure(go.Indicator(
            mode="gauge+number", value=active_inputs["soil_sat_pct"], number={"suffix": "%"},
            title={"text": "Vadose Deep Soil Saturation", "font": {"size": 13, "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 60], "tickcolor": PALETTE["text_secondary"]},
                "bar": {"color": PALETTE["accent_primary"]},
                "steps": [
                    {"range": [0, 25], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [25, 40], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [40, 60], "color": "rgba(239, 68, 68, 0.35)"},
                ],
                "threshold": {"line": {"color": PALETTE["emergency"]["color"], "width": 3}, "value": 42.0},
            },
        ))
        fig_g4.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["bg_surface"], height=200, margin=dict(l=14, r=14, t=28, b=8))
        st.plotly_chart(fig_g4, use_container_width=True)

# ----------------- Tab 4: Corridor Station Telemetry -----------------
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

# ----------------- Tab 5: System Architecture & Contacts -----------------
with tab_method:
    st.markdown("<div class='kpi-label'>Life-Safety System Architecture & Model Limitations</div>", unsafe_allow_html=True)

    meth_col1, meth_col2 = st.columns(2)
    with meth_col1:
        st.markdown(
            f"""
            <div class="command-card" style="font-size: 0.88rem; line-height: 1.6;">
                <div style="font-weight: 700; color: {PALETTE['accent_primary']}; margin-bottom: 8px;">
                    58-Feature Hydrological Pipeline Architecture
                </div>
                <ul style="margin-top: 6px; padding-left: 18px; color: {PALETTE['text_secondary']};">
                    <li><b>Cascading Hydrodynamics:</b> Station flow gradients, velocity ratios, and upstream corridor sums.</li>
                    <li><b>Temporal Acceleration:</b> 1st and 2nd order differencing plus percentage surge acceleration.</li>
                    <li><b>Vadose Runoff Physics:</b> Coupled non-linear infiltration $R = P_{{basin}} \\times (\\text{{Soil}}_{{deep}})^2$.</li>
                    <li><b>Glacial Melt Thermodynamics:</b> Lapse-rate corrected 4,500m positive degree-days.</li>
                    <li><b>Seasonality Fourier Series:</b> Dual-harmonic intra-annual cyclical curves.</li>
                </ul>
                <div style="font-weight: 700; color: {PALETTE['emergency']['color']}; margin-top: 14px; margin-bottom: 6px;">
                    Known Model Uncertainty & Physical Limitations
                </div>
                <ul style="margin-top: 6px; padding-left: 18px; color: {PALETTE['text_secondary']};">
                    <li>Ensemble trained on 42,540 station-days. High uncertainty during unmodeled landslide dam outbursts (LLDOF) or sudden glacial moraine wall collapse.</li>
                    <li>Confidence bounds represent 80% coverage; extreme outliers beyond historical training envelope may exceed intervals.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with meth_col2:
        st.markdown(
            f"""
            <div class="command-card" style="font-size: 0.88rem; line-height: 1.6;">
                <div style="font-weight: 700; color: {PALETTE['warning']['color']}; margin-bottom: 8px;">
                    Official Disaster Response Directory (Dolakha District)
                </div>
                <div style="margin-top: 10px;">
                    <div style="font-size: 0.80rem; font-weight: 600; color: {PALETTE['text_secondary']};">Dolakha District Emergency Operations Center (DEOC):</div>
                    <code style="color:{PALETTE['text_primary']}; background:#1A2642; padding:4px 8px; border-radius:4px; font-weight: 600; display: inline-block; margin-top: 2px;">[PLACEHOLDER: Charikot DEOC VHF Ch 4 / +977-49-421111]</code>
                </div>
                <div style="margin-top: 10px;">
                    <div style="font-size: 0.80rem; font-weight: 600; color: {PALETTE['text_secondary']};">Armed Police Force Disaster Unit (Singati Base):</div>
                    <code style="color:{PALETTE['text_primary']}; background:#1A2642; padding:4px 8px; border-radius:4px; font-weight: 600; display: inline-block; margin-top: 2px;">[PLACEHOLDER: APF Disaster Base / +977-49-421999]</code>
                </div>
                <div style="margin-top: 10px;">
                    <div style="font-size: 0.80rem; font-weight: 600; color: {PALETTE['text_secondary']};">Upper Tamakoshi Hydropower Control Room:</div>
                    <code style="color:{PALETTE['text_primary']}; background:#1A2642; padding:4px 8px; border-radius:4px; font-weight: 600; display: inline-block; margin-top: 2px;">[PLACEHOLDER: Gongar Dam Control Room / +977-49-420000]</code>
                </div>
                <div style="margin-top: 10px;">
                    <div style="font-size: 0.80rem; font-weight: 600; color: {PALETTE['text_secondary']};">National Flood Forecasting Toll-Free Emergency Desk:</div>
                    <code style="color:{PALETTE['text_primary']}; background:#1A2642; padding:4px 8px; border-radius:4px; font-weight: 600; display: inline-block; margin-top: 2px;">[PLACEHOLDER: DHM Emergency Desk / Toll-Free 1155]</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
