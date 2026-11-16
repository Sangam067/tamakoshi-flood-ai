"""Enter Tamakoshi basin measurements and run the flood-risk model."""

import calendar
import os
import sys

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.tamakoshi_api import (
    compute_forecast_from_inputs,
    load_model_bundle,
    route_live_basin_telemetry,
)

st.set_page_config(
    page_title="Data & Risk Predictor | Tamakoshi Flood AI",
    page_icon="🌊",
    layout="wide",
)

st.title("Data Entry & Risk Prediction")
st.caption(
    "Review the latest available measurements, enter updated basin data, "
    "and run the Tamakoshi flood-risk model."
)

try:
    bundle = load_model_bundle()
except Exception as exc:
    st.error(f"The flood-risk model could not be loaded: {exc}")
    st.stop()

try:
    telemetry = route_live_basin_telemetry(past_days=7)
except Exception as exc:
    st.error(f"Could not load default telemetry values: {exc}")
    st.stop()

defaults = telemetry["inputs"]
st.info(
    f"Input defaults: {telemetry['status_label']} · Last synced "
    f"{telemetry['synced_at']}. Edit any field before running a prediction."
)
if telemetry["source_mode"] != "LIVE_API":
    st.warning(
        "The API is unavailable. The form is prefilled with the app's "
        "verified fallback snapshot; confirm or replace these values before use."
    )

widget_keys = {
    "q_mid": "risk_q_mid",
    "q_yesterday": "risk_q_yesterday",
    "q_glacier": "risk_q_glacier",
    "q_upper": "risk_q_upper",
    "q_lower": "risk_q_lower",
    "rain_mid": "risk_rain_mid",
    "rain_glacier": "risk_rain_glacier",
    "rain_3d": "risk_rain_3d",
    "rain_7d": "risk_rain_7d",
    "rain_hours": "risk_rain_hours",
    "soil_sat_pct": "risk_soil_sat_pct",
    "tmax": "risk_tmax",
    "tmin": "risk_tmin",
    "snow_glacier": "risk_snow_glacier",
    "solar_rad": "risk_solar_rad",
    "month": "risk_month",
}
for input_name, key in widget_keys.items():
    st.session_state.setdefault(key, defaults[input_name])

with st.form("tamakoshi_risk_input_form"):
    st.subheader("River discharge")
    flow_col1, flow_col2, flow_col3 = st.columns(3)
    with flow_col1:
        q_mid = st.number_input(
            "Singati flow today (m³/s)", min_value=0.0, step=0.1, key=widget_keys["q_mid"]
        )
        q_yesterday = st.number_input(
            "Singati flow yesterday (m³/s)",
            min_value=0.0,
            step=0.1,
            key=widget_keys["q_yesterday"],
        )
    with flow_col2:
        q_glacier = st.number_input(
            "Tsho Rolpa outflow (m³/s)",
            min_value=0.0,
            step=0.05,
            key=widget_keys["q_glacier"],
        )
        q_upper = st.number_input(
            "Gongar flow (m³/s)", min_value=0.0, step=0.1, key=widget_keys["q_upper"]
        )
    with flow_col3:
        q_lower = st.number_input(
            "Lower Tamakoshi flow (m³/s)",
            min_value=0.0,
            step=0.1,
            key=widget_keys["q_lower"],
        )

    st.subheader("Rainfall and soil")
    rain_col1, rain_col2, rain_col3 = st.columns(3)
    with rain_col1:
        rain_mid = st.number_input(
            "Valley rainfall today (mm)",
            min_value=0.0,
            step=1.0,
            key=widget_keys["rain_mid"],
        )
        rain_glacier = st.number_input(
            "Glacier-area rainfall today (mm)",
            min_value=0.0,
            step=1.0,
            key=widget_keys["rain_glacier"],
        )
    with rain_col2:
        rain_3d = st.number_input(
            "Cumulative rainfall, 3 days (mm)",
            min_value=0.0,
            step=5.0,
            key=widget_keys["rain_3d"],
        )
        rain_7d = st.number_input(
            "Cumulative rainfall, 7 days (mm)",
            min_value=0.0,
            step=5.0,
            key=widget_keys["rain_7d"],
        )
    with rain_col3:
        rain_hours = st.number_input(
            "Rain duration today (hours)",
            min_value=0,
            max_value=24,
            step=1,
            key=widget_keys["rain_hours"],
        )
        soil_sat_pct = st.slider(
            "Deep soil saturation (%)",
            min_value=10,
            max_value=50,
            key=widget_keys["soil_sat_pct"],
        )

    st.subheader("Weather and glacial conditions")
    weather_col1, weather_col2, weather_col3 = st.columns(3)
    with weather_col1:
        tmax = st.number_input(
            "Valley maximum temperature (°C)",
            step=0.5,
            key=widget_keys["tmax"],
        )
        tmin = st.number_input(
            "Valley minimum temperature (°C)",
            step=0.5,
            key=widget_keys["tmin"],
        )
    with weather_col2:
        snow_glacier = st.number_input(
            "Glacier snowfall today (cm)",
            min_value=0.0,
            step=0.5,
            key=widget_keys["snow_glacier"],
        )
        solar_rad = st.number_input(
            "Solar radiation (MJ/m²)",
            min_value=0.0,
            step=0.5,
            key=widget_keys["solar_rad"],
        )
    with weather_col3:
        month = st.selectbox(
            "Month represented by these measurements",
            options=range(1, 13),
            format_func=lambda value: calendar.month_name[value],
            key=widget_keys["month"],
        )

    submitted = st.form_submit_button("Run flood-risk prediction", type="primary")

active_inputs = {
    "q_mid": q_mid,
    "q_yesterday": q_yesterday,
    "q_glacier": q_glacier,
    "q_upper": q_upper,
    "q_lower": q_lower,
    "rain_mid": rain_mid,
    "rain_glacier": rain_glacier,
    "rain_3d": rain_3d,
    "rain_7d": rain_7d,
    "rain_hours": rain_hours,
    "soil_sat_pct": soil_sat_pct,
    "tmax": tmax,
    "tmin": tmin,
    "snow_glacier": snow_glacier,
    "solar_rad": solar_rad,
    "month": month,
}
forecast = compute_forecast_from_inputs(bundle, active_inputs)

if submitted:
    st.success("Prediction updated using the submitted measurements.")
else:
    st.caption("Showing the prediction for the values currently in the form.")

risk_profiles = {
    "RED_ALERT": ("CRITICAL", "#EF4444"),
    "YELLOW_WATCH": ("ELEVATED", "#F59E0B"),
    "GREEN_NORMAL": ("LOW", "#10B981"),
}
risk_label, risk_color = risk_profiles[forecast["alert_level"]]
st.markdown(
    f"""
    <div style="border-left:5px solid {risk_color}; background:#131D31;
                border-radius:8px; padding:16px 20px; margin:18px 0;">
      <div style="font-size:0.78rem; opacity:0.75; letter-spacing:0.08em;">MODEL RISK LEVEL</div>
      <div style="font-size:1.55rem; font-weight:800; color:{risk_color};">{risk_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Predicted next-day flow", f"{forecast['pred_flow']:.2f} m³/s")
metric_col2.metric("Flood probability", f"{forecast['flood_prob']:.1f}%")
metric_col3.metric(
    "Model interval",
    f"{forecast['low_flow']:.2f}–{forecast['high_flow']:.2f} m³/s",
)
metric_col4.metric("Danger threshold", f"{forecast['danger_threshold']:.2f} m³/s")

danger_threshold = forecast["danger_threshold"]
fig = go.Figure()
fig.add_hrect(
    y0=0,
    y1=forecast["yellow_trigger"],
    fillcolor="rgba(16, 185, 129, 0.10)",
    line_width=0,
    layer="below",
)
fig.add_hrect(
    y0=forecast["yellow_trigger"],
    y1=danger_threshold,
    fillcolor="rgba(245, 158, 11, 0.12)",
    line_width=0,
    layer="below",
)
fig.add_trace(
    go.Scatter(
        x=["Next day", "Next day"],
        y=[forecast["low_flow"], forecast["high_flow"]],
        mode="lines",
        line={"color": "#38BDF8", "width": 12},
        name="Model interval",
    )
)
fig.add_trace(
    go.Scatter(
        x=["Next day"],
        y=[forecast["pred_flow"]],
        mode="markers",
        marker={"color": risk_color, "size": 18, "line": {"color": "#F8FAFC", "width": 2}},
        name="Predicted flow",
    )
)
fig.add_hline(
    y=forecast["yellow_trigger"],
    line_dash="dash",
    line_color="#F59E0B",
    annotation_text="Watch threshold",
)
fig.add_hline(
    y=danger_threshold,
    line_dash="dot",
    line_color="#EF4444",
    annotation_text="Danger threshold",
)
fig.update_layout(
    title="Next-day discharge forecast and thresholds",
    template="plotly_dark",
    paper_bgcolor="#131D31",
    plot_bgcolor="#131D31",
    yaxis_title="Discharge (m³/s)",
    xaxis={"visible": False},
    margin={"l": 20, "r": 20, "t": 55, "b": 15},
    height=360,
)
st.plotly_chart(fig, width="stretch")
st.warning(
    "This model output is decision support, not an official flood warning. "
    "Verify measurements and follow local disaster-management authorities."
)
