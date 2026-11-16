"""Show live Tamakoshi telemetry and recent observed discharge risk."""

import html
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.tamakoshi_api import load_model_bundle, route_live_basin_telemetry
from ui_theme import render_page_intro, render_site_navigation, tamakoshi_logo_svg

st.set_page_config(
    page_title="Live Data & History | Tamakoshi Flood AI",
    page_icon=tamakoshi_logo_svg(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_site_navigation("history")
render_page_intro(
    "TELEMETRY · OBSERVED RISK",
    "Follow the river",
    "Explore upstream-to-downstream discharge, rainfall, and observed Singati "
    "risk levels over the latest available months.",
)

try:
    bundle = load_model_bundle()
except Exception as exc:
    st.error(f"The flood-risk model could not be loaded: {exc}")
    st.stop()

controls_col, refresh_col = st.columns([3, 1])
with controls_col:
    lookback_days = st.select_slider(
        "Historical window",
        options=[30, 60, 90],
        value=90,
        format_func=lambda days: f"{days} days",
    )
with refresh_col:
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    refresh = st.button("Refresh live data", width="stretch")
if refresh:
    st.cache_data.clear()

with st.spinner(f"Loading up to {lookback_days} days of Tamakoshi telemetry..."):
    try:
        payload = route_live_basin_telemetry(past_days=lookback_days)
    except Exception as exc:
        st.error(f"Could not retrieve Tamakoshi telemetry: {exc}")
        st.stop()

st.info(f"{payload['status_label']} · Last synced {payload['synced_at']}")
if payload["source_mode"] != "LIVE_API":
    st.warning(
        "Live services did not return this history window. The chart below uses "
        "the app's verified short fallback snapshot; it does not fabricate "
        "the missing months."
    )

history = payload["history_df"].copy()
if history.empty or "Date" not in history:
    st.error("The telemetry response contains no dated history to chart.")
    st.stop()

history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
history = history.dropna(subset=["Date"]).sort_values("Date")
if history.empty:
    st.error("The telemetry response contains no valid dates to chart.")
    st.stop()

station_prefixes = (
    "Tsho Rolpa",
    "Gongar",
    "Singati",
    "Lower Tamakoshi",
)
station_columns = {
    station: next(
        (column for column in history.columns if column.startswith(station)),
        None,
    )
    for station in station_prefixes
}
missing_stations = [station for station, column in station_columns.items() if column is None]
if missing_stations:
    st.error(
        "The telemetry response is missing expected station data: "
        + ", ".join(missing_stations)
    )
    st.stop()

try:
    danger_threshold = float(bundle["threshold"])
except (KeyError, TypeError, ValueError) as exc:
    st.error(f"The model's danger threshold is unavailable: {exc}")
    st.stop()

watch_threshold = danger_threshold * 0.90
warning_threshold = danger_threshold - 0.35
singati_column = station_columns["Singati"]
history[singati_column] = pd.to_numeric(history[singati_column], errors="coerce")
history = history.dropna(subset=[singati_column])
if history.empty:
    st.error("No valid Singati discharge values were returned for this period.")
    st.stop()


def observed_risk(flow):
    if flow >= danger_threshold:
        return "Critical"
    if flow >= warning_threshold:
        return "Warning"
    if flow >= watch_threshold:
        return "Watch"
    return "Normal"


risk_colors = {
    "Normal": "#10B981",
    "Watch": "#F59E0B",
    "Warning": "#F97316",
    "Critical": "#EF4444",
}
history["Observed risk"] = history[singati_column].map(observed_risk)
history["Risk tier"] = history["Observed risk"].map(
    {"Normal": 1, "Watch": 2, "Warning": 3, "Critical": 4}
)
history["Risk color"] = history["Observed risk"].map(risk_colors)

latest_flow = float(history[singati_column].iloc[-1])
latest_risk = history["Observed risk"].iloc[-1]
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Latest Singati discharge", f"{latest_flow:.2f} m³/s")
summary_col2.metric("Latest observed risk tier", latest_risk)
summary_col3.metric("Days available", f"{len(history)}")
summary_col4.metric("Model danger threshold", f"{danger_threshold:.2f} m³/s")

fig = go.Figure()
max_flow = max(float(history[singati_column].max()), danger_threshold)
chart_top = max_flow * 1.08 if max_flow > 0 else danger_threshold + 1
fig.add_hrect(
    y0=0,
    y1=watch_threshold,
    fillcolor="rgba(16, 185, 129, 0.08)",
    line_width=0,
    layer="below",
)
fig.add_hrect(
    y0=watch_threshold,
    y1=warning_threshold,
    fillcolor="rgba(245, 158, 11, 0.10)",
    line_width=0,
    layer="below",
)
fig.add_hrect(
    y0=warning_threshold,
    y1=danger_threshold,
    fillcolor="rgba(249, 115, 22, 0.10)",
    line_width=0,
    layer="below",
)
fig.add_hrect(
    y0=danger_threshold,
    y1=chart_top,
    fillcolor="rgba(239, 68, 68, 0.10)",
    line_width=0,
    layer="below",
)

for station, column in station_columns.items():
    values = pd.to_numeric(history[column], errors="coerce")
    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=values,
            mode="lines",
            name=station,
            line={
                "color": "#38BDF8" if station == "Singati" else "#64748B",
                "width": 3 if station == "Singati" else 1.5,
            },
            connectgaps=False,
        )
    )

fig.add_trace(
    go.Scatter(
        x=history["Date"],
        y=history["Risk tier"],
        mode="lines+markers",
        name="Singati observed risk",
        yaxis="y2",
        line={"shape": "hv", "color": "#F8FAFC", "width": 1.5},
        marker={"color": history["Risk color"], "size": 7},
    )
)
for threshold, label, color in (
    (watch_threshold, "Watch", "#F59E0B"),
    (warning_threshold, "Warning", "#F97316"),
    (danger_threshold, "Danger", "#EF4444"),
):
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color=color,
        annotation_text=f"{label}: {threshold:.2f} m³/s",
    )

fig.update_layout(
    title="Tamakoshi daily discharge and observed Singati risk tier",
    template="plotly_dark",
    paper_bgcolor="#131D31",
    plot_bgcolor="#131D31",
    xaxis_title="Date",
    yaxis_title="River discharge (m³/s)",
    yaxis={"rangemode": "tozero", "range": [0, chart_top]},
    yaxis2={
        "title": "Observed risk",
        "overlaying": "y",
        "side": "right",
        "range": [0.5, 4.5],
        "tickmode": "array",
        "tickvals": [1, 2, 3, 4],
        "ticktext": ["Normal", "Watch", "Warning", "Critical"],
        "showgrid": False,
    },
    legend={"orientation": "h", "y": -0.2},
    margin={"l": 20, "r": 20, "t": 60, "b": 75},
    height=560,
    hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")

st.caption(
    "Historical risk is assigned from observed Singati discharge against the "
    "model's watch, warning, and danger thresholds. It is not a retrospective "
    "model probability. Check the data-source status above before interpreting "
    "the selected time window."
)

rainfall_column = next(
    (column for column in history.columns if column.startswith("Rain (")),
    None,
)
if rainfall_column:
    history[rainfall_column] = pd.to_numeric(history[rainfall_column], errors="coerce")
    rainfall_col, table_col = st.columns([1, 1])
    with rainfall_col:
        rain_fig = go.Figure(
            go.Bar(
                x=history["Date"],
                y=history[rainfall_column],
                marker_color="#38BDF8",
                name="Daily rainfall",
            )
        )
        rain_fig.update_layout(
            title="Daily basin rainfall",
            template="plotly_dark",
            paper_bgcolor="#131D31",
            plot_bgcolor="#131D31",
            xaxis_title="Date",
            yaxis_title="Rainfall (mm)",
            height=330,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
        )
        st.plotly_chart(rain_fig, width="stretch")
    with table_col:
        st.subheader("Recent telemetry")
        st.dataframe(
            history.drop(columns=["Risk tier", "Risk color"], errors="ignore")
            .sort_values("Date", ascending=False)
            .head(20),
            width="stretch",
            hide_index=True,
        )
else:
    st.subheader("Recent telemetry")
    st.dataframe(
        history.drop(columns=["Risk tier", "Risk color"], errors="ignore")
        .sort_values("Date", ascending=False)
        .head(20),
        width="stretch",
        hide_index=True,
    )

st.divider()
st.subheader("Send announcement")
st.caption("Create a demo announcement and preview it in this page's local feed.")

announcement_styles = {
    "Flood warning": ("CRITICAL ALERT", "#EF4444"),
    "High-flow advisory": ("RIVER ADVISORY", "#F97316"),
    "Safety notice": ("SAFETY NOTICE", "#38BDF8"),
    "All clear": ("ALL CLEAR", "#10B981"),
}

with st.form("tamakoshi_announcement_form"):
    alert_type = st.selectbox("Alert type", list(announcement_styles))
    announcement_message = st.text_area(
        "Announcement message",
        placeholder="Write the message to show in the announcement feed...",
        max_chars=500,
    )
    announcement_submitted = st.form_submit_button(
        "Send announcement",
        type="primary",
    )

if announcement_submitted:
    if not announcement_message.strip():
        st.error("Enter an announcement message before sending.")
    else:
        nepal_timezone = timezone(timedelta(hours=5, minutes=45))
        announcement = {
            "alert_type": alert_type,
            "message": announcement_message.strip(),
            "sent_at": datetime.now(nepal_timezone).strftime("%d %b %Y, %I:%M %p NPT"),
        }
        announcements = st.session_state.setdefault("tamakoshi_announcements", [])
        st.session_state["tamakoshi_announcements"] = [announcement] + announcements[:4]
        st.success("Demo announcement published to the in-app feed.")

st.caption(
    "Demo preview only: announcements are visible in this browser session and "
    "are not delivered to external recipients."
)

announcements = st.session_state.setdefault("tamakoshi_announcements", [])
if announcements:
    st.markdown("#### Recent announcements")
    for item in announcements:
        label, color = announcement_styles[item["alert_type"]]
        safe_label = html.escape(label)
        safe_alert_type = html.escape(item["alert_type"])
        safe_message = html.escape(item["message"]).replace("\n", "<br>")
        safe_sent_at = html.escape(item["sent_at"])
        st.markdown(
            f"""
            <article style="border:1px solid #34445A;border-left:5px solid {color};
                            border-radius:10px;background:#142235;padding:16px 18px;
                            margin:10px 0 14px;">
              <div style="display:flex;justify-content:space-between;gap:12px;
                          flex-wrap:wrap;align-items:center;">
                <strong style="color:{color};letter-spacing:.08em;font-size:.78rem;">
                  {safe_label}
                </strong>
                <span style="color:#BDCADB;font-size:.78rem;">{safe_sent_at}</span>
              </div>
              <div style="margin-top:7px;color:#F3F6FB;font-weight:650;">
                {safe_alert_type}
              </div>
              <p style="margin:8px 0 0;color:#E2E8F0;line-height:1.55;
                        white-space:normal;">{safe_message}</p>
              <div style="margin-top:10px;color:#94A3B8;font-size:.74rem;">
                DEMO · In-app preview
              </div>
            </article>
            """,
            unsafe_allow_html=True,
        )
