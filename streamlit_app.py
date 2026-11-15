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
import streamlit as st

st.set_page_config(page_title="Tamakoshi Flood Early Warning AI", page_icon="🏔️", layout="wide")


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

st.title("🏔️ Tamakoshi Basin Flood & Glacial Melt Early Warning AI")
st.markdown(
    "**Watershed:** Tsho Rolpa Glacial Lake → Gongar → Singati → Lower Tamakoshi (Dolakha, Nepal)  \n"
    "**Architecture:** 6-Model Stacking Meta-Ensemble (`XGBoost` + `LightGBM` + `CatBoost` + `HistGradientBoosting` + `ExtraTrees` + `Deep Residual MLP`) trained on 42,540 station-days"
)

SCENARIOS = {
    "Custom Input": None,
    "🟢 Scenario 1: Dry Winter Baseflow (Normal)": (0.35, 0.36, 0.05, 0.22, 0.55, 0.0, 0.0, 0.5, 2.0, 0, 22, 14.0, 2.5, 0.0, 14.5, 2),
    "🟡 Scenario 2: Rising Monsoon Surge (Yellow Watch)": (5.85, 5.20, 0.85, 3.70, 8.40, 28.0, 20.0, 68.0, 135.0, 7, 41, 22.0, 14.0, 0.0, 19.5, 7),
    "🔴 Scenario 3: Cloudburst + Glacial Melt (Red Alert)": (6.60, 5.55, 1.15, 4.35, 9.80, 58.0, 42.0, 135.0, 230.0, 3, 48, 24.0, 15.5, 0.0, 22.5, 8),
}

choice = st.selectbox("Quick-Load a Calibrated Tamakoshi Watershed Scenario:", list(SCENARIOS.keys()), index=3)
defaults = SCENARIOS[choice] or (6.60, 5.55, 1.15, 4.35, 9.80, 58.0, 42.0, 135.0, 230.0, 3, 48, 24.0, 15.5, 0.0, 22.5, 8)

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Multi-Station River Discharge (m³/s)")
    q_mid = st.number_input("Singati Target Discharge Today (m³/s)", value=float(defaults[0]), step=0.1)
    q_yesterday = st.number_input("Singati Discharge Yesterday (m³/s)", value=float(defaults[1]), step=0.1)
    q_glacier = st.number_input("Tsho Rolpa Glacier Outflow (m³/s)", value=float(defaults[2]), step=0.05)
    q_upper = st.number_input("Upper Tamakoshi / Gongar Flow (m³/s)", value=float(defaults[3]), step=0.1)
    q_lower = st.number_input("Lower Tamakoshi Confluence Flow (m³/s)", value=float(defaults[4]), step=0.1)

with col2:
    st.subheader("2. Cloudburst, Soil Saturation & Glacial Melt")
    rain_mid = st.number_input("Valley Rainfall Today (mm)", value=float(defaults[5]), step=1.0)
    rain_glacier = st.number_input("Glacier Headwaters Rain Today (mm)", value=float(defaults[6]), step=1.0)
    rain_3d = st.number_input("3-Day Basin Rain (mm)", value=float(defaults[7]), step=5.0)
    rain_7d = st.number_input("7-Day Basin Rain (mm)", value=float(defaults[8]), step=5.0)
    rain_hours = st.slider("Rain Duration Today (Hours — lower = cloudburst)", 0, 24, int(defaults[9]))
    soil_sat_pct = st.slider("Deep Soil Moisture Saturation (%)", 10, 50, int(defaults[10]))
    tmax = st.number_input("Valley Max Temp (°C)", value=float(defaults[11]), step=0.5)
    tmin = st.number_input("Valley Min Temp (°C)", value=float(defaults[12]), step=0.5)
    snow_glacier = st.number_input("Glacier Snowfall Today (cm)", value=float(defaults[13]), step=0.5)
    solar_rad = st.number_input("Solar Radiation (MJ/m²)", value=float(defaults[14]), step=0.5)
    month = st.slider("Month of Year (1=Jan, 8=Aug)", 1, 12, int(defaults[15]))

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

base_deltas = np.array([[base_models[name].predict(row)[0] for name in model_names]])
delta_pred = float(meta_learner.predict(base_deltas)[0])
pred_flow = max(0.0, q_mid + delta_pred)
low_flow = max(0.0, q_mid + float(q_low_model.predict(row)[0]))
high_flow = max(pred_flow, q_mid + float(q_high_model.predict(row)[0]))
flood_prob = float(flood_clf.predict_proba(row)[0, 1]) * 100.0
yellow_trigger = DANGER_THRESHOLD * 0.90

st.divider()
st.subheader("3. 24-Hour Forecast & Emergency Decision Support")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Tomorrow Flow", f"{pred_flow:.2f} m³/s", f"{delta_pred:+.2f} m³/s surge")
m2.metric("80% Confidence Interval", f"{low_flow:.2f} – {high_flow:.2f} m³/s")
m3.metric("Calibrated Danger Line", f"{DANGER_THRESHOLD:.2f} m³/s")
m4.metric("AI Flood Probability", f"{flood_prob:.1f}%")

if pred_flow >= DANGER_THRESHOLD or flood_prob >= 60.0:
    st.error(
        "🔴 **RED ALERT — EXTREME FLOOD RISK:** Predicted discharge exceeds the 95th-percentile danger threshold. "
        "Trigger SMS evacuation warnings for Singati, Gongar, and Jagat riverside settlements and open Upper Tamakoshi dam spillway gates."
    )
elif pred_flow >= yellow_trigger or flood_prob >= 35.0:
    st.warning(
        "🟡 **YELLOW WATCH — RAPID SURGE DETECTED:** Flow enters the 10% safety buffer zone. "
        "Alert Dolakha District Disaster Management Committee and place rescue teams on standby."
    )
else:
    st.success(
        "🟢 **GREEN NORMAL — SAFE OPERATING RANGE:** River discharge remains safely below flood warning thresholds."
    )
