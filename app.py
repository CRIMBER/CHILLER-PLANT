"""
app.py
------
Chiller Plant AI — Industrial Monitoring Dashboard

UI/UX layer only. All predictions, metrics, and thresholds come from the
existing backend in train.py (models, feature engineering, cooling-load
estimation, plant-intelligence logic) — nothing here retrains or alters
any model.

There is no live BMS/SCADA connection in this prototype. Every value
shown, in every mode, comes from the real historical dataset.

Run:
    streamlit run app.py
"""

import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from train import (
    CHR_MODEL_PATH,
    LOAD_MODEL_PATH,
    LOAD_TARGET_COL,
    METRICS_PATH,
    TARGET_COL,
    TIMESTAMP_COL,
    chronological_split,
    clean_data,
    create_future_target,
    engineer_features,
    load_data,
    plant_intelligence_status,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG + THEME
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chiller Plant AI", layout="wide", initial_sidebar_state="expanded")

BG = "#0d1117"
PANEL = "#151a21"
CARD = "#181d25"
BORDER = "#2a313c"
TEXT = "#dbe1e8"
SUBTEXT = "#8b95a1"
ACCENT = "#5b9bd5"
GREEN = "#4caf6d"
AMBER = "#d9a441"
RED = "#d9534f"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {BORDER}; }}
    h1, h2, h3, h4, h5 {{ color: {TEXT} !important; font-weight: 600 !important; letter-spacing: 0.3px; }}
    p, span, label, div {{ color: {TEXT}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {SUBTEXT} !important; }}
    div[data-testid="stMetricValue"] {{ color: {TEXT}; }}
    div[data-testid="stMetric"] {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;
        padding: 10px 14px;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; color: {SUBTEXT}; border-radius: 4px 4px 0 0;
        font-size: 13px; letter-spacing: 0.5px; text-transform: uppercase; padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT}; }}
    div[data-testid="stExpander"] {{ background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 6px; }}
    .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 6px; }}
    hr {{ border-color: {BORDER}; }}
    .block-container {{ padding-top: 1.5rem; }}
    .sys-header {{
        font-family: 'Courier New', monospace; font-size: 12px; color: {SUBTEXT};
        letter-spacing: 1px; padding: 6px 0 14px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 18px;
    }}
    .sys-header b {{ color: {TEXT}; }}
    .section-label {{
        font-size: 12px; letter-spacing: 2px; color: {SUBTEXT}; text-transform: uppercase;
        margin: 22px 0 8px 0; font-weight: 600;
    }}
    .ind-card {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;
        padding: 14px 16px; margin-bottom: 8px;
    }}
    .ind-card .label {{ font-size: 11px; letter-spacing: 1px; color: {SUBTEXT}; text-transform: uppercase; }}
    .ind-card .value {{ font-size: 28px; font-weight: 600; color: {TEXT}; margin-top: 2px; }}
    .ind-card .unit {{ font-size: 13px; color: {SUBTEXT}; margin-left: 4px; }}
    .forecast-card {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px;
    }}
    .forecast-card .title {{ font-size: 12px; letter-spacing: 1.5px; color: {SUBTEXT}; text-transform: uppercase; margin-bottom: 12px; }}
    .forecast-row {{ display: flex; align-items: baseline; gap: 14px; }}
    .forecast-current {{ font-size: 22px; color: {SUBTEXT}; }}
    .forecast-arrow {{ font-size: 18px; color: {SUBTEXT}; }}
    .forecast-future {{ font-size: 34px; font-weight: 700; color: {TEXT}; }}
    .forecast-change {{ font-size: 13px; margin-top: 8px; }}
    .chip {{
        display: inline-block; padding: 6px 12px; margin: 3px; border-radius: 4px;
        font-size: 13px; border: 1px solid {BORDER}; background-color: {CARD};
    }}
    .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
    .advisory-banner {{
        border-radius: 8px; padding: 18px 22px; border: 1px solid {BORDER};
    }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT, "axes.grid": True,
    "grid.color": "#20262e", "grid.linewidth": 0.6,
    "xtick.color": SUBTEXT, "ytick.color": SUBTEXT, "text.color": TEXT,
    "legend.facecolor": CARD, "legend.edgecolor": BORDER, "legend.labelcolor": TEXT,
    "font.size": 10,
})
LINE_ACTUAL = "#5b9bd5"
LINE_PRED = "#d9a441"
LINE_ERROR = "#d9534f"

# ---------------------------------------------------------------------------
# STATUS COLOR MAP
# ---------------------------------------------------------------------------
STATUS_COLORS = {
    "STABLE": GREEN,
    "INCREASING DEMAND": AMBER,
    "DEMAND RISING": AMBER,
    "HIGH COOLING DEMAND": RED,
    "HIGH DEMAND": RED,
    "EFFICIENCY WARNING": AMBER,
}
STATUS_EXPLANATION = {
    "STABLE": "Predicted CHR and cooling load are within normal short-term variation.",
    "INCREASING DEMAND": "Predicted CHR and/or cooling load are trending upward beyond normal variation.",
    "HIGH COOLING DEMAND": "Predicted cooling load increase exceeds the high-demand threshold for this plant.",
    "EFFICIENCY WARNING": "Current plant kW/RT is notably above the historical norm for this chiller configuration.",
}


# ---------------------------------------------------------------------------
# HELPERS — REUSED / WRAPPED BACKEND CALLS
# ---------------------------------------------------------------------------
@st.cache_resource
def try_load_artifact(path):
    if not os.path.exists(path):
        return None, "missing"
    try:
        return joblib.load(path), "ok"
    except Exception:
        return None, "corrupt"


@st.cache_data
def try_load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def try_load_and_prepare_data(_chr_features, _load_features):
    raw_df, _ = load_data()
    df = clean_data(raw_df, verbose=False)
    df = engineer_features(df)
    needed = list(set(_chr_features + _load_features + [TARGET_COL, LOAD_TARGET_COL]))
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


CHILLER_POWER_COLS = {
    1: ("DPM-CH-1-CP-1-kW", "DPM-CH-1-CP-2-kW"),
    2: ("DPM-CH-2-CP-1-kW", "DPM-CH-2-CP-2-kW"),
    3: ("DPM-CH-3-CP-1-kW", "DPM-CH-3-CP-2-kW"),
    4: ("DPM_CH-4-CP-1-kW", "DPM-CH-4-CP-2-kW"),
    5: ("DPM-CH-5-CP-1-kW", "DPM-CH-5-CP-2-kW"),
}
CHILLER_RUNNING_THRESHOLD_KW = 5.0  # simple, documented, display-only threshold


def get_chiller_statuses(row):
    statuses = []
    for n, (c1, c2) in CHILLER_POWER_COLS.items():
        if c1 not in row.index or c2 not in row.index:
            continue
        power = float(row[c1]) + float(row[c2])
        running = power > CHILLER_RUNNING_THRESHOLD_KW
        statuses.append({"chiller": n, "power_kw": power, "running": running})
    return statuses


def metric_card(label, value, unit="", col=None):
    html = f"""<div class="ind-card"><div class="label">{label}</div>
    <div class="value">{value}<span class="unit">{unit}</span></div></div>"""
    target = col if col is not None else st
    target.markdown(html, unsafe_allow_html=True)


def slice_last_hours(df, hours, ts_col=TIMESTAMP_COL):
    if len(df) == 0:
        return df
    cutoff = df[ts_col].max() - pd.Timedelta(hours=hours)
    return df[df[ts_col] >= cutoff]


def slice_arrays_last_hours(timestamps, arrays, hours):
    ts = pd.to_datetime(pd.Series(timestamps))
    if len(ts) == 0:
        return timestamps, arrays
    cutoff = ts.max() - pd.Timedelta(hours=hours)
    mask = (ts >= cutoff).values
    return timestamps[mask], [a[mask] for a in arrays]


# ---------------------------------------------------------------------------
# DATASET / MODEL AVAILABILITY CHECKS (no tracebacks shown to user)
# ---------------------------------------------------------------------------
dataset_missing = not (os.path.exists("data/dataset.csv") or os.path.exists("data/dataset.xlsx"))
chr_artifact, chr_load_status = try_load_artifact(CHR_MODEL_PATH)
load_artifact_obj, load_load_status = try_load_artifact(LOAD_MODEL_PATH)
models_ok = chr_load_status == "ok" and load_load_status == "ok"

st.markdown("# CHILLER PLANT AI")
st.markdown("##### Predictive Monitoring & Forecasting")

sys_status = "ONLINE" if (not dataset_missing and models_ok) else "DEGRADED"
model_status_txt = "TRAINED" if models_ok else ("NOT TRAINED" if not dataset_missing else "UNAVAILABLE")
st.markdown(
    f"""<div class="sys-header">
    SYSTEM STATUS: <b>{sys_status}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
    DATA SOURCE: <b>BMS / SCADA (historical dataset — no live connection)</b> &nbsp;&nbsp;|&nbsp;&nbsp;
    MODEL STATUS: <b>{model_status_txt}</b>
    </div>""",
    unsafe_allow_html=True,
)

if dataset_missing:
    st.error("**DATASET NOT FOUND**\n\nPlease place the dataset inside:\n\n`data/dataset.csv` or `data/dataset.xlsx`")
    st.stop()

if not models_ok:
    missing = []
    if chr_load_status != "ok":
        missing.append("CHR model (model.pkl)")
    if load_load_status != "ok":
        missing.append("Cooling Load model (load_model.pkl)")
    st.error("**MODEL NOT TRAINED**\n\nMissing or unreadable: " + ", ".join(missing) + "\n\nRun:\n\n`python train.py`")
    st.stop()

metrics_json = try_load_metrics()
df = try_load_and_prepare_data(chr_artifact["feature_cols"], load_artifact_obj["feature_cols"])
horizon = chr_artifact["horizon_minutes"]

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### CHILLER PLANT AI")
    st.caption("Dashboard · Forecasting · Plant Performance · Model Performance · About")
    st.markdown("---")

    mode = st.radio(
        "Data Mode",
        ["Latest Plant Record", "Demo Mode — Live Replay"],
        help="No live BMS connection exists in this prototype. Both modes "
             "use real historical data — 'Latest Plant Record' shows the "
             "most recent row in the dataset; 'Demo Mode' replays the "
             "held-out test period minute-by-minute.",
    )

    st.markdown("---")
    st.caption("FORECAST HORIZON")
    st.radio("Forecast Horizon", ["15 min"], index=0, label_visibility="collapsed", disabled=True)
    st.caption("Only 15-min models are deployed. 30/45/60-min results are evaluated under Model Performance.")

    st.markdown("---")
    st.caption("TREND TIME RANGE")
    time_range_label = st.radio("Trend Time Range", ["1 Hour", "6 Hours", "24 Hours"], index=2, label_visibility="collapsed")
    time_range_hours = {"1 Hour": 1, "6 Hours": 6, "24 Hours": 24}[time_range_label]

    if mode == "Demo Mode — Live Replay":
        st.markdown("---")
        st.caption("DEMO PLAYBACK")
        if "sim_idx" not in st.session_state:
            st.session_state.sim_idx = 0
        if "playing" not in st.session_state:
            st.session_state.playing = False
        step = st.slider("Step size (min/tick)", 1, 30, 5)
        playing = st.toggle("Play", value=st.session_state.playing)
        st.session_state.playing = playing
        if st.button("Reset"):
            st.session_state.sim_idx = 0
            st.session_state.playing = False
            st.rerun()

# ---------------------------------------------------------------------------
# RESOLVE CURRENT ROW (mode-dependent)
# ---------------------------------------------------------------------------
if mode == "Latest Plant Record":
    row = df.iloc[-1]
    is_demo = False
else:
    _, _, test_df = chronological_split(df)
    test_df = test_df.reset_index(drop=True)
    n_test = len(test_df)
    idx = min(st.session_state.get("sim_idx", 0), n_test - 1)
    row = test_df.iloc[idx]
    is_demo = True

# ---------------------------------------------------------------------------
# PREDICTIONS (single source of truth for this render)
# ---------------------------------------------------------------------------
current_chr = float(row[TARGET_COL])
X_chr = row[chr_artifact["feature_cols"]].to_frame().T
predicted_chr = float(chr_artifact["pipeline"].predict(X_chr)[0])
chr_change = predicted_chr - current_chr

current_load = float(row[LOAD_TARGET_COL])
X_load = row[load_artifact_obj["feature_cols"]].to_frame().T
predicted_load = float(load_artifact_obj["pipeline"].predict(X_load)[0])
load_change = predicted_load - current_load

intel = chr_artifact["plant_intelligence"]
n_chillers_now = str(int(row["Running_Chillers"]))
bucket = intel["config_bucket_stats"].get(n_chillers_now)
current_kwrt = row["Plant_kW_per_RT"]
status = plant_intelligence_status(
    chr_change, load_change,
    intel["load_rising_threshold_rt"], intel["load_high_threshold_rt"],
    current_kw_per_rt=current_kwrt if pd.notna(current_kwrt) else None,
    efficiency_mean=bucket["mean_kw_per_rt"] if bucket else None,
    efficiency_std=bucket["std_kw_per_rt"] if bucket else None,
)
display_status = "DEMAND RISING" if status == "INCREASING DEMAND" else ("HIGH DEMAND" if status == "HIGH COOLING DEMAND" else status)

if is_demo:
    st.info(f"DEMO MODE — Using historical plant data · point {idx + 1} of {n_test} · {row[TIMESTAMP_COL]}")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_dash, tab_forecast, tab_plant, tab_model, tab_about = st.tabs(
    ["Dashboard", "Forecasting", "Plant Performance", "Model Performance", "About"]
)

# ============================================================ DASHBOARD ===
with tab_dash:
    st.markdown('<div class="section-label">Current Plant Status</div>', unsafe_allow_html=True)
    st.caption(f"As of {row[TIMESTAMP_COL]}")
    cols = st.columns(6)
    metric_card("CHR", f"{current_chr:.2f}", " °C", cols[0])
    metric_card("CHWS", f"{row[chr_artifact['supply_col']]:.2f}", " °C", cols[1])
    metric_card("Cooling Load", f"{current_load:,.0f}", " RT", cols[2])
    metric_card("Plant Power", f"{row['Total_Plant_Power_kW']:,.0f}", " kW", cols[3])
    metric_card("Running Chillers", f"{int(row['Running_Chillers'])}", "", cols[4])
    kwrt_val = f"{current_kwrt:.3f}" if pd.notna(current_kwrt) else "N/A"
    metric_card("Plant Efficiency", kwrt_val, " kW/RT" if pd.notna(current_kwrt) else "", cols[5])

    st.markdown('<div class="section-label">AI Forecast</div>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        chg_color = RED if chr_change > 0 else GREEN
        st.markdown(f"""
        <div class="forecast-card">
            <div class="title">CHR Forecast</div>
            <div class="forecast-row">
                <div class="forecast-current">{current_chr:.2f} °C</div>
                <div class="forecast-arrow">&rarr;</div>
                <div class="forecast-future">{predicted_chr:.2f} °C</div>
            </div>
            <div class="forecast-change" style="color:{chg_color}">Change: {chr_change:+.2f} °C over {horizon} min</div>
        </div>""", unsafe_allow_html=True)
    with fc2:
        chg_color = RED if load_change > 0 else GREEN
        st.markdown(f"""
        <div class="forecast-card">
            <div class="title">Cooling Load Forecast</div>
            <div class="forecast-row">
                <div class="forecast-current">{current_load:,.0f} RT</div>
                <div class="forecast-arrow">&rarr;</div>
                <div class="forecast-future">{predicted_load:,.0f} RT</div>
            </div>
            <div class="forecast-change" style="color:{chg_color}">Change: {load_change:+.0f} RT over {horizon} min</div>
        </div>""", unsafe_allow_html=True)
    st.caption("Cooling load is ESTIMATED (flow x ΔT), not a calibrated meter reading — see Model Performance tab.")

    st.markdown('<div class="section-label">AI Advisory Status</div>', unsafe_allow_html=True)
    color = STATUS_COLORS.get(status, SUBTEXT)
    st.markdown(f"""
    <div class="advisory-banner" style="border-left: 4px solid {color};">
        <span class="dot" style="background-color:{color};"></span>
        <span style="font-size:20px; font-weight:700; letter-spacing:1px;">{display_status}</span>
        <div style="color:{SUBTEXT}; margin-top:6px; font-size:13px;">{STATUS_EXPLANATION.get(status, "")}</div>
        <div style="color:{SUBTEXT}; margin-top:8px; font-size:11px;">This is an advisory classification, not an MPC recommendation. No commands are sent to any equipment.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Chiller Status</div>', unsafe_allow_html=True)
    chiller_statuses = get_chiller_statuses(row)
    if chiller_statuses:
        chips_html = ""
        for cs in chiller_statuses:
            dot_color = GREEN if cs["running"] else SUBTEXT
            label = "RUNNING" if cs["running"] else "OFF"
            chips_html += f"""<span class="chip"><span class="dot" style="background-color:{dot_color};"></span>
            CHILLER {cs['chiller']:02d} &nbsp; {label}</span>"""
        st.markdown(chips_html, unsafe_allow_html=True)
        st.caption(f"Inferred from per-chiller compressor power (threshold: >{CHILLER_RUNNING_THRESHOLD_KW:.0f} kW = running). Display only — not used by the models.")
    else:
        st.caption("Per-chiller power columns not available in this dataset.")

# ============================================================ FORECASTING =
with tab_forecast:
    st.markdown('<div class="section-label">CHR Forecast — Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
    ts, (act, pred) = slice_arrays_last_hours(
        chr_artifact["test_timestamps"], [chr_artifact["test_actual"], chr_artifact["test_predicted"]], time_range_hours
    )
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(ts, act, label="Actual CHR", color=LINE_ACTUAL, linewidth=1.3)
    ax.plot(ts, pred, label="Predicted CHR", color=LINE_PRED, linewidth=1.3)
    ax.set_ylabel("°C")
    ax.legend(frameon=False)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-label">Cooling Load Forecast — Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
    ts2, (act2, pred2) = slice_arrays_last_hours(
        load_artifact_obj["test_timestamps"], [load_artifact_obj["test_actual"], load_artifact_obj["test_predicted"]], time_range_hours
    )
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(ts2, act2, label="Actual Cooling Load", color=LINE_ACTUAL, linewidth=1.3)
    ax.plot(ts2, pred2, label="Predicted Cooling Load", color=LINE_PRED, linewidth=1.3)
    ax.set_ylabel("RT")
    ax.legend(frameon=False)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"Showing last {time_range_label.lower()} of the held-out test period (change range in the sidebar). Cooling load is estimated — see Model Performance tab for methodology.")

# ======================================================= PLANT PERFORMANCE=
with tab_plant:
    st.markdown('<div class="section-label">Energy & Efficiency</div>', unsafe_allow_html=True)
    ecols = st.columns(5)
    metric_card("Total Chiller Power", f"{row['Total_Chiller_Power_kW']:,.0f}", " kW", ecols[0])
    metric_card("Total Pump Power", f"{row['Total_CHWP_Power_kW'] + row['Total_CWP_Power_kW']:,.0f}", " kW", ecols[1])
    metric_card("Cooling Tower Power", f"{row['Total_CT_Power_kW']:,.0f}", " kW", ecols[2])
    metric_card("Total Plant Power", f"{row['Total_Plant_Power_kW']:,.0f}", " kW", ecols[3])
    metric_card("Plant kW/RT", kwrt_val, " kW/RT" if pd.notna(current_kwrt) else "", ecols[4])
    st.caption("Chiller-level kW/RT is not shown: no per-chiller cooling-load metering exists in this dataset.")

    st.markdown('<div class="section-label">Plant Power Trend</div>', unsafe_allow_html=True)
    trend_df = slice_last_hours(df, time_range_hours)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(trend_df[TIMESTAMP_COL], trend_df["Total_Plant_Power_kW"], color=LINE_ACTUAL, linewidth=1.2)
    ax.set_ylabel("kW")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-label">Thermal Conditions</div>', unsafe_allow_html=True)
    cwst_cols = [c for c in df.columns if c.endswith("_CWST")]
    cwrt_cols = [c for c in df.columns if c.endswith("_CWRT")]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(trend_df[TIMESTAMP_COL], trend_df[chr_artifact["supply_col"]], label="CHWS", linewidth=1.1)
    ax.plot(trend_df[TIMESTAMP_COL], trend_df[TARGET_COL], label="CHR", linewidth=1.1)
    ax.plot(trend_df[TIMESTAMP_COL], trend_df["WetBulb_Avg"], label="Wet Bulb", linewidth=1.1, linestyle="--")
    if cwst_cols and cwrt_cols:
        ax.plot(trend_df[TIMESTAMP_COL], trend_df[cwst_cols].mean(axis=1), label="CWST (avg)", linewidth=1.0, alpha=0.8)
        ax.plot(trend_df[TIMESTAMP_COL], trend_df[cwrt_cols].mean(axis=1), label="CWRT (avg)", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("°C")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-label">Plant Timeline (Normalized Comparison)</div>', unsafe_allow_html=True)
    timeline_options = {
        "CHR": TARGET_COL, "CHWS": chr_artifact["supply_col"], "Cooling Load": LOAD_TARGET_COL,
        "Plant Power": "Total_Plant_Power_kW", "Wet Bulb": "WetBulb_Avg", "CHW ΔT": "CHW_DeltaT",
    }
    chosen = st.multiselect("Variables", list(timeline_options.keys()), default=["CHR", "Cooling Load"])
    if chosen:
        fig, ax = plt.subplots(figsize=(13, 4))
        for name in chosen:
            col = timeline_options[name]
            series = trend_df[col]
            rng = series.max() - series.min()
            norm = (series - series.min()) / rng if rng > 0 else series * 0
            ax.plot(trend_df[TIMESTAMP_COL], norm, label=name, linewidth=1.1)
        ax.set_ylabel("Normalized (0-1)")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        fig.autofmt_xdate(rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Each variable is min-max normalized over the selected time range for shape comparison — see cards above for actual values/units.")

    st.markdown('<div class="section-label">Historical Chiller-Configuration Comparison</div>', unsafe_allow_html=True)
    st.caption("Historical operating analysis only — not an optimization recommendation.")
    if metrics_json and metrics_json.get("chiller_config_analysis", {}).get("table"):
        cfg = metrics_json["chiller_config_analysis"]
        cfg_df = pd.DataFrame(cfg["table"]).rename(columns={
            "running_chillers": "Chillers Running", "n_samples": "Samples",
            "avg_cooling_load_rt": "Avg Load (RT)", "avg_plant_power_kw": "Avg Power (kW)",
            "avg_kw_per_rt": "Avg kW/RT", "std_kw_per_rt": "Std kW/RT",
        })
        st.dataframe(cfg_df.round(1), hide_index=True, width='stretch')
        if cfg.get("skipped_configs"):
            st.caption(f"Skipped {cfg['skipped_configs']} chiller-count(s): {cfg['skip_reason']}")
    else:
        st.info("No reliable historical configuration comparison available.")

# ========================================================= MODEL PERFORMANCE
with tab_model:
    st.markdown('<div class="section-label">CHR Model</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metric_card("MAE", f"{chr_artifact['metrics']['MAE']:.4f}", " °C", c1)
    metric_card("RMSE", f"{chr_artifact['metrics']['RMSE']:.4f}", " °C", c2)
    metric_card("R²", f"{chr_artifact['metrics']['R2']:.4f}", "", c3)
    metric_card("Polynomial Degree", f"{chr_artifact['poly_degree']}", "", c4)
    st.caption(f"Training period: rows 1–{chr_artifact['n_train']} · Testing period: last {chr_artifact['n_test']} rows (chronological, held out)")

    st.markdown('<div class="section-label">Cooling Load Model</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metric_card("MAE", f"{load_artifact_obj['metrics']['MAE']:.2f}", " RT", c1)
    metric_card("RMSE", f"{load_artifact_obj['metrics']['RMSE']:.2f}", " RT", c2)
    metric_card("R²", f"{load_artifact_obj['metrics']['R2']:.4f}", "", c3)
    metric_card("Polynomial Degree", f"{load_artifact_obj['poly_degree']}", "", c4)
    st.caption(f"Training period: rows 1–{load_artifact_obj['n_train']} · Testing period: last {load_artifact_obj['n_test']} rows (chronological, held out)")

    st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
    rows = [
        {"Model": "CHR — Persistence Baseline", **chr_artifact["baseline_metrics"]},
        {"Model": f"CHR — Polynomial Regression (deg {chr_artifact['poly_degree']})", **chr_artifact["metrics"]},
        {"Model": "Load — Persistence Baseline", **load_artifact_obj["baseline_metrics"]},
        {"Model": f"Load — Polynomial Regression (deg {load_artifact_obj['poly_degree']})", **load_artifact_obj["metrics"]},
    ]
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width='stretch')

    if metrics_json:
        with st.expander("Multi-horizon results — CHR (15/30/45/60 min)"):
            hrows = [{"Horizon (min)": h, "Baseline RMSE": round(v["baseline"]["RMSE"], 4),
                      "PolyReg RMSE": round(v["polyreg"]["RMSE"], 4), "PolyReg R2": round(v["polyreg"]["R2"], 4)}
                     for h, v in metrics_json["chr_multi_horizon_results"].items()]
            st.dataframe(pd.DataFrame(hrows), hide_index=True, width='stretch')

        with st.expander("Multi-horizon results — Cooling Load (15/30/45/60 min)"):
            hrows = [{"Horizon (min)": h, "Baseline RMSE": round(v["baseline"]["RMSE"], 2),
                      "PolyReg RMSE": round(v["polyreg"]["RMSE"], 2), "PolyReg R2": round(v["polyreg"]["R2"], 4)}
                     for h, v in metrics_json["load_multi_horizon_results"].items()]
            st.dataframe(pd.DataFrame(hrows), hide_index=True, width='stretch')

        with st.expander("Model interpretability — top terms (both models)"):
            st.caption("Linear-model terms on scaled, polynomial-expanded features — not physical causal effects.")
            ic1, ic2 = st.columns(2)
            ic1.markdown("**CHR**")
            ic1.dataframe(pd.DataFrame(chr_artifact["top_terms"]), hide_index=True, width='stretch')
            ic2.markdown("**Cooling Load**")
            ic2.dataframe(pd.DataFrame(load_artifact_obj["top_terms"]), hide_index=True, width='stretch')

        with st.expander("Cooling Load Methodology"):
            m = metrics_json["cooling_load_methodology"]
            st.markdown(f"**Formula:** `{m['formula']}`")
            st.markdown(f"**Risers summed:** {', '.join(m['risers_used'])}")
            st.markdown(f"**Unit assumption:** {m['unit_assumption']}")
            st.markdown(f"**Sanity check:** mean implied plant kW/RT = {m['sanity_check_kw_per_rt_mean']:.3f} "
                         "(plausible for a real chiller plant, supporting the unit assumption — still an ESTIMATE, not a calibrated meter reading).")

# =================================================================== ABOUT
with tab_about:
    st.markdown('<div class="section-label">About This Prototype</div>', unsafe_allow_html=True)
    st.markdown(
        "An AI-based predictive monitoring prototype for chilled-water plant operation. "
        "The system uses Polynomial Regression to forecast future chilled-water return "
        "temperature and cooling demand from historical BMS/SCADA data. These forecasts "
        "provide the foundation for a future Model Predictive Control layer for "
        "energy-efficient chiller optimization — **MPC itself is not implemented here.**"
    )
    st.markdown("""
- No live BMS/SCADA connection — all data is historical.
- Cooling load is a physical **estimate** (flow × ΔT), not a calibrated meter reading.
- No chiller-level kW/RT — no per-chiller cooling-load metering exists in this dataset.
- The chiller-configuration comparison is descriptive historical analysis, not optimization.
- No energy-savings figures are claimed anywhere in this dashboard.
""")

    st.markdown('<div class="section-label">Next Generation: MPC Optimization</div>', unsafe_allow_html=True)
    st.markdown(f"""
```
CURRENT PLANT
     |
AI FORECAST            <- YOU ARE HERE (CHR + Cooling Load)
     |
MPC OPTIMIZATION        <- FUTURE DEVELOPMENT, NOT BUILT
     |
OPTIMAL CHILLER OPERATION
     |
ENERGY REDUCTION
```
""")
    st.caption("This section is purely explanatory. The current prototype does not perform MPC, "
               "does not optimize equipment, and sends no commands to any controller.")

# ---------------------------------------------------------------------------
# DEMO AUTOPLAY (runs after all tabs are rendered, regardless of active tab)
# ---------------------------------------------------------------------------
if is_demo and st.session_state.get("playing"):
    if idx >= n_test - 1:
        st.session_state.playing = False
        st.warning("Reached the end of the demo period. Press Reset in the sidebar to replay.")
    else:
        time.sleep(1.0)
        st.session_state.sim_idx = idx + step
        st.rerun()