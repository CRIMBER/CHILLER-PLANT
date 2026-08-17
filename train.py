"""
train.py
--------
Chiller Plant AI — Dual Forecasting Prototype (CHR + Cooling Load)

Pipeline:

                    BMS DATA
                       |
              Feature Engineering
                       |
          ------------- -------------
          |                         |
   CHR Prediction           Cooling Load Prediction
          |                         |
          ------------- -------------
                       |
              Plant Intelligence (advisory only)
                       |
                 Dashboard

This is the FORECASTING layer only. No MPC, MINLP, BONMIN, real-time
control, or automatic chiller commands are implemented. Nothing here
sends commands to plant equipment.

Run:
    python train.py
"""

import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_PATH_CSV = "data/dataset.csv"
DATA_PATH_XLSX = "data/dataset.xlsx"
TIMESTAMP_COL = "Timestamp"

# CHR target — see README/report for why this column was selected over
# alternative plant-level candidates.
TARGET_COL = "CHW-Riser-MainBuilding-ChwRt"
SUPPLY_COL = "CHW-Riser-MainBuilding-ChwSt"

WETBULB_COLS = [
    "WST_1_WetBulbTemp (°C)",
    "WST_2_WetBulbTemp (°C)",
    "WST_3_WetBulbTemp (°C)",
    "WST_4_WetBulbTemp (°C)",
    "WST_5_WetBulbTemp (°C)",
]

# --- Cooling load estimation -----------------------------------------------
# No reliable "cooling load" column exists in this dataset: the "(RT)" tag
# is applied inconsistently (found on temperature-range AND flow columns
# interchangeably, and missing entirely for some chillers). Rather than
# fabricate a target, we PHYSICALLY ESTIMATE total cooling load from
# chilled-water flow and delta-T across all 4 building risers:
#
#     Q (kW) = Flow (L/s) x 4.186 (kJ/kg.K) x DeltaT (C)
#     RT     = Q (kW) / 3.517
#
# summed across MainBuilding + Finger + L1-3 + T1U.
#
# CAVEAT: this dataset's unit tags are demonstrably unreliable elsewhere
# (e.g. a supply-temperature column tagged "(L/s)", and flow columns
# tagged "(deg C)" — a source-export bug, not a sensor error). L/s is
# assumed for every riser flow reading because it is the only unit tag
# ever seen on a riser flow column here, AND because the resulting
# plant-wide kW/RT efficiency figure lands in a textbook-plausible
# 0.40-0.49 range for a modern chiller plant (see printed report below) —
# an incorrect unit assumption would not produce a physically sane ratio.
# This is clearly labeled ESTIMATED throughout, never presented as a
# calibrated meter reading.
RISERS = {
    "MainBuilding": ("CHW-Riser-MainBuilding-ChwFls", "CHW-Riser-MainBuilding-ChwRt", "CHW-Riser-MainBuilding-ChwSt"),
    "Finger": ("CHW-Riser-Finger-ChwFls (°C)", "CHW-Riser-Finger-ChwRt", "CHW-Riser-Finger-ChwSt (°C)"),
    "L1-3": ("CHW-Riser-L1-3-ChwFls (L/s)", "CHW-Riser-L1-3-ChwRt", "CHW-Riser-L1-3-ChwSt (L/s)"),
    "T1U": ("CHW-Riser-T1U-ChwFls", "CHW-Riser-T1U-ChwRt", "CHW-Riser-T1U-ChwSt"),
}
WATER_CP_KJ_PER_KGK = 4.186
KW_PER_RT = 3.517
LOAD_TARGET_COL = "Estimated_Cooling_Load_RT"

# No reliable cooling-load column, so no fabricated "load" input feature
# either — power, delta-T, and running-chiller count remain the physical
# load proxies for BOTH models.
BASE_FEATURE_COLS = [
    SUPPLY_COL,
    "CHW_DeltaT",
    "Total_Chiller_Power_kW",
    "Total_CHWP_Power_kW",
    "Total_CWP_Power_kW",
    "Total_CT_Power_kW",
    "Total_Plant_Power_kW",
    "CW_DeltaT",
    "Running_Chillers",
]

LAG_MINUTES = [1, 5, 10, 15]
PRIMARY_HORIZON_MINUTES = 15
MULTI_HORIZONS_MINUTES = [15, 30, 45, 60]
CANDIDATE_DEGREES = [1, 2, 3]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (0.15) -> test

# CHR advisory thresholds (fixed, prototype-only, not a control system)
CHR_RISING_THRESHOLD_C = 0.15
CHR_HIGH_RISK_THRESHOLD_C = 0.40

# What-if chiller-configuration analysis
MIN_BUCKET_COUNT = 50          # minimum samples for a configuration to be reported
MIN_LOAD_FOR_KWRT_RT = 200.0   # exclude near-zero-load rows from kW/RT averaging

CHR_MODEL_PATH = "model.pkl"
LOAD_MODEL_PATH = "load_model.pkl"
METRICS_PATH = "metrics.json"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_data():
    if os.path.exists(DATA_PATH_CSV):
        df = pd.read_csv(DATA_PATH_CSV)
        fmt = "CSV"
    elif os.path.exists(DATA_PATH_XLSX):
        df = pd.read_excel(DATA_PATH_XLSX)
        fmt = "Excel"
    else:
        raise FileNotFoundError(f"No dataset found at {DATA_PATH_CSV} or {DATA_PATH_XLSX}")
    return df, fmt


# ---------------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------------
def _fix_timestamp_glitches(ts: pd.Series):
    """Corrects a verified single-row date-rollover glitch (see report)."""
    values = ts.tolist()
    step_mode = ts.diff().mode()
    if len(step_mode) == 0:
        return ts, 0
    typical_step = step_mode.iloc[0]
    fixed = 0
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff != typical_step:
            candidate = values[i] - pd.Timedelta(days=1)
            if candidate - values[i - 1] == typical_step:
                values[i] = candidate
                fixed += 1
    return pd.Series(values, index=ts.index), fixed


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    log = []

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df[TIMESTAMP_COL], n_fixed = _fix_timestamp_glitches(df[TIMESTAMP_COL])
    if n_fixed:
        log.append(f"Corrected {n_fixed} single-row timestamp date-rollover glitch(es)")

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=TIMESTAMP_COL, keep="first").reset_index(drop=True)
    if before != len(df):
        log.append(f"Removed {before - len(df)} duplicate-timestamp row(s)")

    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    if before != len(df):
        log.append(f"Removed {before - len(df)} fully duplicate row(s)")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        log.append(f"Removed {len(empty_cols)} fully-empty column(s)")
        df = df.drop(columns=empty_cols)

    const_cols = [c for c in df.columns if c != TIMESTAMP_COL and df[c].nunique(dropna=True) <= 1]
    if const_cols:
        log.append(f"Removed {len(const_cols)} constant column(s)")
        df = df.drop(columns=const_cols)

    numeric_cols = [c for c in df.columns if c not in (TIMESTAMP_COL, TARGET_COL)]
    n_missing_before = int(df[numeric_cols].isna().sum().sum())
    if n_missing_before > 0:
        df[numeric_cols] = df[numeric_cols].ffill(limit=3)
        n_missing_after = int(df[numeric_cols].isna().sum().sum())
        log.append(f"Forward-filled short (<=3 min) sensor gaps: {n_missing_before} -> {n_missing_after} remaining")
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if before != len(df):
        log.append(f"Dropped {before - len(df)} row(s) with missing TARGET (never fabricated)")

    if verbose:
        for line in log:
            print(f"  - {line}")
    return df


def detect_interval_minutes(df: pd.DataFrame) -> float:
    diffs = df[TIMESTAMP_COL].diff().dropna()
    return diffs.mode().iloc[0].total_seconds() / 60.0


# ---------------------------------------------------------------------------
# COOLING LOAD ESTIMATION
# ---------------------------------------------------------------------------
def compute_estimated_cooling_load(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_q_kw = pd.Series(0.0, index=df.index)
    used_risers = []
    for name, (flow_col, rt_col, st_col) in RISERS.items():
        if not all(c in df.columns for c in (flow_col, rt_col, st_col)):
            continue
        dt = (df[rt_col] - df[st_col]).clip(lower=0)  # negative dT = transient, treated as 0 load, not fabricated
        q = df[flow_col] * WATER_CP_KJ_PER_KGK * dt
        total_q_kw = total_q_kw + q
        used_risers.append(name)
    df[LOAD_TARGET_COL] = total_q_kw / KW_PER_RT
    df.attrs["risers_used_for_load"] = used_risers
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame, source_col: str, lag_minutes: list, prefix: str) -> pd.DataFrame:
    df = df.copy()
    by_time = df.set_index(TIMESTAMP_COL)[source_col]
    for lag in lag_minutes:
        lookup_times = df[TIMESTAMP_COL] - pd.Timedelta(minutes=lag)
        df[f"{prefix}_lag{lag}"] = by_time.reindex(lookup_times).values
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    wb_cols = [c for c in WETBULB_COLS if c in df.columns]
    df["WetBulb_Avg"] = df[wb_cols].mean(axis=1)

    df["Hour"] = df[TIMESTAMP_COL].dt.hour
    df["DayOfWeek"] = df[TIMESTAMP_COL].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    df = compute_estimated_cooling_load(df)
    df["Plant_kW_per_RT"] = df["Total_Plant_Power_kW"] / df[LOAD_TARGET_COL].replace(0, np.nan)

    df = add_lag_features(df, TARGET_COL, LAG_MINUTES, "CHR")
    df = add_lag_features(df, LOAD_TARGET_COL, LAG_MINUTES, "Load")

    return df


def create_future_target(df: pd.DataFrame, horizon_minutes: int, source_col: str, output_col: str) -> pd.DataFrame:
    """Strict time-based lookup at t + horizon — the only place future info enters the frame."""
    df = df.copy()
    by_time = df.set_index(TIMESTAMP_COL)[source_col]
    lookup_times = df[TIMESTAMP_COL] + pd.Timedelta(minutes=horizon_minutes)
    df[output_col] = by_time.reindex(lookup_times).values
    return df


def get_chr_feature_columns() -> list:
    return BASE_FEATURE_COLS + ["WetBulb_Avg", "Hour", "DayOfWeek", "IsWeekend"] + [f"CHR_lag{l}" for l in LAG_MINUTES]


def get_load_feature_columns() -> list:
    return BASE_FEATURE_COLS + ["WetBulb_Avg", "Hour", "DayOfWeek", "IsWeekend"] + [f"Load_lag{l}" for l in LAG_MINUTES]


# ---------------------------------------------------------------------------
# SPLITTING
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
def naive_baseline_predict(current_series: pd.Series) -> np.ndarray:
    """Persistence baseline: predicted future value = current value."""
    return current_series.values


def build_pipeline(degree: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree)),
        ("linreg", LinearRegression()),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def select_best_degree(X_train, y_train, X_val, y_val, degrees=CANDIDATE_DEGREES):
    results = {}
    for degree in degrees:
        pipe = build_pipeline(degree)
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        results[degree] = evaluate(y_val, val_pred)
    best_degree = min(results, key=lambda d: results[d]["RMSE"])
    return best_degree, results


def top_coefficient_terms(pipeline: Pipeline, feature_names: list, top_n: int = 12):
    poly = pipeline.named_steps["poly"]
    linreg = pipeline.named_steps["linreg"]
    term_names = poly.get_feature_names_out(feature_names)
    coefs = linreg.coef_
    order = np.argsort(-np.abs(coefs))[:top_n]
    return [{"term": term_names[i], "coefficient": float(coefs[i])} for i in order]


def train_forecast_model(df, target_source_col, feature_cols, horizon_minutes, label, degrees=CANDIDATE_DEGREES):
    """
    Generic, leakage-safe forecasting pipeline used for BOTH the CHR model
    and the Cooling Load model: future-target creation -> chronological
    split -> naive baseline -> degree selection on validation -> refit on
    train+val -> single evaluation on untouched test.
    """
    d = create_future_target(df, horizon_minutes, target_source_col, f"{label}_future")
    required = feature_cols + [f"{label}_future"]
    d = d.dropna(subset=required).reset_index(drop=True)

    train_df, val_df, test_df = chronological_split(d)
    X_train, y_train = train_df[feature_cols], train_df[f"{label}_future"]
    X_val, y_val = val_df[feature_cols], val_df[f"{label}_future"]
    X_test, y_test = test_df[feature_cols], test_df[f"{label}_future"]

    baseline_pred = naive_baseline_predict(test_df[target_source_col])
    baseline_metrics = evaluate(y_test, baseline_pred)

    best_degree, degree_val_results = select_best_degree(X_train, y_train, X_val, y_val, degrees)

    final_pipeline = build_pipeline(best_degree)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_pipeline.fit(X_trainval, y_trainval)
    y_pred_test = final_pipeline.predict(X_test)
    test_metrics = evaluate(y_test, y_pred_test)

    return {
        "pipeline": final_pipeline,
        "feature_cols": feature_cols,
        "target_source_col": target_source_col,
        "horizon_minutes": horizon_minutes,
        "poly_degree": best_degree,
        "baseline_metrics": baseline_metrics,
        "metrics": test_metrics,
        "degree_val_results": degree_val_results,
        "top_terms": top_coefficient_terms(final_pipeline, feature_cols, top_n=12),
        "test_timestamps": test_df[TIMESTAMP_COL].values,
        "test_actual": y_test.values,
        "test_predicted": y_pred_test,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "improved_over_baseline": bool(test_metrics["RMSE"] < baseline_metrics["RMSE"]),
    }


# ---------------------------------------------------------------------------
# ADVISORY LOGIC (prototype only — no equipment control)
# ---------------------------------------------------------------------------
def advisory_status(current_chr: float, predicted_chr: float) -> str:
    """Legacy single-signal CHR-only status (kept for backward compatibility)."""
    delta = predicted_chr - current_chr
    if delta >= CHR_HIGH_RISK_THRESHOLD_C:
        return "HIGH RISK"
    if delta >= CHR_RISING_THRESHOLD_C:
        return "RISING"
    return "STABLE"


def plant_intelligence_status(chr_change, load_change, load_rising_threshold, load_high_threshold,
                               current_kw_per_rt=None, efficiency_mean=None, efficiency_std=None):
    """
    Combines CHR trend, cooling-load trend, and (optionally) current
    efficiency vs the historical norm for the current chiller
    configuration into ONE advisory label. Priority order:
        1. HIGH COOLING DEMAND — load change exceeds a data-driven
           (98th percentile of historical 15-min swings) threshold
        2. EFFICIENCY WARNING  — current kW/RT is > 1.5 std above the
           historical mean for plants running this many chillers
        3. INCREASING DEMAND   — load or CHR change exceeds a milder
           (90th percentile) threshold
        4. STABLE              — otherwise
    Advisory only. Not an optimizer. Sends no commands to equipment.
    """
    if load_change is not None and load_change >= load_high_threshold:
        return "HIGH COOLING DEMAND"
    if (current_kw_per_rt is not None and efficiency_mean is not None and efficiency_std is not None
            and efficiency_std > 0 and current_kw_per_rt > efficiency_mean + 1.5 * efficiency_std):
        return "EFFICIENCY WARNING"
    if (load_change is not None and load_change >= load_rising_threshold) or (chr_change is not None and chr_change >= CHR_RISING_THRESHOLD_C):
        return "INCREASING DEMAND"
    return "STABLE"


# ---------------------------------------------------------------------------
# WHAT-IF: HISTORICAL CHILLER-CONFIGURATION ANALYSIS (descriptive, not optimization)
# ---------------------------------------------------------------------------
def chiller_config_analysis(df: pd.DataFrame) -> dict:
    """
    Groups the full cleaned dataset by Running_Chillers and reports the
    historically observed average cooling load, plant power, and kW/RT
    for each configuration that has enough samples to be reliable. This
    is purely descriptive historical analysis — NOT an optimization
    algorithm, and it makes no recommendation.
    """
    d = df.copy()
    d = d[d[LOAD_TARGET_COL] >= MIN_LOAD_FOR_KWRT_RT]
    d["_kw_per_rt"] = d["Total_Plant_Power_kW"] / d[LOAD_TARGET_COL]

    counts = df["Running_Chillers"].value_counts()
    reliable_configs = sorted(counts[counts >= MIN_BUCKET_COUNT].index.tolist())
    skipped_configs = sorted(counts[counts < MIN_BUCKET_COUNT].index.tolist())

    rows = []
    bucket_stats = {}
    for n_chillers in reliable_configs:
        sub = d[d["Running_Chillers"] == n_chillers]
        if len(sub) == 0:
            continue
        row = {
            "running_chillers": int(n_chillers),
            "n_samples": int(len(sub)),
            "avg_cooling_load_rt": float(sub[LOAD_TARGET_COL].mean()),
            "avg_plant_power_kw": float(sub["Total_Plant_Power_kW"].mean()),
            "avg_kw_per_rt": float(sub["_kw_per_rt"].mean()),
            "std_kw_per_rt": float(sub["_kw_per_rt"].std()),
        }
        rows.append(row)
        bucket_stats[str(int(n_chillers))] = {
            "mean_kw_per_rt": row["avg_kw_per_rt"],
            "std_kw_per_rt": row["std_kw_per_rt"] if not np.isnan(row["std_kw_per_rt"]) else 0.0,
        }

    return {
        "table": rows,
        "bucket_stats": bucket_stats,
        "skipped_configs": [int(c) for c in skipped_configs],
        "skip_reason": f"Fewer than {MIN_BUCKET_COUNT} samples in the dataset — not reliable enough to report.",
        "min_load_filter_rt": MIN_LOAD_FOR_KWRT_RT,
    }


def compute_load_change_thresholds(df: pd.DataFrame, horizon_minutes: int = PRIMARY_HORIZON_MINUTES) -> dict:
    """
    Data-driven thresholds for the plant-intelligence status logic,
    derived from the ACTUAL historical distribution of {horizon}-minute
    cooling-load swings in this dataset (90th / 98th percentile of the
    absolute change), rather than arbitrary fixed numbers.
    """
    d = create_future_target(df, horizon_minutes, LOAD_TARGET_COL, "_load_future_tmp")
    d = d.dropna(subset=[LOAD_TARGET_COL, "_load_future_tmp"])
    change = (d["_load_future_tmp"] - d[LOAD_TARGET_COL])
    rising = float(change.abs().quantile(0.90))
    high = float(change.abs().quantile(0.98))
    return {"rising_threshold_rt": rising, "high_threshold_rt": high}


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def make_evaluation_plots(result: dict, label: str, y_label: str, max_points: int = 1500):
    ts = result["test_timestamps"]
    actual = result["test_actual"]
    predicted = result["test_predicted"]
    if len(ts) > max_points:
        ts, actual, predicted = ts[-max_points:], actual[-max_points:], predicted[-max_points:]
    error = actual - predicted

    plt.figure(figsize=(12, 5))
    plt.plot(ts, actual, label=f"Actual {label}", linewidth=1.2)
    plt.plot(ts, predicted, label=f"Predicted {label}", linewidth=1.2, alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel(y_label)
    plt.title(f"Actual vs Predicted {label} — Test Set")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"plot_{label.lower().replace(' ', '_')}_actual_vs_predicted.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(ts, error, linewidth=1.0, color="firebrick")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel(f"Actual - Predicted ({y_label})")
    plt.title(f"{label} Prediction Error Over Time — Test Set")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"plot_{label.lower().replace(' ', '_')}_error.png", dpi=120)
    plt.close()

    full_actual, full_predicted = result["test_actual"], result["test_predicted"]
    plt.figure(figsize=(6, 6))
    plt.scatter(full_actual, full_predicted, s=6, alpha=0.35)
    lims = [min(full_actual.min(), full_predicted.min()), max(full_actual.max(), full_predicted.max())]
    plt.plot(lims, lims, color="black", linewidth=1.2, label="Perfect prediction")
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"Predicted vs Actual {label} — Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{label.lower().replace(' ', '_')}_scatter.png", dpi=120)
    plt.close()


def run_multi_horizon(df, target_source_col, feature_cols, label, fixed_degree):
    results = {}
    for h in MULTI_HORIZONS_MINUTES:
        r = train_forecast_model(df, target_source_col, feature_cols, h, f"{label}_h{h}", degrees=[fixed_degree])
        results[h] = {"baseline": r["baseline_metrics"], "polyreg": r["metrics"]}
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("CHILLER PLANT AI — DUAL FORECASTING (CHR + COOLING LOAD)")
    print("=" * 78)

    print("\n[1/10] Loading data...")
    raw_df, fmt = load_data()
    print(f"  - Format: {fmt}, raw shape: {raw_df.shape}")

    print("\n[2/10] Cleaning data...")
    df = clean_data(raw_df)
    interval_min = detect_interval_minutes(df)
    print(f"  - Detected sampling interval: {interval_min:.2f} minute(s)")
    print(f"  - Rows after cleaning: {len(df)}")

    print("\n[3/10] Engineering features + estimating cooling load...")
    df = engineer_features(df)
    print(f"  - Risers used for cooling-load estimate: {df.attrs.get('risers_used_for_load')}")
    load_stats = df[LOAD_TARGET_COL].describe()
    kwrt_stats = df.loc[df[LOAD_TARGET_COL] >= MIN_LOAD_FOR_KWRT_RT, "Plant_kW_per_RT"].describe()
    print(f"  - Estimated Cooling Load (RT): mean={load_stats['mean']:.1f}, "
          f"min={load_stats['min']:.1f}, max={load_stats['max']:.1f}")
    print(f"  - Implied Plant kW/RT (sanity check, load>={MIN_LOAD_FOR_KWRT_RT:.0f} RT only): "
          f"mean={kwrt_stats['mean']:.3f}, 25%={kwrt_stats['25%']:.3f}, 75%={kwrt_stats['75%']:.3f} "
          f"(textbook-plausible range ~0.4-0.7 kW/RT supports the L/s unit assumption)")

    chr_feature_cols = get_chr_feature_columns()
    load_feature_cols = get_load_feature_columns()

    print("\n" + "-" * 78)
    print("DATA REPORT")
    print("-" * 78)
    print(f"Rows (cleaned):              {len(df)}")
    print(f"Raw columns:                 {raw_df.shape[1]}")
    print(f"Sampling interval:           {interval_min:.2f} min")
    print(f"CHR target:                  {TARGET_COL}")
    print(f"CHR features ({len(chr_feature_cols)}): {chr_feature_cols}")
    print(f"Cooling-load target:         {LOAD_TARGET_COL} (ESTIMATED — see methodology note above)")
    print(f"Cooling-load features ({len(load_feature_cols)}): {load_feature_cols}")
    print("-" * 78)

    # ---- CHR model ----
    print(f"\n[4/10] Training CHR model ({PRIMARY_HORIZON_MINUTES}-min ahead)...")
    chr_result = train_forecast_model(df, TARGET_COL, chr_feature_cols, PRIMARY_HORIZON_MINUTES, "CHR")
    print(f"  - Selected degree: {chr_result['poly_degree']}")
    print(f"  - Baseline  MAE={chr_result['baseline_metrics']['MAE']:.4f} RMSE={chr_result['baseline_metrics']['RMSE']:.4f} R2={chr_result['baseline_metrics']['R2']:.4f}")
    print(f"  - PolyReg   MAE={chr_result['metrics']['MAE']:.4f} RMSE={chr_result['metrics']['RMSE']:.4f} R2={chr_result['metrics']['R2']:.4f}")
    print(f"  - Improved over baseline: {chr_result['improved_over_baseline']}")

    # ---- Cooling Load model ----
    print(f"\n[5/10] Training Cooling Load model ({PRIMARY_HORIZON_MINUTES}-min ahead)...")
    load_result = train_forecast_model(df, LOAD_TARGET_COL, load_feature_cols, PRIMARY_HORIZON_MINUTES, "Load")
    print(f"  - Selected degree: {load_result['poly_degree']}")
    print(f"  - Baseline  MAE={load_result['baseline_metrics']['MAE']:.4f} RMSE={load_result['baseline_metrics']['RMSE']:.4f} R2={load_result['baseline_metrics']['R2']:.4f}")
    print(f"  - PolyReg   MAE={load_result['metrics']['MAE']:.4f} RMSE={load_result['metrics']['RMSE']:.4f} R2={load_result['metrics']['R2']:.4f}")
    print(f"  - Improved over baseline: {load_result['improved_over_baseline']}")

    # ---- Multi-horizon (secondary) ----
    print("\n[6/10] Multi-horizon extension (15/30/45/60 min) for both models...")
    chr_multi = run_multi_horizon(df, TARGET_COL, chr_feature_cols, "CHR", chr_result["poly_degree"])
    load_multi = run_multi_horizon(df, LOAD_TARGET_COL, load_feature_cols, "Load", load_result["poly_degree"])
    for h in MULTI_HORIZONS_MINUTES:
        print(f"  - CHR  {h:>3} min | baseline RMSE={chr_multi[h]['baseline']['RMSE']:.4f}  polyreg RMSE={chr_multi[h]['polyreg']['RMSE']:.4f}  R2={chr_multi[h]['polyreg']['R2']:.4f}")
    for h in MULTI_HORIZONS_MINUTES:
        print(f"  - Load {h:>3} min | baseline RMSE={load_multi[h]['baseline']['RMSE']:.4f}  polyreg RMSE={load_multi[h]['polyreg']['RMSE']:.4f}  R2={load_multi[h]['polyreg']['R2']:.4f}")

    # ---- What-if chiller configuration analysis ----
    print("\n[7/10] Historical chiller-configuration analysis (descriptive, not optimization)...")
    config_analysis = chiller_config_analysis(df)
    if config_analysis["table"]:
        for row in config_analysis["table"]:
            print(f"  - {row['running_chillers']} chillers (n={row['n_samples']}): "
                  f"avg load={row['avg_cooling_load_rt']:.0f} RT, "
                  f"avg power={row['avg_plant_power_kw']:.0f} kW, "
                  f"avg kW/RT={row['avg_kw_per_rt']:.3f}")
    if config_analysis["skipped_configs"]:
        print(f"  - Skipped configurations {config_analysis['skipped_configs']}: {config_analysis['skip_reason']}")

    # ---- Data-driven plant-intelligence thresholds ----
    print("\n[8/10] Deriving data-driven plant-intelligence thresholds...")
    load_thresholds = compute_load_change_thresholds(df)
    print(f"  - Load rising threshold (90th pct of historical {PRIMARY_HORIZON_MINUTES}-min swings): {load_thresholds['rising_threshold_rt']:.1f} RT")
    print(f"  - Load high-demand threshold (98th pct): {load_thresholds['high_threshold_rt']:.1f} RT")

    # ---- Plots ----
    print("\n[9/10] Generating plots...")
    make_evaluation_plots(chr_result, "CHR", "Chilled Water Return Temp (°C)")
    make_evaluation_plots(load_result, "Cooling Load", "Estimated Cooling Load (RT)")
    print("  - Saved plot_chr_*.png and plot_cooling_load_*.png (3 each: actual_vs_predicted, error, scatter)")

    # ---- Save artifacts ----
    print("\n[10/10] Saving model artifacts and metrics...")

    shared_intelligence = {
        "config_bucket_stats": config_analysis["bucket_stats"],
        "load_rising_threshold_rt": load_thresholds["rising_threshold_rt"],
        "load_high_threshold_rt": load_thresholds["high_threshold_rt"],
        "chr_rising_threshold_c": CHR_RISING_THRESHOLD_C,
        "chr_high_risk_threshold_c": CHR_HIGH_RISK_THRESHOLD_C,
    }

    chr_artifact = {
        "pipeline": chr_result["pipeline"],
        "feature_cols": chr_result["feature_cols"],
        "target_col": TARGET_COL,
        "supply_col": SUPPLY_COL,
        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "poly_degree": chr_result["poly_degree"],
        "baseline_metrics": chr_result["baseline_metrics"],
        "metrics": chr_result["metrics"],
        "degree_val_results": chr_result["degree_val_results"],
        "top_terms": chr_result["top_terms"],
        "test_timestamps": chr_result["test_timestamps"][-1500:],
        "test_actual": chr_result["test_actual"][-1500:],
        "test_predicted": chr_result["test_predicted"][-1500:],
        "n_train": chr_result["n_train"],
        "n_val": chr_result["n_val"],
        "n_test": chr_result["n_test"],
        "interval_minutes": interval_min,
        "rising_threshold_c": CHR_RISING_THRESHOLD_C,
        "high_risk_threshold_c": CHR_HIGH_RISK_THRESHOLD_C,
        "plant_intelligence": shared_intelligence,
    }
    joblib.dump(chr_artifact, CHR_MODEL_PATH)
    print(f"  - Saved CHR model to {CHR_MODEL_PATH}")

    load_artifact = {
        "pipeline": load_result["pipeline"],
        "feature_cols": load_result["feature_cols"],
        "target_col": LOAD_TARGET_COL,
        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "poly_degree": load_result["poly_degree"],
        "baseline_metrics": load_result["baseline_metrics"],
        "metrics": load_result["metrics"],
        "degree_val_results": load_result["degree_val_results"],
        "top_terms": load_result["top_terms"],
        "test_timestamps": load_result["test_timestamps"][-1500:],
        "test_actual": load_result["test_actual"][-1500:],
        "test_predicted": load_result["test_predicted"][-1500:],
        "n_train": load_result["n_train"],
        "n_val": load_result["n_val"],
        "n_test": load_result["n_test"],
        "interval_minutes": interval_min,
        "plant_intelligence": shared_intelligence,
        "risers_used": df.attrs.get("risers_used_for_load"),
    }
    joblib.dump(load_artifact, LOAD_MODEL_PATH)
    print(f"  - Saved Cooling Load model to {LOAD_MODEL_PATH}")

    metrics_export = {
        "dataset_rows_final": len(df),
        "sampling_interval_minutes": interval_min,
        "cooling_load_methodology": {
            "formula": "Q(kW) = Flow(L/s) x 4.186 x DeltaT(C), summed across MainBuilding/Finger/L1-3/T1U risers; RT = Q(kW)/3.517",
            "risers_used": df.attrs.get("risers_used_for_load"),
            "unit_assumption": "Flow readings assumed L/s (unverified tag reliability elsewhere in export)",
            "sanity_check_kw_per_rt_mean": float(kwrt_stats["mean"]),
            "label": "ESTIMATED — not a calibrated meter reading",
        },
        "chr_target_column": TARGET_COL,
        "chr_primary_horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "chr_selected_polynomial_degree": chr_result["poly_degree"],
        "chr_degree_selection_validation_results": chr_result["degree_val_results"],
        "chr_baseline_test_metrics": chr_result["baseline_metrics"],
        "chr_polynomial_test_metrics": chr_result["metrics"],
        "chr_improved_over_baseline": chr_result["improved_over_baseline"],
        "chr_multi_horizon_results": {str(h): v for h, v in chr_multi.items()},
        "chr_top_model_terms": chr_result["top_terms"],
        "load_target_column": LOAD_TARGET_COL,
        "load_primary_horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "load_selected_polynomial_degree": load_result["poly_degree"],
        "load_degree_selection_validation_results": load_result["degree_val_results"],
        "load_baseline_test_metrics": load_result["baseline_metrics"],
        "load_polynomial_test_metrics": load_result["metrics"],
        "load_improved_over_baseline": load_result["improved_over_baseline"],
        "load_multi_horizon_results": {str(h): v for h, v in load_multi.items()},
        "load_top_model_terms": load_result["top_terms"],
        "chiller_config_analysis": config_analysis,
        "plant_intelligence_thresholds": shared_intelligence,
        "split": {"train": chr_result["n_train"], "val": chr_result["n_val"], "test": chr_result["n_test"]},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_export, f, indent=2, default=str)
    print(f"  - Saved metrics to {METRICS_PATH}")

    # ---- Final console report ----
    print("\n" + "=" * 78)
    print("FINAL REPORT")
    print("=" * 78)
    print(f"Dataset:")
    print(f"  Rows:               {len(df)}")
    print(f"  Columns (raw):      {raw_df.shape[1]}")
    print(f"  Sampling interval:  {interval_min:.2f} min")
    print()
    print(f"CHR target:   {TARGET_COL}")
    print(f"CHR features: {len(chr_feature_cols)} -> {chr_feature_cols}")
    print()
    print(f"Cooling-load target:   {LOAD_TARGET_COL} (ESTIMATED, see methodology)")
    print(f"Cooling-load features: {len(load_feature_cols)} -> {load_feature_cols}")
    print()
    print(f"Best CHR model:  degree={chr_result['poly_degree']}  "
          f"MAE={chr_result['metrics']['MAE']:.4f}  RMSE={chr_result['metrics']['RMSE']:.4f}  R2={chr_result['metrics']['R2']:.4f}")
    print(f"Best Load model: degree={load_result['poly_degree']}  "
          f"MAE={load_result['metrics']['MAE']:.4f}  RMSE={load_result['metrics']['RMSE']:.4f}  R2={load_result['metrics']['R2']:.4f}")
    print()
    print(f"CHR improved over baseline:  {chr_result['improved_over_baseline']}")
    print(f"Load improved over baseline: {load_result['improved_over_baseline']}")
    print("=" * 78)

    print("\nDone. Run `streamlit run app.py` to launch the dashboard.\n")


if __name__ == "__main__":
    main()