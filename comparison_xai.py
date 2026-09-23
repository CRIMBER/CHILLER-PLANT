"""
comparison_xai.py
-----------------
Head-to-head model comparison (Random Forest vs LSTM vs BiLSTM) with
statistical significance testing, plus a full SHAP explainability study
of the BiLSTM.

Written for publication, so the comparison is built to survive review:

  * All three models are selected on the SAME criterion (validation MAE)
    and scored on the SAME final-test windows. Nothing is tuned on test.
  * Differences are reported with SIGNIFICANCE, not just point estimates.
    A 0.002 RMSE gap on autocorrelated data is not a result, and this
    module says so explicitly rather than letting a bar chart imply it.
      - Diebold-Mariano test (Newey-West HAC variance, Harvey-Leybourne-
        Newbold small-sample correction) -- the standard test for
        comparing two forecast series.
      - Moving-block bootstrap confidence intervals, block length chosen
        to preserve the 15-minute autocorrelation structure.
  * Every pairwise comparison is reported on every metric, so a model
    that wins on RMSE but loses on MAE is visible rather than hidden
    behind a single ranking.

                    Random Forest   LSTM   BiLSTM
                          |          |       |
                    identical test windows, identical baseline
                          |          |       |
                     ------------------------------
                                   |
                    pairwise DM tests + bootstrap CIs
                                   |
                    win matrix / violin / ECDF / radar
                                   |
                         SHAP study of BiLSTM

Run:
    python comparison_xai.py
"""

import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

import deep_models as dm
from deep_models import (
    BiLSTMForecaster,
    LSTMForecaster,
    SeqScaler,
    build_sequences,
    predict,
    set_seed,
    train_model,
)
from train import (
    PRIMARY_HORIZON_MINUTES,
    RANDOM_STATE,
    TARGET_COL,
    TIMESTAMP_COL,
    clean_data,
    detect_interval_minutes,
    engineer_features,
    evaluate,
    load_data,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COMPARE_MODELS = ["RandomForest", "LSTM", "BiLSTM"]
SHAP_TARGET_MODEL = "BiLSTM"          # the model explained in depth

# Random Forest search grid. Kept deliberately small and selected on the
# SAME validation split (by MAE) the deep models use -- a larger grid for
# the classical model and none for the deep ones would bias the contest.
RF_GRID = [
    {"n_estimators": 200, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 12, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 20, "max_features": 0.3},
]

# Statistical testing
DM_LOSSES = ["squared", "absolute"]   # DM is run under both loss functions
BOOTSTRAP_B = 2000                    # resamples for block-bootstrap CIs
BOOTSTRAP_BLOCK = 60                  # 60 rows = 1 hour; >> the 15-min horizon
ALPHA = 0.05

# SHAP
SHAP_N_TEST = 256                     # instances explained
SHAP_N_BACKGROUND = 100               # background/reference windows
SHAP_NSAMPLES = 100                   # expected-gradient draws per instance
SHAP_DEPENDENCE_TOP_N = 4

ARTIFACT_PATH = "comparison_artifact.pkl"
METRICS_PATH = "metrics_comparison.json"
PAIRWISE_CSV = "pairwise_comparison.csv"
LEADERBOARD_CSV = "comparison_results.csv"

# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------
# Categorical slots 1-3 of the design system's validated dark-mode order
# (blue / orange / aqua). That ORDER is pre-validated: worst adjacent CVD
# delta-E 8.4 on dark (OKLab x100, >=8 target). Hues are assigned to models
# in fixed order and never cycled or reassigned, so a model keeps its
# colour in every figure in the paper.
MODEL_COLORS = {
    "RandomForest": "#3987e5",   # slot 1 - blue
    "LSTM": "#d95926",           # slot 2 - orange
    "BiLSTM": "#199e70",         # slot 3 - aqua
}
REFERENCE_COLORS = {
    "Persistence": "#8a8a92",
    "PolyReg_deg1": "#c3c2b7",
}

PLOT_VOID = "#0d0d0f"
PLOT_INK = "#e6e6ea"
PLOT_INK_DIM = "#8a8a92"
PLOT_HAIR = "#2a2a30"
PLOT_ALARM = "#e66767"

plt.rcParams.update({
    "figure.facecolor": PLOT_VOID, "axes.facecolor": PLOT_VOID, "savefig.facecolor": PLOT_VOID,
    "axes.edgecolor": PLOT_HAIR, "axes.labelcolor": PLOT_INK,
    "axes.grid": True, "grid.color": PLOT_HAIR, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": PLOT_INK_DIM, "ytick.color": PLOT_INK_DIM, "text.color": PLOT_INK,
    "legend.facecolor": "#141416", "legend.edgecolor": PLOT_HAIR, "legend.labelcolor": PLOT_INK,
    "font.family": "monospace", "axes.titlecolor": PLOT_INK,
    "figure.dpi": 150,
})


def color_of(name):
    return MODEL_COLORS.get(name, REFERENCE_COLORS.get(name, PLOT_INK_DIM))


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def full_metrics(y_true, y_pred) -> dict:
    """Point metrics plus robust ones -- a skewed error distribution shows up
    as a large mean/median gap, which MAE alone hides."""
    err = y_true - y_pred
    abs_err = np.abs(err)
    base = evaluate(y_true, y_pred)
    base.update({
        "MedAE": float(np.median(abs_err)),
        "P90AE": float(np.percentile(abs_err, 90)),
        "MaxAE": float(abs_err.max()),
        "Bias": float(err.mean()),
    })
    return base


# ---------------------------------------------------------------------------
# STATISTICAL TESTS
# ---------------------------------------------------------------------------
def newey_west_var(d: np.ndarray, lag: int) -> float:
    """
    HAC (Newey-West) long-run variance of the loss-differential series.

    Forecasts made every minute for a point 15 minutes ahead overlap
    heavily, so d_t is strongly autocorrelated. Using the plain sample
    variance here would understate the standard error and manufacture
    significance -- this is the single most common error in published
    forecast comparisons.
    """
    n = len(d)
    d_centered = d - d.mean()
    gamma0 = float(np.dot(d_centered, d_centered) / n)
    total = gamma0
    for k in range(1, lag + 1):
        cov = float(np.dot(d_centered[k:], d_centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)       # Bartlett kernel
        total += 2.0 * weight * cov
    return max(total, 1e-12)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, horizon: int, loss: str = "squared"):
    """
    Diebold-Mariano test of equal predictive accuracy between two forecast
    error series, with the Harvey-Leybourne-Newbold small-sample correction.

    H0: the two models have equal expected loss.
    Negative statistic => model 1 has LOWER loss (model 1 is better).

    Returns the statistic, two-sided p-value, and mean loss differential.
    """
    from scipy import stats

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(loss)

    n = len(d)
    lag = max(horizon - 1, 1)
    var_d = newey_west_var(d, lag)
    dm_stat = d.mean() / np.sqrt(var_d / n)

    # Harvey, Leybourne & Newbold (1997) small-sample correction
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_stat = dm_stat * correction

    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value), float(d.mean())


def block_bootstrap_ci(e1: np.ndarray, e2: np.ndarray, metric: str,
                       b: int = BOOTSTRAP_B, block: int = BOOTSTRAP_BLOCK,
                       alpha: float = ALPHA, seed: int = RANDOM_STATE):
    """
    Moving-block bootstrap CI for the DIFFERENCE in a metric (model1 - model2).

    Blocks (not individual points) are resampled so the autocorrelation of
    overlapping 15-minute-ahead forecasts is preserved; an iid bootstrap
    would give intervals that are far too narrow.

    A CI that straddles zero means the difference is not distinguishable
    from noise at this sample size.
    """
    rng = np.random.default_rng(seed)
    n = len(e1)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block

    def metric_of(e):
        return float(np.sqrt(np.mean(e ** 2))) if metric == "RMSE" else float(np.mean(np.abs(e)))

    observed = metric_of(e1) - metric_of(e2)
    diffs = np.empty(b, dtype=float)
    for i in range(b):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[i] = metric_of(e1[idx]) - metric_of(e2[idx])

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "observed_diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def plot_metric_bars(results, order, baseline, poly):
    """Grouped bars. One panel per metric -- never a dual axis."""
    metrics = [("RMSE", "RMSE (deg C)"), ("MAE", "MAE (deg C)"), ("R2", "R-squared")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (key, label) in zip(axes, metrics):
        vals = [results[m]["metrics"][key] for m in order]
        bars = ax.bar(order, vals, color=[color_of(m) for m in order],
                      width=0.62, edgecolor=PLOT_VOID, linewidth=2)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=PLOT_INK)
        if baseline is not None:
            ax.axhline(baseline[key], color=REFERENCE_COLORS["Persistence"],
                       linestyle="--", linewidth=1.2, label="Persistence")
        if poly is not None:
            ax.axhline(poly[key], color=REFERENCE_COLORS["PolyReg_deg1"],
                       linestyle=":", linewidth=1.2, label="PolyReg deg 1")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", rotation=12)
        if key == "RMSE":
            ax.legend(fontsize=7, loc="lower right")
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.45 + 1e-6
        ax.set_ylim(max(0, lo - pad) if key != "R2" else lo - pad, hi + pad)
    fig.suptitle("Final-Test performance by model", fontsize=12)
    plt.tight_layout()
    plt.savefig("plot_cmp_metric_bars.png")
    plt.close()


def plot_error_violin(results, order):
    """
    Violin of the absolute-error DISTRIBUTION.

    The bar chart shows the mean; this shows the shape. Two models with
    the same MAE can have very different tails, and the tail is what
    matters operationally.
    """
    data = [np.abs(results[m]["errors"]) for m in order]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    parts = ax.violinplot(data, showextrema=False, widths=0.82)
    for body, name in zip(parts["bodies"], order):
        body.set_facecolor(color_of(name))
        body.set_edgecolor(PLOT_VOID)
        body.set_linewidth(2)
        body.set_alpha(0.82)
    # median + IQR overlay
    for i, (d, name) in enumerate(zip(data, order), start=1):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color=PLOT_VOID, linewidth=6)
        ax.vlines(i, q1, q3, color=PLOT_INK, linewidth=2)
        ax.scatter([i], [med], s=34, color=PLOT_VOID, zorder=4)
        ax.scatter([i], [med], s=18, color=PLOT_INK, zorder=5)
        ax.text(i + 0.30, med, f"median {med:.4f}", fontsize=7.5,
                color=PLOT_INK, va="center")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Absolute error (deg C)")
    # Rare excursions run to ~1.4 deg C and would squash every distribution into
    # an unreadable sliver at the baseline. Clip the view to the 99th percentile
    # and SAY SO on the axis -- the tail is described by P90AE/MaxAE in the
    # metric table and by the ECDF, so nothing is hidden by clipping here.
    cut = max(np.percentile(d, 99) for d in data)
    n_hidden = sum(int((d > cut).sum()) for d in data)
    total = sum(len(d) for d in data)
    ax.set_ylim(0, cut * 1.12)
    ax.set_title("Absolute-error distribution  (violin = density, bar = IQR, dot = median)",
                 fontsize=11)
    ax.text(0.99, 0.97, f"view clipped at {cut:.3f} deg C (99th pct)\n"
                        f"{n_hidden} of {total} points above ({100*n_hidden/total:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=PLOT_INK_DIM)
    plt.tight_layout()
    plt.savefig("plot_cmp_error_violin.png")
    plt.close()


def plot_error_ecdf(results, order):
    """
    ECDF of absolute error: 'what fraction of forecasts land within X?'

    Reads directly as an operational tolerance curve, and crossing curves
    reveal that one model is better for small errors while another is
    better in the tail -- invisible in any summary statistic.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name in order:
        e = np.sort(np.abs(results[name]["errors"]))
        y = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, y * 100, linewidth=2, color=color_of(name), label=name)
    for tol in (0.05, 0.10):
        ax.axvline(tol, color=PLOT_HAIR, linewidth=1, linestyle=":")
        ax.text(tol, 2, f" {tol:.2f} deg C", fontsize=7, color=PLOT_INK_DIM, rotation=90)
    ax.set_xlim(0, np.percentile(np.abs(results[order[0]]["errors"]), 99.5))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Absolute error tolerance (deg C)")
    ax.set_ylabel("% of forecasts within tolerance")
    ax.set_title("Cumulative error distribution (higher/left is better)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("plot_cmp_error_ecdf.png")
    plt.close()


def plot_win_matrix(pairwise, order):
    """
    Pairwise outcome matrix. Cell (row, col) = how row fares against col,
    counting only DIFFERENCES THAT ARE STATISTICALLY SIGNIFICANT.

    Diverging encoding (wins vs losses about a neutral midpoint), which is
    the correct job for signed polarity data.
    """
    n = len(order)
    mat = np.full((n, n), np.nan)
    ann = np.empty((n, n), dtype=object)
    idx = {m: i for i, m in enumerate(order)}
    for rec in pairwise:
        i, j = idx[rec["model_a"]], idx[rec["model_b"]]
        score = rec["significant_wins_a"] - rec["significant_wins_b"]
        mat[i, j], mat[j, i] = score, -score
        ann[i, j] = f"{rec['significant_wins_a']}W-{rec['significant_wins_b']}L"
        ann[j, i] = f"{rec['significant_wins_b']}W-{rec['significant_wins_a']}L"
    for i in range(n):
        ann[i, i] = "--"

    lim = max(1, np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-lim, vmax=lim)
    ax.set_xticks(range(n)); ax.set_xticklabels(order, rotation=18, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(order, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, ann[i, j], ha="center", va="center", fontsize=9,
                    color="#101010" if not np.isnan(mat[i, j]) else PLOT_INK_DIM,
                    fontweight="bold")
    ax.set_title("Head-to-head: significant metric wins\n"
                 "(row vs column; DM p<0.05 AND bootstrap CI excluding zero)", fontsize=10.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="net significant wins")
    plt.tight_layout()
    plt.savefig("plot_cmp_win_matrix.png")
    plt.close()


def plot_radar(results, order):
    """Normalised multi-metric profile. All axes oriented so outward = better."""
    keys = ["RMSE", "MAE", "MedAE", "P90AE", "MaxAE"]
    raw = {m: [results[m]["metrics"][k] for k in keys] for m in order}
    arr = np.array([raw[m] for m in order])
    # lower is better for every key -> invert to a 0-1 "goodness" score
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    good = 1.0 - (arr - lo) / span

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.8, 6.4), subplot_kw={"polar": True})
    for row, name in zip(good, order):
        vals = row.tolist() + [row[0]]
        ax.plot(angles, vals, linewidth=2, color=color_of(name), label=name)
        ax.fill(angles, vals, color=color_of(name), alpha=0.13)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["worst", "", "best"], fontsize=7, color=PLOT_INK_DIM)
    ax.set_ylim(0, 1.05)
    ax.set_facecolor(PLOT_VOID)
    ax.set_title("Multi-metric profile (outward = better; scaled within this set)",
                 fontsize=10.5, pad=22)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.22, 1.12))
    plt.tight_layout()
    plt.savefig("plot_cmp_radar.png")
    plt.close()


def plot_forecast_trace(results, order, ts, actual):
    n_show = min(600, len(actual))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts[-n_show:], actual[-n_show:], linewidth=2.2, color=PLOT_INK, label="Actual", zorder=2)
    for name in order:
        ax.plot(ts[-n_show:], results[name]["pred"][-n_show:], linewidth=1.3,
                color=color_of(name), alpha=0.9, label=name)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("CHR (deg C)")
    ax.set_title(f"Final-Test forecast replay (last {n_show} windows) - historical data, not live",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    plt.savefig("plot_cmp_forecast_trace.png")
    plt.close()


def plot_scatter_panels(results, order, actual):
    fig, axes = plt.subplots(1, len(order), figsize=(4.3 * len(order), 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lims = [actual.min(), actual.max()]
    for ax, name in zip(axes, order):
        ax.scatter(actual, results[name]["pred"], s=5, alpha=0.28, color=color_of(name),
                   edgecolors="none")
        ax.plot(lims, lims, color=PLOT_INK, linewidth=1.3)
        ax.set_title(f"{name}\nR2 = {results[name]['metrics']['R2']:.4f}", fontsize=10)
        ax.set_xlabel("Actual CHR (deg C)")
    axes[0].set_ylabel("Predicted CHR (deg C)")
    fig.suptitle("Predicted vs actual, Final Test (line = perfect prediction)", fontsize=11)
    plt.tight_layout()
    plt.savefig("plot_cmp_scatter_panels.png")
    plt.close()


def plot_significance(pairwise):
    """
    Effect size with uncertainty: bootstrap CI for each pairwise RMSE
    difference. A bar chart of metrics cannot show whether a gap is real;
    this can, and an interval crossing zero is drawn explicitly.
    """
    recs = [(f"{p['model_a']}\nvs {p['model_b']}", p["bootstrap_RMSE"]) for p in pairwise]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, (label, bs) in enumerate(recs):
        lo, hi, obs = bs["ci_low"], bs["ci_high"], bs["observed_diff"]
        sig = bs["excludes_zero"]
        col = "#199e70" if (sig and obs < 0) else ("#e66767" if sig else PLOT_INK_DIM)
        ax.plot([lo, hi], [i, i], linewidth=3, color=col, solid_capstyle="round")
        ax.scatter([obs], [i], s=60, color=col, zorder=5, edgecolors=PLOT_VOID, linewidths=1.5)
        ax.text(hi, i + 0.16, "significant" if sig else "not significant",
                fontsize=7.5, color=col, va="bottom")
    ax.axvline(0, color=PLOT_INK, linewidth=1.4)
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels([r[0] for r in recs], fontsize=8.5)
    ax.set_xlabel("RMSE difference (deg C).  Negative => first model better")
    ax.set_title(f"Pairwise RMSE difference, {int((1-ALPHA)*100)}% moving-block bootstrap CI",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig("plot_cmp_significance.png")
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("MODEL COMPARISON (Random Forest / LSTM / BiLSTM) + SHAP EXPLAINABILITY")
    print("=" * 78)
    set_seed(RANDOM_STATE)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    # ---- data ----
    print("\n[1/8] Preparing data...")
    df = engineer_features(clean_data(load_data()[0], verbose=False))
    interval = detect_interval_minutes(df)
    horizon_rows = int(round(PRIMARY_HORIZON_MINUTES / interval))
    df = df.dropna(subset=dm.SEQ_FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    X, y, y_now, ends = build_sequences(
        df, dm.SEQ_FEATURE_COLS, TARGET_COL, dm.SEQ_LEN, horizon_rows, interval)
    n = len(X)
    dev_end = int(n * dm.DEV_FRAC)
    train_end = int(dev_end * (1 - dm.VAL_FRAC_OF_DEV))

    Xtr_raw, ytr = X[:train_end], y[:train_end]
    Xva_raw, yva = X[train_end:dev_end], y[train_end:dev_end]
    Xte_raw, yte = X[dev_end:], y[dev_end:]
    yte_now = y_now[dev_end:]
    test_ts = df[TIMESTAMP_COL].to_numpy()[ends[dev_end:] + horizon_rows]

    scaler = SeqScaler().fit(Xtr_raw)                 # TRAIN ONLY
    Xtr, Xva, Xte = (scaler.transform(a) for a in (Xtr_raw, Xva_raw, Xte_raw))
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    print(f"  - Train {len(Xtr)} | Val {len(Xva)} | Test {len(Xte)}  (chronological)")
    print(f"  - Test range {pd.Timestamp(test_ts[0])} -> {pd.Timestamp(test_ts[-1])}")
    print(f"  - Selection criterion for ALL models: validation MAE")

    results = {}

    # ---- Random Forest ----
    print("\n[2/8] Random Forest (windows flattened to "
          f"{dm.SEQ_LEN}x{len(dm.SEQ_FEATURE_COLS)} = {dm.SEQ_LEN*len(dm.SEQ_FEATURE_COLS)} features)...")
    Xtr_flat = Xtr.reshape(len(Xtr), -1)
    Xva_flat = Xva.reshape(len(Xva), -1)
    Xte_flat = Xte.reshape(len(Xte), -1)

    best_rf, best_cfg, best_val = None, None, np.inf
    for cfg in RF_GRID:
        t0 = time.time()
        rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **cfg)
        rf.fit(Xtr_flat, ytr)
        val_mae = float(np.mean(np.abs(yva - rf.predict(Xva_flat))))
        print(f"    {cfg} -> val MAE={val_mae:.5f}  ({time.time()-t0:.0f}s)")
        if val_mae < best_val:
            best_rf, best_cfg, best_val = rf, cfg, val_mae
    print(f"  - Selected: {best_cfg}  (val MAE={best_val:.5f})")
    rf_pred = best_rf.predict(Xte_flat)
    results["RandomForest"] = {
        "pred": rf_pred, "errors": yte - rf_pred, "metrics": full_metrics(yte, rf_pred),
        "config": best_cfg, "val_MAE": best_val,
        "n_params": int(sum(t.tree_.node_count for t in best_rf.estimators_)),
    }
    m = results["RandomForest"]["metrics"]
    print(f"  - TEST  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

    # ---- LSTM / BiLSTM (retrained under the SAME criterion) ----
    trained_nets = {}
    for step, (name, cls) in enumerate([("LSTM", LSTMForecaster), ("BiLSTM", BiLSTMForecaster)], start=3):
        print(f"\n[{step}/8] {name} (early stopping on validation "
              f"{dm.EARLY_STOP_METRIC}, same split)...")
        info = train_model(name, cls, Xtr, ytr, Xva, yva, y_mean, y_std)
        trained_nets[name] = info
        pred_scaled, _ = predict(info["model"], torch.from_numpy(Xte))
        pred = pred_scaled * y_std + y_mean
        results[name] = {
            "pred": pred, "errors": yte - pred, "metrics": full_metrics(yte, pred),
            "n_params": info["n_params"], "val_MAE": info["best_val_MAE"],
            "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
        }
        mm = results[name]["metrics"]
        print(f"  - best epoch {info['best_epoch']} / {info['epochs_run']} run")
        print(f"  - TEST  MAE={mm['MAE']:.4f}  RMSE={mm['RMSE']:.4f}  R2={mm['R2']:.4f}")

    # ---- reference points ----
    baseline_metrics = full_metrics(yte, yte_now)
    poly_metrics, poly_pred = None, None
    try:
        art = joblib.load("model.pkl")
        poly_pred = art["pipeline"].predict(df.iloc[ends[dev_end:]][art["feature_cols"]])
        poly_metrics = full_metrics(yte, poly_pred)
    except Exception as exc:
        print(f"  (PolyReg reference unavailable: {exc})")

    # ---- pairwise statistics ----
    print("\n[5/8] Pairwise significance testing "
          f"(Diebold-Mariano + {BOOTSTRAP_B}x moving-block bootstrap)...")
    metric_keys = ["RMSE", "MAE", "MedAE", "P90AE", "R2"]
    pairwise = []
    for i in range(len(COMPARE_MODELS)):
        for j in range(i + 1, len(COMPARE_MODELS)):
            a, b = COMPARE_MODELS[i], COMPARE_MODELS[j]
            ea, eb = results[a]["errors"], results[b]["errors"]

            dm_tests = {}
            for loss in DM_LOSSES:
                stat, p, mean_d = diebold_mariano(ea, eb, horizon_rows, loss)
                dm_tests[loss] = {"statistic": stat, "p_value": p, "mean_loss_diff": mean_d,
                                  "better": a if mean_d < 0 else b,
                                  "significant": bool(p < ALPHA)}

            boot = {"RMSE": block_bootstrap_ci(ea, eb, "RMSE"),
                    "MAE": block_bootstrap_ci(ea, eb, "MAE")}

            # A difference counts as significant ONLY if the DM test and the
            # bootstrap CI agree. They can disagree (DM rejects while the CI
            # straddles zero), and in that case the conservative reading is the
            # correct one to publish -- claiming a win on the strength of
            # whichever test happened to pass is exactly the practice that makes
            # forecast comparisons irreproducible.
            per_metric, wins_a, wins_b = {}, 0, 0
            sig_metrics_a, sig_metrics_b = [], []
            for key in metric_keys:
                va, vb = results[a]["metrics"][key], results[b]["metrics"][key]
                higher_better = (key == "R2")
                winner = (a if va > vb else b) if higher_better else (a if va < vb else b)
                loss_for = "squared" if key in ("RMSE", "R2") else "absolute"
                boot_for = "RMSE" if key in ("RMSE", "R2") else "MAE"
                dm_sig = dm_tests[loss_for]["significant"]
                ci_sig = boot[boot_for]["excludes_zero"]
                sig = bool(dm_sig and ci_sig)
                per_metric[key] = {
                    "value_a": va, "value_b": vb, "winner": winner,
                    "abs_diff": abs(va - vb),
                    "pct_improvement": (abs(va - vb) / max(abs(vb if winner == a else va), 1e-12)) * 100,
                    "significant": sig,
                    "dm_significant": bool(dm_sig), "ci_excludes_zero": bool(ci_sig),
                    "significance_source": f"DM({loss_for}) AND bootstrap({boot_for})",
                }
                if sig:
                    wins_a += int(winner == a)
                    wins_b += int(winner == b)
                    (sig_metrics_a if winner == a else sig_metrics_b).append(key)

            # Verdict names the metrics, so a MAE-only advantage can never read
            # as a blanket win.
            if wins_a > wins_b:
                verdict = f"{a} better on {', '.join(sig_metrics_a)} (not significant on the rest)"
            elif wins_b > wins_a:
                verdict = f"{b} better on {', '.join(sig_metrics_b)} (not significant on the rest)"
            else:
                verdict = "no significant difference on any metric"

            rec = {
                "model_a": a, "model_b": b,
                "dm_tests": dm_tests,
                "bootstrap_RMSE": boot["RMSE"], "bootstrap_MAE": boot["MAE"],
                "per_metric": per_metric,
                "significant_wins_a": wins_a, "significant_wins_b": wins_b,
                "significant_metrics_a": sig_metrics_a, "significant_metrics_b": sig_metrics_b,
                "verdict": verdict,
            }
            pairwise.append(rec)
            bs = rec["bootstrap_RMSE"]
            print(f"  - {a} vs {b}: {rec['verdict']}")
            print(f"      DM(squared) stat={dm_tests['squared']['statistic']:+.3f} "
                  f"p={dm_tests['squared']['p_value']:.2e} | "
                  f"dRMSE={bs['observed_diff']:+.5f} "
                  f"CI[{bs['ci_low']:+.5f},{bs['ci_high']:+.5f}] "
                  f"{'EXCLUDES 0' if bs['excludes_zero'] else 'includes 0'}")

    # ---- plots ----
    print("\n[6/8] Comparison figures...")
    plot_metric_bars(results, COMPARE_MODELS, baseline_metrics, poly_metrics)
    plot_error_violin(results, COMPARE_MODELS)
    plot_error_ecdf(results, COMPARE_MODELS)
    plot_win_matrix(pairwise, COMPARE_MODELS)
    plot_radar(results, COMPARE_MODELS)
    plot_forecast_trace(results, COMPARE_MODELS, test_ts, yte)
    plot_scatter_panels(results, COMPARE_MODELS, yte)
    plot_significance(pairwise)
    print("  - Saved plot_cmp_*.png (8)")

    # ---- SHAP ----
    print(f"\n[7/8] SHAP explainability study of {SHAP_TARGET_MODEL}...")
    shap_result = run_shap_study(
        trained_nets[SHAP_TARGET_MODEL]["model"], Xtr, Xte, Xte_raw,
        dm.SEQ_FEATURE_COLS, y_mean, y_std)

    # ---- save ----
    print("\n[8/8] Saving artifacts...")
    TRACE = 1500
    artifact = {
        "compare_models": COMPARE_MODELS,
        "model_colors": MODEL_COLORS,
        "selection_criterion": f"validation {dm.EARLY_STOP_METRIC}",
        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "seq_len": dm.SEQ_LEN,
        "feature_cols": dm.SEQ_FEATURE_COLS,
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "test_date_range": [str(pd.Timestamp(test_ts[0])), str(pd.Timestamp(test_ts[-1]))],
        "metrics": {k: v["metrics"] for k, v in results.items()},
        "model_details": {k: {kk: vv for kk, vv in v.items()
                              if kk not in ("pred", "errors")} for k, v in results.items()},
        "baseline_metrics": baseline_metrics,
        "poly_metrics": poly_metrics,
        "pairwise": pairwise,
        "statistics_note": {
            "dm_test": "Diebold-Mariano with Newey-West HAC variance (Bartlett kernel, "
                       f"lag={horizon_rows-1}) and Harvey-Leybourne-Newbold correction.",
            "bootstrap": f"Moving-block bootstrap, block={BOOTSTRAP_BLOCK} rows (1 h), "
                         f"B={BOOTSTRAP_B}, alpha={ALPHA}.",
            "why": "Overlapping 15-min-ahead forecasts are strongly autocorrelated; "
                   "iid standard errors would overstate significance.",
        },
        "shap": shap_result,
        "test_timestamps": test_ts[-TRACE:],
        "test_actual": yte[-TRACE:],
        "test_predictions": {k: v["pred"][-TRACE:] for k, v in results.items()},
        # Full error series kept so figures can be regenerated or re-scaled
        # without refitting every model (~6.7k floats per model).
        "test_errors": {k: v["errors"] for k, v in results.items()},
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"  - {ARTIFACT_PATH}")

    export = {k: v for k, v in artifact.items()
              if k not in ("test_timestamps", "test_actual", "test_predictions", "shap")}
    export["shap"] = {k: v for k, v in shap_result.items() if k != "values"}
    with open(METRICS_PATH, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"  - {METRICS_PATH}")

    rows = []
    for name in COMPARE_MODELS:
        rows.append({"model": name, **results[name]["metrics"],
                     "val_MAE": results[name]["val_MAE"]})
    rows.append({"model": "Persistence", **baseline_metrics, "val_MAE": None})
    if poly_metrics:
        rows.append({"model": "PolyReg_deg1", **poly_metrics, "val_MAE": None})
    pd.DataFrame(rows).to_csv(LEADERBOARD_CSV, index=False)

    flat = []
    for rec in pairwise:
        for key, d in rec["per_metric"].items():
            flat.append({
                "model_a": rec["model_a"], "model_b": rec["model_b"], "metric": key,
                "value_a": d["value_a"], "value_b": d["value_b"], "winner": d["winner"],
                "abs_diff": d["abs_diff"], "pct_improvement": d["pct_improvement"],
                "significant": d["significant"], "significance_source": d["significance_source"],
            })
    pd.DataFrame(flat).to_csv(PAIRWISE_CSV, index=False)
    print(f"  - {LEADERBOARD_CSV} / {PAIRWISE_CSV}")

    # ---- report ----
    print("\n" + "=" * 78)
    print("FINAL REPORT")
    print("=" * 78)
    print(f"{'Model':<16}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'MedAE':>10}{'P90AE':>10}")
    print("-" * 78)
    ranked = sorted(COMPARE_MODELS, key=lambda m: results[m]["metrics"]["RMSE"])
    for name in ranked:
        mm = results[name]["metrics"]
        print(f"{name:<16}{mm['MAE']:>10.4f}{mm['RMSE']:>10.4f}{mm['R2']:>10.4f}"
              f"{mm['MedAE']:>10.4f}{mm['P90AE']:>10.4f}")
    print("-" * 78)
    print(f"{'Persistence':<16}{baseline_metrics['MAE']:>10.4f}{baseline_metrics['RMSE']:>10.4f}"
          f"{baseline_metrics['R2']:>10.4f}{baseline_metrics['MedAE']:>10.4f}{baseline_metrics['P90AE']:>10.4f}")
    if poly_metrics:
        print(f"{'PolyReg deg1':<16}{poly_metrics['MAE']:>10.4f}{poly_metrics['RMSE']:>10.4f}"
              f"{poly_metrics['R2']:>10.4f}{poly_metrics['MedAE']:>10.4f}{poly_metrics['P90AE']:>10.4f}")
    print("=" * 78)
    for rec in pairwise:
        print(f"{rec['model_a']} vs {rec['model_b']}: {rec['verdict']}")
    print("=" * 78)
    print("\nDone. Run `streamlit run app.py` for the Model Comparison and SHAP tabs.\n")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
class _ShapWrapper(nn.Module):
    """SHAP needs a single tensor output; our forecasters return (pred, attention)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return self.base(x)[0].unsqueeze(-1)


def run_shap_study(model, Xtr, Xte, Xte_raw, feature_names, y_mean, y_std):
    """
    SHAP for a recurrent sequence model.

    Explainer choice: GradientExplainer (expected gradients). SHAP's
    DeepExplainer has no DeepLIFT rule for LSTM gates and fails on this
    architecture; KernelExplainer on 30x14 = 420 inputs is computationally
    hopeless. GradientExplainer is the one that is both valid and feasible
    here, and the choice is recorded rather than left implicit.

    Aggregation: raw SHAP values are (N, timesteps, features). Because SHAP
    values are ADDITIVE, summing over the time axis gives each feature's
    total contribution to the forecast and preserves the additivity
    guarantee -- so the per-feature values still satisfy
    sum(phi) + base = prediction, which is what makes force/waterfall
    plots legitimate after aggregation.
    """
    try:
        import shap
    except ImportError:
        print("  !! shap not installed -- skipping (pip install shap)")
        return {"available": False, "reason": "shap not installed"}

    set_seed(RANDOM_STATE)
    wrapped = _ShapWrapper(model).eval()
    rng = np.random.default_rng(RANDOM_STATE)

    bg_idx = rng.choice(len(Xtr), size=min(SHAP_N_BACKGROUND, len(Xtr)), replace=False)
    te_idx = np.sort(rng.choice(len(Xte), size=min(SHAP_N_TEST, len(Xte)), replace=False))
    background = torch.from_numpy(Xtr[bg_idx])
    to_explain = torch.from_numpy(Xte[te_idx])

    t0 = time.time()
    try:
        explainer = shap.GradientExplainer(wrapped, background)
        raw = explainer.shap_values(to_explain, nsamples=SHAP_NSAMPLES)
    except Exception as exc:
        print(f"  !! GradientExplainer failed: {exc}")
        return {"available": False, "reason": f"GradientExplainer failed: {exc}"}

    if isinstance(raw, list):
        raw = raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 4:               # (N, T, F, 1) -> (N, T, F)
        raw = raw[..., 0]
    print(f"  - GradientExplainer done in {time.time()-t0:.0f}s, shape {raw.shape}")

    # additive aggregation over the time axis (scaled units -> deg C)
    per_feature = raw.sum(axis=1) * y_std              # (N, F)
    # feature value shown to the user = most recent reading, ORIGINAL units
    feat_display = Xte_raw[te_idx][:, -1, :]           # (N, F)

    with torch.no_grad():
        preds = wrapped(to_explain).squeeze(-1).numpy() * y_std + y_mean
    base_value = float(preds.mean() - per_feature.sum(axis=1).mean())

    expl = shap.Explanation(
        values=per_feature, base_values=np.full(len(per_feature), base_value),
        data=feat_display, feature_names=list(feature_names))

    mean_abs = np.abs(per_feature).mean(axis=0)
    order = np.argsort(-mean_abs)
    print("  - top channels by mean |SHAP|: " + ", ".join(
        f"{feature_names[i]} ({mean_abs[i]:.4f})" for i in order[:3]))

    made = _shap_plots(shap, expl, per_feature, feat_display, feature_names,
                       raw, order, base_value, preds)

    return {
        "available": True,
        "model": SHAP_TARGET_MODEL,
        "explainer": "shap.GradientExplainer (expected gradients)",
        "explainer_rationale": (
            "DeepExplainer has no DeepLIFT rule for LSTM gates; KernelExplainer on "
            "420 inputs is intractable. GradientExplainer is valid and feasible."),
        "aggregation": ("SHAP values summed over the 30 timesteps per channel; "
                        "additivity is preserved so sum(phi)+base = prediction."),
        "units": "deg C contribution to the 15-minute-ahead CHR forecast",
        "n_explained": int(len(te_idx)), "n_background": int(len(bg_idx)),
        "nsamples": SHAP_NSAMPLES,
        "base_value": base_value,
        "feature_names": list(feature_names),
        "mean_abs_shap": mean_abs.tolist(),
        "ranking": [feature_names[i] for i in order],
        "plots": made,
        "values": per_feature,
        "data": feat_display,
        "temporal_abs": np.abs(raw).mean(axis=(0, 2)).tolist(),
    }


def _shap_plots(shap, expl, per_feature, feat_display, feature_names,
                raw, order, base_value, preds):
    """Every SHAP figure the paper needs, each saved as its own PNG."""
    made = []

    def _save(name):
        fig = plt.gcf()
        fig.set_facecolor(PLOT_VOID)
        for ax in fig.get_axes():
            ax.set_facecolor(PLOT_VOID)
            ax.tick_params(colors=PLOT_INK_DIM)
            for sp in ax.spines.values():
                sp.set_color(PLOT_HAIR)
            ax.xaxis.label.set_color(PLOT_INK)
            ax.yaxis.label.set_color(PLOT_INK)
            ax.title.set_color(PLOT_INK)
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_color(PLOT_INK)
        plt.tight_layout()
        plt.savefig(name, facecolor=PLOT_VOID)
        plt.close("all")
        made.append(name)
        print(f"    + {name}")

    # 1. beeswarm (the canonical SHAP summary)
    try:
        plt.figure(figsize=(9, 6))
        shap.plots.beeswarm(expl, max_display=len(feature_names), show=False)
        plt.title("SHAP beeswarm - per-forecast channel contributions", fontsize=11)
        _save("plot_shap_beeswarm.png")
    except Exception as exc:
        print(f"    ! beeswarm failed: {exc}")

    # 2. bar (mean |SHAP|)
    try:
        plt.figure(figsize=(9, 6))
        shap.plots.bar(expl, max_display=len(feature_names), show=False)
        plt.title("SHAP global importance (mean |SHAP|)", fontsize=11)
        _save("plot_shap_bar.png")
    except Exception as exc:
        print(f"    ! bar failed: {exc}")

    # 3. violin
    try:
        plt.figure(figsize=(9, 6))
        shap.plots.violin(expl, max_display=len(feature_names), show=False)
        plt.title("SHAP violin - contribution distribution per channel", fontsize=11)
        _save("plot_shap_violin.png")
    except Exception as exc:
        print(f"    ! violin failed: {exc}")

    # 4. layered violin - shows the value-vs-contribution relationship
    try:
        plt.figure(figsize=(9, 6))
        shap.summary_plot(per_feature, features=feat_display, feature_names=list(feature_names),
                          plot_type="layered_violin", show=False)
        plt.title("SHAP layered violin", fontsize=11)
        _save("plot_shap_layered_violin.png")
    except Exception as exc:
        print(f"    ! layered violin failed: {exc}")

    # 5. heatmap (instances x channels, clustered)
    try:
        plt.figure(figsize=(11, 6))
        shap.plots.heatmap(expl, max_display=len(feature_names), show=False)
        plt.title("SHAP heatmap - contribution structure across test windows", fontsize=11)
        _save("plot_shap_heatmap.png")
    except Exception as exc:
        print(f"    ! heatmap failed: {exc}")

    # 6. dependence plots for the strongest channels
    for rank, fi in enumerate(order[:SHAP_DEPENDENCE_TOP_N], start=1):
        try:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(int(fi), per_feature, feat_display,
                                 feature_names=list(feature_names), show=False,
                                 interaction_index="auto")
            plt.title(f"SHAP dependence #{rank}: {feature_names[fi]}", fontsize=11)
            _save(f"plot_shap_dependence_{rank}_{_slug(feature_names[fi])}.png")
        except Exception as exc:
            print(f"    ! dependence {feature_names[fi]} failed: {exc}")

    # 7. force plot for one representative instance (largest |prediction - base|)
    try:
        pick = int(np.argmax(np.abs(preds - base_value)))
        plt.figure()
        shap.force_plot(base_value, per_feature[pick], feat_display[pick],
                        feature_names=list(feature_names), matplotlib=True, show=False,
                        figsize=(16, 3.4), text_rotation=18)
        _save("plot_shap_force_single.png")
    except Exception as exc:
        print(f"    ! force plot failed: {exc}")

    # 8. waterfall for the same instance
    try:
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(expl[pick], max_display=len(feature_names), show=False)
        plt.title("SHAP waterfall - one forecast decomposed", fontsize=11)
        _save("plot_shap_waterfall.png")
    except Exception as exc:
        print(f"    ! waterfall failed: {exc}")

    # 9. temporal SHAP profile (our own figure: |SHAP| by lookback position)
    try:
        temporal = np.abs(raw).mean(axis=(0, 2))
        lag = np.arange(-(len(temporal) - 1), 1)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.fill_between(lag, temporal, color=MODEL_COLORS[SHAP_TARGET_MODEL], alpha=0.30)
        ax.plot(lag, temporal, linewidth=2, color=MODEL_COLORS[SHAP_TARGET_MODEL])
        ax.set_xlabel("Minutes before forecast origin (0 = most recent reading)")
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title(f"{SHAP_TARGET_MODEL}: where in the 30-minute window the signal comes from",
                     fontsize=11)
        _save("plot_shap_temporal.png")
    except Exception as exc:
        print(f"    ! temporal failed: {exc}")

    return made


def _slug(s):
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")[:40]


if __name__ == "__main__":
    main()
