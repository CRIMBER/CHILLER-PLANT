"""
horizon_sweep.py
----------------
Does the best model depend on how far ahead you forecast?

The 15-minute comparison found Random Forest ahead of LSTM and BiLSTM on
typical-case error. That result is specific to a horizon at which chilled-
water return temperature is dominated by persistence -- conditions that
favour models which lean on the most recent reading. As the horizon
lengthens the autoregressive signal decays, and the question becomes
whether sequence models overtake the tree ensemble once it does.

This module runs the SAME three-way comparison at 15 / 30 / 45 / 60
minutes and reports, per horizon:

    * metrics for Random Forest, LSTM, BiLSTM and the persistence baseline
    * skill score against persistence (the fair cross-horizon yardstick --
      raw RMSE rises with horizon for every model, so comparing raw RMSE
      across horizons tells you about the task, not the model)
    * pairwise Diebold-Mariano tests + moving-block bootstrap CIs
    * any RANK CROSSOVER: the horizon at which the leader changes

Every horizon rebuilds its own sequences, refits its own scaler on
training rows only, re-selects Random Forest hyperparameters on its own
validation split, and retrains both networks. Nothing is carried over
between horizons, so each is an independent experiment.

Run:
    python horizon_sweep.py
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
from sklearn.ensemble import RandomForestRegressor

import deep_models as dm
from comparison_xai import (
    ALPHA,
    BOOTSTRAP_BLOCK,
    MODEL_COLORS,
    PLOT_ALARM,
    PLOT_HAIR,
    PLOT_INK,
    PLOT_INK_DIM,
    PLOT_VOID,
    REFERENCE_COLORS,
    RF_GRID,
    block_bootstrap_ci,
    diebold_mariano,
    full_metrics,
)
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
    RANDOM_STATE,
    TARGET_COL,
    TIMESTAMP_COL,
    clean_data,
    detect_interval_minutes,
    engineer_features,
    load_data,
)

warnings.filterwarnings("ignore")

HORIZONS = [15, 30, 45, 60]
MODELS = ["RandomForest", "LSTM", "BiLSTM"]
BOOTSTRAP_B = 1000          # halved vs the single-horizon study: 4x the work here

ARTIFACT_PATH = "horizon_artifact.pkl"
METRICS_PATH = "metrics_horizon.json"
RESULTS_CSV = "horizon_results.csv"
PAIRWISE_CSV = "horizon_pairwise.csv"

plt.rcParams.update({
    "figure.facecolor": PLOT_VOID, "axes.facecolor": PLOT_VOID, "savefig.facecolor": PLOT_VOID,
    "axes.edgecolor": PLOT_HAIR, "axes.labelcolor": PLOT_INK,
    "axes.grid": True, "grid.color": PLOT_HAIR, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": PLOT_INK_DIM, "ytick.color": PLOT_INK_DIM, "text.color": PLOT_INK,
    "legend.facecolor": "#141416", "legend.edgecolor": PLOT_HAIR, "legend.labelcolor": PLOT_INK,
    "font.family": "monospace", "axes.titlecolor": PLOT_INK, "figure.dpi": 150,
})


def color_of(name):
    return MODEL_COLORS.get(name, REFERENCE_COLORS.get(name, PLOT_INK_DIM))


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """
    Percentage reduction in RMSE relative to persistence.

    Raw RMSE grows with horizon for every model, so a raw-RMSE line chart
    across horizons mostly measures how hard each horizon is. Skill score
    divides that out and answers the question actually being asked: how
    much value does this model add over doing nothing?
    """
    return (1.0 - model_rmse / baseline_rmse) * 100.0


def run_one_horizon(df, horizon_min, interval, seq_len=dm.SEQ_LEN):
    """One complete, self-contained experiment at a single horizon."""
    horizon_rows = int(round(horizon_min / interval))
    set_seed(RANDOM_STATE)

    X, y, y_now, ends = build_sequences(
        df, dm.SEQ_FEATURE_COLS, TARGET_COL, seq_len, horizon_rows, interval)
    n = len(X)
    dev_end = int(n * dm.DEV_FRAC)
    train_end = int(dev_end * (1 - dm.VAL_FRAC_OF_DEV))

    Xtr_raw, ytr = X[:train_end], y[:train_end]
    Xva_raw, yva = X[train_end:dev_end], y[train_end:dev_end]
    Xte_raw, yte = X[dev_end:], y[dev_end:]
    yte_now = y_now[dev_end:]

    scaler = SeqScaler().fit(Xtr_raw)                    # TRAIN ONLY, per horizon
    Xtr, Xva, Xte = (scaler.transform(a) for a in (Xtr_raw, Xva_raw, Xte_raw))
    y_mean, y_std = float(ytr.mean()), float(ytr.std())

    out = {"horizon_minutes": horizon_min, "horizon_rows": horizon_rows,
           "n_test": int(len(Xte)), "models": {}, "errors": {}}

    baseline = full_metrics(yte, yte_now)
    out["baseline"] = baseline
    out["errors"]["Persistence"] = yte - yte_now
    print(f"    persistence      RMSE={baseline['RMSE']:.4f}  MAE={baseline['MAE']:.4f}")

    # --- Random Forest (hyperparameters re-selected at THIS horizon) ---
    Xtr_f, Xva_f, Xte_f = (a.reshape(len(a), -1) for a in (Xtr, Xva, Xte))
    best_rf, best_cfg, best_val = None, None, np.inf
    for cfg in RF_GRID:
        rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **cfg)
        rf.fit(Xtr_f, ytr)
        v = float(np.mean(np.abs(yva - rf.predict(Xva_f))))
        if v < best_val:
            best_rf, best_cfg, best_val = rf, cfg, v
    pred = best_rf.predict(Xte_f)
    out["models"]["RandomForest"] = {
        "metrics": full_metrics(yte, pred), "val_MAE": best_val, "config": best_cfg,
        "skill_score": skill_score(full_metrics(yte, pred)["RMSE"], baseline["RMSE"]),
    }
    out["errors"]["RandomForest"] = yte - pred
    m = out["models"]["RandomForest"]["metrics"]
    print(f"    RandomForest     RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"skill={out['models']['RandomForest']['skill_score']:+.1f}%")

    # --- LSTM / BiLSTM ---
    for name, cls in [("LSTM", LSTMForecaster), ("BiLSTM", BiLSTMForecaster)]:
        info = train_model(name, cls, Xtr, ytr, Xva, yva, y_mean, y_std, verbose=False)
        ps, _ = predict(info["model"], torch.from_numpy(Xte))
        p = ps * y_std + y_mean
        mm = full_metrics(yte, p)
        out["models"][name] = {
            "metrics": mm, "val_MAE": info["best_val_MAE"],
            "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
            "n_params": info["n_params"],
            "skill_score": skill_score(mm["RMSE"], baseline["RMSE"]),
        }
        out["errors"][name] = yte - p
        print(f"    {name:<16} RMSE={mm['RMSE']:.4f}  MAE={mm['MAE']:.4f}  "
              f"skill={out['models'][name]['skill_score']:+.1f}%  (epoch {info['best_epoch']})")

    # --- pairwise significance at this horizon ---
    pairs = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            ea, eb = out["errors"][a], out["errors"][b]
            dm_sq = diebold_mariano(ea, eb, horizon_rows, "squared")
            dm_ab = diebold_mariano(ea, eb, horizon_rows, "absolute")
            bs_r = block_bootstrap_ci(ea, eb, "RMSE", b=BOOTSTRAP_B, block=BOOTSTRAP_BLOCK)
            bs_m = block_bootstrap_ci(ea, eb, "MAE", b=BOOTSTRAP_B, block=BOOTSTRAP_BLOCK)
            rmse_sig = bool(dm_sq[1] < ALPHA and bs_r["excludes_zero"])
            mae_sig = bool(dm_ab[1] < ALPHA and bs_m["excludes_zero"])
            better_rmse = a if out["models"][a]["metrics"]["RMSE"] < out["models"][b]["metrics"]["RMSE"] else b
            better_mae = a if out["models"][a]["metrics"]["MAE"] < out["models"][b]["metrics"]["MAE"] else b
            pairs.append({
                "model_a": a, "model_b": b,
                "dm_squared_stat": dm_sq[0], "dm_squared_p": dm_sq[1],
                "dm_absolute_stat": dm_ab[0], "dm_absolute_p": dm_ab[1],
                "bootstrap_RMSE": bs_r, "bootstrap_MAE": bs_m,
                "rmse_winner": better_rmse, "rmse_significant": rmse_sig,
                "mae_winner": better_mae, "mae_significant": mae_sig,
                "verdict": (f"{better_rmse} better on RMSE" if rmse_sig else
                            f"{better_mae} better on MAE" if mae_sig else
                            "no significant difference"),
            })
    out["pairwise"] = pairs
    return out


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def plot_rmse_vs_horizon(sweep):
    hs = [s["horizon_minutes"] for s in sweep]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(hs, [s["baseline"]["RMSE"] for s in sweep], linewidth=2, linestyle="--",
            color=REFERENCE_COLORS["Persistence"], marker="o", markersize=7, label="Persistence")
    for name in MODELS:
        ax.plot(hs, [s["models"][name]["metrics"]["RMSE"] for s in sweep],
                linewidth=2.4, marker="o", markersize=8, color=color_of(name), label=name)
    ax.set_xticks(hs)
    ax.set_xlabel("Forecast horizon (minutes ahead)")
    ax.set_ylabel("Final-Test RMSE (deg C)")
    ax.set_title("Raw error grows with horizon for every model\n"
                 "(use the skill-score chart to compare models across horizons)", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("plot_hz_rmse.png")
    plt.close()


def plot_skill_vs_horizon(sweep):
    """The headline figure: value added over persistence, per horizon."""
    hs = [s["horizon_minutes"] for s in sweep]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.axhline(0, color=REFERENCE_COLORS["Persistence"], linewidth=1.6, linestyle="--")
    ax.text(hs[0], 0.4, "persistence baseline (no skill)", fontsize=7.5,
            color=REFERENCE_COLORS["Persistence"], va="bottom")
    for name in MODELS:
        vals = [s["models"][name]["skill_score"] for s in sweep]
        ax.plot(hs, vals, linewidth=2.6, marker="o", markersize=9, color=color_of(name), label=name)
        ax.annotate(f"{vals[-1]:+.1f}%", (hs[-1], vals[-1]), textcoords="offset points",
                    xytext=(9, 0), fontsize=8.5, color=color_of(name), va="center")
    ax.set_xticks(hs)
    ax.set_xlim(hs[0] - 3, hs[-1] + 9)
    ax.set_xlabel("Forecast horizon (minutes ahead)")
    ax.set_ylabel("Skill score vs persistence (% RMSE reduction)")
    ax.set_title("Skill score by horizon - higher is better\n"
                 "Where lines cross, the best model changes", fontsize=11)
    ax.legend(fontsize=8.5, loc="best")
    plt.tight_layout()
    plt.savefig("plot_hz_skill.png")
    plt.close()


def plot_rank_bump(sweep):
    """
    Bump chart: each model's RANK at each horizon.

    Rank is the right encoding for 'who is winning where' -- it makes a
    crossover unmissable in a way that overlapping metric lines cannot.
    """
    hs = [s["horizon_minutes"] for s in sweep]
    ranks = {m: [] for m in MODELS}
    for s in sweep:
        order = sorted(MODELS, key=lambda m: s["models"][m]["metrics"]["RMSE"])
        for pos, m in enumerate(order, start=1):
            ranks[m].append(pos)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in MODELS:
        ax.plot(hs, ranks[name], linewidth=3, marker="o", markersize=13,
                color=color_of(name), label=name, markeredgecolor=PLOT_VOID, markeredgewidth=2)
        for h, r in zip(hs, ranks[name]):
            ax.text(h, r, str(r), ha="center", va="center", fontsize=8,
                    color=PLOT_VOID, fontweight="bold")
    ax.set_xticks(hs)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1st", "2nd", "3rd"])
    ax.invert_yaxis()
    ax.set_xlabel("Forecast horizon (minutes ahead)")
    ax.set_title("Rank by Final-Test RMSE at each horizon", fontsize=11)
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig("plot_hz_rank_bump.png")
    plt.close()


def plot_significance_grid(sweep):
    """Which comparisons are actually resolved, at which horizon."""
    hs = [s["horizon_minutes"] for s in sweep]
    pair_labels = [f"{p['model_a']}\nvs {p['model_b']}" for p in sweep[0]["pairwise"]]
    n_pairs, n_h = len(pair_labels), len(hs)

    grid = np.zeros((n_pairs, n_h))
    ann = np.empty((n_pairs, n_h), dtype=object)
    for j, s in enumerate(sweep):
        for i, p in enumerate(s["pairwise"]):
            if p["rmse_significant"] or p["mae_significant"]:
                winner = p["rmse_winner"] if p["rmse_significant"] else p["mae_winner"]
                grid[i, j] = 1 if winner == p["model_a"] else -1
                ann[i, j] = f"{winner}\n({'RMSE' if p['rmse_significant'] else 'MAE'})"
            else:
                grid[i, j] = 0
                ann[i, j] = "n.s."

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.imshow(grid, cmap="RdYlGn", vmin=-1.4, vmax=1.4, aspect="auto")
    ax.set_xticks(range(n_h)); ax.set_xticklabels([f"{h} min" for h in hs], fontsize=9)
    ax.set_yticks(range(n_pairs)); ax.set_yticklabels(pair_labels, fontsize=8.5)
    for i in range(n_pairs):
        for j in range(n_h):
            ax.text(j, i, ann[i, j], ha="center", va="center", fontsize=7.5,
                    color="#101010", fontweight="bold")
    ax.set_title("Significant winner by horizon  (n.s. = not statistically separable)\n"
                 "DM p<0.05 AND bootstrap CI excluding zero", fontsize=10.5)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig("plot_hz_significance_grid.png")
    plt.close()


def plot_metric_small_multiples(sweep):
    hs = [s["horizon_minutes"] for s in sweep]
    keys = [("RMSE", "RMSE (deg C)"), ("MAE", "MAE (deg C)"),
            ("R2", "R-squared"), ("P90AE", "P90 abs. error (deg C)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, (key, label) in zip(axes, keys):
        ax.plot(hs, [s["baseline"][key] for s in sweep], linewidth=1.6, linestyle="--",
                color=REFERENCE_COLORS["Persistence"], marker="o", markersize=5, label="Persistence")
        for name in MODELS:
            ax.plot(hs, [s["models"][name]["metrics"][key] for s in sweep],
                    linewidth=2, marker="o", markersize=6, color=color_of(name), label=name)
        ax.set_xticks(hs)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Horizon (min)")
    axes[0].legend(fontsize=7)
    fig.suptitle("All metrics across horizons", fontsize=11)
    plt.tight_layout()
    plt.savefig("plot_hz_small_multiples.png")
    plt.close()


def plot_error_violin_by_horizon(sweep):
    """Error distributions side by side, grouped by horizon."""
    hs = [s["horizon_minutes"] for s in sweep]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    width, gap = 0.24, 1.0
    all_cut = []
    for gi, name in enumerate(MODELS):
        positions = [k * gap + (gi - 1) * width for k in range(len(hs))]
        data = [np.abs(s["errors"][name]) for s in sweep]
        all_cut += [np.percentile(d, 99) for d in data]
        parts = ax.violinplot(data, positions=positions, widths=width * 0.92, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color_of(name)); body.set_edgecolor(PLOT_VOID)
            body.set_linewidth(1.5); body.set_alpha(0.85)
        for pos, d in zip(positions, data):
            ax.scatter([pos], [np.median(d)], s=14, color=PLOT_INK, zorder=5)
        ax.plot([], [], linewidth=8, color=color_of(name), label=name)   # legend proxy
    cut = max(all_cut)
    ax.set_ylim(0, cut * 1.1)
    ax.set_xticks([k * gap for k in range(len(hs))])
    ax.set_xticklabels([f"{h} min" for h in hs])
    ax.set_ylabel("Absolute error (deg C)")
    ax.set_title(f"Error distribution by model and horizon "
                 f"(dot = median; view clipped at {cut:.3f} deg C, 99th pct)", fontsize=11)
    ax.legend(fontsize=8.5, ncol=3)
    plt.tight_layout()
    plt.savefig("plot_hz_violin_by_horizon.png")
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("HORIZON SWEEP - Random Forest / LSTM / BiLSTM at 15/30/45/60 minutes")
    print("=" * 78)
    set_seed(RANDOM_STATE)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    print("\nPreparing data...")
    df = engineer_features(clean_data(load_data()[0], verbose=False))
    interval = detect_interval_minutes(df)
    df = df.dropna(subset=dm.SEQ_FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"  rows={len(df)}  interval={interval:.2f} min  lookback={dm.SEQ_LEN} rows")

    sweep = []
    for h in HORIZONS:
        print(f"\n--- horizon {h} min " + "-" * 50)
        t0 = time.time()
        res = run_one_horizon(df, h, interval)
        print(f"    ({time.time()-t0:.0f}s, n_test={res['n_test']})")
        for p in res["pairwise"]:
            print(f"      {p['model_a']} vs {p['model_b']}: {p['verdict']}")
        sweep.append(res)

    # ---- crossover analysis ----
    print("\n" + "=" * 78)
    print("CROSSOVER ANALYSIS")
    print("=" * 78)
    leaders = []
    for s in sweep:
        order = sorted(MODELS, key=lambda m: s["models"][m]["metrics"]["RMSE"])
        leaders.append(order[0])
        print(f"  {s['horizon_minutes']:>3} min: " +
              " > ".join(f"{m} ({s['models'][m]['metrics']['RMSE']:.4f})" for m in order))

    crossovers = []
    for i in range(1, len(HORIZONS)):
        if leaders[i] != leaders[i - 1]:
            crossovers.append({"between_minutes": [HORIZONS[i - 1], HORIZONS[i]],
                               "from": leaders[i - 1], "to": leaders[i]})
            print(f"  ** RANK CROSSOVER between {HORIZONS[i-1]} and {HORIZONS[i]} min: "
                  f"{leaders[i-1]} -> {leaders[i]}")
    if not crossovers:
        print(f"  No rank crossover: {leaders[0]} leads at every horizon tested.")

    bilstm_beats_rf = [HORIZONS[i] for i, s in enumerate(sweep)
                       if s["models"]["BiLSTM"]["metrics"]["RMSE"]
                       < s["models"]["RandomForest"]["metrics"]["RMSE"]]
    bilstm_sig = []
    for i, s in enumerate(sweep):
        for p in s["pairwise"]:
            if {p["model_a"], p["model_b"]} == {"RandomForest", "BiLSTM"}:
                if (p["rmse_significant"] and p["rmse_winner"] == "BiLSTM") or \
                   (p["mae_significant"] and p["mae_winner"] == "BiLSTM"):
                    bilstm_sig.append(HORIZONS[i])
    print(f"\n  BiLSTM numerically ahead of Random Forest at: "
          f"{bilstm_beats_rf if bilstm_beats_rf else 'no horizon'}")
    print(f"  BiLSTM SIGNIFICANTLY ahead of Random Forest at: "
          f"{bilstm_sig if bilstm_sig else 'no horizon'}")

    # ---- plots ----
    print("\nFigures...")
    plot_rmse_vs_horizon(sweep)
    plot_skill_vs_horizon(sweep)
    plot_rank_bump(sweep)
    plot_significance_grid(sweep)
    plot_metric_small_multiples(sweep)
    plot_error_violin_by_horizon(sweep)
    print("  saved plot_hz_*.png (6)")

    # ---- save ----
    artifact = {
        "horizons": HORIZONS, "models": MODELS, "seq_len": dm.SEQ_LEN,
        "selection_criterion": f"validation {dm.EARLY_STOP_METRIC}",
        "leaders_by_horizon": dict(zip(HORIZONS, leaders)),
        "crossovers": crossovers,
        "bilstm_beats_rf_numerically_at": bilstm_beats_rf,
        "bilstm_beats_rf_significantly_at": bilstm_sig,
        "sweep": [{k: v for k, v in s.items() if k != "errors"} for s in sweep],
        "note": ("Each horizon is an independent experiment: own sequences, own "
                 "train-only scaler, own RF hyperparameter selection, own network "
                 "training. Skill score is the cross-horizon yardstick because raw "
                 "RMSE rises with horizon for every model."),
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    rows = []
    for s in sweep:
        for name in MODELS:
            rows.append({"horizon_min": s["horizon_minutes"], "model": name,
                         **s["models"][name]["metrics"],
                         "skill_score_pct": s["models"][name]["skill_score"]})
        rows.append({"horizon_min": s["horizon_minutes"], "model": "Persistence",
                     **s["baseline"], "skill_score_pct": 0.0})
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    prows = []
    for s in sweep:
        for p in s["pairwise"]:
            prows.append({
                "horizon_min": s["horizon_minutes"], "model_a": p["model_a"], "model_b": p["model_b"],
                "rmse_winner": p["rmse_winner"], "rmse_significant": p["rmse_significant"],
                "dm_squared_p": p["dm_squared_p"],
                "rmse_ci_low": p["bootstrap_RMSE"]["ci_low"], "rmse_ci_high": p["bootstrap_RMSE"]["ci_high"],
                "mae_winner": p["mae_winner"], "mae_significant": p["mae_significant"],
                "dm_absolute_p": p["dm_absolute_p"],
                "mae_ci_low": p["bootstrap_MAE"]["ci_low"], "mae_ci_high": p["bootstrap_MAE"]["ci_high"],
                "verdict": p["verdict"],
            })
    pd.DataFrame(prows).to_csv(PAIRWISE_CSV, index=False)
    print(f"  saved {ARTIFACT_PATH} / {METRICS_PATH} / {RESULTS_CSV} / {PAIRWISE_CSV}")

    # ---- report ----
    print("\n" + "=" * 78)
    print("FINAL REPORT - SKILL SCORE vs PERSISTENCE (%RMSE reduction, higher better)")
    print("=" * 78)
    print(f"{'Horizon':<12}" + "".join(f"{m:>16}" for m in MODELS))
    print("-" * 78)
    for s in sweep:
        print(f"{str(s['horizon_minutes'])+' min':<12}" +
              "".join(f"{s['models'][m]['skill_score']:>15.1f}%" for m in MODELS))
    print("=" * 78)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
