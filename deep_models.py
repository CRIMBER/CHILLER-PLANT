"""
deep_models.py
--------------
Chiller Plant — Deep Sequence Models + Explainable AI (XAI)

Extends the existing Polynomial Regression study (train.py) with six
deep learning architectures for the SAME forecasting question:

    "Forecast chilled-water return temperature (CHR) 15 minutes ahead
     from historical chiller-plant / BMS measurements."

                         BMS DATA (44k rows @ 1 min)
                                   |
                       Cleaning + Feature Engineering
                            (reused from train.py)
                                   |
                     Sliding-window sequence tensors
                        (B, SEQ_LEN, n_features)
                                   |
        -------------------------------------------------------
        |        |        |        |          |               |
     SimpleRNN  LSTM   BiLSTM    GRU      1D-CNN        CNN-LSTM
        |        |        |        |          |               |
        -------------------------------------------------------
                                   |
                     Explainable AI (3 methods)
              Permutation Importance / Integrated Gradients /
                   Temporal Attention + Saliency
                                   |
                              Dashboard

METHODOLOGY NOTES (these matter for the write-up)
-------------------------------------------------
* Same chronological Development/Final-Test split as the CHR research
  pipeline in train.py (DEV_FRAC = 0.85). The Final Test set is touched
  exactly once, at the very end, for every model.
* Early stopping uses a chronological validation slice taken from the END
  of the Development set. The test set never influences training or model
  selection.
* Scalers are fit on the TRAINING rows only, then applied to val/test.
  Fitting a scaler on the full series is a classic leakage bug; it is
  avoided here deliberately.
* Windows that straddle a gap in the timestamp index are DROPPED, not
  silently stitched together. A 30-minute lookback is only valid if all
  30 minutes are actually present and contiguous.
* Every model is compared on the IDENTICAL set of test windows, against
  the identical persistence baseline, so the numbers are commensurable.

This is the FORECASTING layer only. Nothing here controls plant
equipment.

Run:
    python deep_models.py
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

from train import (
    DEV_FRAC,
    PRIMARY_HORIZON_MINUTES,
    RANDOM_STATE,
    TARGET_COL,
    TIMESTAMP_COL,
    BASE_FEATURE_COLS,
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
SEQ_LEN = 30                  # lookback window, in rows (= 30 min @ 1-min data)
VAL_FRAC_OF_DEV = 0.15        # chronological tail of Development -> early-stopping val
BATCH_SIZE = 256
MAX_EPOCHS = 40
PATIENCE = 6                  # early-stopping patience, in epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
HIDDEN_SIZE = 64
CNN_CHANNELS = 64
DROPOUT = 0.1

# XAI configuration
XAI_SAMPLE_SIZE = 1024        # test windows used for gradient-based attribution
IG_STEPS = 50                 # Riemann steps for Integrated Gradients
PERM_REPEATS = 3              # repeats per feature for permutation importance

DEEP_DIR = "deep_models"
DEEP_ARTIFACT_PATH = "deep_artifact.pkl"
DEEP_METRICS_PATH = "metrics_deep.json"
DEEP_EXPERIMENT_PATH = "deep_experiment_results.csv"

DEVICE = torch.device("cpu")

# Sequence feature channels.
#
# NOTE ON FEATURE PARITY WITH THE POLYNOMIAL MODEL:
# train.py's CHR feature set is 13 "static" features + 4 explicit CHR lag
# columns (lag 1/5/10/15). A sequence model does not need hand-built lag
# columns -- the lookback window IS the lag structure, at every timestep,
# not just four hand-picked ones. So the channels below are the same 13
# static features PLUS the raw CHR channel, and the recurrent/convolutional
# layers learn their own temporal weighting over the 30-minute window.
# The comparison against Polynomial Regression therefore uses the same
# underlying measurements, presented in the form each model class expects.
SEQ_FEATURE_COLS = BASE_FEATURE_COLS + [
    "WetBulb_Avg", "Hour", "DayOfWeek", "IsWeekend", TARGET_COL,
]

# Plot palette — identical to train.py so the dashboard reads as one system.
PLOT_VOID = "#0d0d0f"
PLOT_INK = "#e6e6ea"
PLOT_INK_DIM = "#8a8a92"
PLOT_HAIR = "#2a2a30"
PLOT_CHROME = "#c9c9d1"
PLOT_ALARM = "#d9695c"

plt.rcParams.update({
    "figure.facecolor": PLOT_VOID, "axes.facecolor": PLOT_VOID, "savefig.facecolor": PLOT_VOID,
    "axes.edgecolor": PLOT_HAIR, "axes.labelcolor": PLOT_INK, "axes.grid": True,
    "grid.color": PLOT_HAIR, "grid.linewidth": 0.6,
    "xtick.color": PLOT_INK_DIM, "ytick.color": PLOT_INK_DIM, "text.color": PLOT_INK,
    "legend.facecolor": "#141416", "legend.edgecolor": PLOT_HAIR, "legend.labelcolor": PLOT_INK,
    "font.family": "monospace",
    "axes.titlecolor": PLOT_INK,
})


def set_seed(seed: int = RANDOM_STATE):
    """Full reproducibility across numpy + torch (CPU is deterministic here)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # not needed on CPU for these ops


# ---------------------------------------------------------------------------
# SEQUENCE CONSTRUCTION (leakage-safe)
# ---------------------------------------------------------------------------
def _contiguity_mask(df: pd.DataFrame, interval_min: float, seq_len: int, horizon_rows: int) -> np.ndarray:
    """
    True for every row t that can serve as the END of a valid window.

    A window ending at t is valid only if:
      * rows [t-seq_len+1 .. t] are consecutive at the sampling interval, AND
      * rows [t .. t+horizon_rows] are consecutive (so the target at
        t+horizon_rows really is 15 minutes after t, not across a gap).

    Cleaning in train.py drops rows, so gaps genuinely exist. Stitching
    across them would silently fabricate a lookback window that never
    happened.
    """
    n = len(df)
    step_min = df[TIMESTAMP_COL].diff().dt.total_seconds().to_numpy() / 60.0
    good_step = np.isclose(step_min, interval_min, atol=1e-6)
    good_step[0] = False  # first row has no predecessor

    # cum[i] = number of good steps in rows [0..i]
    cum = np.concatenate([[0], np.cumsum(good_step.astype(np.int64))])

    ends = np.arange(n)
    valid = np.zeros(n, dtype=bool)

    lo = seq_len - 1
    hi = n - horizon_rows - 1
    if hi < lo:
        return valid
    t = ends[lo:hi + 1]

    # steps inside the lookback window: indices t-seq_len+2 .. t  -> seq_len-1 steps
    look_ok = (cum[t + 1] - cum[t - seq_len + 2]) == (seq_len - 1)
    # steps from t to t+horizon_rows: indices t+1 .. t+horizon_rows -> horizon_rows steps
    fwd_ok = (cum[t + horizon_rows + 1] - cum[t + 1]) == horizon_rows

    valid[t] = look_ok & fwd_ok
    return valid


def build_sequences(df: pd.DataFrame, feature_cols: list, target_col: str,
                    seq_len: int, horizon_rows: int, interval_min: float):
    """
    Returns
        X       (N, seq_len, n_features) float32 -- lookback windows
        y       (N,)                     float32 -- CHR at t + horizon
        y_now   (N,)                     float32 -- CHR at t (persistence baseline)
        ends    (N,)                     int64   -- row index t of each window's end
    """
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)

    valid = _contiguity_mask(df, interval_min, seq_len, horizon_rows)
    ends = np.nonzero(valid)[0]

    # sliding_window_view gives (n-seq_len+1, n_features, seq_len) as a view
    windows = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0)
    X = np.ascontiguousarray(windows[ends - seq_len + 1].transpose(0, 2, 1))

    y = target[ends + horizon_rows]
    y_now = target[ends]
    return X, y, y_now, ends


class SeqScaler:
    """Standardises (N, T, F) tensors per feature channel. Fit on train only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # constant channel -> no scaling, no div-by-zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)


# ---------------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention pooling over the time axis.

    Serves two purposes: it gives the recurrent models a learned way to
    weight the 30-minute window, and it exposes those weights directly as
    an explanation ("which minutes in the lookback drove this forecast?").
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, H):                       # H: (B, T, h)
        s = self.score(torch.tanh(self.proj(H))).squeeze(-1)   # (B, T)
        a = torch.softmax(s, dim=1)                            # (B, T)
        ctx = torch.bmm(a.unsqueeze(1), H).squeeze(1)          # (B, h)
        return ctx, a


class _RecurrentForecaster(nn.Module):
    """Shared scaffold for SimpleRNN / LSTM / BiLSTM / GRU."""

    def __init__(self, n_features, rnn):
        super().__init__()
        self.rnn = rnn
        out_h = HIDDEN_SIZE * (2 if getattr(rnn, "bidirectional", False) else 1)
        self.attn = TemporalAttention(out_h)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(out_h, 1)

    def forward(self, x):                       # x: (B, T, F)
        H, _ = self.rnn(x)                      # (B, T, h)
        ctx, a = self.attn(H)
        return self.head(self.drop(ctx)).squeeze(-1), a


class SimpleRNNForecaster(_RecurrentForecaster):
    def __init__(self, n_features):
        super().__init__(n_features, nn.RNN(
            n_features, HIDDEN_SIZE, batch_first=True, nonlinearity="tanh"))


class LSTMForecaster(_RecurrentForecaster):
    def __init__(self, n_features):
        super().__init__(n_features, nn.LSTM(
            n_features, HIDDEN_SIZE, batch_first=True))


class BiLSTMForecaster(_RecurrentForecaster):
    def __init__(self, n_features):
        super().__init__(n_features, nn.LSTM(
            n_features, HIDDEN_SIZE, batch_first=True, bidirectional=True))


class GRUForecaster(_RecurrentForecaster):
    def __init__(self, n_features):
        super().__init__(n_features, nn.GRU(
            n_features, HIDDEN_SIZE, batch_first=True))


class CNN1DForecaster(nn.Module):
    """
    Dilated 1D convolutional encoder + global pooling.

    Architecturally distinct from the recurrent family: no hidden state,
    no attention head. Its temporal explanation comes from Integrated
    Gradients rather than attention weights, which is exactly why it is
    worth including in an XAI comparison.
    """

    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, CNN_CHANNELS, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(CNN_CHANNELS * 2, 1)   # max-pool ++ avg-pool

    def forward(self, x):                       # (B, T, F)
        h = self.conv(x.transpose(1, 2))        # (B, C, T)
        pooled = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)
        return self.head(self.drop(pooled)).squeeze(-1), None


class CNNLSTMForecaster(nn.Module):
    """Convolutional feature extractor feeding an attention-pooled LSTM."""

    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(CNN_CHANNELS, HIDDEN_SIZE, batch_first=True)
        self.attn = TemporalAttention(HIDDEN_SIZE)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, T, C)
        H, _ = self.lstm(h)
        ctx, a = self.attn(H)
        return self.head(self.drop(ctx)).squeeze(-1), a


MODEL_REGISTRY = {
    "SimpleRNN": (SimpleRNNForecaster, "Vanilla recurrent network (tanh) with attention pooling"),
    "LSTM": (LSTMForecaster, "Long Short-Term Memory with attention pooling"),
    "BiLSTM": (BiLSTMForecaster, "Bidirectional LSTM with attention pooling"),
    "GRU": (GRUForecaster, "Gated Recurrent Unit with attention pooling"),
    "CNN1D": (CNN1DForecaster, "Dilated 1D convolutional encoder, global max+avg pooling"),
    "CNN_LSTM": (CNNLSTMForecaster, "1D-CNN feature extractor feeding an attention-pooled LSTM"),
}


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def _batches(n, batch_size, shuffle, rng=None):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def predict(model, X: torch.Tensor, batch_size: int = 1024, want_attention: bool = False):
    """Batched inference. Returns (predictions, attention_or_None) as numpy."""
    model.eval()
    preds, attns = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out, a = model(X[i:i + batch_size])
            preds.append(out.cpu().numpy())
            if want_attention and a is not None:
                attns.append(a.cpu().numpy())
    pred = np.concatenate(preds)
    attn = np.concatenate(attns) if attns else None
    return pred, attn


def train_model(name, model_cls, Xtr, ytr, Xva, yva, y_mean, y_std, verbose=True):
    """
    Trains one architecture with Adam + early stopping on validation RMSE
    (in ORIGINAL units, so the number is directly comparable to every
    other model in the study and to the polynomial baseline).
    """
    set_seed(RANDOM_STATE)
    model = model_cls(Xtr.shape[-1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(RANDOM_STATE)

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(((ytr - y_mean) / y_std).astype(np.float32))
    Xva_t = torch.from_numpy(Xva)

    best_rmse, best_state, best_epoch, bad_epochs = np.inf, None, 0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss, seen = 0.0, 0
        for bidx in _batches(len(Xtr_t), BATCH_SIZE, shuffle=True, rng=rng):
            xb, yb = Xtr_t[bidx], ytr_t[bidx]
            opt.zero_grad()
            out, _ = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_loss += float(loss) * len(bidx)
            seen += len(bidx)

        val_pred_scaled, _ = predict(model, Xva_t)
        val_pred = val_pred_scaled * y_std + y_mean
        val_rmse = float(np.sqrt(np.mean((yva - val_pred) ** 2)))
        val_mae = float(np.mean(np.abs(yva - val_pred)))
        history.append({
            "epoch": epoch,
            "train_loss_scaled": epoch_loss / max(seen, 1),
            "val_RMSE": val_rmse,
            "val_MAE": val_mae,
        })

        if val_rmse < best_rmse - 1e-6:
            best_rmse, best_epoch, bad_epochs = val_rmse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or bad_epochs >= PATIENCE):
            print(f"      epoch {epoch:>3}/{MAX_EPOCHS}  train_mse={history[-1]['train_loss_scaled']:.5f}  "
                  f"val_RMSE={val_rmse:.5f}  (best {best_rmse:.5f} @ {best_epoch})")

        if bad_epochs >= PATIENCE:
            if verbose:
                print(f"      early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "name": name,
        "n_params": int(n_params),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_RMSE": float(best_rmse),
        "epochs_run": len(history),
        "train_seconds": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# EXPLAINABLE AI
# ---------------------------------------------------------------------------
def permutation_importance(model, X: np.ndarray, y: np.ndarray, y_mean, y_std,
                           feature_names, n_repeats=PERM_REPEATS, seed=RANDOM_STATE):
    """
    Model-agnostic permutation importance for sequence models.

    For each input channel, the channel's entire time series is shuffled
    ACROSS samples (the within-window temporal structure is preserved, the
    association with the target is destroyed). Importance = increase in
    test RMSE, in degrees C. Averaged over n_repeats shuffles.

    Model-agnostic is the point: this is the one XAI method that produces
    directly comparable numbers across all six architectures AND the
    polynomial model.
    """
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(X)
    base_pred, _ = predict(model, Xt)
    base_rmse = float(np.sqrt(np.mean((y - (base_pred * y_std + y_mean)) ** 2)))

    results = []
    for f_i, fname in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, :, f_i] = Xp[rng.permutation(len(Xp)), :, f_i]
            pred, _ = predict(model, torch.from_numpy(Xp))
            rmse = float(np.sqrt(np.mean((y - (pred * y_std + y_mean)) ** 2)))
            deltas.append(rmse - base_rmse)
        results.append({
            "feature": fname,
            "importance_rmse_increase": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
        })

    results.sort(key=lambda r: -r["importance_rmse_increase"])
    return {"baseline_rmse": base_rmse, "importances": results}


def integrated_gradients(model, X: np.ndarray, steps=IG_STEPS, batch_size=128):
    """
    Integrated Gradients (Sundararajan et al., 2017).

    Baseline = the all-zeros window, which in standardised space is the
    per-channel training mean -- i.e. "a completely average half-hour of
    plant operation". Attribution for input i is

        (x_i - b_i) * average gradient along the straight path b -> x

    Returns a (N, T, F) attribution array in the model's scaled output
    units, plus per-feature and per-timestep aggregates of |attribution|.
    """
    model.eval()
    attributions = np.zeros_like(X, dtype=np.float32)
    alphas = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)

    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        baseline = torch.zeros_like(xb)
        diff = xb - baseline
        total_grad = torch.zeros_like(xb)

        for a in alphas:
            point = (baseline + float(a) * diff).clone().requires_grad_(True)
            out, _ = model(point)
            grad = torch.autograd.grad(out.sum(), point)[0]
            total_grad += grad

        attributions[i:i + batch_size] = (diff * total_grad / steps).detach().numpy()

    abs_attr = np.abs(attributions)
    return {
        "attributions": attributions,
        "per_feature": abs_attr.mean(axis=(0, 1)),   # (F,)
        "per_timestep": abs_attr.mean(axis=(0, 2)),  # (T,)
    }


def temporal_explanation(model, X: np.ndarray, ig_per_timestep: np.ndarray):
    """
    "Which minutes of the 30-minute lookback drove the forecast?"

    Attention-pooled models answer this directly with their attention
    weights. The pure 1D-CNN has no attention, so its temporal profile
    falls back to the Integrated-Gradients per-timestep aggregate. Both
    are reported with an explicit `source` field -- never presented as if
    they were the same quantity.
    """
    _, attn = predict(model, torch.from_numpy(X), want_attention=True)
    if attn is not None:
        return {
            "source": "attention",
            "weights": attn.mean(axis=0).tolist(),
            "std": attn.std(axis=0).tolist(),
            "note": "Mean learned attention weight per lookback timestep (sums to 1).",
        }
    profile = ig_per_timestep / (ig_per_timestep.sum() + 1e-12)
    return {
        "source": "integrated_gradients",
        "weights": profile.tolist(),
        "std": [0.0] * len(profile),
        "note": "No attention head in this architecture; normalised |Integrated Gradients| per timestep.",
    }


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def _barh(ax, labels, values, color=PLOT_CHROME):
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=color, edgecolor=PLOT_HAIR)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()


def make_comparison_plots(rows, baseline_metrics, poly_metrics):
    """Bar charts ranking every model against the two reference points."""
    for metric, better in (("RMSE", "lower"), ("MAE", "lower"), ("R2", "higher")):
        names = [r["model"] for r in rows]
        vals = [r["test_" + metric] for r in rows]
        order = np.argsort(vals) if better == "lower" else np.argsort(vals)[::-1]
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [PLOT_INK if n in ("Persistence", "PolyReg_deg1") else PLOT_CHROME for n in names]
        ax.bar(names, vals, color=colors, edgecolor=PLOT_HAIR)
        ax.axhline(baseline_metrics[metric], color=PLOT_ALARM, linewidth=1.0,
                   linestyle="--", label=f"Persistence baseline ({baseline_metrics[metric]:.4f})")
        ax.axhline(poly_metrics[metric], color=PLOT_INK_DIM, linewidth=1.0,
                   linestyle=":", label=f"PolyReg degree 1 ({poly_metrics[metric]:.4f})")
        ax.set_ylabel(f"Test {metric}" + (" (deg C)" if metric != "R2" else ""))
        ax.set_title(f"Final Test {metric} by model  ({better} is better)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"plot_deep_comparison_{metric.lower()}.png", dpi=150)
        plt.close()


def make_training_curves(trained):
    fig, ax = plt.subplots(figsize=(9, 5))
    shades = np.linspace(0.45, 1.0, len(trained))
    for (name, info), shade in zip(trained.items(), shades):
        epochs = [h["epoch"] for h in info["history"]]
        rmse = [h["val_RMSE"] for h in info["history"]]
        ax.plot(epochs, rmse, linewidth=1.3, label=name, color=plt.cm.gray(shade))
        ax.scatter([info["best_epoch"]], [info["best_val_RMSE"]], s=28,
                   color=PLOT_ALARM, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (deg C)")
    ax.set_title("Validation RMSE per epoch (red dot = early-stopping checkpoint)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("plot_deep_training_curves.png", dpi=150)
    plt.close()


def make_best_model_plots(name, ts, actual, pred):
    resid = actual - pred

    plt.figure(figsize=(11, 4.5))
    n_show = min(1500, len(ts))
    plt.plot(ts[-n_show:], actual[-n_show:], linewidth=1.3, color=PLOT_INK, label="Actual CHR")
    plt.plot(ts[-n_show:], pred[-n_show:], linewidth=1.1, color=PLOT_CHROME,
             linestyle="--", label=f"Predicted CHR ({name})")
    plt.xlabel("Timestamp")
    plt.ylabel("CHR (deg C)")
    plt.title(f"{name} — Final Test forecast vs actual (last {n_show} windows)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_deep_test_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, pred, s=5, alpha=0.3, color=PLOT_CHROME)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    plt.plot(lims, lims, color=PLOT_INK, linewidth=1.2, label="y = x (perfect prediction)")
    plt.xlabel("Actual CHR (deg C)")
    plt.ylabel("Predicted CHR (deg C)")
    plt.title(f"{name} — Predicted vs Actual, Final Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_deep_test_scatter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.scatter(pred, resid, s=5, alpha=0.3, color=PLOT_CHROME)
    plt.axhline(0, color=PLOT_INK, linewidth=1.0)
    plt.xlabel("Predicted CHR (deg C)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(f"{name} — Residuals vs Predicted, Final Test")
    plt.tight_layout()
    plt.savefig("plot_deep_test_residuals.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.hist(resid, bins=60, color=PLOT_CHROME, edgecolor=PLOT_HAIR)
    plt.axvline(0, color=PLOT_ALARM, linewidth=1.0)
    plt.xlabel("Residual (deg C)")
    plt.ylabel("Count")
    plt.title(f"{name} — Final Test error distribution")
    plt.tight_layout()
    plt.savefig("plot_deep_test_error_distribution.png", dpi=150)
    plt.close()


def make_xai_plots(xai, feature_names, seq_len):
    # --- per-model permutation importance ---
    for name, x in xai.items():
        imps = x["permutation"]["importances"]
        labels = [i["feature"] for i in imps][::-1]
        vals = [i["importance_rmse_increase"] for i in imps][::-1]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _barh(ax, labels, vals)
        ax.set_xlabel("RMSE increase when channel is permuted (deg C)")
        ax.set_title(f"{name} — Permutation Importance (Final Test)")
        plt.tight_layout()
        plt.savefig(f"plot_xai_permutation_{name.lower()}.png", dpi=150)
        plt.close()

    # --- per-model integrated gradients (per feature) ---
    for name, x in xai.items():
        vals = np.array(x["integrated_gradients"]["per_feature"])
        order = np.argsort(vals)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _barh(ax, [feature_names[i] for i in order], vals[order])
        ax.set_xlabel("Mean |attribution| (scaled output units)")
        ax.set_title(f"{name} — Integrated Gradients, feature attribution")
        plt.tight_layout()
        plt.savefig(f"plot_xai_integrated_gradients_{name.lower()}.png", dpi=150)
        plt.close()

    # --- temporal profiles, all models on one axis ---
    fig, ax = plt.subplots(figsize=(9, 5))
    lag_axis = np.arange(-(seq_len - 1), 1)
    shades = np.linspace(0.45, 1.0, len(xai))
    for (name, x), shade in zip(xai.items(), shades):
        w = np.array(x["temporal"]["weights"])
        style = "-" if x["temporal"]["source"] == "attention" else "--"
        ax.plot(lag_axis, w, style, linewidth=1.4, label=f"{name} ({x['temporal']['source']})",
                color=plt.cm.gray(shade))
    ax.axhline(1.0 / seq_len, color=PLOT_ALARM, linewidth=0.9, linestyle=":",
               label="Uniform (no temporal preference)")
    ax.set_xlabel("Minutes before forecast origin  (0 = most recent reading)")
    ax.set_ylabel("Normalised temporal weight")
    ax.set_title("Which minutes of the 30-minute lookback drive the forecast?")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("plot_xai_temporal_profiles.png", dpi=150)
    plt.close()

    # --- cross-model permutation importance heatmap ---
    names = list(xai.keys())
    matrix = np.zeros((len(feature_names), len(names)))
    for j, name in enumerate(names):
        lookup = {i["feature"]: i["importance_rmse_increase"]
                  for i in xai[name]["permutation"]["importances"]}
        for i, f in enumerate(feature_names):
            matrix[i, j] = lookup.get(f, 0.0)

    order = np.argsort(-matrix.mean(axis=1))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix[order], aspect="auto", cmap="gray")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_title("Permutation importance across architectures (deg C RMSE increase)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig("plot_xai_feature_agreement.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("CHILLER PLANT — DEEP SEQUENCE MODELS + EXPLAINABLE AI")
    print("=" * 78)
    set_seed(RANDOM_STATE)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    os.makedirs(DEEP_DIR, exist_ok=True)

    # ---- [1] Data ----
    print("\n[1/7] Loading + preparing data (reusing train.py pipeline)...")
    raw_df, fmt = load_data()
    df = clean_data(raw_df, verbose=False)
    df = engineer_features(df)
    interval_min = detect_interval_minutes(df)
    horizon_rows = int(round(PRIMARY_HORIZON_MINUTES / interval_min))
    df = df.dropna(subset=SEQ_FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"  - Format {fmt}, rows after cleaning/engineering: {len(df)}")
    print(f"  - Sampling interval {interval_min:.2f} min -> horizon_rows = {horizon_rows}")
    print(f"  - Sequence channels ({len(SEQ_FEATURE_COLS)}): {SEQ_FEATURE_COLS}")

    # ---- [2] Sequences + splits ----
    print("\n[2/7] Building leakage-safe sequences...")
    X, y, y_now, ends = build_sequences(
        df, SEQ_FEATURE_COLS, TARGET_COL, SEQ_LEN, horizon_rows, interval_min)
    n_possible = max(0, len(df) - SEQ_LEN + 1 - horizon_rows)
    print(f"  - Lookback {SEQ_LEN} rows ({SEQ_LEN * interval_min:.0f} min), shape {X.shape}")
    print(f"  - Dropped {n_possible - len(X)} window(s) that straddled a timestamp gap")

    dev_end = int(len(X) * DEV_FRAC)
    train_end = int(dev_end * (1 - VAL_FRAC_OF_DEV))
    Xtr_raw, ytr = X[:train_end], y[:train_end]
    Xva_raw, yva = X[train_end:dev_end], y[train_end:dev_end]
    Xte_raw, yte = X[dev_end:], y[dev_end:]
    yte_now = y_now[dev_end:]
    test_ts = df[TIMESTAMP_COL].to_numpy()[ends[dev_end:] + horizon_rows]

    print(f"  - Train {len(Xtr_raw)} | Val {len(Xva_raw)} | Final Test {len(Xte_raw)}  (chronological, never shuffled)")
    print(f"  - Test window range: {pd.Timestamp(test_ts[0])} -> {pd.Timestamp(test_ts[-1])}")

    scaler = SeqScaler().fit(Xtr_raw)          # fit on TRAIN ONLY
    Xtr, Xva, Xte = (scaler.transform(a) for a in (Xtr_raw, Xva_raw, Xte_raw))
    y_mean, y_std = float(ytr.mean()), float(ytr.std())

    # ---- [3] Reference points ----
    print("\n[3/7] Reference models on the identical test windows...")
    baseline_metrics = evaluate(yte, yte_now)
    print(f"  - Persistence baseline: MAE={baseline_metrics['MAE']:.4f} "
          f"RMSE={baseline_metrics['RMSE']:.4f} R2={baseline_metrics['R2']:.4f}")

    poly_metrics, poly_pred = None, None
    try:
        chr_artifact = joblib.load("model.pkl")
        poly_feats = chr_artifact["feature_cols"]
        X_poly = df.iloc[ends[dev_end:]][poly_feats]
        poly_pred = chr_artifact["pipeline"].predict(X_poly)
        poly_metrics = evaluate(yte, poly_pred)
        print(f"  - PolyReg degree {chr_artifact['poly_degree']}:  MAE={poly_metrics['MAE']:.4f} "
              f"RMSE={poly_metrics['RMSE']:.4f} R2={poly_metrics['R2']:.4f}")
    except Exception as exc:
        print(f"  - PolyReg comparison unavailable ({exc}). Run `python train.py` first for the full table.")

    # ---- [4] Train every architecture ----
    print(f"\n[4/7] Training {len(MODEL_REGISTRY)} deep architectures "
          f"(Adam, MSE, early stopping patience {PATIENCE})...")
    trained, results = {}, {}
    for name, (cls, desc) in MODEL_REGISTRY.items():
        print(f"\n  >> {name}: {desc}")
        info = train_model(name, cls, Xtr, ytr, Xva, yva, y_mean, y_std)
        trained[name] = info
        pred_scaled, _ = predict(info["model"], torch.from_numpy(Xte))
        pred = pred_scaled * y_std + y_mean
        m = evaluate(yte, pred)
        results[name] = {"metrics": m, "predictions": pred}
        print(f"      params={info['n_params']:,}  epochs={info['epochs_run']}  "
              f"time={info['train_seconds']:.1f}s")
        print(f"      FINAL TEST  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

    ranking = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["RMSE"])
    best_name = ranking[0][0]
    print(f"\n  - Best deep model by Final Test RMSE: {best_name} "
          f"(RMSE={results[best_name]['metrics']['RMSE']:.4f})")

    # ---- [5] XAI ----
    print(f"\n[5/7] Explainable AI on {min(XAI_SAMPLE_SIZE, len(Xte))} test windows...")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = np.sort(rng.choice(len(Xte), size=min(XAI_SAMPLE_SIZE, len(Xte)), replace=False))
    X_xai, y_xai = Xte[sample_idx], yte[sample_idx]

    xai = {}
    for name, info in trained.items():
        print(f"  >> {name}")
        t0 = time.time()
        perm = permutation_importance(info["model"], X_xai, y_xai, y_mean, y_std, SEQ_FEATURE_COLS)
        ig = integrated_gradients(info["model"], X_xai)
        temporal = temporal_explanation(info["model"], X_xai, ig["per_timestep"])
        xai[name] = {
            "permutation": perm,
            "integrated_gradients": {
                "per_feature": ig["per_feature"].tolist(),
                "per_timestep": ig["per_timestep"].tolist(),
                "feature_names": SEQ_FEATURE_COLS,
            },
            "temporal": temporal,
        }
        top3 = ", ".join(f"{i['feature']} (+{i['importance_rmse_increase']:.4f})"
                         for i in perm["importances"][:3])
        print(f"     permutation top-3: {top3}")
        print(f"     temporal source: {temporal['source']}  ({time.time() - t0:.1f}s)")

    # ---- [6] Plots ----
    print("\n[6/7] Generating plots...")
    rows = [{"model": n, "test_MAE": r["metrics"]["MAE"], "test_RMSE": r["metrics"]["RMSE"],
             "test_R2": r["metrics"]["R2"]} for n, r in results.items()]
    rows.append({"model": "Persistence", "test_MAE": baseline_metrics["MAE"],
                 "test_RMSE": baseline_metrics["RMSE"], "test_R2": baseline_metrics["R2"]})
    if poly_metrics:
        rows.append({"model": "PolyReg_deg1", "test_MAE": poly_metrics["MAE"],
                     "test_RMSE": poly_metrics["RMSE"], "test_R2": poly_metrics["R2"]})

    make_comparison_plots(rows, baseline_metrics, poly_metrics or baseline_metrics)
    make_training_curves(trained)
    make_best_model_plots(best_name, test_ts, yte, results[best_name]["predictions"])
    make_xai_plots(xai, SEQ_FEATURE_COLS, SEQ_LEN)
    print("  - Saved plot_deep_*.png and plot_xai_*.png")

    # ---- [7] Save artifacts ----
    print("\n[7/7] Saving artifacts...")
    for name, info in trained.items():
        torch.save(info["model"].state_dict(), os.path.join(DEEP_DIR, f"{name}.pt"))
    print(f"  - Saved {len(trained)} state dict(s) to {DEEP_DIR}/")

    TRACE_N = 1500
    artifact = {
        "seq_feature_cols": SEQ_FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "horizon_rows": horizon_rows,
        "interval_minutes": interval_min,
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.std_,
        "y_mean": y_mean,
        "y_std": y_std,
        "hyperparameters": {
            "hidden_size": HIDDEN_SIZE, "cnn_channels": CNN_CHANNELS, "dropout": DROPOUT,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "grad_clip": GRAD_CLIP, "optimizer": "Adam", "loss": "MSE",
            "random_state": RANDOM_STATE,
        },
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte), "dev_frac": DEV_FRAC},
        "test_date_range": [str(pd.Timestamp(test_ts[0])), str(pd.Timestamp(test_ts[-1]))],
        "model_descriptions": {n: d for n, (_, d) in MODEL_REGISTRY.items()},
        "model_metrics": {n: r["metrics"] for n, r in results.items()},
        "model_training": {n: {k: v for k, v in i.items() if k != "model"} for n, i in trained.items()},
        "baseline_metrics": baseline_metrics,
        "poly_metrics": poly_metrics,
        "best_model": best_name,
        "ranking": [n for n, _ in ranking],
        "xai": xai,
        # trimmed traces so the dashboard can plot without re-running inference
        "test_timestamps": test_ts[-TRACE_N:],
        "test_actual": yte[-TRACE_N:],
        "test_baseline": yte_now[-TRACE_N:],
        "test_predictions": {n: r["predictions"][-TRACE_N:] for n, r in results.items()},
        "test_poly": poly_pred[-TRACE_N:] if poly_pred is not None else None,
    }
    joblib.dump(artifact, DEEP_ARTIFACT_PATH)
    print(f"  - Saved {DEEP_ARTIFACT_PATH}")

    metrics_export = {
        "study": "Deep sequence models + XAI for 15-minute CHR forecasting",
        "sequence_length_rows": SEQ_LEN,
        "sequence_length_minutes": SEQ_LEN * interval_min,
        "sequence_feature_channels": SEQ_FEATURE_COLS,
        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "horizon_rows": horizon_rows,
        "split": artifact["split"],
        "test_date_range": artifact["test_date_range"],
        "windows_dropped_at_gaps": int(n_possible - len(X)),
        "hyperparameters": artifact["hyperparameters"],
        "model_descriptions": artifact["model_descriptions"],
        "baseline_test_metrics": baseline_metrics,
        "polynomial_test_metrics": poly_metrics,
        "deep_model_test_metrics": {n: r["metrics"] for n, r in results.items()},
        "model_training_summary": {
            n: {"n_params": i["n_params"], "epochs_run": i["epochs_run"],
                "best_epoch": i["best_epoch"], "best_val_RMSE": i["best_val_RMSE"],
                "train_seconds": i["train_seconds"]}
            for n, i in trained.items()
        },
        "ranking_by_test_rmse": [n for n, _ in ranking],
        "best_model": best_name,
        "xai_methods": {
            "permutation_importance": f"Channel-wise shuffling across samples, {PERM_REPEATS} repeats, "
                                      f"reported as RMSE increase in deg C.",
            "integrated_gradients": f"Sundararajan et al. 2017, {IG_STEPS} steps, zero (= training-mean) baseline.",
            "temporal_attention": "Learned additive attention weights per lookback timestep; "
                                  "1D-CNN has no attention head and falls back to |IG| per timestep.",
        },
        "xai_results": xai,
    }
    with open(DEEP_METRICS_PATH, "w") as f:
        json.dump(metrics_export, f, indent=2, default=str)
    print(f"  - Saved {DEEP_METRICS_PATH}")

    exp_rows = []
    for name, info in trained.items():
        m = results[name]["metrics"]
        exp_rows.append({
            "model": name, "family": "deep", "n_params": info["n_params"],
            "epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
            "train_seconds": round(info["train_seconds"], 1),
            "val_RMSE": info["best_val_RMSE"],
            "test_MAE": m["MAE"], "test_RMSE": m["RMSE"], "test_R2": m["R2"],
        })
    exp_rows.append({"model": "Persistence", "family": "baseline", "n_params": 0,
                     "epochs_run": 0, "best_epoch": 0, "train_seconds": 0.0, "val_RMSE": None,
                     **{f"test_{k}": v for k, v in baseline_metrics.items()}})
    if poly_metrics:
        exp_rows.append({"model": "PolyReg_deg1", "family": "classical", "n_params": 18,
                         "epochs_run": 0, "best_epoch": 0, "train_seconds": 0.0, "val_RMSE": None,
                         **{f"test_{k}": v for k, v in poly_metrics.items()}})
    exp_df = pd.DataFrame(exp_rows).rename(columns={"test_R2": "test_R2"})
    exp_df.to_csv(DEEP_EXPERIMENT_PATH, index=False)
    print(f"  - Saved {DEEP_EXPERIMENT_PATH}")

    # ---- Report ----
    print("\n" + "=" * 78)
    print("FINAL REPORT — DEEP MODELS")
    print("=" * 78)
    print(f"{'Model':<16}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'Params':>12}{'Epochs':>8}")
    print("-" * 78)
    for name, r in ranking:
        info = trained[name]
        m = r["metrics"]
        print(f"{name:<16}{m['MAE']:>10.4f}{m['RMSE']:>10.4f}{m['R2']:>10.4f}"
              f"{info['n_params']:>12,}{info['epochs_run']:>8}")
    print("-" * 78)
    print(f"{'Persistence':<16}{baseline_metrics['MAE']:>10.4f}{baseline_metrics['RMSE']:>10.4f}"
          f"{baseline_metrics['R2']:>10.4f}{0:>12}{0:>8}")
    if poly_metrics:
        print(f"{'PolyReg deg1':<16}{poly_metrics['MAE']:>10.4f}{poly_metrics['RMSE']:>10.4f}"
              f"{poly_metrics['R2']:>10.4f}{18:>12}{0:>8}")
    print("=" * 78)
    beat_baseline = [n for n, r in ranking if r["metrics"]["RMSE"] < baseline_metrics["RMSE"]]
    print(f"Deep models beating persistence: {beat_baseline if beat_baseline else 'NONE'}")
    if poly_metrics:
        beat_poly = [n for n, r in ranking if r["metrics"]["RMSE"] < poly_metrics["RMSE"]]
        print(f"Deep models beating PolyReg deg 1: {beat_poly if beat_poly else 'NONE'}")
    print("=" * 78)
    print("\nDone. Run `streamlit run app.py` for the Deep Learning and Explainable AI tabs.\n")


if __name__ == "__main__":
    main()
