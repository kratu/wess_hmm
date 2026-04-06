"""
───────────────────────────────────────────────────────────────────────────────
HYBRID WASSERSTEIN + HMM REGIME INFERENCE  —  v2.0  (Evolved Architecture)
───────────────────────────────────────────────────────────────────────────────

Changes from v1.0.1:

  [FIX-1]  Corrected HMM posterior computation.
           v1 called model.score_samples(X[i].reshape(1,-1)) in a loop —
           this discards temporal context because a single-observation
           sequence has no transitions. The HMM's transition matrix was
           being ignored during inference.

           v2 calls model.score_samples(X_full) ONCE on the full sequence.
           hmmlearn's forward-backward algorithm then properly integrates
           transition probabilities when computing posteriors. This is the
           mathematically correct way to use a trained HMM.

           Consequence: the regime governor's role is reduced because the
           HMM now smooths transitions internally. Min-hold logic is kept
           as a final safety net, not the primary smoother.

  [FIX-2]  PCA whitening applied at inference time.
           Loads the pca artifact saved by the trainer and applies the same
           StandardScaler → PCA transform that training used. Without this,
           the HMM receives a different feature distribution than it was
           trained on, causing systematic miscalibration.

  [FIX-3]  Wasserstein context uses multi-feature signature.
           Mirrors the trainer change: returns + ADX + volatility concatenated
           per window, allowing Wasserstein to distinguish Range from Choppy.

  [FIX-4]  Removed is_trend_override() hard override.
           This function was silently converting correctly-identified Range
           and Choppy regimes to Trending-Up/Down whenever slope/R²/ADX
           exceeded thresholds. With PCA whitening and proper posterior
           computation, the HMM should identify trends without manual override.
           Removed to prevent the trending bias identified in v1.

  [FIX-5]  STATE_TO_LABEL loaded from saved artifact.
           No longer hardcoded — loaded from data/state_to_label.pkl which
           is derived from the multi-feature archetype matching in the trainer.

  [KEPT]   Option B structured RegimeState output (cls, direction, confidence).
           ENABLE_STRUCTURED_REGIME flag retained; behaviour unchanged.

  [KEPT]   Multiscale window logic (6/12/24 bars) and RegimeGovernor.
           Now acting on correct posteriors rather than single-bar emissions.

Author: Jeevan Jonas (evolved by Claude, 2025)
License: Apache 2.0
───────────────────────────────────────────────────────────────────────────────
"""

import sys
import joblib
import numpy as np
import pandas as pd
import talib
import os
import threading
from collections import deque
from datetime import datetime
from pytz import timezone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from dataclasses import dataclass
from typing import Literal, Optional

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_FILE_HMM  = os.path.join(DATA_DIR, "regime_hmm.pkl")
MODEL_FILE_WASS = os.path.join(DATA_DIR, "regime_wasserstein.pkl")
SCALER_FILE     = os.path.join(DATA_DIR, "regime_scaler.pkl")
PCA_FILE        = os.path.join(DATA_DIR, "regime_pca.pkl")
STATE_MAP_FILE  = os.path.join(DATA_DIR, "state_to_label.pkl")
WASS_MAP_FILE   = os.path.join(DATA_DIR, "wass_cluster_map.pkl")

# Fallback if state_to_label.pkl not found (should not happen after v2 training)
# 4-state fallback: two Trending states, one Range, one Choppy.
# Direction (Up/Down) annotated per-bar from slope sign.
_FALLBACK_STATE_TO_LABEL = {0: "Trending", 1: "Trending", 2: "Range", 3: "Choppy"}

# Fallback Wasserstein cluster→regime mapping.
# WassersteinClusterer sorts centroids by mean descending, so cluster 0 has
# the highest mean returns (bullish trending). Loaded from wass_cluster_map.pkl
# after training; this literal is only used if that artifact is absent.
_FALLBACK_WASS_CLUSTER_TO_REGIME = {0: "Trending", 1: "Range", 2: "Choppy", 3: "Choppy"}

MIN_HOLD_MIN = 20
IST          = timezone("Asia/Kolkata")

# Post-fusion tuning (kept conservative for cross-day stability)
RANGE_SLOPE_Z_CAP                 = 0.45
DIRECTION_RETZ_DOMINANCE_THR      = 0.80
WLAB1_STRONG_DOWN_SLOPE_Z_THR     = 0.52
WLAB1_STRONG_DOWN_R2_Z_THR        = 0.07
WLAB1_STRONG_DOWN_ADX_Z_THR       = -0.05
WLAB1_STRONG_DOWN_RETZ_SUM_THR    = 1.00
DOWN_DRIFT_LOOKBACK_BARS          = 12
DOWN_DRIFT_RETZ_SUM_THR           = 1.70
DOWN_DRIFT_SLOPE_Z_THR            = 0.26
HMM_TREND_LOCK_CONF_THR           = 0.82
HMM_TREND_LOCK_SLOPE_Z_THR        = 0.16
HMM_TREND_LOCK_RETZ_SUM_THR       = 0.95
CHOPPY_DEMOTE_WEAK_CONF_MAX       = 0.60
CHOPPY_DEMOTE_WEAK_SLOPE_Z_MAX    = 0.12
CHOPPY_DEMOTE_WEAK_RETZ_SUM_MAX   = 0.70

# Option B: structured regime output
ENABLE_STRUCTURED_REGIME = True

# Lazy-load placeholders
_models_loaded = False
_models_lock = threading.Lock()
hmmf      = None
clusterer = None
scaler    = None
pca       = None
STATE_TO_LABEL:         dict = _FALLBACK_STATE_TO_LABEL
WASS_CLUSTER_TO_REGIME: dict = _FALLBACK_WASS_CLUSTER_TO_REGIME


# --------------------------------------------------
# STRUCTURED OUTPUT
# --------------------------------------------------
@dataclass
class RegimeState:
    cls:        Literal["Trending", "Range", "Choppy", "Transitional"]
    direction:  Literal["Up", "Down", "Neutral"]
    confidence: Optional[float] = None


# --------------------------------------------------
# MODEL LOADER  [FIX-2: loads PCA + STATE_TO_LABEL]
# --------------------------------------------------
def load_models_once():
    global _models_loaded, hmmf, clusterer, scaler, pca, STATE_TO_LABEL, WASS_CLUSTER_TO_REGIME
    if _models_loaded:
        return

    with _models_lock:
        if _models_loaded:
            return

        hmmf      = joblib.load(MODEL_FILE_HMM)
        clusterer = joblib.load(MODEL_FILE_WASS)
        scaler    = joblib.load(SCALER_FILE)

        # Load PCA artifact (new in v2)
        if os.path.exists(PCA_FILE):
            pca = joblib.load(PCA_FILE)
        else:
            pca = None
            print("⚠  PCA artifact not found — running without PCA whitening.")
            print("   Re-run hybrid_wes_hmm_trainer.py to generate data/regime_pca.pkl")

        # Load derived HMM state→label mapping
        if os.path.exists(STATE_MAP_FILE):
            STATE_TO_LABEL = joblib.load(STATE_MAP_FILE)
        else:
            STATE_TO_LABEL = _FALLBACK_STATE_TO_LABEL
            print("⚠  state_to_label.pkl not found — using fallback mapping.")

        # Load derived Wasserstein cluster→regime mapping
        if os.path.exists(WASS_MAP_FILE):
            WASS_CLUSTER_TO_REGIME = joblib.load(WASS_MAP_FILE)
        else:
            WASS_CLUSTER_TO_REGIME = _FALLBACK_WASS_CLUSTER_TO_REGIME
            print("⚠  wass_cluster_map.pkl not found — using fallback Wasserstein mapping.")
            print("   Re-run hybrid_wes_hmm_trainer.py to generate data/wass_cluster_map.pkl")

        _models_loaded = True
        print(f"✲✦▾  Models loaded. State mapping: {STATE_TO_LABEL}")
        print(f"     Wasserstein mapping: {WASS_CLUSTER_TO_REGIME}")
        print(f"     [v2.1 — slope normalised by price, 4-state HMM, loaded from: {os.path.abspath(MODEL_FILE_HMM)}]")


# --------------------------------------------------
# FEATURE ENGINE  [must match trainer exactly]
# --------------------------------------------------
def _rolling_slope_r2(close: pd.Series, window: int = 14):
    """
    Compute rolling slope and R² in one pass per window (no duplicated polyfit).
    Slope is normalized by rolling mean(close) to keep scale stable across years.
    """
    y = close.to_numpy(dtype=float)
    n = len(y)
    slope = np.full(n, np.nan, dtype=float)
    r2 = np.full(n, np.nan, dtype=float)
    if n < window:
        return pd.Series(slope, index=close.index), pd.Series(r2, index=close.index)

    x = np.arange(window, dtype=float)
    sx = x.sum()
    sxx = np.dot(x, x)
    den = (window * sxx) - (sx * sx)

    ones = np.ones(window, dtype=float)
    sum_y = np.convolve(y, ones, mode="valid")
    sum_y2 = np.convolve(y * y, ones, mode="valid")
    sum_xy = np.convolve(y, x, mode="valid")

    slope_raw = ((window * sum_xy) - (sx * sum_y)) / den
    mean_y = sum_y / window
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_norm = slope_raw / mean_y

    intercept = (sum_y - (slope_raw * sx)) / window
    ss_tot = sum_y2 - ((sum_y * sum_y) / window)
    ss_res = (
        sum_y2
        - (2.0 * slope_raw * sum_xy)
        - (2.0 * intercept * sum_y)
        + (slope_raw * slope_raw * sxx)
        + (2.0 * slope_raw * intercept * sx)
        + (window * intercept * intercept)
    )
    r2_valid = np.where(ss_tot > 1e-12, 1.0 - (ss_res / ss_tot), 0.0)
    r2_valid = np.clip(r2_valid, -1.0, 1.0)

    start = window - 1
    slope[start:] = slope_norm
    r2[start:] = r2_valid
    return pd.Series(slope, index=close.index), pd.Series(r2, index=close.index)


def compute_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute raw (unamplified) features.
    Matches hybrid_wes_hmm_trainer_v2.py exactly — no amplifiers.
    """
    df = df.copy()

    df["return"]      = np.log(df["close"] / df["close"].shift(1))
    df["adx"]         = talib.ADX(df["high"], df["low"], df["close"], 7)
    df["atr"]         = talib.ATR(df["high"], df["low"], df["close"], 14)
    df["range_ratio"] = (df["high"] - df["low"]) / df["close"]

    df["slope"], df["r2"] = _rolling_slope_r2(df["close"], window=14)
    df["range_vol"] = (df["high"] - df["low"]).rolling(14).mean() / df["close"].rolling(14).mean()
    df["volatility"]= df["close"].pct_change().rolling(14).std()
    df["atr_norm"]  = df["atr"] / df["close"]

    # Forward-fill only to avoid look-ahead leakage from future rows.
    df = df.ffill().dropna()

    if len(df) < 10:
        print(f"⚠  Insufficient rows after feature computation ({len(df)}).")
        return None

    return df[[
        "return", "slope", "r2", "adx", "atr",
        "range_ratio", "range_vol", "volatility", "atr_norm"
    ]]


# Feature column order must match the order used during training.
# Stored here so transform_features can reconstruct a named DataFrame if
# given a raw numpy array, avoiding the sklearn "no feature names" warning.
_FEATURE_COLS = [
    "return", "slope", "r2", "adx", "atr",
    "range_ratio", "range_vol", "volatility", "atr_norm"
]


def transform_features(raw_features) -> np.ndarray:
    """
    Apply the full preprocessing pipeline: StandardScaler → PCA whitening.

    Accepts either a pd.DataFrame (preferred, no sklearn warning) or a
    np.ndarray (converted to DataFrame internally using _FEATURE_COLS).
    If PCA was not saved (legacy model), falls back to StandardScaler only.
    """
    # Ensure DataFrame with named columns so scaler doesn't warn.
    # pd is already imported at module level.
    if not isinstance(raw_features, pd.DataFrame):
        raw_features = pd.DataFrame(raw_features, columns=_FEATURE_COLS)

    X_std = scaler.transform(raw_features)

    if pca is not None:
        X_out = pca.transform(X_std)
    else:
        X_out = X_std

    return X_out


def prepare_inference_inputs(df: pd.DataFrame):
    """
    Shared inference input preparation used by wrappers and callers.
    Returns (raw_features, X_std, X_pca) or (None, None, None).
    """
    raw_features = compute_features(df)
    if raw_features is None or raw_features.empty:
        return None, None, None
    X_std = scaler.transform(raw_features)
    X_pca = transform_features(raw_features)
    return raw_features, X_std, X_pca


# --------------------------------------------------
# REGIME GOVERNOR
# --------------------------------------------------
class RegimeGovernor:
    def __init__(self, min_hold: int = MIN_HOLD_MIN):
        self.last_label  = None
        self.last_change = None
        self.min_hold    = min_hold

    def decide(self, probs: np.ndarray, timestamp: datetime) -> str:
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        conf    = float(np.max(probs))
        label   = "Transitional"

        # 4-state: uniform entropy = ln(4) ≈ 1.386. Ceiling 2.0 allows
        # split-but-leaning posteriors to commit. Floor 0.20 requires plurality.
        if conf >= 0.20 and entropy <= 2.0:
            label = STATE_TO_LABEL.get(int(np.argmax(probs)), "Transitional")

        # Transitional does not enforce min_hold — resolves immediately.
        if self.last_label is not None:
            hold = (timestamp - self.last_change).total_seconds() / 60
            if label != self.last_label:
                if self.last_label != "Transitional" and hold < self.min_hold:
                    label = self.last_label
                else:
                    self.last_change = timestamp
        else:
            self.last_change = timestamp

        self.last_label = label
        return label

    def decide_multiscale(self, labels: list, timestamp: datetime) -> str:
        short, mid, long_ = labels[0], labels[1], labels[2]

        if short == mid == long_:
            label = short
        elif short == mid:
            label = short
        elif labels.count("Trending") >= 2:
            label = "Trending"
        elif short == "Range" and mid == "Range":
            label = "Range"
        elif short == "Choppy" and mid == "Choppy":
            label = "Choppy"
        elif labels.count("Transitional") >= 2:
            label = "Transitional"
        elif short == "Trending" and mid == "Range":
            label = "Mild-Uptrend"
        else:
            label = max(set(labels), key=labels.count)

        if self.last_label is not None:
            hold = (timestamp - self.last_change).total_seconds() / 60
            if label != self.last_label:
                if self.last_label != "Transitional" and hold < self.min_hold:
                    label = self.last_label
                else:
                    self.last_change = timestamp
        else:
            self.last_change = timestamp

        self.last_label = label
        return label


# --------------------------------------------------
# WASSERSTEIN CONTEXT  [FIX-3: multi-feature]
# --------------------------------------------------
def compute_wasserstein_context(
    X_std_raw: np.ndarray,        # StandardScaler output (pre-PCA), shape (N, 9)
    clusterer,
    window: int = 6,
) -> list:
    """
    Compute per-bar Wasserstein cluster label using a multi-feature signature.

    Uses the same feature indices as the trainer:
        return (0), adx (3), volatility (7)

    Note: takes X_std_raw (pre-PCA) — Wasserstein operates on interpretable
    features, not rotated PCA components.
    """
    RETURN_IDX     = 0
    ADX_IDX        = 3
    VOLATILITY_IDX = 7

    buf    = deque(maxlen=window)
    labels = []

    for i in range(len(X_std_raw)):
        buf.append(X_std_raw[i])
        if len(buf) < window:
            labels.append(None)
            continue

        w = np.array(buf)
        signature = np.concatenate([
            w[:, RETURN_IDX],
            w[:, ADX_IDX],
            w[:, VOLATILITY_IDX],
        ])
        labels.append(int(clusterer.predict(signature)))

    return labels


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def _regime_key(label):
    if hasattr(label, "cls"):
        return (label.cls, label.direction)
    return str(label)


def _suppress_single_bar_directional_blips(labels: list) -> list:
    """
    Merge isolated one-bar directional flips into the surrounding direction.
    Example: Down, Up, Down -> Down, Down, Down.
    """
    if len(labels) < 3:
        return labels

    directional = {"Trending-Up", "Trending-Down", "Mild-Uptrend", "Mild-Downtrend"}
    out = list(labels)
    for i in range(1, len(labels) - 1):
        prev_lbl = labels[i - 1]
        cur_lbl = labels[i]
        next_lbl = labels[i + 1]
        if prev_lbl == next_lbl and cur_lbl != prev_lbl:
            if prev_lbl in directional and cur_lbl in directional:
                out[i] = prev_lbl
    return out

def _enforce_trending_minimum_duration(
    labels: list,
    df_index,
    min_trending_minutes: float = 15.0,
    target_class: str = "Transitional",
) -> list:
    """
    Demote short same-direction Trending-Up / Trending-Down runs only when
    there is counter-direction evidence inside the same directional cluster.

    Additional heuristic:
      - If a full directional cluster is entirely short and has 3..5 flips,
        classify it as Range.

    Duration is timestamp-based with inferred bar width; this keeps behavior
    consistent across 1m, 1m->3m resampled, and 5m flows.
    """
    _target_map = {"choppy": "Choppy", "transitional": "Transitional"}
    _target_key = str(target_class).strip().lower()
    if _target_key not in _target_map:
        raise ValueError(
            f"target_class must be 'Choppy' or 'Transitional', got {target_class!r}"
        )
    target_class = _target_map[_target_key]

    directional = {"Trending-Up", "Trending-Down"}
    out = list(labels)
    n = len(labels)

    if n == 0:
        return out

    bar_width = 1.0
    if df_index is not None and len(df_index) >= 2:
        deltas = []
        sample_n = min(20, len(df_index) - 1)
        for i in range(sample_n):
            try:
                delta_min = (df_index[i + 1] - df_index[i]).total_seconds() / 60.0
            except Exception:
                continue
            if delta_min > 0:
                deltas.append(delta_min)
        if deltas:
            bar_width = float(np.median(deltas))

    def _duration_min(start_i: int, end_i_exclusive: int) -> float:
        duration = (end_i_exclusive - start_i) * bar_width
        if df_index is not None and end_i_exclusive <= len(df_index):
            try:
                duration = (
                    (df_index[end_i_exclusive - 1] - df_index[start_i]).total_seconds() / 60.0
                    + bar_width
                )
            except Exception:
                pass
        return max(bar_width, float(duration))

    i = 0
    while i < n:
        if labels[i] not in directional:
            i += 1
            continue

        c_start = i
        while i < n and labels[i] in directional:
            i += 1
        c_end = i

        runs = []
        r_start = c_start
        r_label = labels[c_start]
        j = c_start + 1
        while j <= c_end:
            if j == c_end or labels[j] != r_label:
                runs.append(
                    {
                        "start": r_start,
                        "end": j,
                        "label": r_label,
                        "duration_min": _duration_min(r_start, j),
                    }
                )
                if j < c_end:
                    r_start = j
                    r_label = labels[j]
            j += 1

        run_dirs = {r["label"] for r in runs}
        if len(run_dirs) < 2:
            continue

        short_runs = [r for r in runs if r["duration_min"] < min_trending_minutes]
        short_dirs = {r["label"] for r in short_runs}

        if len(short_runs) < 2 or len(short_dirs) < 2:
            continue

        flip_count = max(0, len(runs) - 1)
        all_short = len(short_runs) == len(runs)
        replacement = "Range" if (all_short and 3 <= flip_count <= 5) else target_class

        for r in short_runs:
            for k in range(r["start"], r["end"]):
                out[k] = replacement

    return out


def summarize_regime_periods(df, label_col="RegimeLabel"):
    if label_col not in df.columns:
        return []
    segs = []
    current_label = df[label_col].iloc[0]
    current_key   = _regime_key(current_label)
    start_time    = df.index[0]
    prev_t        = df.index[0]
    for t, lbl in zip(df.index[1:], df[label_col].iloc[1:]):
        if _regime_key(lbl) != current_key:
            segs.append((start_time, prev_t, current_label))
            start_time    = t
            current_label = lbl
            current_key   = _regime_key(lbl)
        prev_t = t
    segs.append((start_time, df.index[-1], current_label))
    return segs


# --------------------------------------------------
# MULTI-SCALE INFERENCE  [FIX-1: corrected posteriors]
# --------------------------------------------------
def infer_regime_multiscale(
    X_pca:    np.ndarray,
    X_std:    np.ndarray,          # pre-PCA, for Wasserstein context
    df_index,
    model:    GaussianHMM,
    governor: RegimeGovernor,
    clusterer,
    alpha:    float = 0.65,
    return_posteriors: bool = False,
) -> list:
    """
    Multi-scale regime inference using corrected HMM posteriors.

    Key change from v1:
        model.score_samples(X_pca) is called ONCE on the FULL sequence.
        This allows the forward-backward algorithm to use transition
        probabilities, giving temporally coherent posteriors.
    """
    n = len(X_pca)

    # ── [FIX-1] Full-sequence posterior computation ───────────────────────────
    # Returns shape (N, n_components) — posterior P(state_t | all observations)
    _log_prob, posteriors_raw = model.score_samples(X_pca)

    # Normalise each row (should already sum to 1, but numerical safety)
    posteriors_raw = posteriors_raw / (posteriors_raw.sum(axis=1, keepdims=True) + 1e-9)

    # Exponential smoothing across time (preserves low-latency responsiveness)
    post_smooth = np.zeros_like(posteriors_raw)
    post_smooth[0] = posteriors_raw[0]
    for i in range(1, n):
        post_smooth[i] = alpha * posteriors_raw[i] + (1.0 - alpha) * post_smooth[i - 1]

    # ── Wasserstein labels (multi-feature) ───────────────────────────────────
    # The centroid length = 3 × WASS_WINDOW (return + ADX + volatility slices).
    # Dividing by 3 recovers the original bar window used during training.
    # Floor-divide is safe: centroid length is always an exact multiple of 3.
    _centroid_len = len(clusterer.centroids[0]) if clusterer.centroids else 18
    wass_window   = max(6, _centroid_len // 3)   # floor at 6 bars as a safety net

    wlabels = compute_wasserstein_context(
        X_std_raw = X_std,
        clusterer = clusterer,
        window    = wass_window,
    )

    # ── Multi-scale window voting ─────────────────────────────────────────────
    # Keep one governor per window and one for fusion to avoid cross-pass state
    # leakage (a single shared governor can bias labels toward the last window).
    windows = [6, 12, 24]
    min_hold = getattr(governor, "min_hold", MIN_HOLD_MIN)
    window_governors = {w: RegimeGovernor(min_hold=min_hold) for w in windows}
    fusion_governor = RegimeGovernor(min_hold=min_hold)
    regimes_by_window = {w: [] for w in windows}

    for w in windows:
        buf = deque(maxlen=w)
        gov_w = window_governors[w]
        for i in range(n):
            buf.append(post_smooth[i])
            if len(buf) < w:
                regimes_by_window[w].append("Transitional")
                continue
            avg_post = np.mean(np.array(buf), axis=0)
            regimes_by_window[w].append(gov_w.decide(avg_post, df_index[i]))

    # ── Per-bar fusion ────────────────────────────────────────────────────────
    final_labels = []
    for i in range(n):
        local = [regimes_by_window[w][i] for w in windows]
        hmm_label = fusion_governor.decide_multiscale(local, df_index[i])
        wlab = wlabels[i]
        wlab_name = WASS_CLUSTER_TO_REGIME.get(wlab) if wlab is not None else None

        slope_z = float(X_std[i, 1])
        r2_z = float(X_std[i, 2])
        adx_z = float(X_std[i, 3])
        ret_start = max(0, i - DOWN_DRIFT_LOOKBACK_BARS + 1)
        ret_z_sum = float(np.sum(X_std[ret_start:i + 1, 0]))
        hmm_conf = float(np.max(post_smooth[i]))

        trend_up = slope_z >= 0
        if abs(ret_z_sum) >= DIRECTION_RETZ_DOMINANCE_THR:
            trend_up = ret_z_sum >= 0

        strong_down = (
            slope_z <= -WLAB1_STRONG_DOWN_SLOPE_Z_THR
            and r2_z >= WLAB1_STRONG_DOWN_R2_Z_THR
            and adx_z >= WLAB1_STRONG_DOWN_ADX_Z_THR
            and ret_z_sum <= -WLAB1_STRONG_DOWN_RETZ_SUM_THR
        )
        sustained_down = (
            i >= (DOWN_DRIFT_LOOKBACK_BARS - 1)
            and slope_z <= -DOWN_DRIFT_SLOPE_Z_THR
            and ret_z_sum <= -DOWN_DRIFT_RETZ_SUM_THR
        )
        trend_lock = (
            hmm_label == "Trending"
            and hmm_conf >= HMM_TREND_LOCK_CONF_THR
            and (
                abs(slope_z) >= HMM_TREND_LOCK_SLOPE_Z_THR
                or abs(ret_z_sum) >= HMM_TREND_LOCK_RETZ_SUM_THR
            )
        )

        if trend_lock:
            final_label = "Trending-Up" if trend_up else "Trending-Down"
        elif wlab_name is not None:
            if hmm_label == "Trending" and wlab_name == "Choppy":
                weak_trend_evidence = (
                    abs(slope_z) <= CHOPPY_DEMOTE_WEAK_SLOPE_Z_MAX
                    and abs(ret_z_sum) <= CHOPPY_DEMOTE_WEAK_RETZ_SUM_MAX
                )
                if hmm_conf <= CHOPPY_DEMOTE_WEAK_CONF_MAX and weak_trend_evidence:
                    final_label = "Transitional"
                elif strong_down:
                    final_label = "Trending-Down"
                elif sustained_down:
                    final_label = "Mild-Downtrend"
                elif hmm_conf >= HMM_TREND_LOCK_CONF_THR:
                    final_label = "Trending-Up" if trend_up else "Trending-Down"
                else:
                    final_label = "Mild-Uptrend" if trend_up else "Mild-Downtrend"
            elif hmm_label == "Trending" and wlab_name == "Range":
                if strong_down:
                    final_label = "Trending-Down"
                elif abs(slope_z) <= RANGE_SLOPE_Z_CAP:
                    final_label = "Range"
                else:
                    final_label = "Mild-Uptrend" if trend_up else "Mild-Downtrend"
            elif hmm_label == "Range" and wlab_name == "Trending":
                final_label = "Mild-Uptrend" if trend_up else "Mild-Downtrend"
            elif hmm_label == "Transitional" and wlab_name in ("Range", "Choppy"):
                final_label = wlab_name
            else:
                final_label = hmm_label
        else:
            final_label = hmm_label

        # Generic bearish rescue (cross-day): when trend evidence is directional
        # but class fusion still says non-trending, promote to Trending-Down.
        if final_label in ("Range", "Mild-Downtrend", "Transitional", "Choppy") and (strong_down or sustained_down):
            final_label = "Trending-Down"

        # Direction annotation for unresolved "Trending" class.
        if final_label == "Trending":
            final_label = "Trending-Up" if trend_up else "Trending-Down"

        final_labels.append(final_label)

    # Suppress noisy one-bar direction flips inside broader directional runs.
    final_labels = _suppress_single_bar_directional_blips(final_labels)
    final_labels = _enforce_trending_minimum_duration(
        final_labels,
        df_index,
        min_trending_minutes=15.0,
        target_class="Transitional",
    )

    # ── Option B: structured output ───────────────────────────────────────────
    if ENABLE_STRUCTURED_REGIME:
        structured = []
        for i, label in enumerate(final_labels):
            conf = float(np.max(post_smooth[i]))
            if label == "Trending-Up":
                s = RegimeState("Trending",     "Up",      conf)
            elif label == "Trending-Down":
                s = RegimeState("Trending",     "Down",    conf)
            elif label == "Mild-Uptrend":
                s = RegimeState("Range",        "Up",      conf)
            elif label == "Mild-Downtrend":
                s = RegimeState("Range",        "Down",    conf)
            elif label in ("Range", "Choppy", "Transitional"):
                s = RegimeState(label,          "Neutral", conf)
            else:
                s = RegimeState("Transitional", "Neutral", conf)
            structured.append(s)
        if return_posteriors:
            return structured, posteriors_raw
        return structured

    if return_posteriors:
        return final_labels, posteriors_raw
    return final_labels


# --------------------------------------------------
# LIVE WRAPPER
# --------------------------------------------------
def infer_hybrid_regime_for_live_data(df: pd.DataFrame, min_hold: int = MIN_HOLD_MIN) -> dict:
    """
    Full inference pipeline for a live OHLC DataFrame.

    Returns a dict with:
        timestamp, regime (RegimeState or str), confidence,
        wass_label, rolling_labels (last 5), posteriors (last bar)
    """
    load_models_once()

    if df.empty or len(df) < 30:
        return {"error": f"Insufficient data ({len(df)} bars, need ≥30)"}

    raw_features, X_std, X_pca = prepare_inference_inputs(df)
    if raw_features is None:
        return {"error": "Feature computation failed."}

    # Diagnostic: confirm slope normalisation is active
    # Normalised slope should be ~1e-5 scale, not ~0.1-1.0 scale
    _slope_sample = raw_features["slope"].iloc[-1]
    if abs(_slope_sample) > 0.01:
        print(f"  ⚠  SLOPE NOT NORMALISED: slope[-1]={_slope_sample:.4f} (expected ~1e-5)")
        print(f"     Check that hybrid_regime_infer.py has slope / rolling_mean fix.")
    else:
        print(f"  ✔  slope[-1]={_slope_sample:.6f} (normalised ✓)")

    gov    = RegimeGovernor(min_hold=min_hold)
    labels, posteriors = infer_regime_multiscale(
        X_pca    = X_pca,
        X_std    = X_std,
        df_index = df.index,
        model    = hmmf,
        governor = gov,
        clusterer= clusterer,
        return_posteriors=True,
    )

    current_label = labels[-1]

    # Last-bar posterior (for confidence reporting)
    last_post     = posteriors[-1]
    confidence    = float(np.max(last_post))

    return {
        "timestamp":     df.index[-1],
        "regime":        current_label,
        "confidence":    round(confidence, 3),
        "posteriors":    {STATE_TO_LABEL.get(i, f"State{i}"): round(float(p), 3)
                          for i, p in enumerate(last_post)},
        "rolling_labels": labels[-5:],
    }


# --------------------------------------------------
# Embedded WassersteinClusterer
# (defined here for pickle compatibility — same as v1)
# --------------------------------------------------
class WassersteinClusterer:
    def __init__(self, n_clusters: int = 3, refit_freq: int = 50):
        self.k           = n_clusters
        self.refit_freq  = refit_freq
        self.centroids   = None
        self.counter     = 0

    @staticmethod
    def wdist(a: np.ndarray, b: np.ndarray) -> float:
        a, b = np.sort(a), np.sort(b)
        return float(np.mean(np.abs(a - b)))

    def fit(self, distributions: list):
        n = len(distributions)
        if n < self.k:
            raise ValueError(f"Need ≥ {self.k} distributions, got {n}.")

        rng       = np.random.default_rng(42)
        idx       = rng.choice(n, self.k, replace=False)
        centroids = [distributions[i] for i in idx]

        for _ in range(20):
            labels = [np.argmin([self.wdist(d, c) for c in centroids])
                      for d in distributions]
            new_centroids = []
            for j in range(self.k):
                group = [distributions[i] for i in range(n) if labels[i] == j]
                if group:
                    sorted_group = np.array([np.sort(g) for g in group])
                    new_centroids.append(np.median(sorted_group, axis=0))
                else:
                    new_centroids.append(centroids[j])

            loss = sum(self.wdist(a, b)
                       for a, b in zip(centroids, new_centroids))
            centroids = new_centroids
            if loss < 1e-4:
                break

        # Sort by variance then by mean (descending mean = Trending first)
        vars_  = [np.var(c) for c in centroids]
        order  = np.argsort(vars_)
        centroids = [centroids[i] for i in order]
        means_ = [np.mean(c) for c in centroids]
        order  = np.argsort(means_)[::-1]
        self.centroids = [centroids[i] for i in order]

    def predict(self, dist: np.ndarray) -> int:
        if self.centroids is None:
            return 0
        dists = [self.wdist(dist, c) for c in self.centroids]
        return int(np.argmin(dists))

    def should_refit(self) -> bool:
        self.counter += 1
        if self.counter >= self.refit_freq:
            self.counter = 0
            return True
        return False

    def summary(self):
        if self.centroids is None:
            print("WassersteinClusterer: not fitted.")
            return
        print("\nWasserstein Cluster Summary:")
        for i, c in enumerate(self.centroids):
            print(f"  Cluster {i}: len={len(c)}, mean={np.mean(c):.4f}, var={np.var(c):.6f}")
