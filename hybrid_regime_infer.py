
"""
Hybrid Wasserstein + HMM Market Regime Inference
------------------------------------------------
Loads pre-trained Wasserstein & HMM models and applies them dynamically
to new data (API or CSV). Usable as both a standalone diagnostic tool
and an importable inference module for live trading logic.

Version: 1.0
Author: Jeevan Jonas
Date: 2024-06-15
License: MIT License


------------------------------------------------
"""

import sys, joblib, numpy as np, pandas as pd, talib, matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta
from pytz import timezone
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from openalgo import api
import os
from config import API_KEY, API_HOST

# --- Backward compatibility alias for pickled clusterer ---
sys.modules["wasserstein_clusterer"] = sys.modules[__name__]

# --------------------------------------------------
# CONFIG
# --------------------------------------------------



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_FILE_HMM  = os.path.join(DATA_DIR, "regime_hmm.pkl")
MODEL_FILE_WASS = os.path.join(DATA_DIR, "regime_wasserstein.pkl")
SCALER_FILE     = os.path.join(DATA_DIR, "regime_scaler.pkl")
STATE_TO_LABEL  = {0: "Trending", 1: "Range", 2: "Choppy"}

MIN_HOLD_MIN    = 20
IST = timezone("Asia/Kolkata")

# Lazy-load placeholders
_models_loaded = False
hmmf = clusterer = scaler = None


# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------
def load_models_once():
    global _models_loaded, hmmf, clusterer, scaler
    if not _models_loaded:
        hmmf = joblib.load(MODEL_FILE_HMM)
        clusterer = joblib.load(MODEL_FILE_WASS)
        scaler = joblib.load(SCALER_FILE)
        _models_loaded = True
        print("✲✦▾ Models loaded successfully.")


# --------------------------------------------------
# FEATURE ENGINE
# --------------------------------------------------
def compute_features(df):
    df = df.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["adx"] = talib.ADX(df["high"], df["low"], df["close"], 7)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], 14)
    df["range_ratio"] = (df["high"] - df["low"]) / df["close"]

    def slope_and_r2(series):
        if len(series) < 14:
            return np.nan, np.nan
        x = np.arange(14)
        coeffs = np.polyfit(x, series[-14:], 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((series[-14:] - y_pred)**2)
        ss_tot = np.sum((series[-14:] - np.mean(series[-14:]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        return coeffs[0], r2

    df["slope"] = df["close"].rolling(14).apply(lambda x: slope_and_r2(x)[0])
    df["r2"]    = df["close"].rolling(14).apply(lambda x: slope_and_r2(x)[1])

    df["slope"] *= 2.0
    df["r2"]    *= 1.3
    df["adx"]   *= 1.6
    df["range_vol"] = (df["high"] - df["low"]).rolling(14).mean() / df["close"].rolling(14).mean()
    df["volatility"] = df["close"].pct_change().rolling(14).std()
    df["atr_norm"] = df["atr"] / df["close"]

    df = df.bfill().ffill().dropna()
    if len(df) < 10:
        print(f"Insufficient rows after feature computation ({len(df)}).")
        return

    return df[["return", "slope", "r2", "adx", "atr",
               "range_ratio", "range_vol", "volatility", "atr_norm"]]


# --------------------------------------------------
# REGIME GOVERNOR
# --------------------------------------------------
class RegimeGovernor:
    def __init__(self, min_hold=20):
        self.last_label, self.last_change = None, None
        self.min_hold = min_hold

    def decide(self, probs, timestamp):
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        conf = np.max(probs)
        label = "Transitional"
        if conf >= 0.25 and entropy <= 1.6:
            label = STATE_TO_LABEL.get(np.argmax(probs), "Transitional")

        if self.last_label:
            hold = (timestamp - self.last_change).total_seconds() / 60
            if label != self.last_label and hold < self.min_hold:
                label = self.last_label
            elif label != self.last_label:
                self.last_change = timestamp
        else:
            self.last_change = timestamp
        self.last_label = label
        return label

    def decide_multiscale(self, labels, timestamp):
        short, mid, long = labels
        if long == "Trending" and mid in ("Trending", "Mild-Uptrend"):
            label = "Trending"
        elif long == "Range" and mid in ("Range", "Choppy"):
            label = "Range"
        elif long == "Choppy" and mid == "Choppy":
            label = "Choppy"
        elif short == "Trending" and mid == "Range":
            label = "Mild-Uptrend"
        elif "Transitional" in (short, mid, long):
            label = "Transitional"
        else:
            label = max(set(labels), key=labels.count)
        if self.last_label:
            hold = (timestamp - self.last_change).total_seconds() / 60
            if label != self.last_label and hold < self.min_hold:
                label = self.last_label
            elif label != self.last_label:
                self.last_change = timestamp
        else:
            self.last_change = timestamp
        self.last_label = label
        return label


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def compute_wasserstein_context(X_scaled, clusterer, feature_index=0, window=12):
    buf, labels = deque(maxlen=window), []
    for i in range(len(X_scaled)):
        buf.append(X_scaled[i, feature_index])
        if len(buf) < window:
            labels.append(None)
            continue
        labels.append(int(clusterer.predict(np.array(buf))))
    return labels


def summarize_regime_periods(df, label_col="RegimeLabel"):
    if label_col not in df:
        return []
    segs, current_label, start_time = [], df[label_col].iloc[0], df.index[0]
    for t, lbl in zip(df.index[1:], df[label_col].iloc[1:]):
        if lbl != current_label:
            end_time = df.index[df.index.get_loc(t) - 1]
            segs.append((start_time, end_time, current_label))
            start_time, current_label = t, lbl
    segs.append((start_time, df.index[-1], current_label))
    return segs


# --------------------------------------------------
# MULTI-SCALE INFERENCE
# --------------------------------------------------
def infer_regime_multiscale(X_scaled, df_index, model, governor, clusterer, wlabels, alpha=0.65):
    windows = [6, 12, 24]
    post_smooth = np.zeros((len(X_scaled), model.n_components))
    for i in range(len(X_scaled)):
        _, post = model.score_samples(X_scaled[i].reshape(1, -1))
        post = post / (post.sum() + 1e-9)
        if np.max(post) < 0.7:
            post[:] = 1.0 / model.n_components
        post_smooth[i] = post[0] if i == 0 else alpha * post[0] + (1 - alpha) * post_smooth[i - 1]
    regimes_by_window = {w: [] for w in windows}
    for w in windows:
        buf = deque(maxlen=w)
        for i in range(len(X_scaled)):
            buf.append(post_smooth[i])
            if len(buf) < w:
                regimes_by_window[w].append("Transitional")
                continue
            avg_post = np.mean(np.array(buf), axis=0)
            regimes_by_window[w].append(governor.decide(avg_post, df_index[i]))
    final_labels = []
    for i in range(len(X_scaled)):
        local = [regimes_by_window[w][i] for w in windows]
        hmm_label = governor.decide_multiscale(local, df_index[i])
        wlab = wlabels[i]
        if wlab is not None:
            if hmm_label == "Range" and wlab == 0:
                final_label = "Mild-Uptrend"
            elif hmm_label == "Trending" and wlab == 2:
                final_label = "Transitional"
            elif hmm_label == "Choppy" and wlab == 1:
                final_label = "Range"
            else:
                final_label = hmm_label
        else:
            final_label = hmm_label
        if final_label == "Transitional" and wlab == 1:
            final_label = "Range"
        elif final_label == "Transitional" and wlab == 2:
            final_label = "Choppy"
        final_labels.append(final_label)
    return final_labels


# --------------------------------------------------
# LIVE WRAPPER
# --------------------------------------------------
def infer_hybrid_regime_for_live_data(df, min_hold=20):
    load_models_once()
    if df.empty or len(df) < 30:
        return {"error": "Insufficient data for inference"}
    features = compute_features(df)
    X_scaled = np.clip(scaler.transform(features), -4, 4)
    wlabels = compute_wasserstein_context(X_scaled, clusterer, feature_index=0,
                                          window=len(clusterer.centroids[0]))
    gov = RegimeGovernor(min_hold=min_hold)
    labels = infer_regime_multiscale(X_scaled, df.index, hmmf, gov, clusterer, wlabels)
    current_label = labels[-1]
    _, post = hmmf.score_samples(X_scaled[-1].reshape(1, -1))
    confidence = float(np.max(post))
    return {
        "timestamp": df.index[-1],
        "regime": current_label,
        "confidence": round(confidence, 3),
        "wass_label": int(wlabels[-1]) if wlabels[-1] is not None else None,
        "rolling_labels": labels[-5:]
    }


# --------------------------------------------------
# Embedded WassersteinClusterer for pickle compatibility
# --------------------------------------------------
import numpy as np
class WassersteinClusterer:
    def __init__(self, n_clusters=3, refit_freq=50):
        self.k = n_clusters
        self.refit_freq = refit_freq
        self.centroids, self.counter = None, 0
    @staticmethod
    def wdist(a, b):
        a, b = np.sort(a), np.sort(b)
        return np.mean(np.abs(a - b))
    def fit(self, distributions):
        n = len(distributions)
        if n < self.k:
            raise ValueError(f"Not enough distributions ({n}) to form {self.k} clusters")
        rng = np.random.default_rng(42)
        idx = rng.choice(n, self.k, replace=False)
        centroids = [distributions[i] for i in idx]
        for _ in range(20):
            labels = [np.argmin([self.wdist(d, c) for c in centroids]) for d in distributions]
            new_centroids = []
            for j in range(self.k):
                group = [distributions[i] for i in range(n) if labels[i] == j]
                if group:
                    new_centroids.append(np.median(np.array([np.sort(g) for g in group]), axis=0))
                else:
                    new_centroids.append(centroids[j])
            loss = np.sum([self.wdist(a, b) for a, b in zip(centroids, new_centroids)])
            centroids = new_centroids
            if loss < 1e-4: break
        vars_ = [np.var(c) for c in centroids]
        order = np.argsort(vars_)
        self.centroids = [centroids[i] for i in order]
        means = [np.mean(c) for c in self.centroids]
        order = np.argsort(means)[::-1]
        self.centroids = [self.centroids[i] for i in order]
    def predict(self, dist):
        if self.centroids is None: return 0
        d = [self.wdist(dist, c) for c in self.centroids]
        return int(np.argmin(d))
    def should_refit(self):
        self.counter += 1
        if self.counter >= self.refit_freq:
            self.counter = 0; return True
        return False
    def summary(self):
        if self.centroids is None:
            print("No centroids — model not fitted yet."); return
        print("\nWasserstein Cluster Summary:")
        for i, c in enumerate(self.centroids):
            print(f"  Cluster {i}: var={np.var(c):.6f}, mean={np.mean(c):.6f}, len={len(c)}")


