"""
Hybrid Regime Inference — Diagnostic Runner
--------------------------------------------
Standalone utility to test and visualize the
HMM + Wasserstein hybrid model on live or recent data.
--------------------------------------------
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone

# --------------------------------------------------
# INTERNAL IMPORTS
# --------------------------------------------------
from openalgo import api
from config import API_KEY, API_HOST
import hybrid_regime_infer as infer  # ← use module namespace directly

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------
IST = timezone("Asia/Kolkata")
client = api(api_key=API_KEY, host=API_HOST)
SYMBOL = "NIFTY25NOV25FUT"
today = datetime.now(IST).strftime("%Y-%m-%d")

print(f"\n[HYBRID DIAGNOSTICS] Fetching {SYMBOL} for {today}")

df = client.history(
    symbol=SYMBOL,
    exchange="NFO",
    interval="5m",
    start_date=today,
    end_date=today,
)

if df is None or df.empty:
    raise ValueError("✕ No data returned from API.")

# --------------------------------------------------
# DATA NORMALIZATION
# --------------------------------------------------
df.columns = [c.lower() for c in df.columns]
if not isinstance(df.index, pd.DatetimeIndex):
    time_col = next((c for c in ["datetime", "timestamp", "time", "date", "ts"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df.set_index(time_col, inplace=True)
    else:
        raise KeyError("No valid datetime column found in DataFrame.")
df.sort_index(inplace=True)

print(f"✔︎ Fetched {len(df)} bars: {df.index.min()} → {df.index.max()}")

# --------------------------------------------------
# MODEL FILE VALIDATION
# --------------------------------------------------
for fpath in [infer.MODEL_FILE_HMM, infer.MODEL_FILE_WASS, infer.SCALER_FILE]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Missing model file: {fpath}")

# --------------------------------------------------
# INFERENCE PIPELINE
# --------------------------------------------------
t0 = time.time()

infer.load_models_once()  # initialize HMM, Wasserstein, Scaler inside module
assert infer.scaler is not None, "Scaler not loaded — check model paths."
assert infer.clusterer is not None, "Clusterer not loaded — check model paths."
assert infer.hmmf is not None, "HMM model not loaded — check model paths."

# --- Feature Computation ---
features = infer.compute_features(df)
features = features.reindex(df.index).dropna()
df = df.loc[features.index]  # align both

X_scaled = np.clip(infer.scaler.transform(features), -3, 3)

if np.any(np.isnan(X_scaled)):
    raise ValueError("NaN detected in scaled features — check input data integrity.")

# --- Wasserstein Context ---
wlabels = infer.compute_wasserstein_context(
    X_scaled,
    infer.clusterer,
    feature_index=0,
    window=len(infer.clusterer.centroids[0]),
)

# --- Regime Inference ---
gov = infer.RegimeGovernor(min_hold=infer.MIN_HOLD_MIN)
df["RegimeLabel"] = infer.infer_regime_multiscale(
    X_scaled, df.index, infer.hmmf, gov, infer.clusterer, wlabels
)

print(f"✔︎ Inference complete in {time.time() - t0:.2f}s")

# --------------------------------------------------
# SEGMENT SUMMARY
# --------------------------------------------------
segments = infer.summarize_regime_periods(df)
print("\n✦ Regime Segments:")
for start, end, label in segments:
    s = start.strftime("%H:%M")
    e = end.strftime("%H:%M")
    print(f"{s}–{e} – {label}")

print("\n✲ Regime Distribution:")
print(df["RegimeLabel"].value_counts(normalize=True).round(3))

# Dominant regimes
dominant = df["RegimeLabel"].value_counts().sort_values(ascending=False)
print("\n✺ Dominant Regimes:")
print(dominant.head(3))

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
colors = {
    "Trending": "green",
    "Mild-Uptrend": "lime",
    "Range": "gold",
    "Choppy": "red",
    "Transitional": "gray",
}

# VWAP overlay
df["vwap"] = (
    (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum()
    / df["volume"].cumsum()
)

plt.figure(figsize=(18, 6))
plt.plot(df.index, df["close"], color="black", lw=1, alpha=0.6, label="Close")
plt.plot(df.index, df["vwap"], "--", lw=1.2, color="gray", alpha=0.8, label="VWAP")

# Regime scatter overlay
for reg, c in colors.items():
    subset = df[df["RegimeLabel"] == reg]
    if not subset.empty:
        plt.scatter(subset.index, subset["close"], s=14, c=c, label=reg, alpha=0.85)

# Subtle background spans for segment visibility
for start, end, label in segments:
    plt.axvspan(start, end, color=colors.get(label, "gray"), alpha=0.05)

plt.legend(loc="upper left")
plt.title(f"Hybrid Regime Inference — {today}", fontsize=13)
plt.grid(ls="--", alpha=0.3)
plt.tight_layout()
plt.show()

# Wasserstein context plot
plt.figure(figsize=(10, 3.5))
plt.scatter(range(len(wlabels)), wlabels, s=10, c=wlabels, cmap="viridis")
plt.title("Wasserstein Cluster Contexts (0=Trend, 1=Range, 2=Choppy)")
plt.xlabel("Bars")
plt.ylabel("Cluster ID")
plt.tight_layout()
plt.show()
