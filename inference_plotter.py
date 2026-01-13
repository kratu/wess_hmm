"""
Hybrid Regime Inference — Diagnostic Runner
--------------------------------------------
Standalone utility to test and visualize the
HMM + Wasserstein hybrid model on live or recent data.
--------------------------------------------
"""

import os, sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dtime
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
SYMBOL = "NIFTY27JAN26FUT"
TIMEFRAME = "1m"
now = datetime.now(IST)
#today = datetime.now(IST).strftime("%Y-%m-%d")

#--------------------------------------------------
# TEST DATE CONFIG
# --------------------------------------------------
# Toggle between fixed test date and today's date
USE_FIXED_DATE = False          # set to False for live runs
DAYS_AGO = 0                   # how many days back for testing

if USE_FIXED_DATE:
    test_date = (now - timedelta(days=DAYS_AGO))
    today = test_date.strftime("%Y-%m-%d")
    print(f"[TEST MODE] Using fixed date → {today}")
else:
    today = now.strftime("%Y-%m-%d")
    print(f"[LIVE MODE] Using today's date → {today}")

print(f"\n[HYBRID DIAGNOSTICS] Fetching {SYMBOL} for {today}")

# --- Hybrid Rule: Force 1m until 10:30, else 5m ---
if now.time() < dtime(10,30):
    timeframe = "1m"
else:
    timeframe = TIMEFRAME  # usually "5m"

df = client.history(
    symbol=SYMBOL,
    exchange="NFO",
    interval=timeframe,
    start_date=today,
    end_date=today,
)

# # --- Handle API returning dict instead of DataFrame ---
# if isinstance(df, dict):
#     # Extract candle data safely
#     candles = df.get("data") or df.get("result", {}).get("data", [])
#     if not candles:
#         raise ValueError("No data from OpenAlgo. Ensure API key and symbol are correct.")
#     df = pd.DataFrame(candles)

if df is None:
    raise RuntimeError(
        f"No data from Open Algo({len(df)} possibly a market holiday)."
    )

# If API returned dict, extract candles
if isinstance(df, dict):
    candles = df.get("data") or df.get("result", {}).get("data", [])

    if not candles:
        raise RuntimeError(
            f"No data from OpenAlgo (0 candles) — possibly a market holiday."
        )

    df = pd.DataFrame(candles)


elif len(df) > 0 and len(df) < 10:
    raise RuntimeError(
        f"[Hybrid] Insufficient raw bars for regime inference "
        f"({len(df)} bars). Likely early market hours."
    )

# --- Handle API returning dict instead of DataFrame ---
if isinstance(df, dict):
    candles = df.get("data") or df.get("result", {}).get("data", [])
    if not candles:
        print(f"[Hybrid] No data from OpenAlgo. Ensure API_KEY is edited and SYMBOL are correct.")
        raise RuntimeError(
        "[Hybrid] No data from OpenAlgo. Check API_KEY, SYMBOL are correct."
    )
    df = pd.DataFrame(candles)

# --- Empty or invalid data guard ---
if df is None or df.empty:
    print(f"[Hybrid] Empty dataset for {SYMBOL} — exiting gracefully.")
    raise RuntimeError(f"[Hybrid] Empty dataset for {SYMBOL}.")
    #sys.exit(0)

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

if len(features) < 20:
    raise RuntimeError(
        f"[Hybrid] Only {len(features)} usable feature rows after indicator warm-up. "
        "Indicators not fully initialized yet."
    )

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
    "Trending-Down": "red",
    "Mild-Uptrend": "lime",
    "Mild-Downtrend": "orange",
    "Range": "gold",
    "Choppy": "gray",
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

# # Wasserstein context plot
# plt.figure(figsize=(10, 3.5))
# plt.scatter(range(len(wlabels)), wlabels, s=10, c=wlabels, cmap="viridis")
# plt.title("Wasserstein Cluster Contexts (0=Trend, 1=Range, 2=Choppy)")
# plt.xlabel("Bars")
# plt.ylabel("Cluster ID")
# plt.tight_layout()
# plt.show()
