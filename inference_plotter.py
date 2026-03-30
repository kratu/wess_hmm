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
import importlib, hybrid_regime_infer as infer  # ← use module namespace directly
# Diagnostic: confirm which file Python actually loaded
print(f"[MODULE] hybrid_regime_infer loaded from: {infer.__file__}")
if not hasattr(infer, 'summarize_regime_periods'):
    raise AttributeError(
        "\n\nThe loaded hybrid_regime_infer.py is MISSING summarize_regime_periods.\n"
        f"File path: {infer.__file__}\n"
        "Fix: delete __pycache__ next to that file, then re-run.\n"
        "  rm -rf __pycache__  (from the wess_hmm_v2 directory)\n"
    )

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------
IST = timezone("Asia/Kolkata")
client = api(api_key=API_KEY, host=API_HOST)
SYMBOL = "NIFTY30MAR26FUT"
TIMEFRAME = "5m"
now = datetime.now(IST)
#today = datetime.now(IST).strftime("%Y-%m-%d")

#--------------------------------------------------
# TEST DATE CONFIG
# --------------------------------------------------
# Toggle between fixed test date and today's date
USE_FIXED_DATE = True          # set to False for live runs
DAYS_AGO = 7                  # how many days back for testing

if USE_FIXED_DATE:
    test_date = (now - timedelta(days=DAYS_AGO))
    today = test_date.strftime("%Y-%m-%d")
    print(f"[TEST MODE] Using fixed date → {today}")
else:
    today = now.strftime("%Y-%m-%d")
    print(f"[LIVE MODE] Using today's date → {today}")

print(f"\n[HYBRID DIAGNOSTICS] Fetching {SYMBOL} for {today}")


if not USE_FIXED_DATE and now.time() < dtime(10,10):
    print("Inference works best after 10:10AM, please check later")
    

df = client.history(
    symbol=SYMBOL,
    exchange="NFO",
    interval=TIMEFRAME,
    start_date=today,
    end_date=today,
)

if df is None:
    raise RuntimeError("No data from OpenAlgo (None response).")

# If API returned dict, extract candles once.
if isinstance(df, dict):
    candles = df.get("data") or df.get("result", {}).get("data", [])
    if not candles:
        raise RuntimeError("No data from OpenAlgo (0 candles) — possibly a market holiday.")
    df = pd.DataFrame(candles)

if len(df) > 0 and len(df) < 10:
    raise RuntimeError(
        f"[Hybrid] Insufficient raw bars for regime inference "
        f"({len(df)} bars). Likely early market hours."
    )

# --- Empty or invalid data guard ---
if df.empty:
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
features, X_std, X_pca = infer.prepare_inference_inputs(df)
if features is None:
    raise RuntimeError("[Hybrid] Feature computation failed.")
df = df.loc[features.index]  # align both

if len(features) < 20:
    raise RuntimeError(
        f"[Hybrid] Only {len(features)} usable feature rows after indicator warm-up. "
        "Indicators not fully initialized yet."
    )

if np.any(np.isnan(X_std)) or np.any(np.isnan(X_pca)):
    raise ValueError("NaN detected in scaled features — check input data integrity.")

# --- Regime Inference ---
# infer_regime_multiscale computes Wasserstein context internally.
gov = infer.RegimeGovernor(min_hold=infer.MIN_HOLD_MIN)
df["RegimeLabel"] = infer.infer_regime_multiscale(
    X_pca     = X_pca,
    X_std     = X_std,
    df_index  = df.index,
    model     = infer.hmmf,
    governor  = gov,
    clusterer = infer.clusterer,
)

# --- Posterior diagnostic ---
# Shows RAW (unsmoothed) HMM posteriors — isolates whether the HMM itself
# is uncertain vs the multi-scale smoothing window carrying old history.
_, posteriors_raw = infer.hmmf.score_samples(X_pca)
state_labels  = [infer.STATE_TO_LABEL.get(i, f"S{i}") for i in range(infer.hmmf.n_components)]
post_df = pd.DataFrame(posteriors_raw, index=df.index, columns=state_labels)

# Show the critical transition zone: 30 bars around midday
mid = len(df) // 2
diag_slice = post_df.iloc[max(0, mid-5): min(len(post_df), mid+25)]
print("\n─── RAW posterior diagnostics (midday transition zone) ──────────────")
print(diag_slice.round(3).to_string())
print("\n─── RAW posterior diagnostics (last 10 bars) ────────────────────────")
print(post_df.tail(10).round(3).to_string())

print(f"✔︎ Inference complete in {time.time() - t0:.2f}s")


def format_regime(reg):
    """
    Human-friendly representation of regime state.
    """
    if hasattr(reg, "cls"):
        if reg.cls == "Trending":
            return f"Trending-{reg.direction}"
        return reg.cls
    return str(reg)


def regime_to_label(reg):
    """
    Convert RegimeState or legacy string into a display-safe label.
    """
    if hasattr(reg, "cls"):
        if reg.cls == "Trending":
            return "Trending-Up" if reg.direction == "Up" else "Trending-Down"
        return reg.cls
    return reg


# --------------------------------------------------
# SEGMENT SUMMARY
# --------------------------------------------------
segments = infer.summarize_regime_periods(df)
print("\n✦ Regime Segments:")
for start, end, label in segments:
    s = start.strftime("%H:%M")
    e = end.strftime("%H:%M")
    print(f"{s}–{e} – {format_regime(label)}")

df["_regime_str"] = df["RegimeLabel"].apply(format_regime)

print("\n✲ Regime Distribution:")
print(df["_regime_str"].value_counts(normalize=True).round(3))

print("\n✺ Dominant Regimes:")
print(df["_regime_str"].value_counts().head(3))

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
colors = {
    "Trending-Up": "green",
    "Trending-Down": "red",
    "Mild-Uptrend": "lime",
    "Mild-Downtrend": "orange",
    "Range": "gold",
    "Choppy": "yellow",
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
# Regime scatter overlay
df["_reg_label"] = df["RegimeLabel"].apply(regime_to_label)

used_labels = set()   # MUST be a set

for reg, c in colors.items():
    subset = df[df["_reg_label"] == reg]
    if subset.empty:
        continue

    label = reg if reg not in used_labels else None

    plt.scatter(
        subset.index,
        subset["close"],
        s=14,
        c=c,
        label=label,
        alpha=0.85,
    )

    used_labels.add(reg)

# Subtle background spans for segment visibility
for start, end, label in segments:
    lbl = regime_to_label(label)
    plt.axvspan(start, end, color=colors.get(lbl, "gray"), alpha=0.05)


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
