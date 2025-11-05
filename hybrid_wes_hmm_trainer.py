"""
Hybrid Wasserstein + HMM Market Regime Trainer
----------------------------------------------
Trains Wasserstein centroids + HMM model from
5-min historical data and persists model artifacts.
----------------------------------------------
"""

"""
───────────────────────────────────────────────────────────────────────────────
HYBRID WASSERSTEIN + HMM REGIME DETECTION  —  Dynamic + Stable Architecture
───────────────────────────────────────────────────────────────────────────────
Overview:
    • Trains once on long historical data (e.g., 2015-2022 NIFTY 5-min)
    • Saves Wasserstein centroids, HMM parameters, and scaler to disk
    • During inference, fetches new bars via OpenAlgo API
    • Uses stored models for regime inference (no retraining each run)

Why this hybrid approach:
    • Fully static models → fast but quickly stale as volatility regime shifts
    • Fully retrained models → adaptive but slow and unstable
    • Hybrid model → combines stability of historical training with 
      adaptability from live data input.

Key properties:
    • Fast inference using pre-trained parameters
    • Optional drift-triggered re-fit for long-term adaptation
    • No repainting; each 5-min window classified causally
    • Maintains 20-minute regime persistence to filter noise

Result:
    • Computationally light
    • Structurally stable across years
    • Sensitive to current market behaviour
───────────────────────────────────────────────────────────────────────────────
"""
import warnings, joblib, numpy as np, pandas as pd, talib
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from wasserstein_clusterer import WassersteinClusterer

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
CSV_FILE       = "data/nifty_futures_5min.csv"
MODEL_FILE_HMM = "data/regime_hmm.pkl"
MODEL_FILE_WASS= "data/regime_wasserstein.pkl"
SCALER_FILE    = "data/regime_scaler.pkl"

WINDOW_MINUTES = 30
STEP_MINUTES   = 5
REFIT_FREQ     = 50
N_CLUSTERS     = 3

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(CSV_FILE)
df.columns = [c.strip().lower() for c in df.columns]
time_col = next((c for c in ["datetime","timestamp","date","time"] if c in df.columns), None)
if not time_col:
    raise KeyError("No datetime column found.")
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df.dropna(subset=[time_col], inplace=True)
df.set_index(time_col, inplace=True)
df.sort_index(inplace=True)
print(f"✅ Loaded {len(df):,} bars ({df.index.min()} → {df.index.max()})")

# ======================================================
# FEATURE ENGINE
# ======================================================
def compute_features(df):
    """
    Compute technical/statistical features for Wasserstein + HMM training.
    Matches inference version exactly.
    """
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["adx"] = talib.ADX(df["high"], df["low"], df["close"], 14)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], 14)
    df["range_ratio"] = (df["high"] - df["low"]) / df["close"]

    # --- Rolling slope and R² ---
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

    df["slope"] = df["close"].rolling(14).apply(lambda x: slope_and_r2(x)[0], raw=False)
    df["r2"]    = df["close"].rolling(14).apply(lambda x: slope_and_r2(x)[1], raw=False)
    df["range_vol"] = (df["high"] - df["low"]).rolling(14).mean() / df["close"].rolling(14).mean()

    df["volatility"] = df["close"].pct_change().rolling(14).std()
    df["atr_norm"] = df["atr"] / df["close"]
    


    # --- Trend Amplification (moderate) ---
    df["slope"] *= 1.6
    df["r2"]    *= 1.2
    df["adx"]   *= 1.4

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    ### FIXED: include range_vol in return
    return df[["return","slope","r2","adx","atr","range_ratio","range_vol","volatility","atr_norm"]]

# ======================================================
# FEATURE SCALING
# ======================================================
features = compute_features(df)
scaler = StandardScaler()
scaled_feats = scaler.fit_transform(features)

print(f"✅ Features computed: {features.shape[0]} samples × {features.shape[1]} features")

# ======================================================
# TRAIN HYBRID COMPONENTS
# ======================================================
buf, distributions = deque(maxlen=6), []
clusterer = WassersteinClusterer(n_clusters=N_CLUSTERS, refit_freq=REFIT_FREQ)

for i in range(len(scaled_feats)):
    buf.append(scaled_feats[i])
    if len(buf) < 6:
        continue
    window = np.array(buf)
    distributions.append(window[:, 0])
    if clusterer.should_refit() and len(distributions) > 50:
        clusterer.fit(distributions[-100:])

clusterer.fit(distributions[-1000:])
print("✅ Wasserstein centroids trained.")

# ======================================================
# TRAIN HMM
# ======================================================
hmmf = GaussianHMM(
    n_components=N_CLUSTERS,
    covariance_type="full",     # ✅ CHANGE: full covariance for richer dynamics
    n_iter=200,
    random_state=42
)
print("🧠 Training HMM on scaled features...")
hmmf.fit(scaled_feats)
print("✅ HMM training complete.")

# Inspect state usage
pred_states = hmmf.predict(scaled_feats)
print("\n📊 HMM state frequencies:")
print(pd.Series(pred_states).value_counts())

# Optional: compute mean ADX by state to decide mapping
state_means = []
for i in range(N_CLUSTERS):
    state_means.append(features.loc[pred_states == i, "adx"].mean())
print("Mean ADX per state:", state_means)

print("\n⚙️ Suggest STATE_TO_LABEL mapping for inference:")
sorted_states = np.argsort(state_means)[::-1]
print({sorted_states[0]: "Trending", sorted_states[1]: "Range", sorted_states[2]: "Choppy"})

state_stats = pd.DataFrame({
    'mean_adx': [features.loc[pred_states==i,'adx'].mean() for i in range(3)],
    'mean_vol': [features.loc[pred_states==i,'volatility'].mean() for i in range(3)],
    'count': np.bincount(pred_states)
})
print(state_stats)


# ======================================================
# SAVE MODEL FILES
# ======================================================
joblib.dump(clusterer, MODEL_FILE_WASS)
joblib.dump(hmmf, MODEL_FILE_HMM)
joblib.dump(scaler, SCALER_FILE)
print(f"💾 Saved → {MODEL_FILE_WASS}, {MODEL_FILE_HMM}, {SCALER_FILE}")

