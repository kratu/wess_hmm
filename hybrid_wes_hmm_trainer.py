"""
───────────────────────────────────────────────────────────────────────────────
HYBRID WASSERSTEIN + HMM REGIME DETECTION  —  v2.0  (Evolved Architecture)
───────────────────────────────────────────────────────────────────────────────

Changes from v1.0.1:
  [FIX-1]  Removed hardcoded trend amplifiers (slope*2.0, r2*1.3, adx*1.6).
           These artificially collapsed Range/Choppy clusters toward each other.
           PCA whitening now handles feature importance in a principled way.

  [FIX-2]  Added PCA whitening after StandardScaler.
           Decorrelates features and equalises variance across all 9 dimensions
           so the HMM sees them on equal footing. No manual tuning needed.

  [FIX-3]  Wasserstein now clusters on a multi-feature signature (returns +
           ADX + volatility distribution per window), not returns alone.
           This lets Wasserstein distinguish Choppy (high-vol, low-ADX) from
           Range (low-vol, low-ADX), which it previously could not.

  [FIX-4]  STATE_TO_LABEL mapping derived from a multi-feature centroid
           signature (ADX + R² + volatility + range_vol), not ADX alone.
           Each state is scored against regime archetypes using cosine similarity
           to a reference vector, making the mapping robust across retrains.

  [FIX-5]  Supervised anchoring: optional annotation CSV allows human-labeled
           segments to anchor HMM initialisation and apply soft posterior
           correction after training. Two sub-modes:
             (a) Anchor init only  — sets HMM emission means from labeled data.
             (b) Anchor + soft correction  — iterative pull toward human labels.

  [FIX-6]  Corrected HMM inference bug (score_samples on full sequence, not
           per-bar) is in hybrid_regime_infer_v2.py. The trainer now saves
           a PCA model artifact so inference can apply the same transform.

Annotation CSV format (data/regime_annotations.csv):
    datetime_start,       datetime_end,         regime,     confidence
    2023-03-15 09:15:00,  2023-03-15 10:45:00,  Range,      high
    2023-03-15 11:00:00,  2023-03-15 13:30:00,  Trending,   high
    2023-06-07 09:15:00,  2023-06-07 15:30:00,  Choppy,     medium

    confidence: high=1.0, medium=0.6, low=0.3
    regime:     Trending | Range | Choppy

Author: Jeevan Jonas (evolved by Claude, 2025)
License: Apache 2.0
───────────────────────────────────────────────────────────────────────────────
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import talib
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
# WassersteinClusterer is defined in hybrid_regime_infer_v2.py
# Import it from there for pickle compatibility
from hybrid_regime_infer import WassersteinClusterer

warnings.filterwarnings("ignore")


# ======================================================
# CONFIG
# ======================================================
CSV_FILE            = "data/nifty_futures_5min.csv"
ANNOTATION_FILE     = "data/regime_annotations.csv"   # optional; set to None to skip

MODEL_FILE_HMM      = "data/regime_hmm.pkl"
MODEL_FILE_WASS     = "data/regime_wasserstein.pkl"
SCALER_FILE         = "data/regime_scaler.pkl"
PCA_FILE            = "data/regime_pca.pkl"            # new artifact

N_CLUSTERS          = 4      # Trending-Up, Trending-Down, Range, Choppy
REFIT_FREQ          = 50     # Wasserstein periodic refit cadence

# Supervised anchoring strength (0.0 = pure unsupervised, 0.20 = moderate pull)
# Higher values make the model agree more with human labels but generalise less.
ANCHOR_ALPHA        = 0.15

# How many soft-correction EM passes to run after anchoring
ANCHOR_EM_PASSES    = 3

# Wasserstein window: how many bars per distribution snapshot
WASS_WINDOW         = 6


# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(CSV_FILE)
df.columns = [c.strip().lower() for c in df.columns]
time_col = next(
    (c for c in ["datetime", "timestamp", "date", "time"] if c in df.columns), None
)
if not time_col:
    raise KeyError("No datetime column found in CSV.")
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df.dropna(subset=[time_col], inplace=True)
df.set_index(time_col, inplace=True)
df.sort_index(inplace=True)
print(f"✔︎  Loaded {len(df):,} bars  ({df.index.min()} → {df.index.max()})")


# ======================================================
# FEATURE ENGINE  [FIX-1: amplifiers removed]
# ======================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw (unamplified) features.
    Identical feature set to v1 minus the hardcoded multipliers.
    Called by both trainer and inference — must stay in sync.
    """
    df = df.copy()

    df["return"]      = np.log(df["close"] / df["close"].shift(1))
    df["adx"]         = talib.ADX(df["high"], df["low"], df["close"], 7)
    df["atr"]         = talib.ATR(df["high"], df["low"], df["close"], 14)
    df["range_ratio"] = (df["high"] - df["low"]) / df["close"]

    def _slope_r2(series: np.ndarray):
        if len(series) < 14:
            return np.nan, np.nan
        x      = np.arange(14, dtype=float)
        coeffs = np.polyfit(x, series[-14:], 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((series[-14:] - y_pred) ** 2)
        ss_tot = np.sum((series[-14:] - series[-14:].mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return coeffs[0], r2

    df["slope"]    = df["close"].rolling(14).apply(lambda x: _slope_r2(x)[0], raw=False)
    df["r2"]       = df["close"].rolling(14).apply(lambda x: _slope_r2(x)[1], raw=False)
    df["range_vol"]= (df["high"] - df["low"]).rolling(14).mean() / df["close"].rolling(14).mean()
    df["volatility"]= df["close"].pct_change().rolling(14).std()
    df["atr_norm"] = df["atr"] / df["close"]

    # ── NO amplifiers ──────────────────────────────────────────────────────────
    # v1 had:  slope *= 2.0,  r2 *= 1.3,  adx *= 1.6
    # These are removed. PCA whitening replaces them with a principled approach.

    df = df.bfill().ffill().dropna()

    return df[[
        "return", "slope", "r2", "adx", "atr",
        "range_ratio", "range_vol", "volatility", "atr_norm"
    ]]


features = compute_features(df)
print(f"✔︎  Features: {features.shape[0]} samples × {features.shape[1]} features")


# ======================================================
# SCALING + PCA WHITENING  [FIX-2]
# ======================================================
scaler = StandardScaler()
X_std  = scaler.fit_transform(features)

# PCA with whitening: decorrelates features and normalises variance.
# n_components = 9 (keep all) so no information is lost.
# whiten=True divides each component by its std, making the HMM's
# Gaussian emission model work properly across all dimensions equally.
pca    = PCA(n_components=features.shape[1], whiten=True, random_state=42)
X_pca  = pca.fit_transform(X_std)

print(f"✔︎  PCA explained variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.3f} (all components)")
print(f"    Per-component: {np.round(pca.explained_variance_ratio_, 3)}")


# ======================================================
# SUPERVISED ANCHORING  [FIX-5]
# ======================================================

CONFIDENCE_MAP = {"high": 1.0, "medium": 0.6, "low": 0.3}

def load_annotations(annotation_file: str, features_df: pd.DataFrame):
    """
    Load human-labeled annotation CSV and return per-regime feature masks
    and sample weights.

    Returns:
        labeled_masks  : dict  {regime_name: boolean array over features_df index}
        sample_weights : np.ndarray  shape (N,) — weight per training sample
                         1.0 for unlabeled bars, confidence-scaled for labeled.
    """
    try:
        ann = pd.read_csv(annotation_file)
    except FileNotFoundError:
        print(f"⚠  Annotation file not found: {annotation_file}. Running unsupervised.")
        return None, None

    ann["datetime_start"] = pd.to_datetime(ann["datetime_start"])
    ann["datetime_end"]   = pd.to_datetime(ann["datetime_end"])
    ann["confidence"]     = ann["confidence"].str.lower().map(CONFIDENCE_MAP).fillna(0.5)

    n = len(features_df)
    idx = features_df.index
    sample_weights = np.ones(n, dtype=float)
    labeled_masks  = {"Trending": np.zeros(n, bool),
                      "Range":    np.zeros(n, bool),
                      "Choppy":   np.zeros(n, bool)}

    for _, row in ann.iterrows():
        regime     = row["regime"].strip()
        conf       = float(row["confidence"])
        mask       = (idx >= row["datetime_start"]) & (idx <= row["datetime_end"])
        positions  = np.where(mask)[0]

        if regime not in labeled_masks:
            print(f"⚠  Unknown regime in annotation: '{regime}' — skipping.")
            continue

        labeled_masks[regime][positions] = True
        sample_weights[positions] = 1.0 + conf   # labeled bars get up-weighted

    totals = {k: v.sum() for k, v in labeled_masks.items()}
    print(f"✔︎  Annotations loaded: {totals}")
    return labeled_masks, sample_weights


def compute_anchor_means(X_pca, labeled_masks):
    """
    Compute empirical PCA-space centroid for each regime from labeled segments.
    Used to initialise HMM emission means.
    """
    anchor_means = {}
    for regime, mask in labeled_masks.items():
        if mask.sum() < 5:
            print(f"⚠  Too few labeled samples for {regime} ({mask.sum()}) — using random init.")
            anchor_means[regime] = None
        else:
            anchor_means[regime] = X_pca[mask].mean(axis=0)
    return anchor_means


labeled_masks, sample_weights = None, None
anchor_means                  = None

if ANNOTATION_FILE is not None:
    labeled_masks, sample_weights = load_annotations(ANNOTATION_FILE, features)
    if labeled_masks is not None:
        anchor_means = compute_anchor_means(X_pca, labeled_masks)


# ======================================================
# WASSERSTEIN TRAINING  [FIX-3: multi-feature signature]
# ======================================================

def build_wasserstein_distributions(X_scaled_raw, window=WASS_WINDOW):
    """
    Build per-window distribution snapshots for Wasserstein clustering.

    v1: only used returns (column 0).
    v2: uses a composite signature: returns + ADX + volatility concatenated.
        This allows Wasserstein to distinguish:
          - Trending  : high ADX + moderate volatility + directional returns
          - Range     : low ADX + low volatility + near-zero returns
          - Choppy    : low ADX + HIGH volatility + oscillating returns

    Feature indices in the raw (pre-PCA) scaled space:
        0 = return, 3 = adx, 7 = volatility
    """
    # Use raw standardised features (pre-PCA) for Wasserstein so that the
    # distributions are interpretable and the clusterer centroids stable.
    RETURN_IDX     = 0
    ADX_IDX        = 3
    VOLATILITY_IDX = 7

    buf           = deque(maxlen=window)
    distributions = []

    for i in range(len(X_scaled_raw)):
        buf.append(X_scaled_raw[i])
        if len(buf) < window:
            continue
        w = np.array(buf)   # shape: (window, n_features)

        # Concatenate the three distributional slices into one vector
        signature = np.concatenate([
            w[:, RETURN_IDX],
            w[:, ADX_IDX],
            w[:, VOLATILITY_IDX],
        ])
        distributions.append(signature)

    return distributions


distributions = build_wasserstein_distributions(X_std)

clusterer = WassersteinClusterer(n_clusters=N_CLUSTERS, refit_freq=REFIT_FREQ)

# Periodic partial refits during build (mirrors v1 logic)
temp_buf = deque(maxlen=REFIT_FREQ + 10)
for d in distributions:
    temp_buf.append(d)
    if clusterer.should_refit() and len(temp_buf) > 50:
        clusterer.fit(list(temp_buf)[-100:])

# Final full fit on last 1000 snapshots
clusterer.fit(distributions[-1000:])
clusterer.summary()
print("✔︎  Wasserstein centroids trained (multi-feature).")


# ======================================================
# HMM TRAINING WITH SUPERVISED ANCHORING  [FIX-4, FIX-5]
# ======================================================

# ── Step 1: set up init_params based on whether we have anchors ──────────────
REGIME_ORDER = ["Trending", "Range", "Choppy", "Transitional"]   # 4-state order for anchor init

if anchor_means is not None:
    # We will provide means_ manually → tell hmmlearn to NOT init means.
    # "stc" = initialise startprob, transmat, covars (but not means).
    init_params = "stc"
    means_init  = np.array([
        anchor_means[r] if anchor_means[r] is not None
        else X_pca[np.random.choice(len(X_pca), 50)].mean(axis=0)
        for r in REGIME_ORDER
    ])
    print(f"✔︎  Anchor means computed from labeled segments:")
    for i, r in enumerate(REGIME_ORDER):
        n_labeled = labeled_masks[r].sum() if labeled_masks else 0
        print(f"    State {i} ({r:10s}): {n_labeled} labeled bars  "
              f"  mean[:3] = {means_init[i, :3].round(3)}")
else:
    init_params = "stmc"  # fully random init
    means_init  = None
    print("⚠  No annotations — using random HMM initialisation.")


# ── Step 2: HMM fit with multi-seed retry + degenerate state detection ──────
# A degenerate state is one capturing < 1% of training bars — this means
# EM has converged to a poor local minimum. We retry across seeds and
# optionally fall back from "full" to "tied" covariance for stability.

MIN_STATE_FRACTION = 0.05   # each state should cover ≥5% of data with 4 states


def train_hmm(X, n_components, init_params, means_init, covariance_type,
              seeds=(42, 7, 13, 99, 2024), n_iter=200):
    """
    Train GaussianHMM with multi-seed retry to avoid degenerate states.
    Returns best non-degenerate model, or best available if all are degenerate.
    """
    best_model   = None
    best_logprob = -np.inf

    for seed in seeds:
        model = GaussianHMM(
            n_components    = n_components,
            covariance_type = covariance_type,
            n_iter          = n_iter,
            random_state    = seed,
            init_params     = init_params,
            params          = "stmc",
        )
        if means_init is not None:
            model.means_ = means_init.copy()

        try:
            model.fit(X)
        except Exception as e:
            print(f"  ⚠  Seed {seed} failed: {e}")
            continue

        states    = model.predict(X)
        counts    = np.bincount(states, minlength=n_components)
        fractions = counts / len(X)
        min_frac  = float(fractions.min())
        log_prob  = model.score(X)

        print(f"  Seed {seed:5d}: logprob={log_prob:>12.1f}  "
              f"fractions={np.round(fractions, 3)}  min={min_frac:.3f}", end="")

        if min_frac >= MIN_STATE_FRACTION:
            print("  ✔")
            if log_prob > best_logprob:
                best_logprob = log_prob
                best_model   = model
        else:
            print(f"  ✗ degenerate (state {fractions.argmin()} = {min_frac:.3f})")
            if best_model is None or log_prob > best_logprob:
                best_logprob = log_prob
                best_model   = model

    return best_model


print("\nTraining HMM on PCA-whitened features (multi-seed, degenerate detection)...")
hmmf = train_hmm(
    X               = X_pca,
    n_components    = N_CLUSTERS,
    init_params     = init_params,
    means_init      = means_init,
    covariance_type = "full",
)

# If full covariance is still degenerate after all seeds, retry with tied.
# Tied shares one covariance matrix across states — more stable when clusters
# are not well-separated in PCA space.
states_check = hmmf.predict(X_pca)
counts_check = np.bincount(states_check, minlength=N_CLUSTERS)
if counts_check.min() < len(X_pca) * MIN_STATE_FRACTION:
    print(f"\n  ⚠  All seeds produced degenerate states with full covariance.")
    print(f"     Retrying with covariance_type='tied'...")
    hmmf_tied   = train_hmm(
        X               = X_pca,
        n_components    = N_CLUSTERS,
        init_params     = init_params,
        means_init      = means_init,
        covariance_type = "tied",
    )
    states_tied = hmmf_tied.predict(X_pca)
    if np.bincount(states_tied, minlength=N_CLUSTERS).min() > counts_check.min():
        print("  ✔  Tied covariance produced better balance — using it.")
        hmmf = hmmf_tied
    else:
        print("  Using full covariance result (tied was not better).")

print("✔︎  HMM training complete.")


# ── Step 3: soft posterior correction  [FIX-5b] ──────────────────────────────
if anchor_means is not None and ANCHOR_ALPHA > 0.0:
    print(f"\nApplying soft anchor correction (alpha={ANCHOR_ALPHA}, "
          f"passes={ANCHOR_EM_PASSES})...")

    for pass_num in range(ANCHOR_EM_PASSES):
        for state_idx, regime in enumerate(REGIME_ORDER):
            if anchor_means[regime] is None:
                continue
            human_centroid   = anchor_means[regime]
            current_mean     = hmmf.means_[state_idx]
            # Weighted blend: pull current mean (1-α) toward human anchor (α)
            hmmf.means_[state_idx] = (
                (1.0 - ANCHOR_ALPHA) * current_mean
                + ANCHOR_ALPHA       * human_centroid
            )

        # Re-run EM from corrected starting point
        hmmf_warm = GaussianHMM(
            n_components    = N_CLUSTERS,
            covariance_type = "full",
            n_iter          = 30,
            random_state    = 42,
            init_params     = "",       # empty = use all provided params
            params          = "stmc",
        )
        hmmf_warm.startprob_  = hmmf.startprob_
        hmmf_warm.transmat_   = hmmf.transmat_
        hmmf_warm.means_      = hmmf.means_.copy()
        hmmf_warm.covars_     = hmmf.covars_
        hmmf_warm.fit(X_pca)
        hmmf = hmmf_warm
        print(f"    Pass {pass_num + 1}/{ANCHOR_EM_PASSES} complete.")

    print("✔︎  Soft anchor correction applied.")


# ======================================================
# STATE CHARACTERISATION + MAPPING  [FIX-4]
# ======================================================
pred_states = hmmf.predict(X_pca)

print("\n─── HMM State Frequencies ───────────────────────────────────────────")
print(pd.Series(pred_states).value_counts().sort_index())

# Multi-feature state characterisation in the *original* (pre-PCA) space.
# We use the original features DataFrame for interpretable means.
state_profile = pd.DataFrame({
    "mean_adx":       [features.loc[pred_states == i, "adx"].mean()       for i in range(N_CLUSTERS)],
    "mean_r2":        [features.loc[pred_states == i, "r2"].mean()        for i in range(N_CLUSTERS)],
    "mean_volatility":[features.loc[pred_states == i, "volatility"].mean() for i in range(N_CLUSTERS)],
    "mean_range_vol": [features.loc[pred_states == i, "range_vol"].mean() for i in range(N_CLUSTERS)],
    "mean_slope":     [features.loc[pred_states == i, "slope"].mean()     for i in range(N_CLUSTERS)],
    "count":           np.bincount(pred_states),
})

print("\n─── State Profile (original feature space) ──────────────────────────")
print(state_profile.round(4).to_string())

# ── Decision-tree state→label mapping ─────────────────────────────────────────
#
# Step 1 → Range:    lowest mean_volatility (quiet consolidation, unambiguous)
#
# Step 2 → Choppy:   highest volatility AND low R² among remaining
#           R² is the key signal: choppy = erratic moves that don't fit a line.
#           Trending moves have HIGH R² (price fits a linear trend well).
#           If the highest-vol state has high R² (≥ 0.35), it's a volatile
#           trend, not chop — skip it and take the next highest-vol state.
#           If no state qualifies as low-R², use a combined score:
#           score = vol_rank - r2_rank  (high vol, low R² wins).
#
# Step 3 → Trending: all remaining, sorted by |slope| descending.
#           Direction (Up/Down) annotated per-bar at inference from slope_z.

R2_CHOPPY_THRESHOLD = 0.35   # states with R² above this are directional, not choppy


def derive_state_to_label(state_profile: pd.DataFrame) -> dict:
    """
    Assign regime labels to HMM states using a decision tree on interpretable
    features (volatility, R², ADX, slope magnitude). No cosine similarity.

    Choppy is identified by HIGH volatility AND LOW R² — erratic price
    movement that doesn't fit a linear trend. This correctly separates
    volatile downtrends (high vol, HIGH R²) from true chop (high vol, LOW R²).

    Works for any N_CLUSTERS ≥ 2.
    """
    n         = len(state_profile)
    remaining = list(range(n))
    mapping   = {}

    vol  = state_profile["mean_volatility"].values
    adx  = state_profile["mean_adx"].values
    r2   = state_profile["mean_r2"].values
    slp  = np.abs(state_profile["mean_slope"].values)

    # ── Step 1: Range = lowest volatility ─────────────────────────────────
    range_state = int(np.argmin(vol))
    mapping[range_state] = "Range"
    remaining.remove(range_state)
    print(f"  Range   → State {range_state}  "
          f"(vol={vol[range_state]:.4f}, adx={adx[range_state]:.1f}, "
          f"r2={r2[range_state]:.3f}, |slope|={slp[range_state]:.3f})")

    # ── Step 2: Choppy = highest volatility WITH low R² ───────────────────
    # Among remaining states, sort by vol descending.
    # A state qualifies as Choppy if r2 < R2_CHOPPY_THRESHOLD.
    # If the highest-vol state has r2 ≥ threshold, it's a volatile trend —
    # skip it and try the next. If no state qualifies, fall back to a
    # combined score: vol_normalised - r2_normalised (high vol, low r2 wins).

    remaining_by_vol = sorted(remaining, key=lambda s: vol[s], reverse=True)

    choppy_state = None
    for s in remaining_by_vol:
        if r2[s] < R2_CHOPPY_THRESHOLD:
            choppy_state = s
            break

    if choppy_state is None:
        # No clearly low-R² state — use combined score as fallback
        vol_arr = np.array([vol[s] for s in remaining])
        r2_arr  = np.array([r2[s]  for s in remaining])
        # Normalise to [0,1] within remaining set
        vol_norm = (vol_arr - vol_arr.min()) / (vol_arr.ptp() + 1e-9)
        r2_norm  = (r2_arr  - r2_arr.min())  / (r2_arr.ptp()  + 1e-9)
        scores   = vol_norm - r2_norm   # want high vol, low r2
        choppy_state = remaining[int(np.argmax(scores))]
        print(f"  ⚠  No state with R²<{R2_CHOPPY_THRESHOLD} found — using combined vol-r2 score.")
        print(f"     Best candidate: State {choppy_state} "
              f"(vol={vol[choppy_state]:.4f}, r2={r2[choppy_state]:.3f})")

    mapping[choppy_state] = "Choppy"
    remaining.remove(choppy_state)

    # Log with explicit warning if R² is suspiciously high for a Choppy state
    if r2[choppy_state] >= R2_CHOPPY_THRESHOLD:
        print(f"  ⚠  Choppy → State {choppy_state}  r2={r2[choppy_state]:.3f} ≥ {R2_CHOPPY_THRESHOLD} "
              f"— this state may be a volatile trend. Add annotations to correct.")
    else:
        print(f"  Choppy  → State {choppy_state}  "
              f"(vol={vol[choppy_state]:.4f}, adx={adx[choppy_state]:.1f}, "
              f"r2={r2[choppy_state]:.3f}, |slope|={slp[choppy_state]:.3f})")

    # ── Step 3: Trending = all remaining ──────────────────────────────────
    remaining_sorted = sorted(remaining, key=lambda s: slp[s], reverse=True)
    for rank, s in enumerate(remaining_sorted):
        mapping[s] = "Trending"
        tag = "primary" if rank == 0 else f"secondary-{rank}"
        print(f"  Trending ({tag}) → State {s}  "
              f"(vol={vol[s]:.4f}, adx={adx[s]:.1f}, "
              f"r2={r2[s]:.3f}, |slope|={slp[s]:.3f}, "
              f"mean_slope={state_profile['mean_slope'].values[s]:.3f})")

    return mapping


STATE_TO_LABEL = derive_state_to_label(state_profile)

print("\n─── Final STATE_TO_LABEL mapping ────────────────────────────────────")
for state, label in sorted(STATE_TO_LABEL.items()):
    row = state_profile.iloc[int(state)]
    print(f"    State {state} → {label:10s}  "
          f"adx={row['mean_adx']:.1f}  "
          f"vol={row['mean_volatility']:.4f}  "
          f"slope={row['mean_slope']:.3f}  "
          f"r2={row['mean_r2']:.3f}  "
          f"count={int(row['count'])}")


# ======================================================
# TRANSITION MATRIX INSPECTION (useful for future predictor)
# ======================================================
print("\n─── HMM Transition Matrix ───────────────────────────────────────────")
label_order = [STATE_TO_LABEL.get(i, f"State{i}") for i in range(N_CLUSTERS)]
transmat_df = pd.DataFrame(
    hmmf.transmat_,
    index  =[f"{i}:{label_order[i]}" for i in range(N_CLUSTERS)],
    columns=[f"{j}:{label_order[j]}" for j in range(N_CLUSTERS)],
)
print(transmat_df.round(4).to_string())
print("\n(Each row = probability of transitioning FROM that state to each other state on the next bar.)")


# ======================================================
# SAVE ALL ARTIFACTS
# ======================================================
joblib.dump(hmmf,      MODEL_FILE_HMM)
joblib.dump(clusterer, MODEL_FILE_WASS)
joblib.dump(scaler,    SCALER_FILE)
joblib.dump(pca,       PCA_FILE)

# Save the derived mapping so inference doesn't need to recompute
joblib.dump(STATE_TO_LABEL, "data/state_to_label.pkl")

print(f"\n✦  Saved artifacts:")
print(f"    {MODEL_FILE_HMM}")
print(f"    {MODEL_FILE_WASS}")
print(f"    {SCALER_FILE}")
print(f"    {PCA_FILE}")
print(f"    data/state_to_label.pkl")


# ======================================================
# VALIDATION: label distribution on training set
# ======================================================
label_series = pd.Series([STATE_TO_LABEL.get(s, "Unknown") for s in pred_states])
dist = label_series.value_counts(normalize=True)

print("\n─── Training Set Regime Distribution ────────────────────────────────")
for regime, pct in dist.items():
    bar = "█" * int(pct * 40)
    print(f"    {regime:15s}  {pct:.1%}  {bar}")

# Warn if any regime is under-represented (< 5%)
for regime in ["Trending", "Range", "Choppy"]:
    pct = dist.get(regime, 0.0)
    if pct < 0.05:
        print(f"\n  ⚠  WARNING: '{regime}' is only {pct:.1%} of training data.")
        print(f"     Consider adding more labeled samples for this regime in {ANNOTATION_FILE}.")

# Warn if Trending dominates excessively (> 70%)
trending_pct = dist.get("Trending", 0.0)
if trending_pct > 0.70:
    print(f"\n  ⚠  WARNING: Trending = {trending_pct:.1%}. Model may still be biased.")
    print(f"     Try adding Range/Choppy annotations or reducing ANCHOR_ALPHA.")

# Summary note about 4-state mapping
n_trending = sum(1 for v in STATE_TO_LABEL.values() if v == "Trending")
print(f"\n  ℹ  {n_trending} HMM state(s) mapped to Trending — direction (Up/Down) "
      f"assigned per-bar at inference from slope sign.")