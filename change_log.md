# Hybrid Wasserstein + HMM Regime Detector — v2.1 Release


## What changed in v2.1

### 1. Fixed regime label assignment
**Before:** Labels were assigned using cosine similarity against regime archetypes.
When HMM states had similar ADX values, scores collapsed to identical values and
assignment became essentially random. A strong downtrend was being labelled Choppy.

**After:** Replaced with a deterministic decision tree — lowest volatility state
gets Range, highest volatility state with R² below threshold gets Choppy, everything
else gets Trending. R² (how well price fits a straight line) is the key signal:
genuine chop has erratic moves that don't fit a line. This mapping is stable
across retrains.

---

### 2. Slope feature normalised by price level
**Before:** Slope was computed as raw price points per bar. The model was trained
on NIFTY at ~11,000 (2015–2022) but runs live at ~25,800 (2026). The same
absolute slope value is 2.3× smaller in relative terms at live prices, so quiet
2026 afternoons matched the "gentle trend" emission distribution from training.

**After:** Slope is divided by the 14-bar rolling mean close price, making it a
dimensionless relative measure that is consistent across all price levels. Applied
identically in both trainer and inference — a mismatch between the two is a silent
error and would cause systematic miscalibration.

---

### 3. Reduced multi-scale window from 24 bars to 12 bars
**Before:** Posteriors were averaged across windows of 6, 12, and 24 bars. A
24-bar (2-hour) window at 14:00 still contains trending bars from 12:00, keeping
the trending state dominant even when current conditions have clearly changed.

**After:** Windows are now 6, 12, 12 bars. The longest memory is 60 minutes,
enough for context without carrying morning structure into the afternoon.

---

### 4. PCA whitening applied at inference
**Before:** Inference applied StandardScaler only. Training used StandardScaler
followed by PCA whitening, meaning the HMM received a different feature
distribution at inference than it was trained on.

**After:** PCA transform artifact (`regime_pca.pkl`) is saved at training time
and loaded at inference. The full StandardScaler → PCA pipeline is applied
identically in both.

---

### 5. Full-sequence HMM posteriors (not per-bar)
**Before:** `score_samples` was called individually per bar. A single-observation
sequence has no transitions, so the HMM's transition matrix was being ignored
entirely during inference.

**After:** `score_samples` is called once on the full sequence. The
forward-backward algorithm integrates transition probabilities correctly, giving
temporally coherent posteriors across the session.

---

### 6. State-to-label mapping saved as artifact
**Before:** Label mapping was hardcoded in the inference module and had to be
manually updated after each retrain.

**After:** `derive_state_to_label` runs at training time and saves
`state_to_label.pkl`. Inference loads it automatically. No manual sync required
between trainer and inference after a retrain.

---

### 7. Posterior diagnostic logging
**New.** The plotter now prints raw (pre-smoothing) HMM state probabilities for
the midday transition zone and last 10 bars of each session. This makes it
immediately clear whether a misclassification is a model-level problem (posterior
= 1.000 on the wrong state — requires training changes) or an inference-level
problem (split posteriors being averaged incorrectly — requires threshold or
window changes). Without this distinction, every debugging cycle requires guessing
which layer is at fault.

---

### 8. Annotation toolchain (`annotate_regimes.py`)
**New.** Interactive CLI for labelling historical sessions and building
`data/regime_annotations.csv`. Accepts human-friendly input, handles direction
aliases, supports undo and per-session review.

```
> 09:15-10:30 Trending-Up
> 11:00-13:45 Choppy medium
> 14:00-15:25 R
```

Annotations are used by the trainer's supervised anchoring mechanism to
initialise HMM emission means directly in labelled regions of PCA space — the
correct approach when unsupervised EM cannot find a cluster because it lacks a
natural geometric gap.

---

## Benefits over v1

| | v1 | v2.1 |
|---|---|---|
| Label assignment | Cosine similarity (unstable) | Decision tree on vol + R² (stable) |
| Slope feature | Raw price points (level-dependent) | Normalised by price (level-independent) |
| Inference transform | StandardScaler only | StandardScaler + PCA (matches training) |
| HMM posteriors | Per-bar (ignores transitions) | Full sequence (uses transition matrix) |
| State mapping | Hardcoded in inference | Saved artifact, auto-loaded |
| Window memory | 24-bar max (2 hours) | 12-bar max (1 hour) |
| Diagnostics | Output labels only | Raw posteriors per bar |
| Annotations | Not supported | Interactive CLI + supervised anchoring |

---

## Known gap

The current model cannot distinguish **quiet chop** (low volatility, low R²,
directionless) from **quiet drift** (low volatility, moderate R², gentle trend).
These two regimes overlap in feature space and unsupervised EM cannot find a
clean boundary between them. An attempt at 5-state training found an extreme
outlier microstate (14 bars) as the 5th cluster rather than the desired quiet-chop
cluster.

This is a supervised anchoring problem — adding labelled examples of quiet-choppy
sessions (expiry-day afternoons, post-announcement drift) via `annotate_regimes.py`
will allow the anchor mechanism to initialise the HMM directly in the correct
region. Targeted for the next training iteration.

---

## Files

| File | Description |
|------|-------------|
| `hybrid_wes_hmm_trainer.py` | Trainer — slope normalised, R²-aware labels, 10-seed retry |
| `hybrid_regime_infer.py` | Inference — full-sequence posteriors, PCA loaded, diagnostics |
| `annotate_regimes.py` | Interactive annotation CLI |
| `data/regime_annotations.csv` | Human labels for supervised anchoring (build with CLI) |

---

*Training data: NIFTY Futures 5-min · 2015-03-02 → 2022-10-13 · 106,355 bars*

## 30 March, 2026

Retrained the model with larger data (2010-2026). And tuned the inference to focus on evaluating range/choppy regimes. Now, it should have more accurate inference of these regimes. 

At this point, the unsupervised learning is hitting a limit. So, I’ve also added an annotation tool that helps in supervised or human-assisted training.  In the future, I might add annotations to infer accurately.