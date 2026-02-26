# Hybrid Wasserstein + HMM Regime Detector — v2.0 → v2.1 Changelog

## Overview

This document covers the full evolution of the regime detection system from its
initial v2.0 architecture through the diagnostic and debugging work that produced
the current v2.1 stable baseline. The changes span the trainer, the inference
module, and a new annotation toolchain.

---

## Background

The system uses a two-layer architecture: a Wasserstein distance clusterer for
structural market context and a Gaussian HMM for temporal regime sequencing.
Features are derived from 5-minute OHLC bars and include ADX, R², volatility,
slope, range-vol, ATR, and log-returns. Training data covers NIFTY Futures
2015–2022 (~106,000 bars).

---

## Issue 1 — State mislabelling via cosine similarity (v2.0 bug)

**Symptom:** Feb-19 (a clean 400-point sustained downtrend) was labelled
entirely as Choppy. Feb-23 was mostly Transitional despite a visible stepwise
decline.

**Root cause:** `derive_state_to_label` was using cosine similarity against
regime archetypes to assign labels. All three HMM states had similar ADX values
(34–46), causing the cosine similarity scores to collapse to near-identical
values (~1.000 for all). The tiebreaker then assigned State 1 (ADX=45.8,
slope=-1.76 — a strong downtrend) as Choppy because it happened to win the
volatility comparison.

**Fix:** Replaced cosine similarity entirely with a deterministic decision tree:
1. **Range** → lowest mean volatility state (unambiguous, always lowest)
2. **Choppy** → highest volatility with R² < 0.35 (R² measures linearity; choppy
   price has erratic moves that don't fit a line — genuine chop has LOW R²,
   regardless of volatility)
3. **Trending** → all remaining states, sorted by |slope| descending

If no state qualifies for Choppy (R² < threshold), Choppy is skipped entirely
and all remaining states are labelled Trending. This is the honest outcome when
the model hasn't seen enough choppy training examples — forcing a mislabelled
Choppy state is worse than having none.

---

## Issue 2 — Broken import causing silent cache fallback

**Symptom:** Repeated retraining produced identical inference output regardless
of model changes. Training log showed correct state profiles but inference didn't
change.

**Root cause:** The trainer had `from hybrid_regime_infer_v2 import
WassersteinClusterer` but the file had been renamed to `hybrid_regime_infer.py`.
Python silently fell back to `__pycache__`, loading the old compiled bytecode on
every run.

**Fix:** Corrected import to `from hybrid_regime_infer import WassersteinClusterer`.
Added a version stamp to `load_models_once()` that prints the absolute path of
the loaded pkl file, making this class of issue immediately visible in future.

---

## Issue 3 — Slope feature not normalised by price level

**Symptom:** After fixing the label mapping, State 3 (Trending-secondary) still
received posterior = 1.000 for all afternoon bars on Feb-23 — a visibly choppy
oscillating session. The HMM had zero uncertainty despite clear visual chop.

**Root cause:** `slope` was computed as raw price points per bar from `np.polyfit`.
Training data covered NIFTY at 8,000–18,000 (2015–2022 average ~11,000). Live
inference runs on NIFTY at ~25,800 (2026). A slope of +0.18 pts/bar represents
+0.0017%/bar at training prices but only +0.0007%/bar at live prices — 2.3×
smaller in relative terms. The model learned that slope=+0.18 means "gentle
grind" at 11,000 levels, but at 25,800 the same absolute value means
"essentially flat". Flat 2026 afternoons matched State 3's emission distribution
perfectly.

**Fix:** Normalised slope by the 14-bar rolling mean close price in both trainer
and inference feature engines:

```python
# Before (level-dependent)
df["slope"] = df["close"].rolling(14).apply(lambda x: _slope_r2(x)[0], raw=False)

# After (level-independent)
df["slope"] = (
    df["close"].rolling(14).apply(lambda x: _slope_r2(x)[0], raw=False)
    / df["close"].rolling(14).mean()
)
```

Critical: this change must be applied identically in both `hybrid_wes_hmm_trainer.py`
and `hybrid_regime_infer.py`. Mismatch between training and inference feature
distributions is a silent, hard-to-diagnose error.

**Retrain required:** Yes. Existing pkl artifacts are stale after this change.

---

## Issue 4 — 24-bar window carrying trend memory into post-trend chop

**Symptom:** Even after slope normalisation, Transitional dominated the
12:45–14:30 block on Feb-23. The HMM itself was not at fault — raw posteriors
showed State 3 = 1.000 throughout.

**Root cause:** `infer_regime_multiscale` averaged posteriors across windows of
[6, 12, 24] bars. A 24-bar window at 14:10 contains bars from 12:10 onwards —
half of which were during the strong trending-down morning. Simulation confirmed
that 14 trending bars + 10 choppy bars in a 24-bar average keeps State 3 at
0.697 even when the current bars are unambiguously choppy. The 0.20 confidence
floor is then easily cleared and the label stays Trending.

**Fix:** Changed windows from `[6, 12, 24]` to `[6, 12, 12]`. The longest
memory is now 60 minutes. A regime that ends at 13:00 stops influencing labels
by 14:00 at the latest.

---

## Issue 5 — Posterior diagnostic tooling (observability improvement)

**Root cause of multiple debugging cycles:** Every diagnosis required guessing
whether the problem was the HMM, the smoothing, or the label mapping. There was
no way to see the raw HMM belief per bar.

**Fix:** Added posterior diagnostic output to the plotter, showing the raw
(pre-smoothing) HMM state probabilities for the midday transition zone and the
last 10 bars of each session. This immediately distinguishes:
- "HMM is uncertain, smoothing is amplifying noise" (split posteriors ~0.4/0.3/0.2)
- "HMM is wrong at the model level" (posterior = 1.000 on wrong state)

The second case (posterior = 1.000 persistently) tells you the problem is in
training data or feature engineering — no amount of threshold or window tuning
will fix it.

---

## Issue 6 — Unsupervised HMM cannot find quiet-chop cluster (current known gap)

**Symptom:** Feb-23 afternoon posteriors show State 3 = 1.000 on every bar even
after all fixes above. The forward-backward algorithm is 100% certain those bars
belong to State 3 (Trending-secondary). This is correct given the training
data — it is not a bug.

**Root cause:** The 4-state model has:
- State 0: Active trend (high ADX, high R², high vol)
- State 1: Deep range (very low vol)
- State 2: Active chop (moderate vol, low R²)
- State 3: Quiet drift (low vol, moderate R²) — the "miscellaneous" bin

Feb-23 afternoons are **quiet chop**: low vol AND low R². This sits in a gap
between State 2 (too low vol for active chop) and State 3 (too low R² for
quiet drift). State 3 wins because its vol and ADX match better — R² is only
one of nine features in 9D PCA space and cannot overcome the vol/ADX advantage.

**Attempted fix:** Increased to N_CLUSTERS=5. EM found a 14-bar extreme outlier
microstate (ADX=67, vol=0.077) as the 5th cluster across all 10 seeds, and
split the medium-noise space into two near-identical states. The quiet-chop
geometry does not have a natural gap that unsupervised EM can find.

**Current status:** Reverted to 4 states. This gap requires **supervised
anchoring** — adding `regime_annotations.csv` with examples of quiet-choppy
afternoons (expiry days, post-announcement drift, pre-event consolidation) so
the anchor mechanism can initialise an HMM emission mean directly in the
quiet-chop region of PCA space.

---

## New tooling: `annotate_regimes.py`

Interactive CLI for building `data/regime_annotations.csv`. Accepts
human-friendly time range input and appends labeled segments to the annotation
file used by the trainer's supervised anchoring mechanism.

**Supported input formats:**
```
09:15-10:30 Trending
11:03-11:27 Trending-Up        # direction aliases normalise to Trending
14:00-15:25 Choppy medium
10:00-11:00 T-Down             # shorthand accepted
13:00-15:25 R                  # single-letter shorthand
```

Commands: `done`, `skip`, `show`, `undo`, `quit`

The CSV format consumed by the trainer:
```csv
datetime_start,datetime_end,regime,confidence
2022-06-30 09:15:00,2022-06-30 10:30:00,Trending,high
2022-06-30 11:00:00,2022-06-30 13:45:00,Choppy,medium
```

---

## Current state of all files

| File | Status |
|------|--------|
| `hybrid_wes_hmm_trainer.py` | v2.1 — slope normalised, 4-state, R²-aware label mapping, 10-seed retry |
| `hybrid_regime_infer.py` | v2.1 — slope normalised, windows [6,12,12], posterior diagnostics, version stamp |
| `annotate_regimes.py` | New — interactive annotation CLI |
| `data/regime_annotations.csv` | Pending — needs 15–20 Choppy segments from 2015–2022 data |

---

## Next steps

1. Use `annotate_regimes.py` to label 15–20 Choppy sessions from the training
   data (2015–2022). Target: expiry-day afternoons, post-RBI afternoons,
   post-gap-up/down consolidation sessions.

2. Retrain with annotations. The anchor mechanism will initialise a dedicated
   emission mean for quiet-chop in PCA space. With anchoring, 4 states should
   be sufficient — the missing cluster will be found because EM now knows where
   to look.

3. Validate on Feb-23 (quiet-chop afternoon), Feb-25 (trend reversal), and
   Feb-19 (clean downtrend). All three should now produce coherent output.

4. Once annotations are in place, consider adding a slope normalization audit —
   verify the normalised slope distribution is consistent between training data
   (2015–2022) and live inference (2026) to catch any remaining level-dependent
   features.

---

*Last updated: 2026-02-26*
*Training data: NIFTY Futures 5-min, 2015-03-02 → 2022-10-13, 106,355 bars*