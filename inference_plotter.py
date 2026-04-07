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
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from datetime import datetime, timedelta, time as dtime
from pytz import timezone
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------
# INTERNAL IMPORTS
# --------------------------------------------------
from openalgo import api
from config import API_KEY, API_HOST
import hybrid_regime_infer as infer  # ← use module namespace directly
# Diagnostic: confirm which file Python actually loaded
print(f"[MODULE] hybrid_regime_infer loaded from: {infer.__file__}")
if not hasattr(infer, 'summarize_regime_periods'):
    raise AttributeError(
        "\n\nThe loaded hybrid_regime_infer.py is MISSING summarize_regime_periods.\n"
        f"File path: {infer.__file__}\n"
        "Fix: delete __pycache__ next to that file, then re-run.\n"
        "  rm -rf __pycache__  (from the wess_hmm_v2 directory)\n"
    )


def normalize_regime_label(r) -> str:
    if hasattr(r, "cls"):
        if hasattr(r, "direction") and r.direction in ("Up", "Down"):
            return f"{r.cls}-{r.direction}"
        return r.cls
    return str(r)


def _label_cls_direction(label) -> Tuple[str, Optional[str]]:
    if hasattr(label, "cls"):
        cls = getattr(label, "cls")
        direction = getattr(label, "direction", None)
        if direction in ("Up", "Down"):
            return str(cls), str(direction)
        return str(cls), None

    s = str(label)
    if "-" in s:
        cls, tail = s.split("-", 1)
        tail = tail.strip()
        if tail in ("Up", "Down"):
            return cls.strip(), tail
        return cls.strip(), None
    return s.strip(), None


def _replace_trending_direction(label, new_direction: str):
    cls, _ = _label_cls_direction(label)
    if cls != "Trending" or new_direction not in ("Up", "Down"):
        return label

    if hasattr(label, "cls"):
        confidence = getattr(label, "confidence", None)
        bias = getattr(label, "bias", None)
        try:
            return type(label)("Trending", new_direction, confidence, bias)
        except Exception:
            try:
                return type(label)("Trending", new_direction, confidence)
            except Exception:
                pass
    return f"Trending-{new_direction}"


def _apply_trending_direction_hysteresis(
    labels: Sequence,
    *,
    required_consecutive: int = 2,
) -> List:
    out = []
    active_dir: Optional[str] = None
    pending_dir: Optional[str] = None
    pending_count = 0

    for label in labels:
        cls, direction = _label_cls_direction(label)
        if cls != "Trending" or direction not in ("Up", "Down"):
            pending_dir = None
            pending_count = 0
            out.append(label)
            continue

        if active_dir is None:
            active_dir = direction
            pending_dir = None
            pending_count = 0
            out.append(label)
            continue

        if direction == active_dir:
            pending_dir = None
            pending_count = 0
            out.append(label)
            continue

        if pending_dir == direction:
            pending_count += 1
        else:
            pending_dir = direction
            pending_count = 1

        if pending_count >= max(1, int(required_consecutive)):
            active_dir = direction
            pending_dir = None
            pending_count = 0
            out.append(label)
        else:
            out.append(_replace_trending_direction(label, active_dir))

    return out


def _estimate_bar_width_minutes(index: pd.DatetimeIndex) -> float:
    if index is None or len(index) < 2:
        return 1.0
    deltas = []
    sample_n = min(20, len(index) - 1)
    for i in range(sample_n):
        try:
            delta_min = (index[i + 1] - index[i]).total_seconds() / 60.0
        except Exception:
            continue
        if delta_min > 0:
            deltas.append(delta_min)
    if not deltas:
        return 1.0
    return max(1.0, float(np.median(deltas)))


def _segment_duration_minutes(index: pd.DatetimeIndex, start: int, end_exclusive: int, bar_width: float) -> float:
    if end_exclusive <= start:
        return bar_width
    try:
        return max(
            bar_width,
            (index[end_exclusive - 1] - index[start]).total_seconds() / 60.0 + bar_width,
        )
    except Exception:
        return max(bar_width, float(end_exclusive - start) * bar_width)


def _merge_short_regime_segments(
    labels: Sequence,
    index: pd.DatetimeIndex,
    *,
    min_duration_min: float = 3.0,
    trend_extension_cap_min: float = 2.0,
) -> List:
    out = list(labels)
    n = len(out)
    if n < 2:
        return out

    bar_width = _estimate_bar_width_minutes(index)

    for _ in range(n):
        segments = []
        seg_start = 0
        seg_key = normalize_regime_label(out[0])

        for i in range(1, n):
            k = normalize_regime_label(out[i])
            if k != seg_key:
                segments.append((seg_start, i, seg_key))
                seg_start = i
                seg_key = k
        segments.append((seg_start, n, seg_key))

        changed = False
        if len(segments) <= 1:
            break

        for si, (start, end, _key) in enumerate(segments):
            dur = _segment_duration_minutes(index, start, end, bar_width)
            if dur >= float(min_duration_min):
                continue

            seg_cls, _ = _label_cls_direction(out[start])

            candidates = []
            if si > 0:
                prev_start, prev_end, _ = segments[si - 1]
                prev_dur = _segment_duration_minutes(index, prev_start, prev_end, bar_width)
                prev_cls, _ = _label_cls_direction(out[prev_start])
                candidates.append(
                    {"si": si - 1, "start": prev_start, "dur": prev_dur, "cls": prev_cls}
                )
            if si < len(segments) - 1:
                next_start, next_end, _ = segments[si + 1]
                next_dur = _segment_duration_minutes(index, next_start, next_end, bar_width)
                next_cls, _ = _label_cls_direction(out[next_start])
                candidates.append(
                    {"si": si + 1, "start": next_start, "dur": next_dur, "cls": next_cls}
                )

            if not candidates:
                continue

            non_trending = [c for c in candidates if c["cls"] != "Trending"]
            pool = non_trending if non_trending else candidates
            target = max(pool, key=lambda c: c["dur"])

            if (
                seg_cls != "Trending"
                and target["cls"] == "Trending"
                and dur > float(trend_extension_cap_min)
            ):
                alt = [c for c in candidates if c["cls"] != "Trending"]
                if alt:
                    target = max(alt, key=lambda c: c["dur"])
                else:
                    continue

            replacement = out[target["start"]]
            for j in range(start, end):
                out[j] = replacement
            changed = True
            break

        if not changed:
            break

    return out


def _merge_single_bar_segments(
    labels: Sequence,
    index: pd.DatetimeIndex,
) -> List:
    out = list(labels)
    n = len(out)
    if n < 2:
        return out

    for _ in range(n):
        segments = []
        seg_start = 0
        seg_key = normalize_regime_label(out[0])

        for i in range(1, n):
            k = normalize_regime_label(out[i])
            if k != seg_key:
                segments.append((seg_start, i, seg_key))
                seg_start = i
                seg_key = k
        segments.append((seg_start, n, seg_key))

        if len(segments) <= 1:
            break

        changed = False
        for si, (start, end, _key) in enumerate(segments):
            seg_len = end - start
            if seg_len != 1:
                continue

            candidates = []
            if si > 0:
                ps, pe, _ = segments[si - 1]
                p_cls, _ = _label_cls_direction(out[ps])
                candidates.append({"si": si - 1, "start": ps, "len": pe - ps, "cls": p_cls})
            if si < len(segments) - 1:
                ns, ne, _ = segments[si + 1]
                n_cls, _ = _label_cls_direction(out[ns])
                candidates.append({"si": si + 1, "start": ns, "len": ne - ns, "cls": n_cls})

            if not candidates:
                continue

            non_trending = [c for c in candidates if c["cls"] != "Trending"]
            pool = non_trending if non_trending else candidates
            target = max(pool, key=lambda c: c["len"])

            replacement = out[target["start"]]
            out[start] = replacement
            changed = True
            break

        if not changed:
            break

    return out

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------
IST = timezone("Asia/Kolkata")
client = api(api_key=API_KEY, host=API_HOST)
SYMBOL = "NIFTY28APR26FUT"
CANDLESTICKS = False
now = datetime.now(IST)

# Match regime_metrics timeframe policy.
_SESSION_OPEN_MIN = 9 * 60 + 15
_now_min = now.hour * 60 + now.minute
_elapsed = max(0, _now_min - _SESSION_OPEN_MIN)

_resample_1m_to_3m = False
if _elapsed >= 25 * 5:       # >= 11:20 -> native 5m
    TIMEFRAME = "5m"
elif _elapsed >= 25 * 3:     # 10:30-11:19 -> fetch 1m, resample to 3m
    TIMEFRAME = "1m"
    _resample_1m_to_3m = True
else:                        # < 10:30 -> native 1m
    TIMEFRAME = "1m"

_REGIME_PRIMARY_TF = "3m(resampled)" if _resample_1m_to_3m else TIMEFRAME

#--------------------------------------------------
# TEST DATE CONFIG
# --------------------------------------------------
# Toggle between fixed test date and today's date
USE_FIXED_DATE = False          # set to False for live runs
DAYS_AGO = 0                  # how many days back for testing

if USE_FIXED_DATE:
    test_date = (now - timedelta(days=DAYS_AGO))
    today = test_date.strftime("%Y-%m-%d")
    print(f"[TEST MODE] Using fixed date → {today}")
else:
    today = now.strftime("%Y-%m-%d")
    print(f"[LIVE MODE] Using today's date → {today}")

print(f"\n[HYBRID DIAGNOSTICS] Preparing {SYMBOL} for {today}")


if not USE_FIXED_DATE and now.time() < dtime(10,10):
    print("Inference works best after 10:10AM, please check later")

def _normalize_history_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize payload into a timezone-aware, sorted OHLC DataFrame."""
    df_local = df_raw.copy()
    df_local.columns = [c.lower() for c in df_local.columns]
    if not isinstance(df_local.index, pd.DatetimeIndex):
        time_col = next((c for c in ["datetime", "timestamp", "time", "date", "ts"] if c in df_local.columns), None)
        if time_col:
            df_local[time_col] = pd.to_datetime(df_local[time_col], errors="coerce")
            df_local.set_index(time_col, inplace=True)
        else:
            raise KeyError("No valid datetime column found in DataFrame.")

    # OpenAlgo payloads are usually UTC; keep diagnostics in IST.
    if df_local.index.tz is None:
        df_local.index = df_local.index.tz_localize("UTC")
    df_local.index = df_local.index.tz_convert(IST)
    df_local.sort_index(inplace=True)
    return df_local


def _fetch_history(interval: str) -> pd.DataFrame:
    """Fetch one timeframe and return a normalized DataFrame."""
    raw = client.history(
        symbol=SYMBOL,
        exchange="NFO",
        interval=interval,
        start_date=today,
        end_date=today,
    )

    if raw is None:
        raise RuntimeError("No data from OpenAlgo (None response).")

    if isinstance(raw, dict):
        if raw.get("status") == "error":
            raise RuntimeError(raw.get("message") or "OpenAlgo returned error status.")
        candles = raw.get("data") or raw.get("result", {}).get("data", [])
        if not candles:
            raise RuntimeError("No data from OpenAlgo (0 candles).")
        raw = pd.DataFrame(candles)

    if raw.empty:
        raise RuntimeError("Empty dataset from OpenAlgo.")

    return _normalize_history_df(raw)


def _resample_ohlc_to_3m(df_src: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m OHLCV to 3m, matching regime_metrics behavior."""
    if df_src.empty:
        return df_src
    agg = {}
    if "open" in df_src.columns:
        agg["open"] = "first"
    if "high" in df_src.columns:
        agg["high"] = "max"
    if "low" in df_src.columns:
        agg["low"] = "min"
    if "close" in df_src.columns:
        agg["close"] = "last"
    if "volume" in df_src.columns:
        agg["volume"] = "sum"
    if "close" not in agg:
        return df_src.iloc[0:0]
    return df_src.resample("3min").agg(agg).dropna(subset=["close"])

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
if infer.scaler is None:
    raise RuntimeError("Scaler not loaded — check model paths.")
if infer.clusterer is None:
    raise RuntimeError("Clusterer not loaded — check model paths.")
if infer.hmmf is None:
    raise RuntimeError("HMM model not loaded — check model paths.")

# --- Adaptive fetch + feature fallback (aligned with regime_metrics) ---
if TIMEFRAME == "5m":
    _attempt_plan = [
        ("5m", False, "5m"),
        ("3m", False, "3m"),
        ("1m", False, "1m"),
    ]
elif _resample_1m_to_3m:
    _attempt_plan = [
        ("1m", True, "3m(resampled)"),
        ("1m", False, "1m"),
        ("3m", False, "3m"),
    ]
else:
    _attempt_plan = [
        ("1m", False, "1m"),
    ]

attempt_notes = []
selected_tf = None
selected_tf_label = None
df_plot = None
features = X_std = X_pca = None

for tf, use_resample, tf_label in _attempt_plan:
    _resample_note = " [resample->3m]" if use_resample else ""
    print(f"\n[HYBRID DIAGNOSTICS] Fetching {SYMBOL} ({tf}){_resample_note} for {today}")
    try:
        df_candidate = _fetch_history(tf)
    except Exception as exc:
        msg = f"{tf_label}: fetch failed ({exc})"
        attempt_notes.append(msg)
        print(f"⚠ {msg}")
        continue

    print(f"✔︎ Fetched {len(df_candidate)} bars on {tf}: {df_candidate.index.min()} → {df_candidate.index.max()}")

    if use_resample:
        pre_n = len(df_candidate)
        df_candidate = _resample_ohlc_to_3m(df_candidate)
        print(f"✔︎ Resampled 1m -> 3m: {pre_n} -> {len(df_candidate)} bars")

    if len(df_candidate) < 10:
        msg = f"{tf_label}: insufficient raw bars ({len(df_candidate)})."
        attempt_notes.append(msg)
        print(f"⚠ {msg}")
        continue

    features_try, X_std_try, X_pca_try = infer.prepare_inference_inputs(df_candidate)
    if features_try is None:
        msg = f"{tf_label}: feature warm-up incomplete ({len(df_candidate)} raw bars)."
        attempt_notes.append(msg)
        print(f"⚠ {msg}")
        continue

    if len(features_try) < 10:
        msg = f"{tf_label}: only {len(features_try)} feature rows after warm-up (need >=10)."
        attempt_notes.append(msg)
        print(f"⚠ {msg}")
        continue

    if np.any(np.isnan(X_std_try)) or np.any(np.isnan(X_pca_try)):
        msg = f"{tf_label}: NaN detected in transformed features."
        attempt_notes.append(msg)
        print(f"⚠ {msg}")
        continue

    selected_tf = tf
    selected_tf_label = tf_label
    df_plot = df_candidate
    features, X_std, X_pca = features_try, X_std_try, X_pca_try
    break

if features is None or df_plot is None:
    detail = " | ".join(attempt_notes) if attempt_notes else "no valid timeframe attempt"
    raise RuntimeError(f"[Hybrid] Feature computation failed across all timeframes. {detail}")

if selected_tf_label != _REGIME_PRIMARY_TF:
    print(f"✦ Fallback to {selected_tf_label} succeeded for inference.")

df_infer = df_plot.loc[features.index].copy()  # inference-aligned slice only

if np.any(np.isnan(X_std)) or np.any(np.isnan(X_pca)):
    raise ValueError("NaN detected in scaled features — check input data integrity.")

# --- Regime Inference ---
# infer_regime_multiscale computes Wasserstein context internally.
gov = infer.RegimeGovernor(min_hold=infer.MIN_HOLD_MIN)
df_infer["RegimeLabel"] = infer.infer_regime_multiscale(
    X_pca     = X_pca,
    X_std     = X_std,
    df_index  = df_infer.index,
    model     = infer.hmmf,
    governor  = gov,
    clusterer = infer.clusterer,
)

# Keep parity with regime_metrics early-session post-processing.
_EARLY_SEGMENT_MERGE_CUTOFF_MIN = 10 * 60 + 20
if len(df_infer) >= 2:
    _mins = (df_infer.index.hour * 60) + df_infer.index.minute
    _early_n = int((_mins < _EARLY_SEGMENT_MERGE_CUTOFF_MIN).sum())
else:
    _early_n = 0

if _early_n >= 2:
    _orig_labels = df_infer["RegimeLabel"].tolist()
    _early_labels = _orig_labels[:_early_n]
    _hyst_labels = _apply_trending_direction_hysteresis(
        _early_labels,
        required_consecutive=2,
    )
    _merged_early = _merge_short_regime_segments(
        _hyst_labels,
        df_infer.index[:_early_n],
        min_duration_min=3.0,
        trend_extension_cap_min=2.0,
    )
    _merged_labels = _merged_early + _orig_labels[_early_n:]
    _orig_keys = [normalize_regime_label(x) for x in _orig_labels]
    _merged_keys = [normalize_regime_label(x) for x in _merged_labels]
    if _merged_keys != _orig_keys:
        df_infer["RegimeLabel"] = _merged_labels
        print("✦ Plotter parity mode: applied early-session smoothing on <10:20 bars")

# Global cleanup parity: remove 1-bar regime blips.
_lbl_now = df_infer["RegimeLabel"].tolist()
_lbl_ns = _merge_single_bar_segments(_lbl_now, df_infer.index)
_k_now = [normalize_regime_label(x) for x in _lbl_now]
_k_ns = [normalize_regime_label(x) for x in _lbl_ns]
if _k_ns != _k_now:
    df_infer["RegimeLabel"] = _lbl_ns
    print("✦ Plotter parity mode: merged 1-bar regime segments")
print(
    f"✔︎ Inference bars retained: {len(df_infer)} / {len(df_plot)} "
    f"(warm-up dropped: {len(df_plot) - len(df_infer)})"
)

# --- Posterior diagnostic ---
# Shows RAW (unsmoothed) HMM posteriors — isolates whether the HMM itself
# is uncertain vs the multi-scale smoothing window carrying old history.
_, posteriors_raw = infer.hmmf.score_samples(X_pca)
state_labels  = [f"S{i}:{infer.STATE_TO_LABEL.get(i, f'S{i}')}" for i in range(infer.hmmf.n_components)]
post_df = pd.DataFrame(posteriors_raw, index=df_infer.index, columns=state_labels)

# Show the critical transition zone: 30 bars around midday
mid = len(df_infer) // 2
diag_slice = post_df.iloc[max(0, mid-5): min(len(post_df), mid+25)]
print("\n─── RAW posterior diagnostics (midday transition zone) ──────────────")
print(diag_slice.round(3).to_string())
print("\n─── RAW posterior diagnostics (last 10 bars) ────────────────────────")
print(post_df.tail(10).round(3).to_string())

print(f"✔︎ Inference complete in {time.time() - t0:.2f}s")


def regime_to_label(reg):
    """
    Convert RegimeState or legacy string into a display-safe label.
    """
    if hasattr(reg, "cls"):
        if reg.cls == "Trending":
            return "Trending-Up" if reg.direction == "Up" else "Trending-Down"
        return reg.cls
    return str(reg)


# --------------------------------------------------
# SEGMENT SUMMARY
# --------------------------------------------------
segments = infer.summarize_regime_periods(df_infer)
print("\n✦ Regime Segments:")
for start, end, label in segments:
    s = start.strftime("%H:%M")
    e = end.strftime("%H:%M")
    print(f"{s}–{e} – {regime_to_label(label)}")

df_infer["_regime_str"] = df_infer["RegimeLabel"].apply(regime_to_label)

print("\n✲ Regime Distribution:")
print(df_infer["_regime_str"].value_counts(normalize=True).round(3))

print("\n✺ Dominant Regimes:")
print(df_infer["_regime_str"].value_counts().head(3))

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


def lighter_color(color, blend: float = 0.45):
    """Blend an input color toward white for subtle regime connector lines."""
    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1.0 - r) * blend,
        g + (1.0 - g) * blend,
        b + (1.0 - b) * blend,
    )


def plot_candles(ax, ohlc_df: pd.DataFrame, width_frac: float = 0.65):
    """
    Lightweight candlestick renderer using plain Matplotlib.
    Avoids external dependencies (e.g., mplfinance).
    """
    if ohlc_df.empty:
        return

    x = mdates.date2num(ohlc_df.index.to_pydatetime())
    if len(x) >= 2:
        spacing = float(np.median(np.diff(x)))
    else:
        spacing = 1.0 / (24 * 60)  # ~1 minute fallback in day units
    body_w = spacing * width_frac

    candle_color = "#CCCCCC"
    median_range = float((ohlc_df["high"] - ohlc_df["low"]).median()) if len(ohlc_df) else 0.0
    min_body = max(0.05, median_range * 0.05)

    for xi, o, h, l, c in zip(
        x,
        ohlc_df["open"].to_numpy(dtype=float),
        ohlc_df["high"].to_numpy(dtype=float),
        ohlc_df["low"].to_numpy(dtype=float),
        ohlc_df["close"].to_numpy(dtype=float),
    ):
        is_down = c < o

        # Wick
        ax.vlines(xi, l, h, color=candle_color, linewidth=1.0, alpha=1.0, zorder=2)

        # Body
        lower = min(o, c)
        height = max(abs(c - o), min_body)
        ax.add_patch(
            Rectangle(
                (xi - body_w / 2.0, lower),
                body_w,
                height,
                facecolor="none" if is_down else candle_color,
                edgecolor=candle_color,
                linewidth=1.0,
                alpha=1.0,
                zorder=3,
            )
        )

    ax.xaxis_date()


# VWAP overlay
df_plot["vwap"] = (
    (df_plot["volume"] * (df_plot["high"] + df_plot["low"] + df_plot["close"]) / 3).cumsum()
    / df_plot["volume"].cumsum()
)

fig, ax = plt.subplots(figsize=(12, 8))
if CANDLESTICKS:
    plot_candles(ax, df_plot)
close_line, = ax.plot(
    df_plot.index,
    df_plot["close"],
    color="#6f6f6f",
    lw=1.0,
    alpha=0.7,
    label="Close",
    zorder=3.5,
)
vwap_line, = ax.plot(df_plot.index, df_plot["vwap"], "--", lw=1.2, color="gray", alpha=0.9, label="VWAP")

# Regime scatter overlay
df_plot["_reg_label"] = df_infer["RegimeLabel"].apply(regime_to_label).reindex(df_plot.index)

used_labels = set()   # MUST be a set
legend_handles = [close_line, vwap_line]
if CANDLESTICKS:
    legend_handles.insert(0, Line2D([0], [0], color="#01E4AC", lw=1.2, label="Candles"))

for reg, c in colors.items():
    # Regime-specific close connector; NaNs break lines between disjoint segments.
    ax.plot(
        df_plot.index,
        df_plot["close"].where(df_plot["_reg_label"] == reg, np.nan),
        color=lighter_color(c),
        lw=1.2,
        alpha=0.75,
        zorder=3.8,
    )

    subset = df_plot[df_plot["_reg_label"] == reg]
    if subset.empty:
        continue

    sc = ax.scatter(
        subset.index,
        subset["close"],
        s=14,
        c=c,
        label=reg,
        alpha=0.85,
        zorder=4,
    )

    if reg not in used_labels:
        legend_handles.append(sc)
    used_labels.add(reg)

# Subtle background spans for segment visibility
for start, end, label in segments:
    lbl = regime_to_label(label)
    ax.axvspan(start, end, color=colors.get(lbl, "gray"), alpha=0.05, zorder=0)

ax.legend(handles=legend_handles, loc="upper left")
ax.set_title(f"Hybrid Regime Inference — {today}", fontsize=13)
ax.grid(ls="--", alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M", tz=IST))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
