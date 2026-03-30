"""
Hybrid Regime Inference — Integration Example
---------------------------------------------
This script demonstrates how to integrate the Hybrid HMM + Wasserstein
Regime Inference model into a live trading or research environment.

It fetches intraday market data using the OpenAlgo API, computes
technical features (returns, ADX, ATR, slope, R², volatility),
and applies the pre-trained hybrid model to classify the current
market regime as one of:
    • Trending
    • Range / Transitional
    • Choppy

The script prints time-segmented regime summaries and updates the
latest regime label in real time.

Key Features:
    • Uses pre-trained HMM + Wasserstein cluster models
    • Handles insufficient early-session data gracefully
    • Designed for scheduled or periodic execution
    • Can be extended to enable or disable trading strategies
      based on inferred market conditions

Dependencies:
    - hybrid_regime_infer.py  (core inference logic)
    - config.py               (API key and connection settings)
    - OpenAlgo API            (for live market data)

Intended Use:
    For educational and research purposes only. Not financial advice.
"""


import hybrid_regime_infer as infer
from datetime import datetime, timedelta, time as dtime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import SchedulerAlreadyRunningError
import pandas as pd
import numpy as np
import sys
from openalgo import api
from pytz import timezone
from config import API_KEY, API_HOST

client = api(api_key=API_KEY, host=API_HOST)

latest_regime = "Unknown" #Default initial regime
# Persistent regime governor (IMPORTANT)
regime_governor = infer.RegimeGovernor(min_hold=infer.MIN_HOLD_MIN)

SYMBOL = "NIFTY30MAR26FUT"
IST = timezone("Asia/Kolkata")
TIMEFRAME = "5m"


def duration_bar(
    duration_min: int,
    max_ref_min: int = 50,
    width: int = 10
) -> str:
    """
    Convert duration (minutes) into a block bar.

    - max_ref_min: duration that maps to full bar
    - width: number of blocks
    """
    if duration_min <= 0:
        return "░" * width

    ratio = min(duration_min / max_ref_min, 1.0)
    filled = int(round(ratio * width))

    if filled == 0:
        return "░" + " " * (width - 1)

    return "█" * filled + "░" * (width - filled)


def regime_inference():
    global latest_regime

    now = datetime.now(IST)
    today = datetime.now(IST).strftime("%Y-%m-%d")

    # --- Hybrid Rule: Force 1m until 10:20, else 5m ---
    if now.time() < dtime(10, 20):
        timeframe = "1m"
        print("Inference works best after 10:20AM, using fallback 1min")
    else:
        timeframe = TIMEFRAME

    df = client.history(
        symbol=SYMBOL,
        exchange="NFO",
        interval=timeframe,
        start_date=today,
        end_date=today
    )

    if df.empty:
        raise ValueError("No data from OpenAlgo.")

    # Normalize datetime
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)


    # --------------------------------------------------
    # INFERENCE PIPELINE  (v2.1)
    # --------------------------------------------------
    infer.load_models_once()

    raw_features, X_std, X_pca = infer.prepare_inference_inputs(df)

    if raw_features is None or len(raw_features) < 10:
        print(f"[RegimeGuard] Not enough bars yet ({len(df)}). Waiting for more data.")
        return  # exit gracefully, scheduler will retry on next cycle

    df = df.loc[raw_features.index]

    # v2.1: infer_regime_multiscale takes X_pca and X_std separately.
    # Wasserstein context is computed internally from X_std.
    # wlabels is no longer a caller responsibility.
    df["RegimeLabel"] = infer.infer_regime_multiscale(
        X_pca     = X_pca,
        X_std     = X_std,
        df_index  = df.index,
        model     = infer.hmmf,
        governor  = regime_governor,
        clusterer = infer.clusterer,
    )

    # SEGMENT SUMMARY
    segments = infer.summarize_regime_periods(df)
    print("\n✲✦▾ Hybrid Wasserstein + HMM Regime Inference:\n")
    print(f"{'Time':<13} {'Regime':<14} {'Direction':<12} {'Duration':<12}")
    print("⎯" * 100)

    for start, end, reg in segments:
        s = start.strftime("%H:%M")
        e = end.strftime("%H:%M")
        time_str = f"{s}–{e}"

        # Regime
        if hasattr(reg, "cls"):
            regime_str = f"{reg.cls:<14}"

            # Direction
            if reg.direction == "Up":
                dir_str = "Up ↑"
            elif reg.direction == "Down":
                dir_str = "Down ↓"
            else:
                dir_str = "Neutral →"
            dir_str = f"{dir_str:<12}"
        else:
            regime_str = f"{str(reg):<14}"
            dir_str = f"{'':<12}"

        # Duration
        duration_min = max(1, int((end - start).total_seconds() // 60))
        bar = duration_bar(duration_min, 50, width=10)
        dur_str = f"{duration_min:>3}m {bar}"

        line = f"{time_str:<13} {regime_str} {dir_str} {dur_str:>18}"
        print(line)
    print("⎯" * 100)
    latest_regime = df["RegimeLabel"].iloc[-1]



# CORE STRATEGY ---------------------------------------------------------------------------------

def normalize_regime_label(reg):
    """
    Convert either RegimeState or legacy string label into a comparable string.
    """
    if hasattr(reg, "cls"):
        if reg.cls == "Trending":
            if reg.direction == "Down":
                return "Trending-Down"
            if reg.direction == "Up":
                return "Trending-Up"
            return "Trending"
        if reg.cls == "Range":
            if reg.direction == "Up":
                return "Mild-Uptrend"
            if reg.direction == "Down":
                return "Mild-Downtrend"
            return "Range"
        if reg.cls in ("Choppy", "Transitional"):
            return reg.cls
        return "Unknown"
    return str(reg)

def core_step():
    """
    Core step function to be scheduled periodically.
    """
    global latest_regime

    print(f"\n[Hybrid Regime Inference] Running core step at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    latest_label = normalize_regime_label(latest_regime)
    print(f"\nLatest regime: {latest_label}")
    print("⎯" * 100)

    # ---- TREND REGIMES (direction-agnostic at gate level) ----
    if latest_label in ("Trending", "Trending-Up", "Trending-Down"):
        msg = f"Regime : {latest_label} → proceed with trend strategy"
        print(msg)
        # trend_strategy(direction="down" if latest_label.endswith("Down") else "up")
        return

    # ---- TRANSITIONAL (explicit block) ----
    elif latest_label == "Transitional":
        msg = "Regime : Transitional → halt (no new risk)"
        print(msg)
        return

    # ---- RANGE / MILD TREND ----
    elif latest_label in ("Range", "Mild-Uptrend", "Mild-Downtrend"):
        msg = f"Regime : {latest_label} → proceed with range / mean-reversion strategy"
        print(msg)
        return

    # ---- CHOPPY ----
    elif latest_label == "Choppy":
        msg = "Regime : Choppy → skip / hold"
        print(msg)
        return

    # ---- FALLBACK ----
    elif latest_label == "Unknown":
        msg = f"Unknown regime label: {latest_label} → default hold"
        print(msg)
        return

    print("⎯" * 100)


# Note: concurrent job updates may require a threading.Lock in live use
# Run regime inference every 5 minutes
# Core step executes every 30 seconds using latest regime state


# SETUP SCHEDULER ---------------------------------------------------------------------------------
def setup_scheduler(scheduler):
    """
    Configures jobs with optimized intervals, misfire grace, and logging.
    """

    scheduler.add_job(
        core_step,
        trigger='interval',
        seconds=30,
        id='trend_step',
        max_instances=1,
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=5
    )

    scheduler.add_job(
        regime_inference,
        trigger='interval',
        seconds=300,  # run every 5 minutes
        id='regime_inference',
        replace_existing=True
    )
    print("⚙ Scheduler configured")

if __name__ == "__main__":
    print("⚙ Initializing Hybrid Regime Inference Scheduler...")
    print("⎯" * 100)
    print("\n⎯  Change API_KEY and update SYMBOL if outdated ⎯ \n")
    scheduler = BackgroundScheduler(timezone=IST)
    setup_scheduler(scheduler)

    regime_inference()  # Initial run

    from apscheduler.jobstores.base import JobLookupError
    try:
        scheduler.start()
    except SchedulerAlreadyRunningError:
        print("Scheduler already running; continuing.")

    try:
        print("OpenAlgo Python Bot is running.")
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("◘ Signal received. Initiating shutdown...")
        sys.exit(0)
