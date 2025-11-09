
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

SYMBOL = "NIFTY25NOV25FUT"
IST = timezone("Asia/Kolkata")

def regime_inference():
    global latest_regime

    now = datetime.now(IST)
    today = datetime.now(IST).strftime("%Y-%m-%d")

    # --- Hybrid Rule: Force 1m until 10:30, else 5m ---
    if now.time() < dtime(10,30):
        timeframe = "1m"
    else:
        timeframe = "5m"  # usually "5m"

    df = client.history(
        symbol=SYMBOL,
        exchange="NFO",
        interval=timeframe,
        start_date=today,
        end_date=today
    )
    # --- Handle API returning dict instead of DataFrame ---
    if isinstance(df, dict):
        candles = df.get("data") or df.get("result", {}).get("data", [])
        if not candles:
            print(f"[Hybrid] No data from OpenAlgo. Ensure API_KEY is edited and SYMBOL is valid.")
            sys.exit(0)
        df = pd.DataFrame(candles)

    # --- Empty or invalid data guard ---
    if df is None or df.empty:
        print(f"[Hybrid] Empty dataset for {SYMBOL} — exiting gracefully.")
        return  # exit gracefully, scheduler will retry on next cycle

    # Normalize datetime
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    print(f"Fetched {len(df)} bars: {df.index.min()} → {df.index.max()}")

    # --------------------------------------------------
    # INFERENCE PIPELINE
    # --------------------------------------------------
    infer.load_models_once()

    features = infer.compute_features(df)

    if features is None or len(features) < 10:
        print(f"[RegimeGuard] Not enough bars yet ({len(df)}). Waiting for more data.")
        return  # exit gracefully, scheduler will retry on next cycle

    features = features.reindex(df.index).dropna()
    df = df.loc[features.index]

    X_scaled = np.clip(infer.scaler.transform(features), -3, 3)
    wlabels = infer.compute_wasserstein_context(
        X_scaled, infer.clusterer, feature_index=0,
        window=len(infer.clusterer.centroids[0])
    )

    gov = infer.RegimeGovernor(min_hold=infer.MIN_HOLD_MIN)
    df["RegimeLabel"] = infer.infer_regime_multiscale(
        X_scaled, df.index, infer.hmmf, gov, infer.clusterer, wlabels
    )

    # --------------------------------------------------
    # SEGMENT SUMMARY
    # --------------------------------------------------
    segments = infer.summarize_regime_periods(df)
    print("\n✲✦▾ Hybrid Wasserstein + HMM Regime Inference:")
    print("⎯" * 100)
    for start, end, label in segments:
        s = start.strftime("%H:%M")
        e = end.strftime("%H:%M")
        print(f"{s}–{e} – {label}")
    print("⎯" * 100)
    latest_regime = df["RegimeLabel"].iloc[-1]



# CORE STRATEGY ---------------------------------------------------------------------------------

def core_step():
    """
    Core step function to be scheduled periodically.
    """
    global latest_regime

    print(f"\n[Hybrid Regime Inference] Running core step at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nLatest regime: {latest_regime}")
    print("⎯" * 100)

    if latest_regime == "Trending":
        msg = "Regime : Trending → proceed with trend strategy"
        print(msg) 
      
    elif latest_regime == "Transitional":
        msg = "Regime : Transitional → halt "
        print(msg)
        #return
    
    elif latest_regime in ("Range", "Mild-Uptrend"):
        msg = "Regime : Range/Mild-Uptrend → proceeding with range strategy"
        print(msg) 

    elif latest_regime == "Choppy":
        msg = "Regime : Choppy → skip / hold "
        print(msg)
        return 
    
    else:
        msg = "Unknown regime label."
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

    # Monitor structure: every 5 seconds
    scheduler.add_job(
        regime_inference,
        trigger='interval',
        seconds=300, #run every 5 minutes 
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

    regime_inference() #Initial run

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
