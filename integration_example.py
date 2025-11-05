
import hybrid_regime_infer as infer
from datetime import datetime, timedelta, time as dtime
import pandas as pd
import numpy as np
from openalgo import api
from pytz import timezone
from config import API_KEY, API_HOST

client = api(api_key=API_KEY, host=API_HOST)

SYMBOL = "NIFTY"
IST = timezone("Asia/Kolkata")

def regime_inference():
    global latest_regime

    today = datetime.now(IST).strftime("%Y-%m-%d")

    df = client.history(
        symbol=SYMBOL,
        exchange="NFO",
        interval="5m",
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
        print(f"{s}–{e} – {label}")
    print("⎯" * 100)
    latest_regime = df["RegimeLabel"].iloc[-1]

if __name__ == "__main__":
    regime_inference()
