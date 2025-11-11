

# **Hybrid Regime Inference**

### *HMM + Wasserstein Hybrid Model for Market Structure Detection*

---

## **Overview**

This repository implements a **hybrid market regime inference system** combining **Hidden Markov Models (HMM)** with a **Wasserstein-based clustering algorithm** to classify live market behavior into *Trending, Range, Choppy,* and *Transitional* states.

It is designed for **real-time regime detection** and **quantitative market structure analysis** using compact, interpretable features derived from price, volatility, and momentum dynamics.

---

## **Key Features**

* **Hybrid Modeling Architecture** — integrates *probabilistic inference (HMM)* and *distributional clustering (Wasserstein)*
* **Multi-Scale Regime Logic** — short, medium, and long-term smoothing with entropy-confidence filtering
* **Context-Aware Transitions** — persistence governor to reduce spurious flips between regimes
* **Production-Ready Integration** — easily connects with live data APIs (OpenAlgo, Upstox, etc.)
* **Graceful Fallbacks** — handles early-session and insufficient-data scenarios without breaking scheduler
* **Diagnostics & Visualization** — supports clean visual overlays of market structure and regime segments

---


### **Core Concept**

The hybrid model combines two complementary inference layers:

1. **HMM Layer (Temporal Probabilistic Inference):**
   Learns transition probabilities between hidden regimes (using GaussianHMM). Captures sequential dependencies — how likely the market is to stay or switch regimes.

2. **Wasserstein Layer (Distributional Geometry):**
   Clusters normalized market features (returns, volatility, momentum) using Wasserstein distance — a geometry-aware metric sensitive to the *shape* of price distributions.
   It provides *structural validation* — whether a current feature distribution aligns with known regime archetypes.

3. **Hybrid Integration Logic:**

   * Posterior probabilities from the HMM are weighted by Wasserstein proximity scores.
   * Confidence (`max(p)`) and entropy (`-∑p log p`) metrics are computed to gauge decisiveness.
   * Final regime labels are assigned through a gating logic:

     * If Wasserstein and HMM disagree, confidence or entropy thresholds decide.
     * Transitional states are recognized where entropy is high and distance scores conflict.

---

### **Trainer Module (Offline Phase)**

* Input features: momentum, volatility ratio, ATR-based normalization, rolling returns.
* StandardScaler normalization.
* Separate unsupervised training:

  * `GaussianHMM(n_components=3–4)` for temporal pattern learning.
  * `WassersteinClusterer(n_clusters=3–4)` for geometric clustering.
* Model artifacts stored via `joblib`:

  * `regime_hmm.pkl`
  * `regime_wasserstein.pkl`
  * `regime_scaler.pkl`

---

### **Inference Module (Online Phase)**

* Loads pretrained models once (`load_models_once()`).
* Fetches live data from OpenAlgo API (1m until 10:30, then 5m).
* Scales input features and computes:

  * HMM posterior probabilities
  * Wasserstein cluster distances
  * Confidence + entropy diagnostics
* Labels regimes dynamically and visualizes regime segments (e.g., 09:15–09:55 Uptrend).

---

### **Key Output Example**

```
✦✱▴ Regime Segments:
09:15–09:55 – Uptrend
10:00–10:30 – Transitional
11:10–15:25 – Uptrend

Regime Distribution:
Uptrend         0.813
Choppy          0.107
Transitional    0.080
```

---

### **Strengths**

* Merges **temporal memory** (HMM) with **distributional geometry** (Wasserstein).
* More stable than pure HMM during structural breaks or volatile transitions.
* Better generalization to unseen market behavior.
* Provides interpretable diagnostics (confidence, entropy).

### **Weaknesses**

* Requires consistent feature scaling and windowed retraining.
* Sensitive to drift — Wasserstein clusters may need periodic recalibration.
* Increased compute vs standalone models.

### **Why It Works Better**

The hybrid approach brings the *best of both worlds* —
**HMM** captures the *when*, **Wasserstein** captures the *what*.
Together they yield a regime map that is both *temporally coherent* and *geometrically accurate*.

---


## **Repository Overview**

| File / Module                             | Description                                                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **`hybrid_regime_infer.py`** | Core inference engine combining Wasserstein clustering and Gaussian HMM logic. Includes lazy model loading and multi-scale smoothing. |
| **`hybrid_wes_hmm_trainer.py`** | Training module |
| **`inference_plotter.py`**               | Diagnostic runner and visualizer. Fetches recent data, runs full inference, and plots regime labels.                                  |
| **`usage_example.py`**               | Diagnostic runner and visualizer. Fetches recent data, runs full inference, and plots regime labels.                                  |
| **`config.py`**                           | Stores API credentials and server details. Replace `YOUR_API_KEY` with your actual OpenAlgo key before running.                       |
| **`data/`**                              | Contains pre-trained `.pkl` model files — HMM, Wasserstein centroids, and StandardScaler.                                             |
| **`LICENSE`**                             | Legal license (Apache V2.0).                                                                                               |
| **`README.md`**                           | This documentation.                                                                                                                   |
| **`Hybrid_Wasserstein_HMM_Regime_Detection.pdf`**                           | Paper describing the workings of the methodology         |



---

## **Conceptual Model**

1. **Feature Extraction**
   Computes normalized volatility, slope, ADX, ATR, and R² features from OHLC data.

2. **Hidden Markov Model (HMM)**
   Learns temporal patterns and smooths state transitions probabilistically.

3. **Wasserstein Clusterer**
   Classifies recent return distributions into *trend-like*, *range-like*, or *choppy* clusters.

4. **Regime Governor**
   Enforces persistence rules (minimum hold duration) and smooths transitions using entropy–confidence logic.

5. **Final Regime Output**
   Each bar receives one of four hybrid labels:

   * **Trending**
   * **Range**
   * **Choppy**
   * **Transitional**

---


## **Installation**

```bash
git clone git@github.com:kratu/wess_hmm.git

# Install dependencies
pip install -r requirements.txt
```

---

## **Configuration**

All runtime settings are stored in `config.py`:

For running integration_example.py replace this your OpenAlgo API_KEY

```python
API_KEY  = os.getenv("OPENALGO_API_KEY", "YOUR_API_KEY")
API_HOST = os.getenv("OPENALGO_API_HOST", "http://127.0.0.1:5000")
```

---

## **Usage**

### **1. Run the Diagnostic Inference**

```bash
python run_inference.py
```

This script:

* Fetches live or recent data from OpenAlgo
* Runs multi-scale HMM + Wasserstein regime inference
* Prints segment breakdowns
* Optionally plots regime labels over price

---

### **2. Integrate Into Trading Logic**

A complete example is provided in **`usage_example.py`**.
It demonstrates:

* Fetching 5-minute intraday data
* Computing features (returns, ADX, ATR, slope, R², volatility)
* Running Hybrid inference on live data
* Falling back to 1m timeframe during early session
* Printing current regime + segment summary in real time

---

## **Ongoing Refinement / Future Improvements**

* Retrain using expanded historical data (2008–2025)
* Improve evaluation accuracy for Range and Choppy
* Apply additional smoothing for cleaner boundary transitions

---

## **Intended Use**

This repository is meant for **quantitative research, algorithmic experimentation, and educational use**.
It provides a reproducible framework to explore regime-aware trading, model validation, and adaptive system design — not a production trading signal generator.

---

## **Legal Disclaimer**

```
DISCLAIMER

This repository and all associated code, documentation, and examples
are provided strictly for educational, research, and training purposes.

The author makes no warranty regarding accuracy, completeness,
reliability, or fitness for any trading or financial application.

All trading decisions are made at your own risk. The author
assumes no liability for any financial loss, damage, or misuse.

This project does not constitute financial advice or an invitation to trade.
Users are responsible for verifying correctness, suitability, and regulatory compliance.
```

---

## **Citation**

If you use this framework:

> **Hybrid Regime Inference: A Probabilistic-Distributional Model for Market Structure Detection (2025)**
> [https://github.com/kratu/wess_hmm](https://github.com/kratu/wess_hmm)

---

## **Author & License**

Developed by **Jeevan Jonas**
Artist · UX  · Quant Designer

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

© 2025 Jeevan Jonas — Licensed under the Apache License, Version 2.0.  
You may use, modify, and distribute this software under the terms of the Apache 2.0 License.  
Open-source; suitable for research, education, and derivative development.

