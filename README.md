
# **Hybrid Regime Inference**

### *HMM + Wasserstein Hybrid Model for Market Structure Detection*

---

### **Overview**

This repository implements a **hybrid market regime inference system** combining **Hidden Markov Models (HMM)** with a **Wasserstein-based clustering algorithm** to classify live market behavior into *Trending, Range, Choppy,* and *Transitional* states.

It is designed for **real-time regime detection** and **quantitative market structure analysis** using compact, interpretable features derived from price, volatility, and momentum dynamics.

---

### **Key Features**

* **Hybrid Modeling Architecture** — integrates *probabilistic inference (HMM)* and *distributional clustering (Wasserstein)*
* **Multi-Scale Regime Logic** — short, medium, and long-term smoothing with entropy-confidence filtering
* **Context-Aware Transitions** — persistence governor to reduce spurious flips between regimes
* **Production-Ready Integration** — easily connects with live data APIs (OpenAlgo, Upstox, etc.)
* **Graceful Fallbacks** — handles early-session and insufficient-data scenarios without breaking scheduler
* **Diagnostics & Visualization** — supports clean visual overlays of market structure and regime segments


---

### **Repository Overview**

| File / Module                             | Description                                                                                                                                                          |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`intelligence/hybrid_regime_infer.py`** | Core inference engine combining Wasserstein clustering and Gaussian HMM logic. Includes lazy model loading, multi-scale smoothing, and real-time confidence metrics. |
| **`run_inference_plot.py`**                    | Diagnostic runner and visualizer. Fetches recent data, runs full inference, and plots regime labels on the price chart. Ideal for testing and demonstration.         |
| **`modules/config.py`**                   | Configuration file storing API credentials and server details. Replace `YOUR_API_KEY` with your actual OpenAlgo key before running.                                  |
| **`intelligence/data/`**                  | Directory containing pre-trained `.pkl` model files — HMM, Wasserstein cluster centroids, and StandardScaler.                                                        |
| **`LICENSE`**                             | Legal license (MIT or CC BY-NC-SA 4.0). See section below.                                                                                                           |
| **`README.md`**                           | This documentation.                                                                                                                                                  |


### **Conceptual Model**

1. **Feature Extraction**
   Computes normalized volatility, slope, ADX, ATR, and R² features from OHLC data.

2. **Hidden Markov Model (HMM)**
   Learns temporal patterns and smooths state transitions probabilistically.

3. **Wasserstein Clusterer**
   Classifies recent return distributions into *trend-like*, *range-like*, or *choppy* clusters based on distributional geometry.

4. **Regime Governor**
   Enforces persistence rules (minimum hold duration) and smooths transitions using entropy–confidence logic.

5. **Final Regime Output**
   Each bar receives a hybrid label:

   * **Trending** (strong directional bias)
   * **Range** (balanced movement)
   * **Choppy** (high volatility noise)
   * **Transitional** (indeterminate or conflicting signals)

---

### **Example Output**

```
Regime Segments:
09:15–09:55 – Trending
10:00–10:05 – Transitional
10:10–10:25 – Choppy
10:30–11:05 – Transitional
11:10–11:35 – Trending
11:40–11:50 – Choppy

Regime Distribution:
Trending        0.469
Transitional    0.312
Choppy          0.219
```

---

### **Installation**

```bash
git clone git@github.com:kratu/wess_hmm.git
```

Config

Change 'config.py' with your OpenAlgo API KEY

For visualization:


### **Configuration**

All runtime settings are stored in `modules/config.py`:

```python
import os

API_KEY  = os.getenv("OPENALGO_API_KEY", "YOUR_API_KEY")
API_HOST = os.getenv("OPENALGO_API_HOST", "http://127.0.0.1:5000")
WS_URL   = "ws://127.0.0.1:8765"
```

Before running the inference, **replace** `"YOUR_API_KEY"` with your valid key or export it securely as an environment variable:

```bash
export OPENALGO_API_KEY="your_real_key_here"
export OPENALGO_API_HOST="https://api.openalgo.in"
```

---

### **Usage**

#### **1. Run the diagnostic inference**

Use the standalone runner to test the hybrid model on recent 5-minute bars:

```bash
python run_inference.py
```

This script:

* Fetches live or recent data from OpenAlgo,
* Runs multi-scale HMM + Wasserstein regime inference,
* Prints segment breakdowns (e.g., Trending / Range / Choppy),
* Optionally plots regime labels over price.


### **2. Integrate into Trading Logic**

A complete working example is provided in **`integration_example.py`**.
It demonstrates how to:

* Fetch 5-minute intraday data from **OpenAlgo**.
* Compute required technical features (returns, ADX, ATR, slope, R², volatility).
* Run the **Hybrid HMM + Wasserstein** inference on live data.
* Gracefully skip execution when early-session data is insufficient.
* Print regime segments and current label in real time.

#### **Typical Integration Flow**

```python
import hybrid_regime_infer as infer
infer.load_models_once()

df = client.history(symbol="NIFTY", exchange="NFO", interval="5m", ...)
features = infer.compute_features(df)

if features is not None and len(features) > 10:
    result = infer.infer_hybrid_regime_for_live_data(df)
    if result["regime"] in ["Trending", "Range"]:
        proceed_with_strategy()
    else:
        skip_trade()
```

---

### **Intended Use**

This repository is meant for **quantitative research, algorithmic experimentation, and educational use**.
It provides a reproducible framework to explore regime-aware trading, model validation, and adaptive system design — not an end-user trading signal generator.

---

### **Legal Disclaimer**

```
DISCLAIMER

This repository and all associated code, documentation, and examples are provided
strictly for educational, research, and training purposes.

The author does not make any representation or warranty, express or implied,
regarding the accuracy, completeness, reliability, or suitability of the code
for any financial or trading application.

By using this code, you acknowledge and agree that all trading and investment
decisions are made solely at your own risk. The author assumes no liability for
any financial loss, trading losses, or damages of any kind arising from the use,
misuse, or interpretation of this code, its outputs, or related materials.

This project does not constitute financial advice or an invitation to trade.
Users are solely responsible for verifying the correctness, fitness, and
regulatory compliance of their own implementations.
```

---

### **License**

MIT License (recommended)
You are free to use, modify, and distribute the code provided that attribution is maintained and the same license is included in derivative works.

---

### **Citation**

If you use this framework in your research or implementation, please cite it as:

> **Hybrid Regime Inference: A Probabilistic-Distributional Model for Market Structure Detection (2025)**
> [GitHub Repository](https://github.com/kratu/wess_hmm)

---

### **Author & License**

Developed by **Jeevan Jonas**  
Visual Artist · UX Designer · Algorithmic Trader

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
© 2025 Jeevan Jonas — Released under the MIT License.  
This repository is open-source and may be used, modified, and distributed freely with proper attribution.  
Use responsibly; this project is intended strictly for **educational and research purposes**.

