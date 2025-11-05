
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

### **Core Components**

| Module                                | Description                                                 |
| ------------------------------------- | ----------------------------------------------------------- |
| `intelligence/hybrid_regime_infer.py` | Core inference engine integrating HMM + Wasserstein models  |
| `intelligence/data/`                  | Pre-trained model files (`.pkl` for HMM, clusterer, scaler) |
| `modules/config.py`                   | API configuration and constants                             |
| `test_infer.py`                       | Standalone diagnostic runner with visualization support     |
| `trend_adaptive.py`                   | Example integration with real-time trading system           |

---

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
cd hybrid-regime-inference
pip install -r requirements.txt
```

For visualization:

```bash
python test_infer.py
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

