# 📈 Stock Price Predictor v2

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange.svg)](https://www.tensorflow.org/)
[![NLP](https://img.shields.io/badge/NLP-FinBERT%20%28Hugging%20Face%29-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![Testing](https://img.shields.io/badge/Testing-Pytest%20%28139%20Tests%29-066da5.svg)](https://docs.pytest.org/)

An enterprise-grade **AI-powered financial analytics and time-series forecasting engine**. This system pairs a hybrid **Deep Learning (BiLSTM + Attention + GRU)** network with domain-optimized **Natural Language Processing (FinBERT)** and quantitative **Technical Indicators** to predict daily equity movements and execute systematic trading backtests.

---

## 🚀 Architectural Overview

This system coordinates multi-source data extraction pipelines, text analytics operations, and recurrent tensor deep learning models inside a modular, decoupled framework designed for high performance and clean segregation of concerns.

```
┌────────────────────────┐      ┌────────────────────────┐
│    Yahoo Finance API   │      │     News Aggregator    │
│  (Historical Price)    │      │   (Financial Feeds)    │
└───────────┬────────────┘      └───────────┬────────────┘
│                               │
▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│  Technical Indicators  │      │   FinBERT NLP Engine   │
│ (pandas-ta Oscillator) │      │  (Confidence Weighted) │
└───────────┬────────────┘      └───────────┬────────────┘
│                               │
└───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│    Data Preprocessing Layer   │
│  (Left Join & Temporal Align) │
└───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│  Hybrid Neural Network Matrix │
│   (BiLSTM + Attention + GRU)  │
└───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│    System Metrics Dashboard   │
│  (Streamlit UI & Backtesting) │
└───────────────────────────────┘
```

---

## 🧠 Core Methodology & Models

### 1. Hybrid Neural Network Architecture
Rather than utilizing basic recurrent structures, the predictive framework leverages a custom-engineered, multi-tiered architecture built using the **Keras Functional API**:
* **Bidirectional LSTM Layer 1 (128 units):** Processes input sequences forward and backward through time to capture comprehensive sequential contexts.
* **Spatial Regularization (Dropout = 0.3):** Minimizes node co-dependency and model overfitting across reverse backpropagation iterations.
* **Bidirectional LSTM Layer 2 (64 units):** Compresses sequence representations into deep hierarchical temporal structures.
* **Dot-Product Attention Layer:** Evaluates dynamic contextual weights across lookback horizons, successfully mitigating vanishing gradient memory loss over extended sequences.
* **Gated Recurrent Unit (GRU) Core Layer (32 units):** Condenses dense attention feature tensors into refined target mappings.
* **Dense Output Configurations:** Routes activations through a hidden Dense layer (32 units, ReLU) before map projection down to a singular linear continuous asset closing price forecast.

#### Robust Loss Metrics
Optimization runs on **Huber Loss** rather than standard Mean Squared Error (MSE) to maximize modeling tolerance against sudden market anomalies and volatile outlier tail risks:

$$L_{\delta}(y, f(x)) = \begin{cases} \frac{1}{2}(y - f(x))^2 & \text{for } |y - f(x)| \le \delta, \\ \delta \left(|y - f(x)| - \frac{1}{2}\delta\right) & \text{otherwise.} \end{cases}$$

### 2. Multi-Stage Walk-Forward Cross Validation
To preserve authentic chronological causality and eliminate historical data lookahead bias, training utilizes a rolling 3-Fold **Walk-Forward Validation Framework**:
* **Fold 1:** Train on [0 $\to$ 60%] of timeline $\to$ Validate on [60% $\to$ 72%].
* **Fold 2:** Train on [0 $\to$ 72%] of timeline $\to$ Validate on [72% $\to$ 84%].
* **Fold 3:** Train on [0 $\to$ 84%] of timeline $\to$ Validate on [84% $\to$ 100%].

### 3. Quantitative Feature Engineering Pipeline
The feature extraction layer tracking computes 19 independent coordinates via the `pandas-ta` library engine:
* **Trend Indicators:** 20-Day Simple Moving Average (`MA20`), 50-Day Simple Moving Average (`MA50`), and Average Directional Index (`ADX`).
* **Momentum Oscillators:** Relative Strength Index (`RSI`) and Moving Average Convergence Divergence (`MACD`, `MACD_SIGNAL`, `MACD_HIST`).
* **Volatility Constraints:** Bollinger Bands (`BB_UPPER`, `BB_LOWER`, `BB_MIDDLE`, `BB_WIDTH`, `BB_PERCENT`) and Average True Range (`ATR`).
* **Volume Overlays:** Volume vectors and On-Balance Volume (`OBV`) capital flow accumulation curves.

### 4. Confidence-Weighted Sentiment Ingestion
Unstructured news headline payloads are fetched from market endpoints and processed in memory-safe, hardware-accelerated batches via a **FinBERT** transformer model pipeline (`ProsusAI/finbert`):
1. Categorical class strings (`positive`, `negative`, `neutral`) map to directional polarity signs: $S \in \{-1, 0, 1\}$.
2. Continuous sentiment indices are generated by scaling signs against their softmax probability distributions: $\text{Score} = S \times \text{Probability}$.
3. Multiple intra-day metrics are downsampled using mathematical daily pooling means into a clean timeline data column:

$$\text{Sentiment}_t = \frac{1}{N}\sum_{i=1}^{N} \left( S_i \times \text{Probability}_i \right)$$

---

## 📊 Quantitative Backtesting Engine

Predictive coordinate outputs flow directly into a systematic backtesting loop to check strategy profit performance against standard **Buy-and-Hold** baseline indices:
* **Execution Signaling:** Evaluates trading signals using an absolute $t-1$ index step-shift to lock operational orders exclusively to historical metrics, preventing lookahead leakage.
* **Risk Profile KPIs:** Compiles continuous cumulative equity curves, maximum drawdowns (peak-to-trough drop periods), and the annualized **Sharpe Ratio** to evaluate risk-adjusted performance:

$$\text{Sharpe Ratio} = \frac{\mu_{\text{strategy}}}{\sigma_{\text{strategy}}} \times \sqrt{252}$$

---

## 🏗️ Project Directory Structure

Stock-Price-Predictor-v2/
│
├── src/                            # Core Pipeline Layer Modules
│   ├── init.py                 # Packages namespace control exposure
│   ├── backtesting.py              # Financial backtests, Sharpe ratios, drawdowns
│   ├── data_collection.py          # Exponential backoff Yahoo Finance data downloads
│   ├── data_preprocessing.py       # Left-join vector adjustments & time alignment
│   ├── evaluation.py               # Calculation models (MAE, RMSE, MAPE, Directional Accuracy)
│   ├── feature_engineering.py      # Quantitative indicators calculation loops
│   ├── logger.py                   # Thread-safe color console and file telemetry
│   ├── model_utils.py              # Functional architecture setups & joblib serialization
│   └── sentiment_analysis.py       # Batch FinBERT NLP pipelines & pooling aggregation
│
├── tests/                          # Automated Pytest Framework Layer
│   ├── init.py                 # Testing environment namespace configurations
│   ├── conftest.py                 # Deterministic time-series fixtures & mocking engines
│   ├── test_backtesting.py         # Strategy pipeline verification checks
│   ├── test_data_collection.py     # Network exception handles and schema check tests
│   ├── test_data_preprocessing.py  # Validation constraints and missing data fill sweeps
│   ├── test_evaluation.py          # Regression errors and lift metrics assertions
│   ├── test_feature_engineering.py # Technical indicators execution tests
│   ├── test_model_utils.py         # Topology compilation and dataset shapes verification
│   └── test_sentiment_analysis.py  # Text batch chunking and index parsing validation
│
├── app.py                          # Streamlit UI dashboard and charting pipeline
├── config.py                       # Global settings parameters and env variable casts
├── train_model.py                  # Core cross-validation pipeline optimization controller
├── requirements.txt                # Enterprise third-party dependency list
└── README.md                       # Comprehensive project blueprint documentation

---

## ⚙️ Configuration Setup (`.env`)

The system's optimization and dashboard execution bounds are controlled via a centralized configuration manager. Create a `.env` file in your root folder to customize pipeline hyperparameters:

```ini
# Target Asset Metrics
TICKER=AAPL
START_DATE=2015-01-01
END_DATE=2025-12-31

# Model Topology Constraints
TIME_STEP=60
LSTM_UNITS=128
DROPOUT_RATE=0.3
DENSE_UNITS=32
LEARNING_RATE=0.0001

# Training Optimization Settings
EPOCHS=100
BATCH_SIZE=64
VALIDATION_SPLIT=0.2
EARLY_STOPPING_PATIENCE=10

# Web Dashboard Interface Settings
PAGE_TITLE="AI Stock Price Predictor"
HISTORY_DAYS=90
DATA_BUFFER_DAYS=100
SENTIMENT_MODEL_NAME="ProsusAI/finbert"
```

---

## ▶️ Installation & Execution

1. Clone the Workspace Repository

```Bash
git clone [https://github.com/ItzzKK03/stock-price-predictor-KK-v3.git](https://github.com/ItzzKK03/stock-price-predictor-KK-v3.git)
cd stock-price-predictor-KK-v3
```

2. Set Up a Virtual Environment & Dependencies

Initialize your isolated environment and install the fresh virtual environment dependencies instantly by executing:

```Bash
# Initialize isolated Python environment 
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install compiled dependencies from requirements.txt
pip install -r requirements.txt
```

3. Run the Cross-Validation Training Pipeline

Launches the comprehensive data retrieval, indicator construction, left-join text merging, and walk-forward neural network compilation loops.

```Bash
python train_model.py
```

Outputs: Fitted model files (`stock_predictor.keras`), input/target scalers (`*.joblib`), field schema specifications (`*.json`), and training validation evaluation logs (`*_evaluation.csv`) directly to disk.

4. Launch the Interactive Analytical Dashboard

Starts the dark-themed Streamlit user interface, hosting configuration sliders, live forecasting engines, interactive charts, and quantitative strategy backtesting tabs.

```Bash
streamlit run app.py
```

---

## 🧪 Comprehensive Verification Suite

The engineering workspace incorporates a robust, defensive test framework containing 139 distinct unit and integration checks mapped via Pytest to guarantee multi-source pipeline stability. Mock networks verify yfinance connection throttling behavior, data interpolation steps, network training shapes, and categorical text scoring parameters.

To run the complete verification matrix and check pipeline code coverage, execute:

```Bash
pytest --cov=src tests/ -v
```

---

## 📊 System Performance Metrics

* Mean Absolute Percentage Error (MAPE): $\sim 10\%$ standard trading variation accuracy.
* Directional Trend Accuracy Tracking: $\sim 53\%$ correctness tracking discrete up/down price trajectories on out-of-sample datasets.
* System Optimization Checkpoint: Walk-forward modeling constraints and Huber loss boundaries eliminate historical training lookahead leakages and stabilize weights against unpredictable extreme market tail risks.

---

## 👨‍💻 Engineering Core

Kartik Kant — Artificial Intelligence & Machine Learning Engineer
