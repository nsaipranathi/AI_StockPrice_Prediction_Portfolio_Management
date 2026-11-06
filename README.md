# StockMind AI: AI-Powered Stock Price Prediction and Portfolio Management Using LSTM

[![License: MIT](https://img.shields.io/badge/License-MIT-indigo.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00.svg)](https://tensorflow.org/)

**Application Tagline**: *Intelligent Stock Analytics & Portfolio Management*  
**Academic Level**: Major B.Tech Data Science Project  

---

## 1. Abstract & Project Context
Financial stock markets exhibit non-linear, non-stationary temporal dynamics that present significant challenges for standard quantitative modeling. **StockMind AI** is an intelligent analytics platform centered exclusively around **Long Short-Term Memory (LSTM)** deep learning neural networks to learn temporal dependencies from historical Open-High-Low-Close-Volume (OHLCV) market data and produce multi-step short-term stock price forecasts.

The platform integrates these LSTM predictions into a complete financial suite featuring:
- **Rule-Based Analytical Signals** (BUY, HOLD, SELL with transparent reasoning)
- **Quantitative Portfolio Management & Real-Time P/L Tracking**
- **Statistical Risk Analysis** (Annualized Volatility, Sharpe Ratio, Maximum Drawdown, HHI Concentration)
- **Historical Trading Simulation (Backtesting Engine)**
- **Watchlist & Prediction Audit Logs**
- **Modern Dark Fintech SaaS Glassmorphism Dashboard**

> **CRITICAL ARCHITECTURAL CONSTRAINT**:  
> In compliance with academic requirements, **LSTM is the ONLY AI/DL prediction model** utilized in this system. No secondary machine learning models (XGBoost, Random Forest, SVM, Ensembles) are introduced. All risk metrics and portfolio statistics rely strictly on standard quantitative financial mathematics.

---

## 2. Research Problem & Core Idea
The core research problem addressed by this project is:
> *"Can a Stacked LSTM Deep Learning model effectively learn temporal sequential patterns from historical stock data and yield short-term price forecasts that enhance quantitative portfolio management and risk analytics?"*

### End-to-End System Pipeline
```text
Historical Market Data (yfinance)
            │
            ▼
Chronological Cleaning & Validation
            │
            ▼
Feature Engineering & Technical Indicators (SMA, EMA, RSI, MACD, Volatility)
            │
            ▼
Leak-Free MinMaxScaler & 60-Day Window Sequence Generation
            │
            ▼
Stacked LSTM Deep Learning Neural Network
            │
            ▼
Multi-Step Price Forecast Generation
            │
            ▼
Transparent Rule-Based Signal Engine (BUY / HOLD / SELL + Reasons)
            │
            ▼
Portfolio Valuation & Risk Analytics (Volatility, Sharpe Ratio, Max Drawdown)
            │
            ▼
Modern Dark Glassmorphism SaaS Interface (React + Recharts)
```

---

## 3. Key System Features

| Module | Features & Capabilities |
| :--- | :--- |
| **LSTM Prediction Engine** | 60-day historical windowing, stacked LSTM layers (64 & 32 units) with dropout, EarlyStopping, ModelCheckpointing, multi-step autoregressive forecasting. |
| **Model Evaluation** | Evaluation on unseen test sets reporting MAE, RMSE, MAPE (%), and $R^2$ Score (distinctly separated from "accuracy"). |
| **Technical Indicators** | Real-time calculation of SMA 20/50, EMA 20, RSI (14), MACD (12, 26, 9), Bollinger Bands, Daily Returns, and Volatility. |
| **Analytical Recommendations** | Rule-based decision module combining predicted price direction, RSI boundaries, MACD momentum, and trend alignment with bulleted explanations. |
| **Portfolio Management** | Dynamic tracking of holdings, quantity, purchase price basis, current market valuation, net P/L ($ and %), and asset allocation. |
| **Risk Analysis** | Statistical calculation of annualized volatility ($\sigma_{\text{ann}} = \sigma_{\text{daily}} \sqrt{252}$), Sharpe Ratio, Maximum Drawdown (MDD), and concentration index (HHI). |
| **Historical Simulation** | Backtesting engine executing moving-average momentum rules over custom date ranges without look-ahead bias, displaying equity curves and trade history. |
| **Fintech UI/UX** | Dark glassmorphism theme (`#0B0F19`), responsive Recharts visualizers, skeleton loading states, and search navigation. |

---

## 4. LSTM Model Methodology & Architecture

### Stacked LSTM Architecture (Keras / TensorFlow)
```text
┌─────────────────────────────────────────────────────────┐
│ Input Layer: Sequence Tensor (Batch, 60, 10 Features)   │
├─────────────────────────────────────────────────────────┤
│ Layer 1: LSTM (64 Units, return_sequences=True)         │
│ Layer 2: Dropout (Rate = 0.20)                          │
├─────────────────────────────────────────────────────────┤
│ Layer 3: LSTM (32 Units, return_sequences=False)        │
│ Layer 4: Dropout (Rate = 0.20)                          │
├─────────────────────────────────────────────────────────┤
│ Layer 5: Dense (16 Units, Activation='relu')            │
│ Layer 6: Dense Output (1 Unit, Predicted Scaled Price)  │
└─────────────────────────────────────────────────────────┘
```

### Data Pipeline & Data Leakage Prevention
1. **Chronological Split**: Data is split into **70% Training**, **15% Validation**, and **15% Unseen Testing** strictly chronologically without random shuffling.
2. **MinMaxScaler Integrity**: The `MinMaxScaler` is fitted **EXCLUSIVELY on the Training set**. Validation and Test sets are transformed using the fitted training scalers to prevent data leakage.
3. **Loss Function & Optimizer**: Adam Optimizer ($lr=0.001$), Huber Loss (robust against extreme market outliers).

---

## 5. Evaluation Metrics
The model is evaluated on unseen test data using four quantitative metrics:
- **MAE (Mean Absolute Error)**: $\frac{1}{n} \sum |y_i - \hat{y}_i|$
- **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
- **MAPE (Mean Absolute Percentage Error)**: $\frac{100\%}{n} \sum \left| \frac{y_i - \hat{y}_i}{y_i} \right|$
- **$R^2$ Score (Coefficient of Determination)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

---

## 6. Installation & Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js v18+** and **npm**

### Step 1: Clone Repository & Setup Backend
```bash
git clone https://github.com/your-username/StockMind-AI.git
cd StockMind-AI/backend

# Create virtual environment
python -m venv venv
# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 2: Start Backend Server
```bash
# From the project root or backend folder:
uvicorn backend.app.main:app --reload --port 8000
```
- API Documentation (Swagger): `http://127.0.0.1:8000/docs`

### Step 3: Setup & Start Frontend
```bash
cd ../frontend

# Install node dependencies
npm install

# Start Vite development server
npm run dev
```
- Open Application: `http://localhost:5173`

---

## 7. Viva Explanation Guide

### Quick viva pitch for project reviewers:
1. **Input**: Fetch historical OHLCV data for stock tickers via `MarketDataService`.
2. **Preprocessing**: Compute technical indicators (SMA 20/50, EMA 20, RSI 14, MACD, Volatility) and normalize features using `MinMaxScaler`.
3. **Sequence Windowing**: Construct 60-day historical window tensors for input to the neural network.
4. **Deep Learning**: Train a Stacked 2-Layer LSTM in TensorFlow/Keras with early stopping to learn temporal market patterns.
5. **Prediction**: Autoregressively forecast future stock prices for 1 to 7 trading days.
6. **Analytical Signal**: Combine forecast direction with RSI and MACD rules to output transparent BUY / HOLD / SELL signals with explanations.
7. **Portfolio & Risk**: Compute dynamic holdings P/L, annualized volatility, Sharpe ratio, max drawdown, and strategy backtest equity curves.

---

## 8. Limitations & Future Scope

### Limitations
- Stock market prices are influenced by unpredictable macro-economic events, sentiment spikes, and high volatility.
- LSTM forecasts rely on historical patterns and cannot predict sudden black swan events.
- Model forecasts represent probabilistic estimates and **do NOT constitute financial advice**.

### Future Scope
- Incorporating macroeconomic interest rate data and earnings calendar signals.
- Multi-asset expansion (cryptocurrencies, forex pairs, commodities).
- Automated portfolio rebalancing execution integration.

---

## 9. Academic Disclaimer
*This platform is developed strictly for B.Tech Data Science major project demonstration and academic research purposes. All predictions, simulated backtest results, and analytical recommendations are educational simulations and should not be used as actual investment advice.*
