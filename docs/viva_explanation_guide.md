# StockMind AI — Viva & Technical Defense Guide

This document prepares you for your B.Tech Data Science major project viva examination.

---

## 1. Top 5 Viva Questions & Recommended Answers

### Q1: Why did you choose LSTM over traditional Machine Learning models like XGBoost or Random Forest?
**Answer**:
Stock price time-series data possesses temporal sequential dependencies where future states depend heavily on historical sequences. Standard decision tree models (XGBoost, Random Forest) treat observations as independent feature vectors and fail to retain memory over sequential time windows. **LSTM (Long Short-Term Memory)** is a recurrent neural network architecture specifically designed with input, forget, and output gates that retain long-term cell states while avoiding the vanishing gradient problem.

---

### Q2: How did you prevent data leakage during time-series preprocessing?
**Answer**:
To eliminate look-ahead bias and data leakage:
1. **Chronological Splitting**: We split our historical dataset into 70% Training, 15% Validation, and 15% Testing strictly chronologically without random shuffling.
2. **Scaler Fitting**: The `MinMaxScaler` parameters ($\min, \max$) were fitted **EXCLUSIVELY on the training set**. Validation and test set features were scaled using the pre-fitted training parameters.

---

### Q3: Why is $R^2$ Score used instead of Accuracy for model evaluation?
**Answer**:
Accuracy is a metric for discrete classification tasks (e.g. predicting classes 0 or 1). Stock price forecasting is a continuous regression problem. We evaluate our model using:
- **MAE (Mean Absolute Error)**: Dollar error magnitude.
- **RMSE (Root Mean Squared Error)**: Penalizes larger price deviations.
- **MAPE (Mean Absolute Percentage Error)**: Relative percentage error.
- **$R^2$ Score**: Measures the proportion of variance in stock prices explained by the LSTM model compared to a simple mean baseline.

---

### Q4: How does your recommendation engine work without introducing a second AI model?
**Answer**:
The project strictly obeys the single-AI-model constraint. The recommendation engine is an **analytical rule-based expert module** that evaluates quantitative thresholds:
- **BUY**: Predicted price gain $\ge +2.5\%$, RSI $< 65$ (not overbought), MACD histogram $> 0$ (bullish momentum).
- **SELL**: Predicted price decline $\le -2.5\%$, RSI $> 35$, MACD histogram $< 0$ (bearish momentum).
- **HOLD**: Moderate predicted price change within $[-2.5\%, +2.5\%]$ or conflicting momentum signals.
Each recommendation outputs clear, human-readable explanatory reasons for complete transparency.

---

### Q5: How are portfolio risk metrics calculated?
**Answer**:
Risk analytics use standard statistical financial math:
- **Annualized Volatility**: $\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}$
- **Sharpe Ratio**: $S = \frac{R_p - R_f}{\sigma_p}$, measuring risk-adjusted excess return over a $4.0\%$ risk-free rate.
- **Maximum Drawdown**: Largest percentage drop from historical peak to trough.
- **HHI Concentration**: $\sum w_i^2 \times 10000$, measuring portfolio diversification across holdings.
