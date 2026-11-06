import pandas as pd
from typing import Dict, Any, List
from backend.app.services.market_data import MarketDataService
from backend.app.ml.indicators import TechnicalIndicators

class AnalyticalRecommendationEngine:
    """
    Transparent rule-based analytical signal engine.
    Combines LSTM predicted price direction with RSI, MACD, and volatility signals.
    Does NOT use a secondary AI model.
    """

    @staticmethod
    def generate_recommendation(
        ticker: str,
        current_price: float,
        predicted_price: float,
        horizon_days: int = 5
    ) -> Dict[str, Any]:
        ticker = ticker.upper().strip()
        
        # Calculate predicted price movement percentage
        predicted_change_pct = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0.0

        # Fetch market data and calculate indicators
        df = MarketDataService.get_stock_data(ticker, period="6mo")
        df_ind = TechnicalIndicators.add_all_indicators(df)
        latest_ind = df_ind.iloc[-1]

        rsi = float(latest_ind.get('RSI_14', 50.0))
        macd = float(latest_ind.get('MACD', 0.0))
        macd_hist = float(latest_ind.get('MACD_Hist', 0.0))
        sma_20 = float(latest_ind.get('SMA_20', current_price))
        sma_50 = float(latest_ind.get('SMA_50', current_price))
        volatility = float(latest_ind.get('Volatility_20', 0.20))

        reasons: List[str] = []
        buy_score = 0
        sell_score = 0

        # Rule 1: Predicted Price Change
        if predicted_change_pct >= 2.5:
            buy_score += 2
            reasons.append(f"LSTM forecasts price gain of +{predicted_change_pct:.2f}% over next {horizon_days} trading days.")
        elif predicted_change_pct <= -2.5:
            sell_score += 2
            reasons.append(f"LSTM forecasts price decline of {predicted_change_pct:.2f}% over next {horizon_days} trading days.")
        else:
            reasons.append(f"LSTM forecasts moderate price movement of {predicted_change_pct:+.2f}% (within neutral range).")

        # Rule 2: RSI Signals
        if rsi < 35:
            buy_score += 1
            reasons.append(f"RSI is {rsi:.1f}, indicating oversold condition (potential bullish reversal).")
        elif rsi > 65:
            sell_score += 1
            reasons.append(f"RSI is {rsi:.1f}, indicating overbought condition (potential downside pressure).")
        else:
            reasons.append(f"RSI is {rsi:.1f} (in neutral territory).")

        # Rule 3: MACD Momentum
        if macd_hist > 0:
            buy_score += 1
            reasons.append(f"MACD histogram is positive ({macd_hist:+.3f}), confirming bullish momentum.")
        elif macd_hist < 0:
            sell_score += 1
            reasons.append(f"MACD histogram is negative ({macd_hist:+.3f}), indicating bearish momentum.")

        # Rule 4: Moving Average Trend Alignment
        if current_price > sma_20 > sma_50:
            buy_score += 1
            reasons.append(f"Price (${current_price:.2f}) is above 20-day (${sma_20:.2f}) and 50-day (${sma_50:.2f}) SMAs (strong uptrend).")
        elif current_price < sma_20 < sma_50:
            sell_score += 1
            reasons.append(f"Price (${current_price:.2f}) is below 20-day (${sma_20:.2f}) and 50-day (${sma_50:.2f}) SMAs (downtrend).")

        # Determine Final Analytical Signal
        if buy_score >= 3 and buy_score > sell_score:
            signal = "BUY"
            confidence = min(0.95, 0.60 + (buy_score * 0.08))
        elif sell_score >= 3 and sell_score > buy_score:
            signal = "SELL"
            confidence = min(0.95, 0.60 + (sell_score * 0.08))
        else:
            signal = "HOLD"
            confidence = 0.50 + abs(buy_score - sell_score) * 0.05

        return {
            "ticker": ticker,
            "signal": signal,
            "confidence_score": round(confidence, 2),
            "reasons": reasons,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "predicted_change_pct": round(predicted_change_pct, 2)
        }
