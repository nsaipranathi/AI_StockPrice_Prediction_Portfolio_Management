import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    """
    Technical indicators calculator for stock market data.
    Calculates SMA 20/50, EMA 20, RSI 14, MACD (12, 26, 9), Bollinger Bands,
    Daily Returns, and Volatility.
    """

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends technical indicators as new columns to the stock DataFrame.
        Operates on a copy of the dataframe to avoid side effects.
        """
        data = df.copy()
        
        # Ensure Close price is present
        if 'Close' not in data.columns:
            raise KeyError("DataFrame must contain 'Close' column to compute technical indicators.")

        close = data['Close']

        # Simple Moving Averages (SMA)
        data['SMA_20'] = close.rolling(window=20, min_periods=1).mean()
        data['SMA_50'] = close.rolling(window=50, min_periods=1).mean()

        # Exponential Moving Average (EMA)
        data['EMA_20'] = close.ewm(span=20, adjust=False).mean()

        # Relative Strength Index (RSI - 14 period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        data['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # Bollinger Bands (20 period, 2 std dev)
        sma20 = data['SMA_20']
        rolling_std = close.rolling(window=20, min_periods=1).std().fillna(0)
        data['BB_Upper'] = sma20 + (rolling_std * 2)
        data['BB_Lower'] = sma20 - (rolling_std * 2)

        # Daily Returns & Volatility
        data['Daily_Return'] = close.pct_change().fillna(0)
        data['Volatility_20'] = data['Daily_Return'].rolling(window=20, min_periods=1).std().fillna(0) * np.sqrt(252) # Annualized

        # Volume Change
        if 'Volume' in data.columns:
            data['Volume_Change'] = data['Volume'].pct_change().fillna(0)
        else:
            data['Volume_Change'] = 0.0

        # Fill any remaining NaNs cleanly
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        
        return data

    @staticmethod
    def get_latest_indicator_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Returns a formatted dictionary of the latest technical indicator values."""
        df_ind = TechnicalIndicators.add_all_indicators(df)
        latest = df_ind.iloc[-1]
        
        return {
            "sma_20": round(float(latest.get('SMA_20', 0)), 2),
            "sma_50": round(float(latest.get('SMA_50', 0)), 2),
            "ema_20": round(float(latest.get('EMA_20', 0)), 2),
            "rsi_14": round(float(latest.get('RSI_14', 50)), 2),
            "macd": round(float(latest.get('MACD', 0)), 4),
            "macd_signal": round(float(latest.get('MACD_Signal', 0)), 4),
            "macd_hist": round(float(latest.get('MACD_Hist', 0)), 4),
            "bb_upper": round(float(latest.get('BB_Upper', 0)), 2),
            "bb_lower": round(float(latest.get('BB_Lower', 0)), 2),
            "daily_return_percent": round(float(latest.get('Daily_Return', 0)) * 100, 2),
            "annualized_volatility": round(float(latest.get('Volatility_20', 0)), 4)
        }
