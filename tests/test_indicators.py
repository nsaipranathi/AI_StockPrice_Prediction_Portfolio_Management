import unittest
import pandas as pd
import numpy as np
from backend.app.ml.indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        # Generate 100 days of mock stock data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100))
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_p = close + np.random.randn(100)
        volume = np.random.randint(1000, 100000, size=100)

        self.df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume
        }, index=dates)

    def test_add_all_indicators(self):
        df_ind = TechnicalIndicators.add_all_indicators(self.df)
        expected_cols = [
            "SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD",
            "MACD_Signal", "MACD_Hist", "BB_Upper", "BB_Lower",
            "Daily_Return", "Volatility_20", "Volume_Change"
        ]
        for col in expected_cols:
            self.assertIn(col, df_ind.columns)
            self.assertFalse(df_ind[col].isnull().any(), f"NaN values found in column {col}")

    def test_latest_indicator_summary(self):
        summary = TechnicalIndicators.get_latest_indicator_summary(self.df)
        self.assertIn("rsi_14", summary)
        self.assertIn("macd", summary)
        self.assertIn("sma_20", summary)
        self.assertTrue(0 <= summary["rsi_14"] <= 100)

if __name__ == "__main__":
    unittest.main()
