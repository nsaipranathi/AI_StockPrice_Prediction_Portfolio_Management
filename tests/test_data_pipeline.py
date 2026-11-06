import unittest
import pandas as pd
import numpy as np
from backend.app.ml.data_pipeline import TimeSeriesDataPipeline

class TestTimeSeriesDataPipeline(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
        np.random.seed(42)
        close = 150 + np.cumsum(np.random.randn(300))
        high = close + np.random.rand(300) * 3
        low = close - np.random.rand(300) * 3
        open_p = close + np.random.randn(300)
        volume = np.random.randint(5000, 500000, size=300)

        self.df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume
        }, index=dates)

    def test_pipeline_preparation(self):
        pipeline = TimeSeriesDataPipeline(lookback_window=30)
        splits, metadata = pipeline.prepare_data(self.df, train_ratio=0.7, val_ratio=0.15)
        
        self.assertEqual(metadata["lookback_window"], 30)
        self.assertEqual(metadata["num_features"], 10)
        
        # Verify 3D tensor shapes: (samples, lookback, features)
        self.assertEqual(splits["X_train"].ndim, 3)
        self.assertEqual(splits["X_train"].shape[1], 30)
        self.assertEqual(splits["X_train"].shape[2], 10)
        self.assertEqual(splits["y_train"].ndim, 1)

if __name__ == "__main__":
    unittest.main()
