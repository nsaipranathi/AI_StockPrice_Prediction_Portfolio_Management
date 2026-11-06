import unittest
import numpy as np
from backend.app.ml.evaluate_lstm import LSTMEvaluator

class TestLSTMEvaluator(unittest.TestCase):
    def test_calculate_metrics(self):
        y_true = np.array([100.0, 102.0, 105.0, 103.0, 108.0])
        y_pred = np.array([101.0, 101.5, 104.0, 103.5, 107.0])
        
        metrics = LSTMEvaluator.calculate_metrics(y_true, y_pred)
        
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("r2", metrics)
        
        self.assertGreaterEqual(metrics["r2"], -1.0)
        self.assertLessEqual(metrics["r2"], 1.0)
        self.assertGreater(metrics["mae"], 0.0)

if __name__ == "__main__":
    unittest.main()
