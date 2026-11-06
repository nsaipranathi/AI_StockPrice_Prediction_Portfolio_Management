import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LSTMEvaluator:
    """
    Standard evaluation metrics for time-series stock price forecasting.
    Includes MAE, RMSE, MAPE (%), and R² Score.
    """

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates MAE, RMSE, MAPE, and R2 for actual vs predicted stock prices.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        if len(y_true) == 0 or len(y_pred) == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))

        # Handle zero divisions for MAPE safely
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
        else:
            mape = 0.0

        r2 = float(r2_score(y_true, y_pred))

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "r2": round(r2, 4)
        }
