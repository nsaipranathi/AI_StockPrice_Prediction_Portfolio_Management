import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from backend.app.ml.indicators import TechnicalIndicators

SCALER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "scalers")
os.makedirs(SCALER_DIR, exist_ok=True)

class TimeSeriesDataPipeline:
    """
    Time-series data preprocessing pipeline for LSTM Deep Learning.
    Guarantees strict chronological train/val/test splits without data leakage.
    """

    FEATURE_COLS = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD']
    TARGET_COL = 'Close'

    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        save_scaler_prefix: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Processes stock DataFrame into scaled chronological sequences for LSTM.
        Returns:
            splits: dict containing X_train, y_train, X_val, y_val, X_test, y_test
            metadata: information about dataset sizes and feature columns
        """
        # Ensure indicators are calculated
        data = TechnicalIndicators.add_all_indicators(df)

        # Validate feature columns existence
        for col in self.FEATURE_COLS:
            if col not in data.columns:
                raise KeyError(f"Missing required feature column '{col}' in data pipeline.")

        feature_matrix = data[self.FEATURE_COLS].values
        target_vector = data[[self.TARGET_COL]].values

        n_total = len(data)
        if n_total < self.lookback_window + 50:
            raise ValueError(f"Dataset too small ({n_total} rows). Need at least {self.lookback_window + 50} rows.")

        # Compute split indices chronologically
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # Chronological raw slices for scaling
        train_x_raw = feature_matrix[:n_train]
        train_y_raw = target_vector[:n_train]

        val_x_raw = feature_matrix[n_train - self.lookback_window : n_train + n_val]
        val_y_raw = target_vector[n_train - self.lookback_window : n_train + n_val]

        test_x_raw = feature_matrix[n_train + n_val - self.lookback_window :]
        test_y_raw = target_vector[n_train + n_val - self.lookback_window :]

        # Fit scalers ONLY on training set to prevent data leakage!
        scaled_train_x = self.scaler_x.fit_transform(train_x_raw)
        scaled_train_y = self.scaler_y.fit_transform(train_y_raw)

        scaled_val_x = self.scaler_x.transform(val_x_raw)
        scaled_val_y = self.scaler_y.transform(val_y_raw)

        scaled_test_x = self.scaler_x.transform(test_x_raw)
        scaled_test_y = self.scaler_y.transform(test_y_raw)

        # Save scalers if requested
        if save_scaler_prefix:
            scaler_x_path = os.path.join(SCALER_DIR, f"{save_scaler_prefix}_scaler_x.pkl")
            scaler_y_path = os.path.join(SCALER_DIR, f"{save_scaler_prefix}_scaler_y.pkl")
            joblib.dump(self.scaler_x, scaler_x_path)
            joblib.dump(self.scaler_y, scaler_y_path)

        # Generate windowed sequences
        X_train, y_train = self._create_sequences(scaled_train_x, scaled_train_y)
        X_val, y_val = self._create_sequences(scaled_val_x, scaled_val_y)
        X_test, y_test = self._create_sequences(scaled_test_x, scaled_test_y)

        metadata = {
            "total_records": n_total,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "lookback_window": self.lookback_window,
            "num_features": len(self.FEATURE_COLS),
            "feature_columns": self.FEATURE_COLS,
            "train_start_date": data.index[0].strftime('%Y-%m-%d'),
            "train_end_date": data.index[n_train - 1].strftime('%Y-%m-%d'),
            "val_start_date": data.index[n_train].strftime('%Y-%m-%d'),
            "val_end_date": data.index[n_train + n_val - 1].strftime('%Y-%m-%d'),
            "test_start_date": data.index[n_train + n_val].strftime('%Y-%m-%d'),
            "test_end_date": data.index[-1].strftime('%Y-%m-%d'),
        }

        splits = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "unscaled_test_y": target_vector[n_train + n_val:]
        }

        return splits, metadata

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates 3D window sequences (samples, lookback, features) and targets."""
        X, y = [], []
        for i in range(self.lookback_window, len(features)):
            X.append(features[i - self.lookback_window : i])
            y.append(targets[i, 0])
        return np.array(X), np.array(y)

    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """Converts normalized predictions back to original stock price ($)."""
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        return self.scaler_y.inverse_transform(scaled_values).flatten()

    def prepare_inference_sequence(self, df: pd.DataFrame, ticker: str) -> np.ndarray:
        """Prepares the most recent window of stock data for live inference."""
        data = TechnicalIndicators.add_all_indicators(df)
        if len(data) < self.lookback_window:
            raise ValueError(f"Need at least {self.lookback_window} rows for inference window.")

        scaler_x_path = os.path.join(SCALER_DIR, f"{ticker}_scaler_x.pkl")
        if os.path.exists(scaler_x_path):
            self.scaler_x = joblib.load(scaler_x_path)

        latest_window = data[self.FEATURE_COLS].iloc[-self.lookback_window:].values
        scaled_window = self.scaler_x.transform(latest_window)
        return np.expand_dims(scaled_window, axis=0)
