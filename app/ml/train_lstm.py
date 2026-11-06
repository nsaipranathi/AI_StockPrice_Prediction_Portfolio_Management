import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from backend.app.services.market_data import MarketDataService
from backend.app.ml.data_pipeline import TimeSeriesDataPipeline
from backend.app.ml.evaluate_lstm import LSTMEvaluator

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "lstm")
os.makedirs(MODEL_DIR, exist_ok=True)

# Set reproducible random seeds
tf.random.set_seed(42)
np.random.seed(42)

class LSTMTrainer:
    """
    Keras/TensorFlow LSTM model building, training, and persistence manager.
    """

    @staticmethod
    def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
        """
        Constructs the Stacked LSTM Deep Learning Architecture.
        Input Shape: (lookback_window, num_features)
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        return model

    @staticmethod
    def train_ticker_model(
        ticker: str = "AAPL",
        lookback_window: int = 60,
        epochs: int = 25,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Full end-to-end pipeline: fetch market data, prepare sequences, train LSTM, evaluate, and save.
        """
        ticker = ticker.upper().strip()
        print(f"[LSTMTrainer] Fetching market data for {ticker}...")
        df = MarketDataService.get_stock_data(ticker, period="5y")

        pipeline = TimeSeriesDataPipeline(lookback_window=lookback_window)
        splits, metadata = pipeline.prepare_data(df, save_scaler_prefix=ticker)

        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        model = LSTMTrainer.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
        ]

        print(f"[LSTMTrainer] Training LSTM for {ticker} over {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Save model explicitly
        model.save(model_path)

        # Evaluate on unseen Test Set
        scaled_preds = model.predict(X_test, verbose=0).flatten()
        unscaled_preds = pipeline.inverse_transform_target(scaled_preds)
        unscaled_actuals = splits["unscaled_test_y"][-len(unscaled_preds):].flatten()

        metrics = LSTMEvaluator.calculate_metrics(unscaled_actuals, unscaled_preds)

        # Save metrics log
        metrics_path = os.path.join(MODEL_DIR, f"{ticker}_metrics.json")
        summary = {
            "ticker": ticker,
            "metadata": metadata,
            "metrics": metrics,
            "epochs_trained": len(history.history['loss']),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1])
        }
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[LSTMTrainer] Completed {ticker} LSTM training. Test R²: {metrics['r2']:.4f}, MAE: ${metrics['mae']:.2f}")
        return summary

if __name__ == "__main__":
    LSTMTrainer.train_ticker_model("AAPL", epochs=10)
