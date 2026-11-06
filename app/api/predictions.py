import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import Prediction, User
from backend.app.api.auth import get_current_user
from backend.app.ml.predict_lstm import LSTMPredictor
from backend.app.services.recommendation import AnalyticalRecommendationEngine
from backend.app.schemas.stock import RecommendationResponse

router = APIRouter(prefix="/predictions", tags=["LSTM Predictions"])

@router.get("/forecast/{ticker}")
def get_lstm_forecast(
    ticker: str,
    horizon: int = Query(5, ge=1, le=30),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    try:
        forecast_data = LSTMPredictor.get_forecast(ticker=ticker, horizon_days=horizon)
        
        # Log prediction to database
        try:
            pred_record = Prediction(
                user_id=current_user.id if current_user else None,
                ticker=ticker.upper(),
                current_price=forecast_data["current_price"],
                predicted_price=forecast_data["predicted_price"],
                forecast_horizon_days=horizon,
                model_name="LSTM-DeepLearning",
                recommendation_signal="BUY" if forecast_data["price_change_percent"] > 2 else ("SELL" if forecast_data["price_change_percent"] < -2 else "HOLD"),
                metrics_json=json.dumps(forecast_data["metrics"])
            )
            db.add(pred_record)
            db.commit()
        except Exception as db_err:
            print(f"[Warning] Failed to log prediction to DB: {db_err}")

        return forecast_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM prediction engine error for {ticker}: {str(e)}")

@router.get("/recommendation/{ticker}", response_model=RecommendationResponse)
def get_recommendation(ticker: str, horizon: int = Query(5, ge=1, le=30)):
    try:
        forecast_data = LSTMPredictor.get_forecast(ticker=ticker, horizon_days=horizon)
        rec = AnalyticalRecommendationEngine.generate_recommendation(
            ticker=ticker,
            current_price=forecast_data["current_price"],
            predicted_price=forecast_data["predicted_price"],
            horizon_days=horizon
        )
        return rec
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation for {ticker}: {str(e)}")

@router.get("/history")
def get_prediction_history(db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    user_id = current_user.id if current_user else 1
    preds = db.query(Prediction).filter(
        (Prediction.user_id == user_id) | (Prediction.user_id.is_(None))
    ).order_by(Prediction.prediction_date.desc()).limit(50).all()

    results = []
    for p in preds:
        metrics = json.loads(p.metrics_json) if p.metrics_json else {}
        results.append({
            "id": p.id,
            "ticker": p.ticker,
            "prediction_date": p.prediction_date.strftime('%Y-%m-%d %H:%M'),
            "current_price": round(p.current_price, 2),
            "predicted_price": round(p.predicted_price, 2),
            "forecast_horizon_days": p.forecast_horizon_days,
            "model_name": p.model_name,
            "recommendation_signal": p.recommendation_signal,
            "metrics": metrics
        })
    return {"history": results}
