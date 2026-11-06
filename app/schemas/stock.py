from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class StockQuote(BaseModel):
    ticker: str
    current_price: float
    prev_close: float
    change: float
    change_percent: float
    open: float
    high: float
    low: float
    volume: int
    date: str

class TechnicalIndicatorsSummary(BaseModel):
    sma_20: float
    sma_50: float
    ema_20: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_lower: float
    daily_return_percent: float
    annualized_volatility: float

class RecommendationResponse(BaseModel):
    ticker: str
    signal: str  # BUY, HOLD, SELL
    confidence_score: float
    reasons: List[str]
    current_price: float
    predicted_price: float
    predicted_change_pct: float
