from pydantic import BaseModel
from typing import List, Dict, Any

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0

class BacktestTrade(BaseModel):
    date: str
    type: str  # BUY, SELL, HOLD
    price: float
    shares: float
    cash: float
    portfolio_value: float
    reason: str

class BacktestResult(BaseModel):
    ticker: str
    initial_capital: float
    final_value: float
    total_return_percent: float
    max_drawdown_percent: float
    benchmark_return_percent: float
    win_rate_percent: float
    trade_count: int
    trades: List[BacktestTrade]
    equity_curve: List[Dict[str, Any]]
