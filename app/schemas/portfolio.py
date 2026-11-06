from pydantic import BaseModel
from typing import List, Optional

class HoldingCreate(BaseModel):
    ticker: str
    quantity: float
    purchase_price: float
    purchase_date: str  # YYYY-MM-DD

class HoldingUpdate(BaseModel):
    quantity: Optional[float] = None
    purchase_price: Optional[float] = None

class HoldingOut(BaseModel):
    id: int
    portfolio_id: int
    ticker: str
    quantity: float
    purchase_price: float
    purchase_date: str
    current_price: float
    invested_value: float
    current_value: float
    profit_loss: float
    return_percent: float

    class Config:
        from_attributes = True

class PortfolioPerformance(BaseModel):
    portfolio_id: int
    portfolio_name: str
    total_invested: float
    total_current_value: float
    total_profit_loss: float
    total_return_percent: float
    daily_change_dollars: float
    daily_change_percent: float
    holdings: List[HoldingOut]
