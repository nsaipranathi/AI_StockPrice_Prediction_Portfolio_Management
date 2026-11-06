from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import User, Portfolio, Holding
from backend.app.api.auth import get_current_user
from backend.app.services.market_data import MarketDataService
from backend.app.schemas.portfolio import HoldingCreate, HoldingUpdate, PortfolioPerformance

router = APIRouter(prefix="/portfolio", tags=["Portfolio Management"])

@router.get("", response_model=PortfolioPerformance)
def get_portfolio_performance(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    user_id = current_user.id if current_user else 1
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
    
    if not portfolio:
        portfolio = Portfolio(user_id=user_id, name="Primary Growth Portfolio")
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)

    holdings_db = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    
    # If portfolio is empty, add default representative holdings for demo
    if not holdings_db:
        default_holdings = [
            Holding(portfolio_id=portfolio.id, ticker="AAPL", quantity=15.0, purchase_price=170.0, purchase_date="2023-05-15"),
            Holding(portfolio_id=portfolio.id, ticker="MSFT", quantity=8.0, purchase_price=330.0, purchase_date="2023-06-20"),
            Holding(portfolio_id=portfolio.id, ticker="GOOGL", quantity=12.0, purchase_price=125.0, purchase_date="2023-08-10"),
            Holding(portfolio_id=portfolio.id, ticker="NVDA", quantity=5.0, purchase_price=450.0, purchase_date="2023-09-01"),
        ]
        db.add_all(default_holdings)
        db.commit()
        holdings_db = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()

    total_invested = 0.0
    total_current = 0.0
    daily_change_dollars = 0.0
    holdings_out = []

    for h in holdings_db:
        try:
            quote = MarketDataService.get_latest_quote(h.ticker)
            current_price = quote["current_price"]
            day_change = quote["change"]
        except Exception:
            current_price = h.purchase_price
            day_change = 0.0

        invested_val = h.quantity * h.purchase_price
        current_val = h.quantity * current_price
        p_l = current_val - invested_val
        ret_pct = (p_l / invested_val * 100) if invested_val > 0 else 0.0
        
        total_invested += invested_val
        total_current += current_val
        daily_change_dollars += h.quantity * day_change

        holdings_out.append({
            "id": h.id,
            "portfolio_id": h.portfolio_id,
            "ticker": h.ticker,
            "quantity": h.quantity,
            "purchase_price": round(h.purchase_price, 2),
            "purchase_date": h.purchase_date,
            "current_price": round(current_price, 2),
            "invested_value": round(invested_val, 2),
            "current_value": round(current_val, 2),
            "profit_loss": round(p_l, 2),
            "return_percent": round(ret_pct, 2)
        })

    total_p_l = total_current - total_invested
    total_ret_pct = (total_p_l / total_invested * 100) if total_invested > 0 else 0.0
    daily_change_pct = (daily_change_dollars / (total_current - daily_change_dollars) * 100) if total_current > daily_change_dollars else 0.0

    return {
        "portfolio_id": portfolio.id,
        "portfolio_name": portfolio.name,
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current, 2),
        "total_profit_loss": round(total_p_l, 2),
        "total_return_percent": round(total_ret_pct, 2),
        "daily_change_dollars": round(daily_change_dollars, 2),
        "daily_change_percent": round(daily_change_pct, 2),
        "holdings": holdings_out
    }

@router.post("/holdings")
def add_holding(
    holding_in: HoldingCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    user_id = current_user.id if current_user else 1
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
    if not portfolio:
        portfolio = Portfolio(user_id=user_id, name="Primary Growth Portfolio")
        db.add(portfolio)
        db.commit()

    holding = Holding(
        portfolio_id=portfolio.id,
        ticker=holding_in.ticker.upper().strip(),
        quantity=holding_in.quantity,
        purchase_price=holding_in.purchase_price,
        purchase_date=holding_in.purchase_date
    )
    db.add(holding)
    db.commit()
    db.refresh(holding)
    return {"status": "success", "holding_id": holding.id}

@router.put("/holdings/{holding_id}")
def update_holding(holding_id: int, holding_in: HoldingUpdate, db: Session = Depends(get_db)):
    holding = db.query(Holding).filter(Holding.id == holding_id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found.")
    
    if holding_in.quantity is not None:
        holding.quantity = holding_in.quantity
    if holding_in.purchase_price is not None:
        holding.purchase_price = holding_in.purchase_price

    db.commit()
    return {"status": "success", "message": "Holding updated"}

@router.delete("/holdings/{holding_id}")
def delete_holding(holding_id: int, db: Session = Depends(get_db)):
    holding = db.query(Holding).filter(Holding.id == holding_id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found.")
    
    db.delete(holding)
    db.commit()
    return {"status": "success", "message": "Holding deleted"}
