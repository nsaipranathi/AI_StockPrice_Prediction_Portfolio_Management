from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import User, Watchlist
from backend.app.api.auth import get_current_user
from backend.app.services.market_data import MarketDataService

router = APIRouter(prefix="/watchlist", tags=["Watchlist Management"])

@router.get("")
def get_user_watchlist(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    user_id = current_user.id if current_user else 1
    items = db.query(Watchlist).filter(Watchlist.user_id == user_id).all()
    
    # If empty, populate default watch items for demo
    if not items:
        defaults = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        for t in defaults:
            db.add(Watchlist(user_id=user_id, ticker=t))
        db.commit()
        items = db.query(Watchlist).filter(Watchlist.user_id == user_id).all()

    watchlist_data = []
    for item in items:
        try:
            quote = MarketDataService.get_latest_quote(item.ticker)
            watchlist_data.append({
                "id": item.id,
                "ticker": item.ticker,
                "current_price": quote["current_price"],
                "change": quote["change"],
                "change_percent": quote["change_percent"],
                "added_at": item.added_at.strftime('%Y-%m-%d')
            })
        except Exception:
            watchlist_data.append({
                "id": item.id,
                "ticker": item.ticker,
                "current_price": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "added_at": item.added_at.strftime('%Y-%m-%d')
            })
    return {"watchlist": watchlist_data}

@router.post("/{ticker}")
def add_to_watchlist(
    ticker: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    user_id = current_user.id if current_user else 1
    ticker_clean = ticker.upper().strip()
    
    existing = db.query(Watchlist).filter(
        Watchlist.user_id == user_id, Watchlist.ticker == ticker_clean
    ).first()
    if existing:
        return {"status": "exists", "message": f"{ticker_clean} is already in watchlist."}

    item = Watchlist(user_id=user_id, ticker=ticker_clean)
    db.add(item)
    db.commit()
    return {"status": "success", "message": f"Added {ticker_clean} to watchlist."}

@router.delete("/{ticker}")
def remove_from_watchlist(
    ticker: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    user_id = current_user.id if current_user else 1
    ticker_clean = ticker.upper().strip()
    
    item = db.query(Watchlist).filter(
        Watchlist.user_id == user_id, Watchlist.ticker == ticker_clean
    ).first()
    if item:
        db.delete(item)
        db.commit()
        return {"status": "success", "message": f"Removed {ticker_clean} from watchlist."}
    raise HTTPException(status_code=404, detail=f"Ticker {ticker_clean} not found in watchlist.")
