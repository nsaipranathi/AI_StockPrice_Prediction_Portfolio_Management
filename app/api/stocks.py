from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.services.market_data import MarketDataService
from backend.app.ml.indicators import TechnicalIndicators
from backend.app.schemas.stock import StockQuote, TechnicalIndicatorsSummary

router = APIRouter(prefix="/stocks", tags=["Stocks & Market Data"])

@router.get("/{ticker}", response_model=StockQuote)
def get_stock_quote(ticker: str):
    try:
        quote = MarketDataService.get_latest_quote(ticker)
        return quote
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch stock quote for {ticker}: {str(e)}")

@router.get("/{ticker}/history")
def get_stock_history(
    ticker: str,
    period: str = Query("1y", description="Data period: 1mo, 3mo, 6mo, 1y, 2y, 5y"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    try:
        df = MarketDataService.get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period)
        history = []
        for idx, row in df.iterrows():
            history.append({
                "date": idx.strftime('%Y-%m-%d'),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume'])
            })
        return {"ticker": ticker, "count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load historical market data for {ticker}: {str(e)}")

@router.get("/{ticker}/indicators", response_model=TechnicalIndicatorsSummary)
def get_stock_indicators(ticker: str):
    try:
        df = MarketDataService.get_stock_data(ticker, period="6mo")
        summary = TechnicalIndicators.get_latest_indicator_summary(df)
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to calculate indicators for {ticker}: {str(e)}")
