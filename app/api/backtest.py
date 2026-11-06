from fastapi import APIRouter, HTTPException
from backend.app.schemas.backtest import BacktestRequest, BacktestResult
from backend.app.services.backtest_service import HistoricalBacktestingEngine

router = APIRouter(prefix="/backtest", tags=["Historical Simulation & Backtesting"])

@router.post("", response_model=BacktestResult)
def run_backtest_simulation(req: BacktestRequest):
    try:
        res = HistoricalBacktestingEngine.run_backtest(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            initial_capital=req.initial_capital
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Backtesting simulation failed: {str(e)}")
