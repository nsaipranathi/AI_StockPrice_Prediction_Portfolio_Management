from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import User
from backend.app.api.auth import get_current_user
from backend.app.api.portfolio import get_portfolio_performance
from backend.app.services.risk_service import PortfolioRiskAnalyzer

router = APIRouter(prefix="/risk", tags=["Risk Analysis"])

@router.get("", response_model=None)
def get_portfolio_risk(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    perf = get_portfolio_performance(db=db, current_user=current_user)
    risk_summary = PortfolioRiskAnalyzer.calculate_portfolio_risk(perf["holdings"])
    return risk_summary
