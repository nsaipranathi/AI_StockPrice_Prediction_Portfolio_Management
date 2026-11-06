import numpy as np
import pandas as pd
from typing import List, Dict, Any
from backend.app.services.market_data import MarketDataService

class PortfolioRiskAnalyzer:
    """
    Quantitative risk analysis service for stock portfolios.
    Calculates annualized volatility, Sharpe ratio, maximum drawdown, and concentration.
    """

    @staticmethod
    def calculate_portfolio_risk(
        holdings: List[Dict[str, Any]],
        risk_free_rate: float = 0.04  # 4% annual risk-free rate
    ) -> Dict[str, Any]:
        if not holdings:
            return {
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_percent": 0.0,
                "concentration_score": 0.0,
                "risk_level": "Low",
                "asset_allocations": [],
                "explanations": {
                    "volatility": "No active holdings in portfolio.",
                    "sharpe_ratio": "Add holdings to compute risk-adjusted return.",
                    "max_drawdown": "Maximum peak-to-trough decline over historical window.",
                    "concentration": "Asset diversification measure."
                }
            }

        # Compute total current value & weights
        total_value = sum(h["current_value"] for h in holdings)
        if total_value <= 0:
            total_value = 1.0

        allocations = []
        weights = []
        tickers = []
        for h in holdings:
            weight = h["current_value"] / total_value
            weights.append(weight)
            tickers.append(h["ticker"])
            allocations.append({
                "ticker": h["ticker"],
                "value": round(h["current_value"], 2),
                "weight_percent": round(weight * 100, 2)
            })

        # Calculate Concentration (Herfindahl-Hirschman Index)
        hhi = sum(w ** 2 for w in weights) * 10000  # Scale 0 to 10,000

        # Fetch historical price data for holdings (last 1 year)
        combined_returns = []
        for h in holdings:
            try:
                df = MarketDataService.get_stock_data(h["ticker"], period="1y")
                daily_ret = df['Close'].pct_change().dropna()
                combined_returns.append(daily_ret)
            except Exception as e:
                print(f"[Warning] Risk analysis could not load data for {h['ticker']}: {e}")

        if not combined_returns:
            portfolio_daily_returns = pd.Series([0.0])
        else:
            # Align by date
            ret_df = pd.concat(combined_returns, axis=1, keys=tickers).fillna(0)
            portfolio_daily_returns = ret_df.dot(weights)

        # 1. Annualized Volatility
        daily_std = float(portfolio_daily_returns.std())
        annualized_vol = daily_std * np.sqrt(252)

        # 2. Portfolio Mean Return (Annualized)
        mean_daily_ret = float(portfolio_daily_returns.mean())
        annualized_return = mean_daily_ret * 252

        # 3. Sharpe Ratio
        if annualized_vol > 0:
            sharpe = (annualized_return - risk_free_rate) / annualized_vol
        else:
            sharpe = 0.0

        # 4. Maximum Drawdown
        cum_returns = (1 + portfolio_daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = float(drawdowns.min()) * 100 if not drawdowns.empty else 0.0

        # Determine overall Risk Level badge
        if annualized_vol > 0.30 or abs(max_drawdown) > 25:
            risk_level = "High Risk"
        elif annualized_vol > 0.18 or abs(max_drawdown) > 15:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Conservative / Low Risk"

        return {
            "annualized_volatility": round(annualized_vol * 100, 2),  # In %
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_percent": round(abs(max_drawdown), 2),
            "concentration_hhi": round(hhi, 1),
            "risk_level": risk_level,
            "asset_allocations": allocations,
            "explanations": {
                "volatility": f"Annualized Volatility of {annualized_vol*100:.1f}% measures price fluctuation risk over a 252-trading-day year.",
                "sharpe_ratio": f"Sharpe Ratio of {sharpe:.2f} measures excess return per unit of risk above the {risk_free_rate*100:.0f}% benchmark. (> 1.0 is considered good).",
                "max_drawdown": f"Maximum Drawdown of {abs(max_drawdown):.1f}% represents the largest peak-to-trough decline in historical simulation.",
                "concentration": f"HHI score of {hhi:.0f} measures portfolio diversification (lower is more diversified)."
            }
        }
