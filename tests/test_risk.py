import unittest
from backend.app.services.risk_service import PortfolioRiskAnalyzer

class TestRiskAnalyzer(unittest.TestCase):
    def test_risk_calculations(self):
        holdings = [
            {"ticker": "AAPL", "current_value": 3000.0},
            {"ticker": "MSFT", "current_value": 2000.0}
        ]
        risk = PortfolioRiskAnalyzer.calculate_portfolio_risk(holdings)
        self.assertIn("annualized_volatility", risk)
        self.assertIn("sharpe_ratio", risk)
        self.assertIn("max_drawdown_percent", risk)
        self.assertIn("concentration_hhi", risk)
        self.assertIn("risk_level", risk)

if __name__ == "__main__":
    unittest.main()
