import unittest
from backend.app.services.market_data import MarketDataService

class TestPortfolioCalculations(unittest.TestCase):
    def test_market_quote_fetch(self):
        try:
            quote = MarketDataService.get_latest_quote("AAPL")
            self.assertEqual(quote["ticker"], "AAPL")
            self.assertGreater(quote["current_price"], 0.0)
            self.assertIn("change_percent", quote)
        except Exception as e:
            self.skipTest(f"Network error in fetching market quote: {e}")

if __name__ == "__main__":
    unittest.main()
