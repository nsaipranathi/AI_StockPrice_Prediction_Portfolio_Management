import pandas as pd
import numpy as np
from typing import Dict, Any, List
from backend.app.services.market_data import MarketDataService
from backend.app.ml.indicators import TechnicalIndicators

class HistoricalBacktestingEngine:
    """
    Historical trading simulation engine.
    Simulates a moving-average crossover & momentum strategy without look-ahead bias.
    """

    @staticmethod
    def run_backtest(
        ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        ticker = ticker.upper().strip()
        
        # Fetch historical data spanning the date range
        df = MarketDataService.get_stock_data(ticker, start_date=start_date, end_date=end_date)
        if len(df) < 30:
            raise ValueError(f"Insufficient historical data points ({len(df)}) between {start_date} and {end_date}.")

        df_ind = TechnicalIndicators.add_all_indicators(df)

        cash = initial_capital
        shares = 0.0
        trades: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []
        peak_value = initial_capital
        max_drawdown = 0.0

        wins = 0
        total_trades = 0

        # Run day-by-day simulation strictly using historical data prior to day t
        for i in range(20, len(df_ind)):
            row = df_ind.iloc[i]
            date_str = row.name.strftime('%Y-%m-%d')
            price = float(row['Close'])
            sma20 = float(row['SMA_20'])
            rsi = float(row['RSI_14'])
            macd_hist = float(row['MACD_Hist'])

            # Generate Signal based ONLY on day t-1 / day t indicators
            # Signal Rules:
            # BUY: Price > SMA20, RSI < 60, MACD_Hist > 0
            # SELL: Price < SMA20 or RSI > 75
            
            signal = "HOLD"
            if price > sma20 and rsi < 60 and macd_hist > 0:
                signal = "BUY"
            elif price < sma20 or rsi > 70:
                signal = "SELL"

            # Execute trade if signal changes state
            if signal == "BUY" and cash > 50:
                # Buy maximum affordable shares
                shares_to_buy = cash / price
                shares += shares_to_buy
                trade_cost = shares_to_buy * price
                cash -= trade_cost
                total_trades += 1
                
                trades.append({
                    "date": date_str,
                    "type": "BUY",
                    "price": round(price, 2),
                    "shares": round(shares_to_buy, 4),
                    "cash": round(cash, 2),
                    "portfolio_value": round(cash + (shares * price), 2),
                    "reason": f"Bullish breakout above SMA20 (${sma20:.2f}), RSI={rsi:.1f}"
                })

            elif signal == "SELL" and shares > 0.001:
                # Sell all held shares
                sell_proceeds = shares * price
                entry_trade = [t for t in trades if t["type"] == "BUY"]
                if entry_trade and sell_proceeds > entry_trade[-1]["price"] * shares:
                    wins += 1

                cash += sell_proceeds
                trades.append({
                    "date": date_str,
                    "type": "SELL",
                    "price": round(price, 2),
                    "shares": round(shares, 4),
                    "cash": round(cash, 2),
                    "portfolio_value": round(cash, 2),
                    "reason": f"Downtrend cross below SMA20 (${sma20:.2f}) or overbought RSI={rsi:.1f}"
                })
                shares = 0.0

            # Current Portfolio Value
            current_portfolio_value = cash + (shares * price)
            if current_portfolio_value > peak_value:
                peak_value = current_portfolio_value
            
            dd = (peak_value - current_portfolio_value) / peak_value * 100
            if dd > max_drawdown:
                max_drawdown = dd

            equity_curve.append({
                "date": date_str,
                "portfolio_value": round(current_portfolio_value, 2),
                "benchmark_price": round(price, 2),
                "cash": round(cash, 2),
                "holdings_value": round(shares * price, 2)
            })

        final_value = equity_curve[-1]["portfolio_value"] if equity_curve else initial_capital
        total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
        
        # Benchmark Buy-and-Hold return
        start_price = float(df_ind.iloc[20]['Close'])
        end_price = float(df_ind.iloc[-1]['Close'])
        benchmark_return_pct = ((end_price - start_price) / start_price) * 100
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        return {
            "ticker": ticker,
            "initial_capital": round(initial_capital, 2),
            "final_value": round(final_value, 2),
            "total_return_percent": round(total_return_pct, 2),
            "max_drawdown_percent": round(max_drawdown, 2),
            "benchmark_return_percent": round(benchmark_return_pct, 2),
            "win_rate_percent": round(win_rate, 1),
            "trade_count": len(trades),
            "trades": trades,
            "equity_curve": equity_curve,
            "disclaimer": "Historical Simulation — Past simulation results do not guarantee future performance."
        }
