import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache")
DATASET_DIR1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data")
DATASET_DIR2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

os.makedirs(CACHE_DIR, exist_ok=True)

class MarketDataService:
    """
    Dedicated market-data service for reproducible stock market data fetching,
    cleaning, disk-caching, and offline CSV dataset loading.
    """
    
    @staticmethod
    def get_stock_data(
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        ticker = ticker.upper().strip()
        cache_file = os.path.join(CACHE_DIR, f"{ticker}.parquet")
        csv_file1 = os.path.join(DATASET_DIR1, f"{ticker}.csv")
        csv_file2 = os.path.join(DATASET_DIR2, f"{ticker}.csv")
        
        # 1. Check local CSV Datasets directory first (full 5y 1254 records)
        for csv_path in [csv_file1, csv_file2]:
            if os.path.exists(csv_path):
                try:
                    df_csv = pd.read_csv(csv_path)
                    df_csv['Date'] = pd.to_datetime(df_csv['Date'])
                    df_csv.set_index('Date', inplace=True)
                    df_csv.sort_index(inplace=True)
                    df_filtered = MarketDataService._filter_dates(df_csv, start_date, end_date)
                    if len(df_filtered) >= 100 or period == "1mo":
                        return df_filtered
                except Exception as csv_err:
                    print(f"[Warning] Failed to read dataset CSV {csv_path}: {csv_err}")

        # 2. Check local Parquet cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                df_filtered = MarketDataService._filter_dates(df, start_date, end_date)
                if len(df_filtered) >= 100 or period == "1mo":
                    return df_filtered
            except Exception as e:
                print(f"[Warning] Failed to read parquet cache for {ticker}: {e}")

        # 3. Download from yfinance API
        try:
            print(f"[MarketDataService] Downloading online market data for {ticker} (period={period})...")
            stock = yf.Ticker(ticker)
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
                
            if df.empty:
                raise ValueError(f"No stock market data found for ticker '{ticker}'")

            df.reset_index(inplace=True)
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' in downloaded data for {ticker}")
            
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df.sort_values('Date', inplace=True)
            df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().bfill()
            df = df[df['Close'] > 0]
            df.set_index('Date', inplace=True)
            
            # Save to Parquet cache
            try:
                df.to_parquet(cache_file)
            except Exception as cache_err:
                print(f"[Warning] Could not cache parquet for {ticker}: {cache_err}")
                
            return MarketDataService._filter_dates(df, start_date, end_date)
            
        except Exception as err:
            print(f"[Error] yfinance download failed for {ticker}: {err}")
            if os.path.exists(cache_file):
                df = pd.read_parquet(cache_file)
                return MarketDataService._filter_dates(df, start_date, end_date)
            raise RuntimeError(f"Unable to fetch market data for '{ticker}': {str(err)}")

    @staticmethod
    def get_latest_quote(ticker: str) -> Dict[str, Any]:
        df = MarketDataService.get_stock_data(ticker, period="1mo")
        if df.empty:
            raise ValueError(f"No quote data available for {ticker}")
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        current_price = float(latest['Close'])
        prev_price = float(prev['Close'])
        change = current_price - prev_price
        change_pct = (change / prev_price * 100) if prev_price > 0 else 0.0
        
        date_str = latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)[:10]

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "prev_close": round(prev_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "open": round(float(latest['Open']), 2),
            "high": round(float(latest['High']), 2),
            "low": round(float(latest['Low']), 2),
            "volume": int(latest['Volume']),
            "date": date_str
        }

    @staticmethod
    def _filter_dates(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df
