# src/data/fetch_data.py
import yfinance as yf
import pandas as pd
import time
from typing import Optional

def fetch_data(ticker: str, start_date: str, end_date: Optional[str]=None, retries: int = 3, pause: int = 3):
    """
    Robust fetch wrapper:
      - retries a few times on network errors
      - flattens MultiIndex columns that yfinance may produce
      - returns DataFrame with columns ['Open','High','Low','Close']
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    for attempt in range(1, retries+1):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if df is None or df.empty:
                raise ValueError(f"No data for ticker {ticker} (attempt {attempt})")
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Open","High","Low","Close"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            if attempt < retries:
                time.sleep(pause)
            else:
                raise
