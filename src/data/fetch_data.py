# src/utils/fetch_data.py
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_data(ticker, start_date=None, end_date=None, max_retries=3, debug=False):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * 20)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, group_by="ticker")
            if df is None or df.empty:
                raise ValueError("No data received")

            # Flatten any multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]

            # Map possible names to core names
            rename_map = {}
            for col in df.columns:
                base = col.split("_")[-1]
                if base in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                    rename_map[col] = base
            df.rename(columns=rename_map, inplace=True)

            # If Adj Close present and Close missing
            if "Adj Close" in df.columns and "Close" not in df.columns:
                df.rename(columns={"Adj Close": "Close"}, inplace=True)

            # Keep only required columns if available
            required = ["Open", "High", "Low", "Close", "Volume"]
            present = [c for c in required if c in df.columns]
            if not present:
                raise ValueError(f"No recognizable columns. got: {list(df.columns)}")

            df = df[present].copy()
            # Only require Close to be present and non-null
            if "Close" not in df.columns or df["Close"].dropna().empty:
                raise ValueError("Close missing or empty")

            df.index = pd.to_datetime(df.index)
            # drop weekends
            df = df[~df.index.dayofweek.isin([5,6])]
            if debug:
                print(f"DEBUG fetch {ticker} shape={df.shape} cols={df.columns.tolist()}")
            return df

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                return pd.DataFrame()
