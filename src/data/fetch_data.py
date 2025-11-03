import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_data(ticker, start_date=None, end_date=None, max_retries=3, debug=True):
    """
    Fetch historical stock data for a ticker using yfinance.
    Handles:
      - Multi-index flattening (e.g. ('TCS.NS', 'Close'))
      - Adj Close fallback
      - .BO fallback if .NS fails
      - Missing data gracefully
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * 20)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    tried_bse = False
    tickers_to_try = [ticker, ticker.replace(".NS", ".BO")] if ticker.endswith(".NS") else [ticker]

    for attempt in range(1, max_retries + 1):
        for t in tickers_to_try:
            try:
                df = yf.download(t, start=start_date, end=end_date, progress=False, group_by="ticker")

                if df is None or df.empty:
                    raise ValueError(f"No data received for {t}")

                # ✅ Handle MultiIndex by flattening completely
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]

                # ✅ Handle cases where ticker prefix remains (like TCS.NS_Close)
                rename_map = {}
                for col in df.columns:
                    base = col.split("_")[-1].title()
                    if base in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                        rename_map[col] = base
                df.rename(columns=rename_map, inplace=True)

                # ✅ If Adj Close exists but Close doesn't, use it
                if "Adj Close" in df.columns and "Close" not in df.columns:
                    df.rename(columns={"Adj Close": "Close"}, inplace=True)

                # ✅ Keep only necessary columns
                cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[cols]

                if df.empty:
                    raise ValueError(f"{t}: Cleaned data is empty after filtering")

                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["Close"])
                df = df[~df.index.dayofweek.isin([5, 6])]  # drop weekends

                if debug:
                    print(f"📊 DEBUG [{t}] shape={df.shape}, columns={df.columns.tolist()}")
                    print(df.tail(3))

                return df

            except Exception as e:
                print(f"⚠️ Attempt {attempt}/{max_retries} failed for {t}: {e}")
                if ".BO" in t:
                    tried_bse = True
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    continue

    print(f"❌ Failed to fetch usable data for {ticker} (tried BSE fallback={tried_bse})")
    return pd.DataFrame()
