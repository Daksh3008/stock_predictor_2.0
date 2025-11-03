# src/data/features.py
import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df with columns: Open, High, Low, Close, Index = Date
    Adds: returns, MA5/10/20, vol_5/20, RSI14, MACD, day_of_week, month, is_month_start
    """
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_5"] = out["Close"].pct_change(5)

    for w in [5,10,20]:
        out[f"ma_{w}"] = out["Close"].rolling(w).mean()

    out["vol_5"] = out["Close"].pct_change().rolling(5).std()
    out["vol_20"] = out["Close"].pct_change().rolling(20).std()

    # RSI(14)
    delta = out["Close"].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26

    # calendar
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_month_start"] = out.index.is_month_start.astype(int)

    out = out.dropna()
    return out
