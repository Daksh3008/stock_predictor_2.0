# src/data/features.py
"""
Robust feature builder that avoids leakage.
Creates next-day return target and rolling features computed only on past data.
"""
import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Input:
      df - DataFrame with index datetime and columns Open, High, Low, Close, Volume
    Output:
      DataFrame with features and 'target' column which is next-day return over horizon:
         target = (Close_{t+horizon} / Close_t) - 1
    Features:
      ret_1, ret_5, ma_5, ma_10, ma_ratio5, vol_5, rsi_14, macd, dayofweek, month
    Notes:
      - Returns are clipped to reduce explosive outliers
      - Rows with NaN due to rolling windows are dropped
    """
    df = df.copy().sort_index()
    if "Close" not in df.columns:
        raise ValueError("Close column required")

    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"].astype(float)

    # basic returns
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)

    # clip extreme returns for stability (±50% daily)
    out["ret_1"] = out["ret_1"].clip(-0.5, 0.5)
    out["ret_5"] = out["ret_5"].clip(-0.8, 0.8)

    # moving averages and ratios
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_ratio_5"] = out["Close"] / (out["ma_5"] + 1e-9)
    out["ma_ratio_10"] = out["Close"] / (out["ma_10"] + 1e-9)

    # volatility
    out["vol_5"] = out["ret_1"].rolling(5).std().fillna(0)

    # RSI 14
    delta = out["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # calendar
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month

    # target: horizon-day forward return (no leakage since shift(-horizon))
    out["target"] = out["Close"].shift(-horizon) / out["Close"] - 1.0

    out = out.dropna().copy()
    return out
