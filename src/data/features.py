#added indicators(MA5/10/20/50, ROC, Vol, Bollinger Bands, returns, volume ratios)
import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df with columns: Open, High, Low, Close, (Volume optional), index = Date
    Adds: lag returns, MA5/10/20/50, vol_5/20, RSI14, MACD, Bollinger bands, ROC, volume ratio,
    day_of_week, month, is_month_start.
    Returns dataframe with new features (drops initial NaNs).
    """
    out = df.copy().sort_index()
    # basic returns
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_5"] = out["Close"].pct_change(5)
    out["log_return_1"] = np.log(out["Close"] / out["Close"].shift(1))

    # moving averages
    for w in [5, 10, 20, 50]:
        out[f"ma_{w}"] = out["Close"].rolling(window=w, min_periods=1).mean()

    # Rolling volatility (std of returns)
    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # Momentum / Rate of Change
    for w in [5, 10, 20]:
        out[f"roc_{w}"] = out["Close"].pct_change(w)

    # RSI(14)
    delta = out["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26) and signal (9)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20,2)
    mb = out["Close"].rolling(20).mean()
    mstd = out["Close"].rolling(20).std()
    out["bb_upper"] = mb + 2 * mstd
    out["bb_lower"] = mb - 2 * mstd
    out["bb_pct"] = (out["Close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    # Volume-based (if Volume present)
    if "Volume" in out.columns:
        out["vol_ma_20"] = out["Volume"].rolling(20).mean()
        out["vol_ratio"] = out["Volume"] / (out["vol_ma_20"] + 1e-9)
    else:
        out["vol_ratio"] = 1.0  # fallback neutral

    # Calendar features
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_month_start"] = out.index.is_month_start.astype(int)
    out["is_month_end"] = out.index.is_month_end.astype(int)

    # Drop rows with NaN (created by indicators)
    out = out.dropna().copy()
    return out
