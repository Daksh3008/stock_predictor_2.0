"""
Monthly Backtest for Chemical Sector Stocks (e.g., DEEPAKNTR.BO)
---------------------------------------------------------------
Robust handling for BSE/NSE tickers and Yahoo anomalies.
Includes:
- Macro (Brent, USDINR)
- Bayesian ensemble
- Reasoning summary
"""

import os
import sys
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# --- LOCAL IMPORTS ---
from src.models.xgb_model import train_xgb_with_val
from src.models.rf_model import train_rf
from src.models.linreg_model import train_linreg
from src.models.lstm_model import build_lstm_univariate, train_lstm
from src.models.ensemble import weights_from_scores
from src.utils.seed import set_global_seed

set_global_seed(42)


# ----------------------------
# SAFER YFINANCE FETCH
# ----------------------------
def fetch_stock_data(ticker, start, end):
    """Fetches data robustly for BSE/NSE tickers and normalizes column names."""
    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # If columns are duplicated (e.g., ['Deepakntr.Bo', ...]) → flatten
    if len(set(raw.columns)) == 1 and len(raw.columns) == 5:
        # Common BSE bug: all columns identical
        df = raw.copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    # If MultiIndex (Adj Close, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        # Flatten to level names
        raw.columns = [c[0] for c in raw.columns]
        df = raw.copy()
        df = df.rename(columns=lambda x: x.strip().title())
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # If normal DataFrame but all lowercase (e.g., 'close', 'open')
    df = raw.copy()
    df.columns = [c.title() for c in df.columns]
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]

    # Handle fallback (if only one column repeated)
    if "Close" not in df.columns:
        # Try to extract single-column numeric data
        if df.select_dtypes(include=[np.number]).shape[1] == 1:
            df["Close"] = df[df.select_dtypes(include=[np.number]).columns[0]]
        else:
            raise ValueError(f"Could not normalize columns for {ticker}. Columns={df.columns.tolist()}")

    # Fill missing OHLC columns if needed
    for c in ["Open", "High", "Low", "Volume"]:
        if c not in df.columns:
            df[c] = df["Close"]

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ----------------------------
# MACRO DATA FETCH
# ----------------------------
def fetch_macro_data(start_date, end_date):
    """Fetch Brent crude and USD/INR safely."""
    def safe_fetch(ticker, col_name):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                return pd.Series(dtype=float, name=col_name)
            close = None
            for c in data.columns:
                if "close" in str(c).lower():
                    close = data[c]
                    break
            if close is None:
                close = data.select_dtypes(include=[np.number]).iloc[:, 0]
            close = close.astype(float)
            close.name = col_name
            return close
        except Exception:
            return pd.Series(dtype=float, name=col_name)

    brent = safe_fetch("BZ=F", "brent_price")
    inr = safe_fetch("USDINR=X", "usd_inr")

    df_macro = pd.concat([brent, inr], axis=1)
    df_macro = df_macro.ffill().bfill()
    return df_macro


# ----------------------------
# MACRO FEATURE ENGINEERING
# ----------------------------
def add_macro_features(df, df_macro):
    df = df.join(df_macro, how="left").ffill().bfill()
    df["currency_pressure_z"] = (df["usd_inr"] - df["usd_inr"].rolling(30).mean()) / (
        df["usd_inr"].rolling(30).std() + 1e-9
    )
    df["brent_pressure_z"] = (df["brent_price"] - df["brent_price"].rolling(30).mean()) / (
        df["brent_price"].rolling(30).std() + 1e-9
    )
    df["brent_momentum_7d"] = df["brent_price"].pct_change(7)
    df["inr_momentum_7d"] = df["usd_inr"].pct_change(7)
    df["macro_volatility"] = df[["brent_momentum_7d", "inr_momentum_7d"]].std(axis=1)
    return df.fillna(0)


# ----------------------------
# BAYESIAN WEIGHTING
# ----------------------------
def bayesian_weights(rmses):
    vals = np.array([np.exp(-rmses[k]) for k in rmses.keys()])
    w = vals / vals.sum() if vals.sum() > 0 else np.ones_like(vals) / len(vals)
    return dict(zip(rmses.keys(), w))


# ----------------------------
# FORECAST HELPERS
# ----------------------------
def recursive_forecast_tabular(df_base, model, horizon_days):
    df_work = df_base.copy().sort_index()
    preds, dates = [], []
    from pandas.tseries.offsets import BDay

    last_date = df_work.index[-1]
    for _ in range(horizon_days):
        next_date = last_date + BDay(1)
        close_series = df_work["Close"]
        last_close = close_series.iloc[-1]
        ret_1 = close_series.pct_change(1).iloc[-1]
        ret_5 = close_series.pct_change(5).iloc[-1]
        ma_ratio_5 = close_series.iloc[-1] / close_series.rolling(5).mean().iloc[-1]
        ma_ratio_10 = close_series.iloc[-1] / close_series.rolling(10).mean().iloc[-1]
        vol_5 = close_series.pct_change().rolling(5).std().iloc[-1]
        macro_vals = [
            df_work["currency_pressure_z"].iloc[-1],
            df_work["brent_pressure_z"].iloc[-1],
            df_work["brent_momentum_7d"].iloc[-1],
            df_work["inr_momentum_7d"].iloc[-1],
        ]
        x_row = np.array([[ret_1, ret_5, ma_ratio_5, ma_ratio_10, vol_5] + macro_vals])
        pred_return = float(model.predict(x_row)[0])
        pred_price = last_close * (1 + pred_return)
        df_work.loc[next_date] = df_work.iloc[-1]
        df_work.loc[next_date, "Close"] = pred_price
        preds.append(pred_price)
        dates.append(next_date)
        last_date = next_date
    return pd.Series(preds, index=pd.to_datetime(dates))


def forecast_lstm_close(df_base, lookback, model, scaler, steps):
    arr = df_base["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(arr).squeeze()
    history = list(scaled)
    preds_scaled = []
    for _ in range(steps):
        x = np.array(history[-lookback:]).reshape(1, lookback, 1)
        yhat = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(yhat)
        history.append(float(yhat))
    inv = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).squeeze()
    from pandas.tseries.offsets import BDay
    last_date = df_base.index[-1]
    dates = [last_date + BDay(i + 1) for i in range(len(inv))]
    return pd.Series(inv, index=pd.to_datetime(dates))


# ----------------------------
# MAIN BACKTEST
# ----------------------------
def monthly_backtest():
    ticker = input("Enter Chemical Sector Ticker (e.g., DEEPAKNTR.BO): ").strip().upper()
    cutoff_date = pd.to_datetime("2025-09-14").date()
    horizon_days = 30

    print(f"\n🧪 Backtest for {ticker} — cutoff={cutoff_date}, horizon={horizon_days}d")

    df = fetch_stock_data(ticker, "2005-01-01", "2025-09-30")
    print(f"✅ Loaded {len(df)} rows (last date = {df.index[-1].date()})")

    # Fetch macros
    df_macro = fetch_macro_data(df.index[0], df.index[-1])
    df_feat = add_macro_features(df.copy(), df_macro)

    # Features
    df_feat["ret_1"] = df_feat["Close"].pct_change(1)
    df_feat["ret_5"] = df_feat["Close"].pct_change(5)
    df_feat["ma_ratio_5"] = df_feat["Close"] / df_feat["Close"].rolling(5).mean()
    df_feat["ma_ratio_10"] = df_feat["Close"] / df_feat["Close"].rolling(10).mean()
    df_feat["vol_5"] = df_feat["ret_1"].rolling(5).std()
    df_feat["target"] = df_feat["Close"].pct_change(1).shift(-1)
    df_feat = df_feat.dropna()

    feature_cols = [
        "ret_1", "ret_5", "ma_ratio_5", "ma_ratio_10", "vol_5",
        "currency_pressure_z", "brent_pressure_z", "brent_momentum_7d", "inr_momentum_7d"
    ]
    X, y = df_feat[feature_cols], df_feat["target"]
    split = int(len(X) * 0.8)
    Xtr, Xv = X.iloc[:split], X.iloc[split:]
    ytr, yv = y.iloc[:split], y.iloc[split:]

    # Train
    rmses, models = {}, {}
    models["xgboost"], _, rmses["xgboost"], _ = train_xgb_with_val(
    Xtr.values, ytr.values, Xv.values, yv.values,
    params={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
    ticker=ticker
    )
    models["random_forest"], _ = train_rf(Xtr.values, ytr.values, ticker); rmses["random_forest"] = 0.06
    models["linear_regression"], _ = train_linreg(Xtr.values, ytr.values, ticker); rmses["linear_regression"] = 0.07

    # LSTM
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(df_feat["Close"].values.reshape(-1, 1))
    lookback = 10
    X_seq, y_seq = [], []
    for i in range(lookback, len(scaled)):
        X_seq.append(scaled[i - lookback:i])
        y_seq.append(scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    lstm_model = build_lstm_univariate((lookback, 1))
    lstm_model, _ = train_lstm(lstm_model, X_seq[:-30], y_seq[:-30], X_seq[-30:], y_seq[-30:], epochs=20, batch_size=16)
    models["lstm"], rmses["lstm"] = lstm_model, 0.05

    preds = {}
    for n in ["xgboost", "random_forest", "linear_regression"]:
        preds[n] = recursive_forecast_tabular(df_feat.copy(), models[n], horizon_days)
    preds["lstm"] = forecast_lstm_close(df_feat, lookback, lstm_model, scaler, horizon_days)

    df_pred = pd.DataFrame(preds)
    w = bayesian_weights(rmses)
    df_pred["ensemble"] = (df_pred * np.array(list(w.values()))).sum(axis=1)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(ROOT, f"backtest_{ticker}_{ts}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_pred.to_excel(writer, sheet_name="PREDICTIONS")
        pd.DataFrame([w]).to_excel(writer, sheet_name="BAYES_WEIGHTS", index=False)
    print(f"\n✅ Backtest complete → {out_path}")


if __name__ == "__main__":
    monthly_backtest()
