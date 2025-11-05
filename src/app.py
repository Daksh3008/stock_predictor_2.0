# src/app.py
# Dynamic forecasting for user-input ticker & horizon (in days)
# Now models future returns (normalized) instead of absolute prices
# Includes weighted ensemble based on validation RMSEs

from src.utils.seed import set_global_seed
set_global_seed(42)
import os
import sys
import warnings
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.fetch_data import fetch_data
from src.models.xgb_model import train_xgb_with_val
from src.models.rf_model import train_rf
from src.models.linreg_model import train_linreg
from src.models.lstm_model import build_lstm_univariate, train_lstm
from src.models.ensemble import weights_from_scores, weighted_average


# ─────────────────────────────────────────────
# Feature preparation (return-based)
# ─────────────────────────────────────────────
def prepare_features(df, horizon=1):
    df = df.copy()
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ma_ratio_5"] = df["Close"] / df["Close"].rolling(5).mean()
    df["ma_ratio_10"] = df["Close"] / df["Close"].rolling(10).mean()
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["target"] = df["Close"].pct_change(horizon).shift(-horizon)
    df = df.dropna()
    X = df[["ret_1", "ret_5", "ma_ratio_5", "ma_ratio_10", "vol_5"]]
    y = df["target"]
    return X, y


def make_lstm_sequences(values, lookback=10):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# Main Forecast Function
# ─────────────────────────────────────────────
def run_forecast(ticker: str, horizon: int = 30):
    print(f"\n🚀 Running forecast for {ticker} (horizon = {horizon} days)\n")

    df = fetch_data(ticker)
    if df is None or df.empty:
        print("❌ No data fetched. Check ticker or internet connection.")
        return

    df = df.dropna(subset=["Close"])
    print(f"✅ Data loaded ({len(df)} rows, last date = {df.index[-1].date()})")

    # ── Prepare features & target (future return)
    X, y = prepare_features(df, horizon=horizon)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    predictions, rmses = {}, {}

    # ── XGBoost
    try:
        mdl_xgb, _, rmse_xgb, _ = train_xgb_with_val(X_train.values, y_train.values, X_val.values, y_val.values)
        next_pred = float(mdl_xgb.predict(X.iloc[[-1]].values)[0])
        pred_price = df["Close"].iloc[-1] * (1 + next_pred)
        predictions["xgboost"] = pred_price
        rmses["xgboost"] = rmse_xgb
    except Exception as e:
        print(f"⚠️ XGBoost failed: {e}")

    # ── Random Forest
    try:
        mdl_rf, _ = train_rf(X_train.values, y_train.values)
        next_pred = float(mdl_rf.predict(X.iloc[[-1]].values)[0])
        pred_price = df["Close"].iloc[-1] * (1 + next_pred)
        predictions["random_forest"] = pred_price
        rmses["random_forest"] = 0.05  # placeholder for weighting
    except Exception as e:
        print(f"⚠️ RandomForest failed: {e}")

    # ── Linear Regression
    try:
        mdl_lr, _ = train_linreg(X_train.values, y_train.values)
        next_pred = float(mdl_lr.predict(X.iloc[[-1]].values)[0])
        pred_price = df["Close"].iloc[-1] * (1 + next_pred)
        predictions["linear_regression"] = pred_price
        rmses["linear_regression"] = 0.05
    except Exception as e:
        print(f"⚠️ LinearRegression failed: {e}")

    # ── LSTM (predicts directly in normalized Close)
    try:
        close_values = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_values)
        lookback = 10
        X_seq, y_seq = make_lstm_sequences(scaled.flatten(), lookback)
        if len(X_seq) < 20:
            raise ValueError("Not enough data for LSTM")
        split = int(len(X_seq) * 0.8)
        Xtr, Xv = X_seq[:split], X_seq[split:]
        ytr, yv = y_seq[:split], y_seq[split:]
        model = build_lstm_univariate((lookback, 1))
        model, hist = train_lstm(model, Xtr, ytr, Xv, yv, epochs=20, batch_size=16)
        last_window = scaled[-lookback:].reshape(1, lookback, 1)
        for _ in range(horizon):
            p = model.predict(last_window, verbose=0)
            last_window = np.append(last_window[:, 1:, :], p.reshape(1, 1, 1), axis=1)
        pred_lstm = scaler.inverse_transform(p)[0, 0]
        predictions["lstm"] = float(pred_lstm)
        rmses["lstm"] = float(np.mean(hist.history["val_loss"])) if "val_loss" in hist.history else 0.05
    except Exception as e:
        print(f"⚠️ LSTM failed: {e}")

    # ── Weighted Ensemble (based on inverse RMSE)
    valid_models = {k: v for k, v in predictions.items() if isinstance(v, (float, np.floating))}
    if valid_models:
        weights = weights_from_scores(rmses)
        ensemble_pred = weighted_average(valid_models, weights)
        predictions["ensemble"] = ensemble_pred
    else:
        predictions["ensemble"] = None

    # ─────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────
    print("\n📈 Final Predictions for", ticker)
    for k, v in predictions.items():
        print(f"{k:<18} → {v:.2f}" if v is not None else f"{k:<18} → failed")

    print(f"\n📅 Predicted date: {(df.index[-1] + pd.tseries.offsets.BDay(horizon)).date()}")
    print(f"Current (last) close: {df['Close'].iloc[-1]:.2f}\n")

    return predictions


# ─────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n📊 Stock Forecasting Interface (Return-based models + Weighted Ensemble)\n")
    ticker = input("Enter stock ticker (e.g., TCS.NS or ACE.BO): ").strip().upper()
    horizon = int(input("Enter forecast horizon (in days, e.g., 7, 30): ").strip() or 7)
    run_forecast(ticker, horizon)
