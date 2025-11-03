# Main entrypoint for running prediction (smoke test) on a single ticker.
# Usage : python -m src.main --ticker TCS.NS --horizon 7


import os
import sys
import warnings
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ensure src in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# =======================
# Import project modules
# =======================
from src.data.fetch_data import fetch_data
from src.models.xgb_model import train_xgb_with_val
from src.models.rf_model import train_rf
from src.models.linreg_model import train_linreg
from src.models.lstm_model import build_lstm_univariate, train_lstm
from src.explain.shap_explainer import explain_model


# =======================
# Helper functions
# =======================
def prepare_features(df, horizon=1):
    df = df.copy()
    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag2"] = df["Close"].shift(2)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["target"] = df["Close"].shift(-horizon)
    df = df.dropna()
    X = df[["Close", "Close_lag1", "Close_lag2", "MA5", "MA10"]]
    y = df["target"]
    return X, y


def make_lstm_sequences(values, lookback=10):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


# =======================
# Main smoke function
# =======================
def smoke(ticker, horizon=1, return_results=False):
    logging.info(f"🚀 Running smoke for {ticker}")

    # 1️⃣ Fetch data
    df = fetch_data(ticker)
    if df is None or df.empty:
        raise ValueError(f"No data found for {ticker}")

    df = df.dropna(subset=["Close"])
    logging.info(f"Data loaded: {len(df)} rows")

    # 2️⃣ Prepare features
    X, y = prepare_features(df, horizon=horizon)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    last_row = X.iloc[[-1]]

    # 3️⃣ Initialize predictions dictionary
    predictions = {"xgboost": None, "random_forest": None, "linear_regression": None, "lstm": None, "ensemble": None}

    # 4️⃣ XGBoost
    try:
        mdl_xgb, _, rmse, _ = train_xgb_with_val(X_train.values, y_train.values, X_val.values, y_val.values)
        pred = mdl_xgb.predict(last_row.values)[0]
        predictions["xgboost"] = float(pred)
    except Exception as e:
        logging.warning(f"⚠️ XGBoost failed: {e}")

    # 5️⃣ Random Forest
    try:
        mdl_rf, _ = train_rf(X_train.values, y_train.values)
        pred = mdl_rf.predict(last_row.values)[0]
        predictions["random_forest"] = float(pred)
    except Exception as e:
        logging.warning(f"⚠️ RandomForest failed: {e}")

    # 6️⃣ Linear Regression
    try:
        mdl_lr, _ = train_linreg(X_train.values, y_train.values)
        pred = mdl_lr.predict(last_row.values)[0]
        predictions["linear_regression"] = float(pred)
    except Exception as e:
        logging.warning(f"⚠️ LinearRegression failed: {e}")

    # 7️⃣ LSTM
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
        model, _ = train_lstm(model, Xtr, ytr, Xv, yv, epochs=20, batch_size=16)

        # Predict horizon days ahead
        last_window = scaled[-lookback:].reshape(1, lookback, 1)
        for _ in range(horizon):
            p = model.predict(last_window, verbose=0)
            last_window = np.append(last_window[:, 1:, :], p.reshape(1, 1, 1), axis=1)
        pred_lstm = scaler.inverse_transform(p)[0, 0]
        predictions["lstm"] = float(pred_lstm)
    except Exception as e:
        logging.warning(f"⚠️ LSTM failed: {e}")

    # 8️⃣ Ensemble
    valid_preds = [v for v in predictions.values() if isinstance(v, (int, float))]
    if valid_preds:
        predictions["ensemble"] = float(np.mean(valid_preds))

    logging.info(f"✅ Predictions: {predictions}")
    if return_results:
        return predictions
    return predictions


# =======================
# CLI
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    smoke(args.ticker, args.horizon)
