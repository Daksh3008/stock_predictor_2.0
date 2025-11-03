# src/automation/predict_daily.py
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json, pandas as pd
from datetime import datetime
from src.data.fetch_data import fetch_data
from src.data.features import add_technical_indicators
from src.models.xgb_model import train_xgb_with_val
from src.models.rf_model import train_rf
from src.models.linreg_model import train_linreg
from src.models.lstm_model import build_lstm_univariate, train_lstm
from src.models.ensemble import weights_from_scores, weighted_average
from src.utils.logger import get_logger

logger = get_logger("predict_daily")

STOCK_LIST_FILE = os.path.join(os.path.dirname(__file__), "stock_list.json")
OUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../predictions_log.xlsx"))

with open(STOCK_LIST_FILE, "r") as f:
    stocks = json.load(f)["stocks"]

records = []
for ticker in stocks:
    try:
        logger.info(f"Predicting {ticker}")
        df = fetch_data(ticker, start_date="2015-01-01")
        df_feat = add_technical_indicators(df)
        # simple tabular X creation
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_feat[["Close"]].values)

        # very simple supervised: last 10 closes flattened
        lookback = 10
        X = []
        y = []
        arr = scaled.squeeze()
        for i in range(lookback, len(arr)):
            X.append(arr[i-lookback:i])
            y.append(arr[i])
        import numpy as np
        X = np.array(X); y = np.array(y)
        split = int(0.8*len(X))
        Xtr, Xv = X[:split], X[split:]
        ytr, yv = y[:split], y[split:]

        mdl_xgb, _, rm_xgb = train_xgb_with_val(Xtr, ytr, Xv, yv, params=None, random_state=0)
        mdl_rf = train_rf(Xtr, ytr)
        mdl_lr = train_linreg(Xtr, ytr)

        last_window = X[-1].reshape(1, -1)
        p_xgb = float(mdl_xgb.predict(last_window)[0])
        p_rf = float(mdl_rf.predict(last_window)[0])
        p_lr = float(mdl_lr.predict(last_window)[0])
        preds = {"xgboost": p_xgb, "random_forest": p_rf, "linear_regression": p_lr}
        scores = {"xgboost": rm_xgb, "random_forest": 0.05, "linear_regression": 0.05}
        w = weights_from_scores(scores)
        ensemble = weighted_average(preds, w)

        records.append({
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Predicted_Today": float(ensemble),
            "Confidence": "High",
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Failed for {ticker}: {e}")

# save to excel (append)
df_out = pd.DataFrame(records)
if os.path.exists(OUT_FILE):
    existing = pd.read_excel(OUT_FILE)
    df_out = pd.concat([existing, df_out], ignore_index=True)
df_out.to_excel(OUT_FILE, index=False)
logger.info(f"Saved predictions to {OUT_FILE}")
