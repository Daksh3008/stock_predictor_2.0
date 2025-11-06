import os
import sys
import warnings
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime

# --- setup path ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.fetch_data import fetch_data
from src.data.features import add_technical_indicators
from src.models.xgb_model import train_xgb_with_val
from src.models.rf_model import train_rf
from src.models.linreg_model import train_linreg
from src.models.lstm_model import build_lstm_univariate, train_lstm
from src.explain.shap_explainer import explain_model
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def run_monthly_backtest(ticker, horizon=30):
    logging.info(f"📊 Running monthly backtest for {ticker} (horizon={horizon} days)")

    df = fetch_data(ticker, start_date="2015-01-01")
    df = add_technical_indicators(df)
    df = df.dropna()

    # Split training till Sept 2025
    cutoff_date = pd.Timestamp("2025-09-30")
    train_df = df[df.index <= cutoff_date].copy()
    test_df = df[(df.index > cutoff_date) & (df.index <= cutoff_date + pd.Timedelta(days=horizon * 2))].copy()

    if len(train_df) < 200 or len(test_df) == 0:
        raise ValueError("Insufficient data for monthly backtest")

    # Prepare features
    X = train_df.drop(columns=["Close"])
    y = train_df["Close"]
    Xv = test_df.drop(columns=["Close"])
    yv = test_df["Close"]

    last_date = train_df.index[-1]
    last_close = train_df["Close"].iloc[-1]

    results = {}

    # =============== XGBoost ===============
    try:
        mdl_xgb, _, rmse, _ = train_xgb_with_val(X.values, y.values, Xv.values, yv.values, ticker=ticker)
        pred_xgb = float(mdl_xgb.predict(Xv.values)[-1])
        shap_data, _ = explain_model(mdl_xgb, Xv, "xgboost", ticker)
        results["xgboost"] = (pred_xgb, shap_data[0]["Feature"] if shap_data else "N/A")
    except Exception as e:
        logging.warning(f"⚠️ XGBoost failed: {e}")
        results["xgboost"] = (None, "N/A")

    # =============== Random Forest ===============
    try:
        mdl_rf, _ = train_rf(X.values, y.values, ticker=ticker)
        pred_rf = float(mdl_rf.predict(Xv.values)[-1])
        shap_data, _ = explain_model(mdl_rf, Xv, "random_forest", ticker)
        results["random_forest"] = (pred_rf, shap_data[0]["Feature"] if shap_data else "N/A")
    except Exception as e:
        logging.warning(f"⚠️ RF failed: {e}")
        results["random_forest"] = (None, "N/A")

    # =============== Linear Regression ===============
    try:
        mdl_lr, _ = train_linreg(X.values, y.values, ticker=ticker)
        pred_lr = float(mdl_lr.predict(Xv.values)[-1])
        shap_data, _ = explain_model(mdl_lr, Xv, "linear_regression", ticker)
        results["linear_regression"] = (pred_lr, shap_data[0]["Feature"] if shap_data else "N/A")
    except Exception as e:
        logging.warning(f"⚠️ LR failed: {e}")
        results["linear_regression"] = (None, "N/A")

    # =============== LSTM ===============
    try:
        close_values = train_df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_values)
        lookback = 10
        X_seq, y_seq = [], []
        for i in range(lookback, len(scaled)):
            X_seq.append(scaled[i - lookback:i])
            y_seq.append(scaled[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        X_train, X_val = X_seq[:-horizon], X_seq[-horizon:]
        y_train, y_val = y_seq[:-horizon], y_seq[-horizon:]

        model = build_lstm_univariate((lookback, 1))
        model, _ = train_lstm(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=16)

        # Forecast next horizon days
        last_window = scaled[-lookback:].reshape(1, lookback, 1)
        for _ in range(horizon):
            p = model.predict(last_window, verbose=0)
            last_window = np.append(last_window[:, 1:, :], p.reshape(1, 1, 1), axis=1)
        pred_lstm = scaler.inverse_transform(p)[0, 0]
        results["lstm"] = (float(pred_lstm), "Close")
    except Exception as e:
        logging.warning(f"⚠️ LSTM failed: {e}")
        results["lstm"] = (None, "N/A")

    # =============== Ensemble ===============
    valid_preds = [v[0] for v in results.values() if isinstance(v[0], (float, int))]
    ensemble_pred = float(np.mean(valid_preds)) if valid_preds else None
    results["ensemble"] = (ensemble_pred, "weighted mean")

    # =============== Compare with Actual ===============
    actual_next = yv.iloc[-1]
    rows = []
    for model, (pred, feature) in results.items():
        if pred:
            err = abs((pred - actual_next) / actual_next) * 100
        else:
            err = None
        rows.append({
            "Model": model,
            "Predicted_Price": pred,
            "Actual_Price": actual_next,
            "% Error": err,
            "Most_Influential_Feature": feature
        })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(REPORT_DIR, f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
    df_out.to_excel(out_path, index=False)
    logging.info(f"✅ Saved backtest results to {out_path}")
    return df_out


if __name__ == "__main__":
    ticker = input("Enter Chemical Sector Ticker (e.g., DEEPAKNTR.BO): ").strip()
    run_monthly_backtest(ticker)
