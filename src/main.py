# src/main.py
import argparse
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.data.fetch_data import fetch_data
from src.data.features import add_technical_indicators
from src.models.xgb_model import train_xgb_with_val
from src.models.rf_model import train_rf
from src.models.linreg_model import train_linreg
from src.models.lstm_model import build_lstm_univariate, train_lstm
from src.models.predict import predict_tabular_series, predict_lstm_series
from src.models.ensemble import weights_from_scores, weighted_average
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = get_logger("main")

def prepare_tabular(df_feat, lookback):
    """
    For tabular models: create X matrix of last lookback closes flattened.
    This is a simplification — you can replace with richer supervised creation.
    """
    arr = df_feat["Close"].values
    X = []
    y = []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i])
        y.append(arr[i])
    import numpy as np
    return np.array(X), np.array(y)

def smoke(ticker, horizon, seed=42):
    set_global_seed(seed)
    logger.info(f"Running smoke for {ticker}")
    df = fetch_data(ticker, start_date="2008-01-01")
    dff = add_technical_indicators(df)
    # simple scalar target: Close
    lookback = 10
    # scale full
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dff[["Close"]].values)
    # prepare sequences (for demo, re-use functions lightly)
    X_tab, y_tab = prepare_tabular(dff, lookback)
    # split
    split = int(len(X_tab)*0.8)
    Xtr, Xv = X_tab[:split], X_tab[split:]
    ytr, yv = y_tab[:split], y_tab[split:]

    # XGBoost
    #mdl_xgb, _, rm_xgb = train_xgb_with_val(Xtr, ytr, Xv, yv, params=None, random_state=seed)
    mdl_xgb, _, rm_xgb, xgb_path = train_xgb_with_val(Xtr, ytr, Xv, yv, params=None, random_state=seed, ticker=ticker)
    
    # RF
    #mdl_rf = train_rf(Xtr, ytr)
    mdl_rf, rf_path = train_rf(Xtr, ytr, ticker=ticker)

    # LR
    mdl_lr = train_linreg(Xtr, ytr)
    mdl_lr, lr_path = train_linreg(Xtr, ytr, ticker=ticker)

    # LSTM: prepare sequences shaped for LSTM
    mdl_lstm, lstm_path = train_lstm(mdl_lstm, Xtr_seq, ytr_seq, Xv_seq, yv_seq, epochs=10, batch_size=16, ticker=ticker)

    # build one-day predictions: use last lookback window
    last_window = X_tab[-1].reshape(1, -1)
    p_xgb = float(mdl_xgb.predict(last_window)[0])
    p_rf = float(mdl_rf.predict(last_window)[0])
    p_lr = float(mdl_lr.predict(last_window)[0])
    preds = {"xgboost": p_xgb, "random_forest": p_rf, "linear_regression": p_lr}
    # compute weights
    scores = {"xgboost": rm_xgb, "random_forest": 0.05, "linear_regression": 0.05}
    w = weights_from_scores(scores)
    ensemble_pred = weighted_average(preds, w)
    logger.info(f"Predictions: {preds}, ensemble: {ensemble_pred:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="TCS.NS")
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    smoke(args.ticker, args.horizon)
