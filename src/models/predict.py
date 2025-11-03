# src/models/predict.py
import numpy as np

def predict_tabular_series(model, scaler, scaled_all, lookback, steps):
    """
    Recursive / iterative prediction for tabular models.
    - model: trained sklearn/xgboost regressor
    - scaler: scaler used for inverse transform (expects 2D)
    - scaled_all: full scaled series used as baseline (numpy array shape (N,1))
    - lookback: window size
    - steps: number of future steps to forecast (int)
    returns: array of length steps in original price scale
    """
    history = list(scaled_all.squeeze())  # last known points
    preds = []
    for _ in range(steps):
        # build last lookback as features (flatten)
        x = np.array(history[-lookback:]).reshape(1, -1)
        # if model expects X_tab shaped differently, adapt upstream
        yhat_scaled = model.predict(x)
        preds.append(yhat_scaled)
        history.append(float(yhat_scaled))
    preds = np.array(preds).reshape(-1,1)
    return scaler.inverse_transform(preds).squeeze()

def predict_lstm_series(model, scaler, scaled_all, lookback, steps):
    history = list(scaled_all.squeeze())
    preds = []
    for _ in range(steps):
        x = np.array(history[-lookback:]).reshape(1, lookback, 1)
        yhat = model.predict(x, verbose=0)
        preds.append(yhat[0,0])
        history.append(float(yhat[0,0]))
    preds = np.array(preds).reshape(-1,1)
    return scaler.inverse_transform(preds).squeeze()
