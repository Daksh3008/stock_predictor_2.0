# src/models/predict.py
import numpy as np
import pandas as pd

def predict_tabular_series(model, feature_cols, df_hist, steps, horizon_features_fn=None):
    """
    Iteratively forecast prices using same features used during training.
    """
    if horizon_features_fn is None:
        raise ValueError("horizon_features_fn must be provided")

    history = df_hist.copy()
    preds = []
    from pandas.tseries.offsets import BDay

    for _ in range(steps):
        # recompute features from current history
        df_feat = horizon_features_fn(history, horizon=1)
        X_last = df_feat[feature_cols].iloc[-1:].values
        pred_ret = float(model.predict(X_last)[0])
        last_price = history["Close"].iloc[-1]
        next_price = last_price * (1.0 + pred_ret)

        next_date = history.index[-1] + BDay(1)
        history.loc[next_date] = {
            "Open": next_price, "High": next_price, "Low": next_price,
            "Close": next_price, "Volume": 0
        }
        preds.append(next_price)

    dates = pd.bdate_range(df_hist.index[-1] + BDay(1), periods=steps)
    return pd.Series(preds, index=dates)


def predict_lstm_series(model, scaler, scaled_all, lookback, steps):
    """Predict future scaled returns using LSTM then inverse-scale."""
    history = list(scaled_all.squeeze())
    preds = []
    for _ in range(steps):
        x = np.array(history[-lookback:]).reshape(1, lookback, 1)
        yhat = model.predict(x, verbose=0)[0, 0]
        preds.append(yhat)
        history.append(float(yhat))
    preds = np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(preds).squeeze()
    return inv
