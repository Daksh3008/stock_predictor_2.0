# src/analysis/monthly_backtest.py
"""
Monthly backtest:
- Train on data up to 2025-09-30
- Predict next 30 business days
- Compare predictions with actuals from Yahoo Finance
- Output Excel with one sheet per model + ensemble
"""

import os
import sys
import json
import time
from datetime import datetime
import warnings

# --- Make sure repo root is on path so we can import src modules ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

from src.data.fetch_data import fetch_data
from src.models.xgb_model import train_xgb_with_val
from src.models.rf_model import train_rf
from src.models.linreg_model import train_linreg
from src.models.lstm_model import build_lstm_univariate, train_lstm
from src.models.ensemble import weights_from_scores, weighted_average
from src.explain.shap_explainer import explain_model as shap_explain
from src.utils.seed import set_global_seed

# reproducibility
set_global_seed(42)


# -------------------------
# Helper: create engineered features (returns-based)
# -------------------------
def create_features(df, horizon=1):
    df = df.copy().sort_index()
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ma_ratio_5"] = df["Close"] / df["Close"].rolling(5).mean()
    df["ma_ratio_10"] = df["Close"] / df["Close"].rolling(10).mean()
    df["vol_5"] = df["ret_1"].rolling(5).std()
    # target is future pct change over `horizon` days
    df["target"] = df["Close"].pct_change(horizon).shift(-horizon)
    return df.dropna()


# -------------------------
# Helper: iterative forecast for tabular models (predicting returns)
# -------------------------
def recursive_forecast_tabular(df_base, model, horizon_days, feature_cols):
    """
    df_base: dataframe containing historical OHLCV indexed by Date (training + any appended pred rows)
    model: trained model that takes feature vector and outputs predicted return (scalar)
    horizon_days: integer (number of business days to forecast)
    feature_cols: list of columns used as features in model (must be present in df_base for existing rows)
    Returns: pandas Series of predicted prices indexed by future business dates (length = horizon_days)
    """
    df_work = df_base.copy().sort_index()
    preds = []
    dates = []

    # compute last available business day
    last_date = df_work.index[-1]
    # We'll append predicted rows to df_work one by one
    from pandas.tseries.offsets import BDay
    for step in range(horizon_days):
        # next date = next business day
        next_date = last_date + BDay(1)
        # compute features for the row to be predicted based on df_work (which contains previous preds)
        # we compute feature columns on the fly from Close series
        close_series = df_work["Close"]
        # create temp row features
        # ret_1, ret_5, ma_ratio_5, ma_ratio_10, vol_5
        last_close = close_series.iloc[-1]
        # ret_1
        if len(close_series) >= 1:
            ret_1 = (last_close / close_series.iloc[-1]) - 1.0  # zero
        else:
            ret_1 = 0.0
        # ret_5
        if len(close_series) >= 5:
            ret_5 = (last_close / close_series.iloc[-5]) - 1.0
        else:
            ret_5 = last_close / close_series.iloc[0] - 1.0
        # ma_ratio_5
        if len(close_series) >= 5:
            ma5 = close_series.iloc[-5:].mean()
            ma_ratio_5 = last_close / (ma5 + 1e-9)
        else:
            ma_ratio_5 = 1.0
        # ma_ratio_10
        if len(close_series) >= 10:
            ma10 = close_series.iloc[-10:].mean()
            ma_ratio_10 = last_close / (ma10 + 1e-9)
        else:
            ma_ratio_10 = 1.0
        # vol_5
        if len(close_series) >= 6:
            ret_series = close_series.pct_change().dropna()
            vol_5 = ret_series.iloc[-5:].std()
            vol_5 = vol_5 if not pd.isna(vol_5) else 0.0
        else:
            vol_5 = 0.0

        x_row = np.array([[ret_1, ret_5, ma_ratio_5, ma_ratio_10, vol_5]])
        try:
            pred_return = float(model.predict(x_row)[0])
        except Exception:
            # If model expects flattened input, try predict on 1d
            pred_return = float(model.predict(x_row.reshape(1, -1))[0])

        pred_price = last_close * (1.0 + pred_return)

        # append to df_work as a new row so next iter uses it
        new_row = {
            "Open": pred_price, "High": pred_price, "Low": pred_price, "Close": pred_price, "Volume": 0
        }
        df_work.loc[next_date] = pd.Series(new_row)

        preds.append(pred_price)
        dates.append(next_date)
        last_date = next_date

    return pd.Series(data=preds, index=pd.to_datetime(dates))


# -------------------------
# Helper: LSTM sequence predictor (univariate close series)
# -------------------------
def forecast_lstm_close(df_base, lookback, model, scaler, steps):
    """
    model: trained LSTM that predicts next step given (1, lookback, 1)
    scaler: MinMaxScaler used to scale Close
    returns: pandas Series indexed by next `steps` business days
    """
    history = list(scaler.transform(df_base["Close"].values.reshape(-1, 1)).squeeze())
    preds_scaled = []
    for _ in range(steps):
        x = np.array(history[-lookback:]).reshape(1, lookback, 1)
        yhat = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(yhat)
        history.append(float(yhat))
    # invert scale
    inv = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).squeeze()
    # Create business day dates starting after last index
    from pandas.tseries.offsets import BDay
    last_date = df_base.index[-1]
    dates = [last_date + BDay(i + 1) for i in range(len(inv))]
    return pd.Series(data=inv, index=pd.to_datetime(dates))


# -------------------------
# Top-level backtest routine
# -------------------------
def monthly_backtest():
    ticker = input("Enter stock ticker (e.g., TCS.NS or ACE.BO): ").strip().upper()
    # training cutoff: upto 2025-09-30
    cutoff_date = pd.to_datetime("2025-09-15").date()
    horizon_days = 30

    print(f"\nRunning monthly backtest for {ticker}")
    print(f"Training cutoff: {cutoff_date} -> forecasting next {horizon_days} business days\n")

    # 1) fetch training data up to cutoff_date inclusive
    df_train = fetch_data(ticker, start_date="2005-01-01", end_date=(pd.to_datetime(cutoff_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if df_train is None or df_train.empty:
        print("Failed to fetch training data. Exiting.")
        return

    # ensure index is business-day datetime index
    df_train = df_train.sort_index()
    print(f"Training data rows: {len(df_train)}, last_date = {df_train.index[-1].date()}")

    # 2) create features and split train/val (80/20)
    df_feat = create_features(df_train, horizon=1)  # horizon for training target is 1-day for models that predict one-day return
    feature_cols = ["ret_1", "ret_5", "ma_ratio_5", "ma_ratio_10", "vol_5"]

    X = df_feat[feature_cols]
    y = df_feat["target"]
    split = int(len(X) * 0.8)
    Xtr, Xv = X.iloc[:split], X.iloc[split:]
    ytr, yv = y.iloc[:split], y.iloc[split:]

    # 3) Train models
    print("Training tabular models...")
    # XGBoost
    try:
        mdl_xgb, _, rmse_xgb, _ = train_xgb_with_val(Xtr.values, ytr.values, Xv.values, yv.values, ticker=ticker)
        print("  XGB trained")
    except Exception as e:
        print("  XGB training failed:", e)
        mdl_xgb = None
        rmse_xgb = 9999.0

    # Random Forest
    try:
        mdl_rf, _ = train_rf(Xtr.values, ytr.values, ticker=ticker)
        rmse_rf = 0.05  # placeholder: RF training helper currently doesn't return rmse
        print("  RF trained")
    except Exception as e:
        print("  RF training failed:", e)
        mdl_rf = None
        rmse_rf = 9999.0

    # Linear Regression
    try:
        mdl_lr, _ = train_linreg(Xtr.values, ytr.values, ticker=ticker)
        rmse_lr = 0.05
        print("  LR trained")
    except Exception as e:
        print("  LR training failed:", e)
        mdl_lr = None
        rmse_lr = 9999.0

    # LSTM (univariate close)
    print("Training LSTM (univariate on Close)...")
    try:
        close_values = df_train["Close"].values.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_values)
        lookback = 10
        # create sequences
        X_seq = []
        y_seq = []
        for i in range(lookback, len(scaled)):
            X_seq.append(scaled[i - lookback:i].squeeze())
            y_seq.append(scaled[i].squeeze())
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        if len(X_seq) < 30:
            raise ValueError("Not enough data for LSTM training")

        split_seq = int(len(X_seq) * 0.8)
        Xtr_seq, Xv_seq = X_seq[:split_seq], X_seq[split_seq:]
        ytr_seq, yv_seq = y_seq[:split_seq], y_seq[split_seq:]

        # reshape to (samples, lookback, 1)
        Xtr_seq = Xtr_seq.reshape((Xtr_seq.shape[0], lookback, 1))
        Xv_seq = Xv_seq.reshape((Xv_seq.shape[0], lookback, 1))

        lstm_model = build_lstm_univariate((lookback, 1))
        lstm_model, history = train_lstm(lstm_model, Xtr_seq, ytr_seq, Xv_seq, yv_seq, epochs=20, batch_size=16)
        print("  LSTM trained")
    except Exception as e:
        print("  LSTM training failed:", e)
        lstm_model = None
        scaler = None
        lookback = 10

    # 4) Forecast next 30 business days using each model
    print("\nForecasting next 30 business days...")
    # Setup predictions container
    preds_dict = {}
    # Tabular models: use recursive_forecast_tabular
    if mdl_xgb is not None:
        try:
            preds_xgb = recursive_forecast_tabular(df_train[["Close", "Open", "High", "Low", "Volume"]].copy(), mdl_xgb, horizon_days, feature_cols)
            preds_dict["xgboost"] = preds_xgb
        except Exception as e:
            print("  XGB forecast failed:", e)
            preds_dict["xgboost"] = pd.Series(dtype=float)
    else:
        preds_dict["xgboost"] = pd.Series(dtype=float)

    if mdl_rf is not None:
        try:
            preds_rf = recursive_forecast_tabular(df_train[["Close", "Open", "High", "Low", "Volume"]].copy(), mdl_rf, horizon_days, feature_cols)
            preds_dict["random_forest"] = preds_rf
        except Exception as e:
            print("  RF forecast failed:", e)
            preds_dict["random_forest"] = pd.Series(dtype=float)
    else:
        preds_dict["random_forest"] = pd.Series(dtype=float)

    if mdl_lr is not None:
        try:
            preds_lr = recursive_forecast_tabular(df_train[["Close", "Open", "High", "Low", "Volume"]].copy(), mdl_lr, horizon_days, feature_cols)
            preds_dict["linear_regression"] = preds_lr
        except Exception as e:
            print("  LR forecast failed:", e)
            preds_dict["linear_regression"] = pd.Series(dtype=float)
    else:
        preds_dict["linear_regression"] = pd.Series(dtype=float)

    # LSTM
    if lstm_model is not None and scaler is not None:
        try:
            preds_lstm = forecast_lstm_close(df_train[["Close"]].copy(), lookback, lstm_model, scaler, horizon_days)
            preds_dict["lstm"] = preds_lstm
        except Exception as e:
            print("  LSTM forecast failed:", e)
            preds_dict["lstm"] = pd.Series(dtype=float)
    else:
        preds_dict["lstm"] = pd.Series(dtype=float)

    # 5) Get actuals from Yahoo for the prediction date range
    # determine forecast date range from any non-empty preds_series
    any_preds = [s for s in preds_dict.values() if isinstance(s, pd.Series) and not s.empty]
    if not any_preds:
        print("No predictions were produced by any model. Exiting.")
        return
    # Use the first non-empty series to get date index
    dates = any_preds[0].index
    start_actual = dates[0].strftime("%Y-%m-%d")
    end_actual = dates[-1].strftime("%Y-%m-%d")
    print(f"Fetching actual prices from Yahoo: {start_actual} -> {end_actual}")
    df_actual = fetch_data(ticker, start_date=start_actual, end_date=(pd.to_datetime(end_actual) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if df_actual is None or df_actual.empty:
        print("Failed to fetch actual prices. Continuing but actuals will be empty.")
        df_actual = pd.DataFrame()

    # 6) SHAP top feature extraction (best-effort)
    shap_top = {}
    for model_name, model_obj in [("xgboost", mdl_xgb), ("random_forest", mdl_rf), ("linear_regression", mdl_lr)]:
        top_f = "N/A"
        if model_obj is not None:
            try:
                # use the validation set Xv (if available) to compute SHAP
                X_for_shap = Xv if "Xv" in locals() else X  # fallback
                top_feats, _ = shap_explain(model_obj, X_for_shap, model_name, ticker)
                if isinstance(top_feats, list) and len(top_feats) > 0:
                    top_f = top_feats[0].get("Feature", "N/A")
            except Exception:
                top_f = "N/A"
        shap_top[model_name] = top_f

    # LSTM top feature is Close
    shap_top["lstm"] = "Close"

    # Ensemble - which model gets highest weight?
    # Form RMS dictionary: prefer computed rmse_xgb else placeholders
    rmses = {"xgboost": rmse_xgb if 'rmse_xgb' in locals() else 9999.0,
             "random_forest": rmse_rf if 'rmse_rf' in locals() else 9999.0,
             "linear_regression": rmse_lr if 'rmse_lr' in locals() else 9999.0,
             "lstm": float(np.mean(history.history["val_loss"])) if 'history' in locals() and hasattr(history, "history") else 9999.0}
    # compute weights
    try:
        weights = weights_from_scores(rmses)
        # top model = model with highest weight
        top_model = max(weights.items(), key=lambda x: x[1])[0]
        shap_top["ensemble"] = shap_top.get(top_model, "N/A")
    except Exception:
        shap_top["ensemble"] = "N/A"

    # compute ensemble series: weighted average of model price series (align by index)
    # convert all preds to dataframe with same index (dates). Missing values => NaN
    pred_df = pd.DataFrame(index=dates)
    for k, s in preds_dict.items():
        if isinstance(s, pd.Series):
            pred_df[k] = s.reindex(dates).values
        else:
            pred_df[k] = np.nan
    # weights (normalize)
    try:
        arr = np.array([weights.get(k, 0.0) for k in ["xgboost", "random_forest", "linear_regression", "lstm"]], dtype=float)
        if arr.sum() == 0:
            norm_weights = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            norm_weights = arr / arr.sum()
    except Exception:
        norm_weights = np.array([0.25, 0.25, 0.25, 0.25])
    # compute ensemble
    # handle NaNs by treating them as zero contribution and renormalizing weights for available models
    ensemble_vals = []
    for idx in range(len(dates)):
        row = pred_df.iloc[idx]
        vals = row.values.astype(float)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() == 0:
            ensemble_vals.append(np.nan)
            continue
        w = norm_weights.copy()
        # zero out weights for missing models and renorm
        w = w * valid_mask
        if w.sum() == 0:
            w = np.array([1.0 / valid_mask.sum() if valid_mask.sum() > 0 else 0.0 for _ in range(len(w))])
        else:
            w = w / w.sum()
        ensemble_vals.append(np.nansum(vals * w))
    pred_df["ensemble"] = ensemble_vals

    # 7) Build Excel workbook: one sheet per model: Date | Predicted | Actual | % Error | Top_Feature
    out_rows = {}
    for model_name in ["xgboost", "random_forest", "linear_regression", "lstm", "ensemble"]:
        preds = pred_df.get(model_name, pd.Series(index=dates, dtype=float))
        actuals = []
        for d in dates:
            a = df_actual["Close"].get(d, np.nan) if (isinstance(df_actual, pd.DataFrame) and "Close" in df_actual.columns) else np.nan
            actuals.append(a)
        actuals = pd.Series(actuals, index=dates)
        pct_err = (np.abs((preds - actuals) / (actuals + 1e-9))) * 100.0
        df_tab = pd.DataFrame({
            "Date": dates,
            "Predicted": np.round(preds.values, 4),
            "Actual": np.round(actuals.values, 4),
            "Pct_Error": np.round(pct_err.values, 4),
            "Top_Feature": [shap_top.get(model_name, "N/A")] * len(dates)
        })
        out_rows[model_name] = df_tab

    # 8) Save excel workbook
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f"backtest_{ticker}_{ts}.xlsx"
    out_path = os.path.join(ROOT, out_name)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Summary sheet
        summary = {
            "Ticker": ticker,
            "Training Cutoff": str(cutoff_date),
            "Prediction Start": str(dates[0].date()),
            "Prediction End": str(dates[-1].date()),
            "Models": "xgboost, random_forest, linear_regression, lstm, ensemble"
        }
        pd.DataFrame([summary]).to_excel(writer, sheet_name="SUMMARY", index=False)
        # Write each model sheet
        for model_name, df_tab in out_rows.items():
            # sheet names must be <=31 chars
            sheet = model_name[:31]
            df_tab.to_excel(writer, sheet_name=sheet, index=False)
    print(f"\nBacktest saved → {out_path}")
    print("Done.")


if __name__ == "__main__":
    monthly_backtest()
