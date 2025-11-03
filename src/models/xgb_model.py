# src/models/xgb_model.py
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_xgb_with_val(X_train, y_train, X_val, y_val, params=None, random_state=0):
    params = params or {}
    model = XGBRegressor(
        n_estimators = params.get("n_estimators", 500),
        learning_rate = params.get("learning_rate", 0.05),
        max_depth = params.get("max_depth", 4),
        subsample = params.get("subsample", 0.8),
        colsample_bytree = params.get("colsample_bytree", 0.8),
        random_state = random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    from math import sqrt
    rmse = sqrt(mean_squared_error(y_val, preds))
    return model, params, rmse
