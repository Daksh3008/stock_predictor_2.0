# src/models/xgb_model.py
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt
from src.models.trainer import save_model_artifact, make_artifact_path
from src.utils.logger import get_logger

logger = get_logger("xgb_model")

def train_xgb_with_val(X_train, y_train, X_val, y_val, params=None, random_state=42, ticker="UNKNOWN"):
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
    # compute RMSE robustly
    try:
        rmse = float(mean_squared_error(y_val, preds, squared=False))
    except TypeError:
        rmse = float(sqrt(mean_squared_error(y_val, preds)))
    # save
    try:
        path = make_artifact_path("xgb", ticker)
        metadata = {"model":"xgboost","ticker":ticker,"rmse":rmse}
        save_model_artifact(model, path, metadata)
    except Exception as e:
        logger.warning("Failed saving xgb artifact: %s", e)
        path = None
    return model, params, rmse, path
