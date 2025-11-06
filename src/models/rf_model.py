# src/models/rf_model.py
from sklearn.ensemble import RandomForestRegressor
from src.models.trainer import save_model_artifact, make_artifact_path
from src.utils.logger import get_logger
import numpy as np

logger = get_logger("rf_model")

def train_rf(X_train, y_train, ticker="UNKNOWN"):
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    try:
        path = make_artifact_path("rf", ticker)
        metadata = {"model":"random_forest","ticker":ticker,"train_rows":int(X_train.shape[0])}
        save_model_artifact(rf, path, metadata)
    except Exception as e:
        logger.warning("Failed save rf: %s", e)
        path = None
    return rf, path
