# src/models/trainer.py
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
import joblib, os
from datetime import datetime
logger = get_logger("trainer")

def save_model_artifact(model, path: str, metadata: dict):
    """
    Save model + metadata. For sklearn/xgboost use joblib; for keras, model.save separately.
    """
    base = os.path.splitext(path)[0]
    # save model
    try:
        joblib.dump(model, path)
    except Exception:
        # maybe keras: try save method
        try:
            model.save(path)
        except Exception:
            raise
    # save metadata
    meta_path = base + ".meta.json"
    import json
    with open(meta_path, "w") as f:
        json.dump(metadata, f, default=str)
    logger.info(f"Saved model artifact: {path}, meta: {meta_path}")

def train_with_seed(train_fn, seed: int, *args, **kwargs):
    """
    Call this to train with deterministic seed set.
    train_fn should be a callable returning a trained model and a dict of metrics.
    """
    set_global_seed(seed)
    model, metrics = train_fn(*args, **kwargs)
    return model, metrics
