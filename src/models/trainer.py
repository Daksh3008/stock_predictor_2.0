#ensure reusable helper to save artifacts and metadata
import os
import joblib
import json
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("trainer")

def save_model_artifact(model, path: str, metadata: dict):
    """
    Save model artifact and metadata.
    - For sklearn/xgboost models we use joblib.
    - For keras models, pass a model that has `.save()` method and supply path ending in '/'
    Metadata saved to JSON sidecar.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)

    # Try joblib save first (works for sklearn/xgb)
    try:
        joblib.dump(model, path)
        saved = True
    except Exception:
        # Keras fallback: try model.save to a directory or h5
        try:
            # if path looks like directory (no ext) or ends with .h5
            if path.endswith("/") or ext in ("", ".dir"):
                model.save(path)
            else:
                model.save(path + ".h5")
            saved = True
        except Exception as ex:
            saved = False
            logger.error(f"Failed to save model artifact via both joblib and keras.save: {ex}")
            raise

    # metadata sidecar
    meta_path = base + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, default=str, indent=2)
    logger.info(f"Saved artifact: {path} and metadata: {meta_path}")
    return path, meta_path

def make_artifact_path(model_name: str, ticker: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{model_name}_{ticker}_{ts}.pkl"
    directory = os.path.abspath(os.path.join(os.getcwd(), "models_cache"))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, fname)
