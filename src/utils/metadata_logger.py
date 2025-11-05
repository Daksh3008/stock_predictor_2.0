# src/utils/metadata_logger.py
import os, json
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("metadata_logger")

def save_metadata(ticker, model_name, params=None, metrics=None, shap_summary=None):
    """
    Save metadata per model to JSON file under reports/metadata.
    """
    os.makedirs("reports/metadata", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    record = {
        "ticker": ticker,
        "model": model_name,
        "timestamp": ts,
        "params": params or {},
        "metrics": metrics or {},
        "top_shap_features": shap_summary or [],
    }

    path = os.path.join("reports/metadata", f"{ticker}_{model_name}_{ts}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"📄 Metadata saved → {path}")
    return path
