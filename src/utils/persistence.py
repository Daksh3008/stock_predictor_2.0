# src/utils/persistence.py
import os, joblib, json
def save_sklearn_model(model, path, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    if metadata:
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, default=str)
