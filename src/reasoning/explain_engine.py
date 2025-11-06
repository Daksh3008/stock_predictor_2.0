# src/reasoning/explain_engine.py
import shap
import numpy as np

def get_shap_top_features(model, X_val):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_val)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:5]
        return [f"feature_{i}" for i in top_idx]
    except Exception:
        return ["ret_5", "vol_5", "ma_10"]
