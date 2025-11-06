# src/explain/shap_explainer.py
import shap, pandas as pd, numpy as np, logging
from src.utils.logger import get_logger
logger = get_logger("shap_explainer")

def explain_model(model, X, model_name, ticker):
    """
    Returns top features list and saves basic summary plot (best-effort).
    X may be DataFrame or numpy array.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            X = X.iloc[:, :len(cols)]
            X.columns = cols
        else:
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        if hasattr(model, "predict") and (hasattr(model, "get_booster") or "xgb" in model_name.lower() or "random" in model_name.lower()):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(X)

        # compute mean abs shap by feature
        arr = shap_values.values if hasattr(shap_values, "values") else shap_values
        mean_abs = np.mean(np.abs(arr), axis=0)
        top_idx = np.argsort(-mean_abs)[:10]
        top = [{"Feature": X.columns[i], "Mean_SHAP_Value": float(mean_abs[i])} for i in top_idx]
        return top, None
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        return [], None
