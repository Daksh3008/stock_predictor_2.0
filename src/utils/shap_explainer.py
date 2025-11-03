# src/utils/shap_explainer.py
import shap, numpy as np
def explain_model_xgb(model, X_sample):
    """
    Return SHAP values for a sample X_sample (pd.DataFrame).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # return the explainer & values; caller can plot or format
    return explainer, shap_values
