import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def explain_model(model, X, model_name, ticker):
    """
    Generate SHAP explainability report for the given model.
    Supports tree-based and linear models.
    """
    os.makedirs("reports", exist_ok=True)

    try:
        # Ensure valid DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Select SHAP explainer type
        if isinstance(model, (XGBRegressor, RandomForestRegressor)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif isinstance(model, LinearRegression):
            explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(X)
        else:
            # generic fallback
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

        # SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        try:
            shap.summary_plot(
                shap_values,
                X,
                feature_names=X.columns,
                show=False
            )
        except Exception:
            # Sometimes SHAP returns an object with .values attr
            shap.summary_plot(
                shap_values.values,
                X,
                feature_names=X.columns,
                show=False
            )

        shap_plot_path = os.path.join("reports", f"{ticker}_{model_name}_shap_summary.png")
        plt.tight_layout()
        plt.savefig(shap_plot_path)
        plt.close()

        # Compute Top Features
        shap_array = shap_values if isinstance(shap_values, np.ndarray) else shap_values.values
        top_features = pd.DataFrame({
            "Feature": X.columns,
            "Mean_SHAP_Value": np.abs(shap_array).mean(axis=0)
        }).sort_values(by="Mean_SHAP_Value", ascending=False)
        top_features["Rank"] = range(1, len(top_features) + 1)

        top_path = os.path.join("reports", f"{ticker}_{model_name}_shap_top_features.csv")
        top_features.to_csv(top_path, index=False)

        # Save JSON Summary (Top 10)
        shap_json = {
            "ticker": ticker,
            "model": model_name,
            "top_features": top_features.head(10).to_dict(orient="records")
        }

        shap_json_path = os.path.join("reports", f"{ticker}_{model_name}_shap_summary.json")
        with open(shap_json_path, "w") as f:
            json.dump(shap_json, f, indent=2)

        logging.info(f"Saved SHAP plot → {shap_plot_path}")
        logging.info(f"Saved top features → {top_path}")

        return top_features.head(10).to_dict(orient="records"), shap_plot_path

    except Exception as e:
        logging.warning(f"⚠️ SHAP failed for {model_name}: {e}")
        return {}, None


def combine_shap_reports(ticker):
    """
    Combine all SHAP summaries for a given ticker into one file.
    """
    summary = {}
    os.makedirs("reports", exist_ok=True)

    for file in os.listdir("reports"):
        if file.startswith(ticker) and file.endswith("_shap_summary.json"):
            try:
                path = os.path.join("reports", file)
                with open(path, "r") as f:
                    data = json.load(f)
                model_name = data.get("model", "unknown")
                summary[model_name] = data.get("top_features", [])
            except Exception as e:
                logging.warning(f"Error reading {file}: {e}")

    combined_path = os.path.join("reports", f"{ticker}_shap_combined_summary.json")
    with open(combined_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"📊 Combined SHAP explainability report saved to {combined_path}")
    return summary
