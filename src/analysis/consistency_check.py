import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.main import smoke   # uses your existing main ensemble pipeline
import logging

# -------------------------------------------------
# Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

RUNS = 5
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------------------------------------
# Load ticker list
# -------------------------------------------------
stock_file = os.path.join("src", "data", "stock_list.json")
with open(stock_file, "r") as f:
    tickers = json.load(f)["tickers"]

logging.info(f"Loaded {len(tickers)} tickers for consistency check: {tickers}")

# -------------------------------------------------
# Run consistency analysis
# -------------------------------------------------
records = []

for ticker in tickers:
    logging.info(f"🔁 Running {RUNS} predictions for {ticker}")

    preds_xgb, preds_rf, preds_lr, preds_lstm, preds_ensemble = [], [], [], [], []

    for i in range(RUNS):
        logging.info(f"Run {i+1}/{RUNS} for {ticker}")
        try:
            # smoke() already trains, predicts, and returns predictions
            result = smoke(ticker, horizon=7, return_results=True)

            preds_xgb.append(result["xgboost"])
            preds_rf.append(result["random_forest"])
            preds_lr.append(result["linear_regression"])
            preds_lstm.append(result["lstm"])
            preds_ensemble.append(result["ensemble"])

        except Exception as e:
            logging.error(f"⚠️ Error in run {i+1} for {ticker}: {e}")
            continue

    # Helper to compute stats
    def stability_stats(name, preds):
        preds = np.array(preds)
        mean_ = np.mean(preds)
        std_ = np.std(preds)
        var_pct = (std_ / mean_) * 100 if mean_ != 0 else 0
        stability = 1 - (std_ / mean_) if mean_ != 0 else 0
        return {
            "model": name,
            "ticker": ticker,
            "mean_pred": mean_,
            "std_dev": std_,
            "variation_pct": var_pct,
            "stability_score": stability,
        }

    # Collect stats
    models = {
        "xgboost": preds_xgb,
        "random_forest": preds_rf,
        "linear_regression": preds_lr,
        "lstm": preds_lstm,
        "ensemble": preds_ensemble,
    }

    for model_name, pred_list in models.items():
        if len(pred_list) > 0:
            records.append(stability_stats(model_name, pred_list))

    # Plot distribution for each ticker
    plt.figure(figsize=(8, 5))
    for model_name, pred_list in models.items():
        if len(pred_list) > 0:
            plt.plot(range(1, len(pred_list)+1), pred_list, marker='o', label=model_name)
    plt.title(f"Prediction Consistency for {ticker}")
    plt.xlabel("Run Number")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"consistency_plot_{ticker}.png"))
    plt.close()

# -------------------------------------------------
# Save aggregated results
# -------------------------------------------------
df_results = pd.DataFrame(records)
df_results = df_results.sort_values(["ticker", "model"])
out_path = os.path.join(REPORT_DIR, "consistency_results.csv")
df_results.to_csv(out_path, index=False)

logging.info(f"✅ Consistency results saved to {out_path}")
logging.info("Top 5 results:\n" + str(df_results.head()))
