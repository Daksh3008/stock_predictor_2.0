# src/run_batch.py
# Run batch predictions for multiple tickers.

import os
import sys
import json
import time
import warnings
import logging
from datetime import datetime
import numpy as np

# setup path for src imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

from src.main import smoke


# -------------
# CONFIG
# -------------
STOCK_FILE = os.path.join(ROOT, "src", "automation", "stock_list.json")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
RUNS_PER_TICKER = 5
HORIZON = 7


# -------------
# Load tickers
# -------------
def load_tickers():
    if not os.path.exists(STOCK_FILE):
        raise FileNotFoundError(f"❌ stock_list.json not found at {STOCK_FILE}")
    with open(STOCK_FILE, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "stocks" in data:
        return data["stocks"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("❌ stock_list.json must contain a list or dict with 'stocks' key")


# -------------
# Run batch
# -------------
def run_batch():
    tickers = load_tickers()
    batch_summary = []

    logging.info(f"📊 Starting batch run for {len(tickers)} tickers, {RUNS_PER_TICKER} runs each.")

    for ticker in tickers:
        ticker_preds = []
        for i in range(RUNS_PER_TICKER):
            try:
                logging.info(f"▶️ Run {i+1}/{RUNS_PER_TICKER} for {ticker}")
                result = smoke(ticker, horizon=HORIZON, return_results=True)
                ticker_preds.append(result)
                time.sleep(2)  # polite pause
            except Exception as e:
                logging.warning(f"⚠️ Error processing {ticker}: {e}")

        if ticker_preds:
            avg_preds = {}
            for key in ["xgboost", "random_forest", "linear_regression", "lstm", "ensemble"]:
                vals = [r[key] for r in ticker_preds if r[key] is not None]
                avg_preds[key] = float(np.mean(vals)) if vals else None
        else:
            avg_preds = {k: None for k in ["xgboost", "random_forest", "linear_regression", "lstm", "ensemble"]}

        batch_summary.append({"ticker": ticker, **avg_preds})

    # Save to timestamped JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(REPORT_DIR, f"batch_summary_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    logging.info(f"✅ Batch run completed. Results saved to {out_path}")
    return batch_summary


if __name__ == "__main__":
    run_batch()
