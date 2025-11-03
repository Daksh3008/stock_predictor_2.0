# src/automation/validate_close.py
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd, yfinance as yf
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("validate_daily")
OUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../predictions_log.xlsx"))

df = pd.read_excel(OUT_FILE)
today_str = datetime.now().strftime("%Y-%m-%d")
today_preds = df[df["Date"] == today_str].copy()
actuals = []
for _, row in today_preds.iterrows():
    t = row["Ticker"]
    try:
        data = yf.download(t, period="2d", progress=False)
        if not data.empty:
            actual = data["Close"].iloc[-1]
            pred = row["Predicted_Today"]
            pct_err = abs((pred - actual)/actual)*100.0
            row_idx = row.name
            df.loc[row_idx, "Actual"] = actual
            df.loc[row_idx, "Pct_Error"] = pct_err
    except Exception as e:
        logger.error(f"Failed fetch {t}: {e}")

df.to_excel(OUT_FILE, index=False)
logger.info("Validation appended")
