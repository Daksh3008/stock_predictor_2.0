# src/reasoning/macro_intents.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_macro_signals():
    """
    Fetch recent Brent Crude and USD/INR movements.
    Generate macroeconomic interpretation for chemical sector stocks.
    """
    today = datetime.now().date()
    start = today - timedelta(days=60)

    signals = {
        "brent_change": 0.0,
        "inr_change": 0.0,
        "macro_reasoning": "",
    }

    # --- Brent crude ---
    try:
        brent = yf.download("BZ=F", start=start, end=today, progress=False)
        if not brent.empty and "Close" in brent.columns:
            brent = brent.dropna(subset=["Close"])
            if len(brent) >= 5:
                last = float(brent["Close"].iloc[-1])
                prev = float(brent["Close"].iloc[-5])
                brent_ret = ((last / prev) - 1) * 100
                signals["brent_change"] = round(brent_ret, 2)
            else:
                signals["brent_change"] = 0.0
    except Exception:
        signals["brent_change"] = 0.0

    # --- USD/INR ---
    try:
        inr = yf.download("USDINR=X", start=start, end=today, progress=False)
        if not inr.empty and "Close" in inr.columns:
            inr = inr.dropna(subset=["Close"])
            if len(inr) >= 5:
                last = float(inr["Close"].iloc[-1])
                prev = float(inr["Close"].iloc[-5])
                inr_ret = ((last / prev) - 1) * 100
                signals["inr_change"] = round(inr_ret, 2)
            else:
                signals["inr_change"] = 0.0
    except Exception:
        signals["inr_change"] = 0.0

    # --- Reasoning Summary ---
    brent_change = float(signals["brent_change"])
    inr_change = float(signals["inr_change"])

    reasoning = []

    # Crude interpretation
    if brent_change > 3.0:
        reasoning.append("🛢️ Crude oil has surged >3% recently — potential input cost pressure on chemical producers.")
    elif brent_change < -3.0:
        reasoning.append("🛢️ Crude oil has declined sharply, lowering raw material costs and improving margins.")
    else:
        reasoning.append("🛢️ Crude oil is stable, minimal input cost volatility expected.")

    # INR interpretation
    if inr_change > 1.0:
        reasoning.append("💱 INR weakened vs USD — imports costlier, mild margin risk for crude-dependent firms.")
    elif inr_change < -1.0:
        reasoning.append("💱 INR strengthened vs USD — lower import cost pressure.")
    else:
        reasoning.append("💱 INR stable — currency effects neutral for input costs.")

    # General economic tone
    reasoning.append("🏭 Macro backdrop: industrial production steady; domestic demand resilience key to stock performance.")

    signals["macro_reasoning"] = " ".join(reasoning)
    return signals
