# src/reasoning/text_summary.py
def generate_reasoning_text(ticker, predicted_price, confidence, macro_info, shap_features, horizon_days):
    """
    Generate human-readable reasoning combining macro and micro explanations.
    """
    macro_lines = " ".join(macro_info.get("macro_interpretation", []))
    top_feats = ", ".join(shap_features[:3])

    tone = "bullish" if "relief" in macro_lines or "boost" in macro_lines else "neutral"
    if "pressure" in macro_lines:
        tone = "bearish"

    reasoning = (
        f"Model predicts ₹{predicted_price:.2f} ({horizon_days} days ahead) with confidence {confidence:.1f}%.\n"
        f"Top drivers (micro): {top_feats}.\n"
        f"Macro context: {macro_lines or 'Stable macro conditions observed.'}\n"
        f"Outlook tone: {tone.capitalize()}.\n"
        f"Interpretation: The model expects {ticker} to remain {tone} under current market conditions, "
        f"with chemical margins linked primarily to crude and INR movement."
    )
    return reasoning
