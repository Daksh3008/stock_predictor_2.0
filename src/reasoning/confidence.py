# src/reasoning/confidence.py
import numpy as np

def compute_confidence(pred_list):
    """
    Compute confidence score (0–100) based on ensemble stability.
    """
    if len(pred_list) < 2:
        return 70.0  # default
    mean_val = np.mean(pred_list)
    std_val = np.std(pred_list)
    if mean_val == 0:
        return 50.0
    raw_conf = 100 * (1 - (std_val / mean_val))
    return float(np.clip(raw_conf, 10, 99))
