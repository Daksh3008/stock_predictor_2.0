# src/models/ensemble.py
import numpy as np

def weights_from_scores(scores: dict, priorities: dict = None, temp: float = 0.5):
    """
    Convert RMSE-like scores (lower better) to normalized weights.
    priorities optional to bias certain models.
    """
    pri = priorities or {k:1 for k in scores}
    inv = {k: (1.0/(scores[k] + 1e-9)) * pri.get(k,1) for k in scores}
    arr = np.array(list(inv.values()), dtype=float)
    ex = np.exp((np.log(arr + 1e-12))/temp)
    w = ex / ex.sum()
    return dict(zip(list(inv.keys()), w))

def weighted_average(preds: dict, weights: dict):
    # preds are scalars or arrays (same shape)
    keys = list(preds.keys())
    first = preds[keys[0]]
    if hasattr(first, "__len__"):
        res = sum(preds[k] * weights.get(k, 0.0) for k in keys)
        return res
    else:
        return sum(preds[k] * weights.get(k, 0.0) for k in keys)
