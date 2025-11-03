# src/models/ensemble.py
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred)**2).mean())

def weights_from_scores(scores: dict, priorities: dict = None, temp: float = 0.5):
    """
    scores: model->rmse (lower better)
    priorities: model->priority (higher = more important). Optional.
    Returns normalized weights.
    """
    pri = priorities or {k:1 for k in scores}
    inv = {k: (1.0/(scores[k] + 1e-9)) * pri.get(k,1) for k in scores}
    arr = np.array(list(inv.values()), dtype=float)
    # softmax-like with temperature
    ex = np.exp((np.log(arr + 1e-12))/temp)
    w = ex / ex.sum()
    return dict(zip(list(inv.keys()), w))

def weighted_average(preds: dict, weights: dict):
    """
    preds: model->scalar prediction (float) or array (same length)
    weights: model->weight
    If preds are arrays, they must be same shape and we return array.
    """
    models = list(preds.keys())
    first = preds[models[0]]
    if hasattr(first, "__len__"):
        # vector combine
        res = sum(preds[m]*weights.get(m,0) for m in models)
        return res
    else:
        return sum(preds[m]*weights.get(m,0) for m in models)
