# src/utils/seed.py
import random, os
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int):
    """
    Set seeds across numpy, python random, tf, and set environment flags for determinism.
    Call this at the top of any training/inference run.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    # for XGBoost & sklearn pass random_state where possible
