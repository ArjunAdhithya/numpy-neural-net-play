import numpy as np
import math

def mean_squared_error(target, pred):
    target = np.asarray(target)
    pred = np.asarray(pred)
    mse = np.mean((target - pred) ** 2)
    return mse

def binary_cross_entropy(target, pred):
    target = np.asarray(target, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-12, 1 - 1e-12)
    loss = -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    return np.mean(loss)