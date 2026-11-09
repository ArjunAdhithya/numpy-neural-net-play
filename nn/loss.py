import numpy as np

from nn.activations import softmax

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

def softmax_cross_entropy(target, logits):
    target = np.asarray(target, dtype=float)
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max()
    log_probs = shifted - np.log(np.sum(np.exp(shifted)))
    return 0.0 - np.sum(target * log_probs)

# dL/dpred, one value per output; both losses average over the outputs, hence / pred.size

def mean_squared_error_derivative(target, pred):
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return 2 * (pred - target) / pred.size

def binary_cross_entropy_derivative(target, pred):
    target = np.asarray(target, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-12, 1 - 1e-12)
    return (pred - target) / (pred * (1 - pred)) / pred.size

# target must be one-hot (sums to 1) for this to be the exact gradient
def softmax_cross_entropy_derivative(target, logits):
    return softmax(logits) - np.asarray(target, dtype=float)

LOSS_DERIVATIVES = {
    mean_squared_error: mean_squared_error_derivative,
    binary_cross_entropy: binary_cross_entropy_derivative,
    softmax_cross_entropy: softmax_cross_entropy_derivative,
}
