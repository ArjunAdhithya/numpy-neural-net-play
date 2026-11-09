import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return np.asarray(x, dtype=float)

# softmax mixes all outputs of a layer, so it can't be a per-neuron activation;
# use a linear output layer and softmax_cross_entropy, which applies it inside the loss
def softmax(z):
    z = np.asarray(z, dtype=float)
    shifted = np.exp(z - z.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return (np.asarray(x) > 0).astype(float)

def linear_derivative(x):
    return np.ones_like(np.asarray(x, dtype=float))

DERIVATIVES = {
    sigmoid: sigmoid_derivative,
    tanh: tanh_derivative,
    relu: relu_derivative,
    linear: linear_derivative,
}

 