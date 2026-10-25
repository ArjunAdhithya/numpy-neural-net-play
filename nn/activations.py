import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return (np.asarray(x) > 0).astype(float)

DERIVATIVES = {
    sigmoid: sigmoid_derivative,
    tanh: tanh_derivative,
    relu: relu_derivative,
}

 