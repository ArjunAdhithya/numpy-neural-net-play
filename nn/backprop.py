import numpy as np

from nn.activations import DERIVATIVES

# dL/dz = dL/dout * activation'(z)
# dL/dw = dL/dz * x
# dL/db = dL/dz
# dL/dx = dL/dz * w

def single_neuron_backprop(neuron, dL_dout):
    if not hasattr(neuron, "z"):
        raise RuntimeError("call forward before single_neuron_backprop")

    derivative = DERIVATIVES[neuron.activation]
    dL_dz = dL_dout * derivative(neuron.z)
    neuron.dL_dz = dL_dz
    neuron.dW = dL_dz * neuron.last_x
    neuron.db = dL_dz
    return dL_dz * neuron.w


def multi_neuron_backprop(layer, dL_dout):
    dL_dout = np.asarray(dL_dout, dtype=float)
    if dL_dout.shape != (layer.n,):
        raise ValueError(f"expected one gradient per neuron, shape ({layer.n},), got {dL_dout.shape}")

    dL_dx = np.zeros(layer.n_inputs)
    for neuron, grad in zip(layer.neurons, dL_dout):
        dL_dx += single_neuron_backprop(neuron, grad)
    return dL_dx
