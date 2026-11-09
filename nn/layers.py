import numpy as np

from nn.backprop import multi_neuron_backprop

class stackLayers():
    def __init__(self, layers):
        for i in range(len(layers)-1):
            if layers[i].n != layers[i+1].n_inputs:
                raise ValueError(
                    f"layer {i} outputs {layers[i].n}, "
                    f"layer {i + 1} expects {layers[i + 1].n_inputs}"
                )
                
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dL_dout):
        for layer in reversed(self.layers):
            dL_dout = multi_neuron_backprop(layer, dL_dout)
        return dL_dout

    def __repr__(self):
        parts = [str(self.layers[0].n_inputs)]
        parts += [f"{layer.n} {layer.activation.__name__}" for layer in self.layers]
        return f"stackLayers({' -> '.join(parts)})"
