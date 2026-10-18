import numpy as np

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