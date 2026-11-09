import json
from pathlib import Path

import numpy as np

from nn.activations import DERIVATIVES
from nn.conv import convLayer
from nn.layers import stackLayers
from nn.neuron import multipleNeuron

ACTIVATIONS_BY_NAME = {fn.__name__: fn for fn in DERIVATIVES}


def save_checkpoint(net, path, **meta):
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    layers = []
    for i, layer in enumerate(net.layers):
        if hasattr(layer, "neurons"):
            arrays[f"L{i}_W"] = np.stack([neuron.w for neuron in layer.neurons])
            arrays[f"L{i}_b"] = np.array([float(neuron.b) for neuron in layer.neurons])
            layers.append({
                "kind": "dense",
                "n": layer.n,
                "n_inputs": layer.n_inputs,
                "activation": layer.activation.__name__,
            })
        else:
            arrays[f"L{i}_W"] = layer.W
            arrays[f"L{i}_b"] = layer.b
            layers.append({
                "kind": "conv",
                "n_filters": layer.n_filters,
                "side": layer.side,
                "kernel": layer.kernel,
                "stride": layer.stride,
                "in_channels": layer.in_channels,
                "activation": layer.activation.__name__,
            })

    meta = {"layers": layers, **meta}
    arrays["meta"] = np.array(json.dumps(meta, default=lambda value: value.tolist()))
    np.savez(path, **arrays)
    return path


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        layers = []
        for i, spec in enumerate(meta.pop("layers")):
            activation = ACTIVATIONS_BY_NAME[spec["activation"]]
            if spec.get("kind", "dense") == "conv":
                layers.append(convLayer(
                    n_filters=spec["n_filters"],
                    side=spec["side"],
                    kernel=spec["kernel"],
                    stride=spec["stride"],
                    in_channels=spec["in_channels"],
                    activation=activation,
                    W=data[f"L{i}_W"],
                    b=data[f"L{i}_b"],
                ))
            else:
                layers.append(multipleNeuron(
                    n=spec["n"],
                    n_inputs=spec["n_inputs"],
                    activation=activation,
                    W=data[f"L{i}_W"],
                    b=data[f"L{i}_b"],
                ))
    return stackLayers(layers), meta
