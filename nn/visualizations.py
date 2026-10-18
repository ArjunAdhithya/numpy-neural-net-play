import graphviz
import matplotlib.pyplot as plt
import numpy as np

def plot_activations(*activations, z_min=-8, z_max=8, n=200):
    if not activations:
        raise TypeError("pass the activations to plot, for example plot_activations(sigmoid, relu)")

    z = np.linspace(z_min, z_max, n)
    fig, axes = plt.subplots(1, len(activations), figsize=(4 * len(activations), 3.5), squeeze=False)
    for ax, fn in zip(axes[0], activations):
        ax.plot(z, fn(z))
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.axvline(0, color="0.75", linewidth=0.8)
        ax.set_title(fn.__name__)
        ax.set_xlabel("z")
        ax.set_ylabel(f"{fn.__name__}(z)")
    fig.tight_layout()
    plt.close(fig)
    return fig


def visualize_neuron(neuron):
    if not hasattr(neuron, "last_x"):
        raise RuntimeError("call forward before visualize_neuron")

    x = np.asarray(neuron.last_x, dtype=float)
    w = np.asarray(neuron.w, dtype=float)
    if x.ndim != 1 or w.ndim != 1 or x.shape != w.shape:
        raise ValueError("visualize_neuron draws one sample, with one weight per input")

    b = float(np.asarray(neuron.b))
    graph = graphviz.Digraph(graph_attr={"rankdir": "LR", "nodesep": "0.25", "ranksep": "0.6"})

    def box(target, node_id, label):
        target.node(node_id, label, shape="box")

    def value_label(name, value):
        return f"{name}\\n{float(value):.4f}"

    def stack(target, node_ids):
        for left, right in zip(node_ids, node_ids[1:]):
            target.edge(left, right, style="invis")

    input_ids = []
    mul_ids = []
    product_ids = []
    with graph.subgraph(name="inputs") as column:
        column.attr(rank="same")
        for i, (xi, wi) in enumerate(zip(x, w)):
            box(column, f"x{i}", value_label(f"x{i}", xi))
            box(column, f"w{i}", value_label(f"w{i}", wi))
            input_ids.extend([f"x{i}", f"w{i}"])
        box(column, "b", value_label("b", b))
        input_ids.append("b")
        stack(column, input_ids)

    with graph.subgraph(name="muls") as column:
        column.attr(rank="same")
        for i in range(len(x)):
            box(column, f"mul{i}", "*")
            mul_ids.append(f"mul{i}")
        stack(column, mul_ids)

    with graph.subgraph(name="products") as column:
        column.attr(rank="same")
        for i in range(len(x)):
            box(column, f"p{i}", value_label(f"x{i}*w{i}", x[i] * w[i]))
            product_ids.append(f"p{i}")
        stack(column, product_ids)

    for i in range(len(x)):
        graph.edge(f"x{i}", f"mul{i}")
        graph.edge(f"w{i}", f"mul{i}")
        graph.edge(f"mul{i}", f"p{i}")

    box(graph, "sum", "+")
    for i in range(len(x)):
        graph.edge(f"p{i}", "sum")
    graph.edge("b", "sum")
    box(graph, "z", value_label("z", neuron.z))
    graph.edge("sum", "z")
    box(graph, "act", neuron.activation.__name__)
    box(graph, "out", value_label("out", neuron.out))
    graph.edge("z", "act")
    graph.edge("act", "out")
    return graph

def _column(graph, name, node_ids):
    with graph.subgraph(name=name) as column:
        column.attr(rank="same")
        for node_id in node_ids:
            column.node(node_id)
        for left, right in zip(node_ids, node_ids[1:]):
            column.edge(left, right, style="invis")


def visualize_layer(layer):
    if not layer.neurons or not hasattr(layer.neurons[0], "last_x"):
        raise RuntimeError("call forward before visualize_layer")

    x = layer.neurons[0].last_x
    graph = graphviz.Digraph(graph_attr={"rankdir": "LR"})

    for j, value in enumerate(x):
        graph.node(f"x{j}", f"x{j}\\n{value:.4f}", shape="box")

    for i, neuron in enumerate(layer.neurons):
        graph.node(f"h{i}", f"neuron {i}\\n{neuron.activation.__name__}", shape="circle")
        graph.node(f"y{i}", f"out{i}\\n{float(neuron.out):.4f}", shape="box")
        for j in range(layer.n_inputs):
            graph.edge(f"x{j}", f"h{i}")
        graph.edge(f"h{i}", f"y{i}")

    return graph


def visualize_net(net):
    layers = net.layers
    if not layers or not hasattr(layers[0].neurons[0], "last_x"):
        raise RuntimeError("call forward before visualize_net")

    graph = graphviz.Digraph(graph_attr={"rankdir": "LR", "nodesep": "0.28", "ranksep": "0.75"})
    x = np.asarray(layers[0].neurons[0].last_x, dtype=float)

    input_ids = []
    for j, value in enumerate(x):
        node_id = f"x{j}"
        graph.node(node_id, f"x{j}\\n{float(value):.4f}", shape="box")
        input_ids.append(node_id)
    _column(graph, "inputs", input_ids)

    previous_ids = input_ids
    for layer_i, layer in enumerate(layers):
        neuron_ids = []
        for i, neuron in enumerate(layer.neurons):
            node_id = f"L{layer_i}h{i}"
            graph.node(
                node_id,
                f"L{layer_i} n{i}\\n{neuron.activation.__name__}\\n{float(neuron.out):.4f}",
                shape="circle",
            )
            neuron_ids.append(node_id)
        _column(graph, f"layer{layer_i}", neuron_ids)
        for src in previous_ids:
            for dst in neuron_ids:
                graph.edge(src, dst)
        previous_ids = neuron_ids

    out_ids = []
    for i, neuron in enumerate(layers[-1].neurons):
        node_id = f"y{i}"
        graph.node(node_id, f"out{i}\\n{float(neuron.out):.4f}", shape="box")
        out_ids.append(node_id)
    _column(graph, "outputs", out_ids)
    for src, dst in zip(previous_ids, out_ids):
        graph.edge(src, dst)
    return graph