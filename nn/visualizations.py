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


def plot_history(history):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 3.5))
    epochs = history["epoch"]
    loss_ax.plot(epochs, history["loss"], label="train")
    acc_ax.plot(epochs, history["accuracy"], label="train")
    if history.get("val_loss"):
        loss_ax.plot(epochs, history["val_loss"], label="val")
        acc_ax.plot(epochs, history["val_accuracy"], label="val")
    loss_ax.set_title("loss")
    acc_ax.set_title("accuracy")
    acc_ax.set_ylim(0, 1.05)
    for ax in (loss_ax, acc_ax):
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_decision_boundary(net, X, y, resolution=80, padding=0.5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary draws 2-feature inputs only")

    xs = np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, resolution)
    ys = np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    probs = np.array([net.forward(p)[0] for p in points]).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    filled = ax.contourf(grid_x, grid_y, probs, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.6)
    ax.contour(grid_x, grid_y, probs, levels=[0.5], colors="k", linewidths=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=25)
    fig.colorbar(filled, ax=ax, label="network output")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_digits(X, labels, preds=None, n_cols=10, side=28):
    X = np.asarray(X)
    n_rows = int(np.ceil(len(X) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.1 * n_cols, 1.3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, ax in enumerate(axes.ravel()[:len(X)]):
        ax.imshow(X[i].reshape(side, side), cmap="gray")
        if preds is None:
            ax.set_title(str(labels[i]), fontsize=10)
        elif preds[i] == labels[i]:
            ax.set_title(str(preds[i]), color="green", fontsize=10)
        else:
            ax.set_title(f"{preds[i]} (true {labels[i]})", color="red", fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_confusion_matrix(y_true, y_pred, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    for t in range(n_classes):
        for p in range(n_classes):
            if matrix[t, p]:
                color = "white" if matrix[t, p] > matrix.max() / 2 else "black"
                ax.text(p, t, matrix[t, p], ha="center", va="center", color=color)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    fig.tight_layout()
    plt.close(fig)
    return fig