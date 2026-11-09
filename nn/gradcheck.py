import numpy as np

from nn.loss import LOSS_DERIVATIVES


def relative_error(a, b):
    return abs(a - b) / max(abs(a) + abs(b), 1e-8)


def numerical_gradients(net, x, target, loss_fn, eps=1e-5):
    def loss():
        return float(loss_fn(target, net.forward(x)))

    grads = []
    for layer in net.layers:
        layer_grads = []
        for neuron in layer.neurons:
            w, b = neuron.w, neuron.b
            dW = np.zeros_like(w)
            for j in range(w.size):
                step = np.zeros_like(w)
                step[j] = eps
                neuron.w = w + step
                plus = loss()
                neuron.w = w - step
                minus = loss()
                dW[j] = (plus - minus) / (2 * eps)
            neuron.w = w

            neuron.b = b + eps
            plus = loss()
            neuron.b = b - eps
            minus = loss()
            neuron.b = b
            layer_grads.append((dW, (plus - minus) / (2 * eps)))
        grads.append(layer_grads)
    return grads


def analytic_gradients(net, x, target, loss_fn):
    pred = net.forward(x)
    net.backward(LOSS_DERIVATIVES[loss_fn](target, pred))
    return [
        [(neuron.dW.copy(), float(neuron.db)) for neuron in layer.neurons]
        for layer in net.layers
    ]


def gradient_check(net, x, target, loss_fn, eps=1e-5):
    x = np.asarray(x, dtype=float)
    target = np.asarray(target, dtype=float)

    numeric = numerical_gradients(net, x, target, loss_fn, eps)
    analytic = analytic_gradients(net, x, target, loss_fn)

    rows = []
    for li, (layer_a, layer_n) in enumerate(zip(analytic, numeric)):
        for ni, ((dW_a, db_a), (dW_n, db_n)) in enumerate(zip(layer_a, layer_n)):
            for j, (a, n) in enumerate(zip(dW_a, dW_n)):
                rows.append({"param": f"L{li} n{ni} w{j}", "analytic": float(a), "numeric": float(n),
                             "rel_error": relative_error(a, n)})
            rows.append({"param": f"L{li} n{ni} b", "analytic": db_a, "numeric": float(db_n),
                         "rel_error": relative_error(db_a, db_n)})
    return rows


def max_relative_error(rows):
    return max(row["rel_error"] for row in rows)


def print_gradient_check(rows):
    print(f"{'param':<12} {'analytic':>12} {'numeric':>12} {'rel error':>11}")
    for row in rows:
        print(f"{row['param']:<12} {row['analytic']:>12.8f} {row['numeric']:>12.8f} {row['rel_error']:>11.2e}")
    print(f"max relative error: {max_relative_error(rows):.2e}")
