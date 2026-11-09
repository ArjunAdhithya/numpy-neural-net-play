import numpy as np

from nn.loss import LOSS_DERIVATIVES


def relative_error(a, b):
    return abs(a - b) / max(abs(a) + abs(b), 1e-8)


def numerical_gradients(net, x, target, loss_fn, eps=1e-5):
    def loss():
        return float(loss_fn(target, net.forward(x)))

    grads = []
    for layer in net.layers:
        if hasattr(layer, "neurons"):
            params = [(neuron, "w", "b") for neuron in layer.neurons]
        else:
            params = [(layer, "W", "b")]
        layer_grads = []
        for owner, w_name, b_name in params:
            w = getattr(owner, w_name)
            b = np.asarray(getattr(owner, b_name), dtype=float)
            dW = np.zeros_like(w)
            flat = dW.ravel()
            for j in range(flat.size):
                step = np.zeros(flat.size)
                step[j] = eps
                setattr(owner, w_name, w + step.reshape(w.shape))
                plus = loss()
                setattr(owner, w_name, w - step.reshape(w.shape))
                minus = loss()
                flat[j] = (plus - minus) / (2 * eps)
            setattr(owner, w_name, w)

            if b.ndim == 0:
                setattr(owner, b_name, b + eps)
                plus = loss()
                setattr(owner, b_name, b - eps)
                minus = loss()
                setattr(owner, b_name, b)
                layer_grads.append((dW, (plus - minus) / (2 * eps)))
            else:
                db = np.zeros_like(b)
                for j in range(b.size):
                    step = np.zeros_like(b)
                    step[j] = eps
                    setattr(owner, b_name, b + step)
                    plus = loss()
                    setattr(owner, b_name, b - step)
                    minus = loss()
                    db[j] = (plus - minus) / (2 * eps)
                setattr(owner, b_name, b)
                layer_grads.extend((dW[j], float(db[j])) for j in range(db.size))
        grads.append(layer_grads)
    return grads


def analytic_gradients(net, x, target, loss_fn):
    pred = net.forward(x)
    net.backward(LOSS_DERIVATIVES[loss_fn](target, pred))
    grads = []
    for layer in net.layers:
        if hasattr(layer, "neurons"):
            grads.append([(neuron.dW.copy(), float(neuron.db)) for neuron in layer.neurons])
        else:
            grads.append([(layer.dW[j].copy(), float(layer.db[j])) for j in range(layer.n_filters)])
    return grads


def gradient_check(net, x, target, loss_fn, eps=1e-5):
    x = np.asarray(x, dtype=float)
    target = np.asarray(target, dtype=float)

    numeric = numerical_gradients(net, x, target, loss_fn, eps)
    analytic = analytic_gradients(net, x, target, loss_fn)

    rows = []
    for li, (layer_a, layer_n) in enumerate(zip(analytic, numeric)):
        for ni, ((dW_a, db_a), (dW_n, db_n)) in enumerate(zip(layer_a, layer_n)):
            for j, (a, n) in enumerate(zip(np.ravel(dW_a), np.ravel(dW_n))):
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
