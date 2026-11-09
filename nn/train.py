import numpy as np

from nn.loss import LOSS_DERIVATIVES


def sgd_step(net, lr):
    # rebind instead of -= so layers built from one shared W array stay independent
    for layer in net.layers:
        if hasattr(layer, "neurons"):
            for neuron in layer.neurons:
                neuron.w = neuron.w - lr * neuron.dW
                neuron.b = neuron.b - lr * neuron.db
        else:
            layer.W = layer.W - lr * layer.dW
            layer.b = layer.b - lr * layer.db


def train_step(net, x, target, loss_fn, lr):
    pred = net.forward(x)
    loss = float(loss_fn(target, pred))
    net.backward(LOSS_DERIVATIVES[loss_fn](target, pred))
    sgd_step(net, lr)
    return loss


def predict(net, X):
    return np.array([net.forward(x) for x in np.asarray(X, dtype=float)])


def _as_targets(Y):
    Y = np.asarray(Y, dtype=float)
    return Y[:, None] if Y.ndim == 1 else Y


def evaluate(net, X, Y, loss_fn):
    preds = predict(net, X)
    Y = _as_targets(Y)
    loss = float(np.mean([loss_fn(t, p) for t, p in zip(Y, preds)]))
    if preds.shape[1] > 1:
        accuracy = float(np.mean(preds.argmax(axis=1) == Y.argmax(axis=1)))
    else:
        accuracy = float(np.mean((preds > 0.5) == (Y > 0.5)))
    return loss, accuracy


def train(net, X, Y, loss_fn, lr=0.1, epochs=100, X_val=None, Y_val=None,
          log_every=10, seed=0, history=None):
    X = np.asarray(X, dtype=float)
    Y = _as_targets(Y)
    rng = np.random.default_rng(seed)
    if history is None:
        history = {"epoch": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    start = history["epoch"][-1] if history["epoch"] else 0

    for epoch in range(start + 1, start + epochs + 1):
        for i in rng.permutation(len(X)):
            train_step(net, X[i], Y[i], loss_fn, lr)

        loss, accuracy = evaluate(net, X, Y, loss_fn)
        history["epoch"].append(epoch)
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)
        line = f"epoch {epoch:>4}  loss {loss:.4f}  acc {accuracy:.3f}"
        if X_val is not None:
            val_loss, val_accuracy = evaluate(net, X_val, Y_val, loss_fn)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            line += f"  val_loss {val_loss:.4f}  val_acc {val_accuracy:.3f}"
        if log_every and (epoch == start + 1 or epoch % log_every == 0):
            print(line)
    return history
