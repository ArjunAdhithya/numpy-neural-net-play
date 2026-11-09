import numpy as np

from nn import multipleNeuron, stackLayers, sigmoid, tanh, relu, linear
from nn.activations import DERIVATIVES
from nn.gradcheck import gradient_check, max_relative_error
from nn.loss import LOSS_DERIVATIVES, binary_cross_entropy, mean_squared_error, softmax_cross_entropy

TOLERANCE = 1e-6


def build_net(sizes, activations, seed):
    rng = np.random.default_rng(seed)
    layers = [
        multipleNeuron(n=n_out, n_inputs=n_in, activation=act,
                       W=rng.normal(0, 1, (n_out, n_in)), b=rng.normal(0, 0.5, n_out))
        for n_in, n_out, act in zip(sizes, sizes[1:], activations)
    ]
    return stackLayers(layers)


def assert_gradients_match(net, x, target, loss_fn):
    rows = gradient_check(net, x, target, loss_fn)
    worst = max(rows, key=lambda row: row["rel_error"])
    assert worst["rel_error"] < TOLERANCE, (
        f"{worst['param']}: analytic {worst['analytic']:.8f}, "
        f"numeric {worst['numeric']:.8f}, rel error {worst['rel_error']:.2e}"
    )


def test_single_neuron_matches_hand_example():
    layer = multipleNeuron(n=1, n_inputs=3, activation=sigmoid, W=[[0.5, -1.0, 2.0]], b=[0.25])
    net = stackLayers([layer])
    assert_gradients_match(net, [1.0, 2.0, -1.0], [1.0], mean_squared_error)
    np.testing.assert_allclose(layer.neurons[0].dW, [-0.069185, -0.138369, 0.069185], atol=1e-6)


def test_one_layer_sigmoid_mse():
    net = build_net([3, 4], [sigmoid], seed=1)
    assert_gradients_match(net, [1.0, 2.0, -1.0], [1.0, 0.0, 0.0, 1.0], mean_squared_error)


def test_two_layers_sigmoid_mse():
    net = build_net([3, 4, 2], [sigmoid, sigmoid], seed=2)
    assert_gradients_match(net, [0.3, -1.2, 0.8], [1.0, 0.0], mean_squared_error)


def test_tanh_then_sigmoid_bce():
    net = build_net([2, 5, 1], [tanh, sigmoid], seed=3)
    assert_gradients_match(net, [0.5, -0.7], [1.0], binary_cross_entropy)


def test_three_layers_mixed():
    net = build_net([3, 4, 3, 1], [tanh, sigmoid, sigmoid], seed=4)
    assert_gradients_match(net, [1.0, -0.5, 0.25], [0.0], binary_cross_entropy)


def test_relu_hidden_layer():
    net = build_net([3, 6, 1], [relu, sigmoid], seed=5)
    x = [0.9, -0.4, 1.3]
    net.forward(x)
    hidden_z = np.array([neuron.z for neuron in net.layers[0].neurons])
    # a nudge that crosses relu's kink at z = 0 makes the numerical slope meaningless
    assert np.min(np.abs(hidden_z)) > 1e-3, f"pick another seed, a hidden z sits at the kink: {hidden_z}"
    assert_gradients_match(net, x, [1.0], mean_squared_error)


def test_tanh_then_linear_softmax_cross_entropy():
    net = build_net([4, 5, 3], [tanh, linear], seed=7)
    assert_gradients_match(net, [0.4, -1.1, 0.7, 0.2], [0.0, 1.0, 0.0], softmax_cross_entropy)


def test_loss_derivatives_match_numerical():
    target = np.array([1.0, 0.0, 0.0])
    pred = np.array([0.7, 0.2, 0.4])
    eps = 1e-6
    for loss_fn, derivative in LOSS_DERIVATIVES.items():
        numeric = np.array([
            (loss_fn(target, pred + eps * e) - loss_fn(target, pred - eps * e)) / (2 * eps)
            for e in np.eye(3)
        ])
        np.testing.assert_allclose(derivative(target, pred), numeric, rtol=1e-6, err_msg=loss_fn.__name__)


def test_wrong_derivative_is_caught():
    net = build_net([3, 4, 1], [sigmoid, sigmoid], seed=6)
    original = DERIVATIVES[sigmoid]
    DERIVATIVES[sigmoid] = lambda z: sigmoid(z)
    try:
        rows = gradient_check(net, [1.0, 2.0, -1.0], [1.0], mean_squared_error)
    finally:
        DERIVATIVES[sigmoid] = original
    assert max_relative_error(rows) > 1e-2


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
