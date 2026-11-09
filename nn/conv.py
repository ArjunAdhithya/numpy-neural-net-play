import numpy as np

from nn.activations import DERIVATIVES, relu


class convLayer:
    # one filter is a small pattern. stride slides that same pattern across the photo.
    # in_channels is 1 for a gray photo, or the number of maps from the conv in front.
    def __init__(self, n_filters, side, kernel=3, stride=2, in_channels=1, activation=relu, W=None, b=None):
        self.n_filters = n_filters
        self.side = side
        self.kernel = kernel
        self.stride = stride
        self.in_channels = in_channels
        self.activation = activation
        self.out_side = (side - kernel) // stride + 1
        self.n_inputs = in_channels * side * side
        self.n = n_filters * self.out_side * self.out_side

        if W is None:
            fan_in = in_channels * kernel * kernel
            W = np.random.randn(n_filters, in_channels, kernel, kernel) / np.sqrt(fan_in)
        W = np.asarray(W, dtype=float)
        if W.ndim == 3:
            W = W[:, None, :, :]
        if b is None:
            b = np.zeros(n_filters)
        self.W = W
        self.b = np.asarray(b, dtype=float)

    def _places(self):
        for i in range(self.out_side):
            for j in range(self.out_side):
                yield i, j, i * self.stride, j * self.stride

    def forward(self, x):
        maps = np.asarray(x, dtype=float).reshape(self.in_channels, self.side, self.side)
        self.last_x = maps
        z = np.zeros((self.n_filters, self.out_side, self.out_side))
        k = self.kernel
        for i, j, r, c in self._places():
            patch = maps[:, r:r + k, c:c + k]
            z[:, i, j] = np.sum(self.W * patch, axis=(1, 2, 3)) + self.b
        self.z = z
        self.out = self.activation(z)
        return self.out.ravel()

    def backward(self, dL_dout):
        dL_dout = np.asarray(dL_dout, dtype=float).reshape(self.out.shape)
        dL_dz = dL_dout * DERIVATIVES[self.activation](self.z)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.last_x)
        k = self.kernel
        for i, j, r, c in self._places():
            patch = self.last_x[:, r:r + k, c:c + k]
            grad = dL_dz[:, i, j]
            weighted = grad[:, None, None, None]
            dW += weighted * patch
            db += grad
            dX[:, r:r + k, c:c + k] += np.sum(weighted * self.W, axis=0)
        self.dW = dW
        self.db = db
        return dX.ravel()
