import numpy as np

# input = x
# weight = w
# z = np.dot(x,w) + bias
# out = activation(z)

class singleNeuron():
    def __init__(self, w, b, activation):
        self.w = np.asarray(w, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.activation = activation

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        z = np.dot(x, self.w) + self.b
        out = self.activation(z)
        self.last_x = x
        self.z = z
        self.out = out
        return out

    def __repr__(self):
        w = np.array2string(self.w, precision=4, separator=", ")
        b = np.array2string(self.b, precision=4)
        return f"singleNeuron(activation={self.activation.__name__}, w={w}, b={b})"