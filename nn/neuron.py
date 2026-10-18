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


class multipleNeuron():

    # n = number of neurons in a layer
    # n_inputs = how long each input vector is
    # sample, x = [1.0, 2.0, -1.0] has length 3, so n_inputs = 3. 
    # Two neurons looking at that same x means n = 2. 
    # The layer then returns a vector of length 2.
    
    def __init__(self, n, n_inputs, activation, W=None, b=None):
        self.n = n
        self.n_inputs = n_inputs
        self.activation = activation

        if W is None:
            W = np.random.randn(n, n_inputs)
        if b is None:
            b = np.zeros(n)

        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)

        self.neurons = [
            singleNeuron(w=W[i], b=b[i], activation=activation)
            for i in range(n)
        ]

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        outs = [neuron.forward(x) for neuron in self.neurons]
        return np.asarray(outs)