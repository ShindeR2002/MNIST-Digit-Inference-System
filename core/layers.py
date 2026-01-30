import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # He Initialization: Proven for ReLU stability
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim)
        self.b = np.zeros((output_dim, 1))

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, A_prev) + self.b
        return self.Z

class ReLULayer:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

class SoftmaxLayer:
    def forward(self, Z):
        # Stable Softmax
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

class DropoutLayer:
    def __init__(self, rate=0.1):
        self.rate = rate
    def forward(self, A, training=True):
        if not training: return A
        self.mask = (np.random.rand(*A.shape) > self.rate).astype(float)
        return (A * self.mask) / (1 - self.rate)