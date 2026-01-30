import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.t = 0 # Timestep for ADAM
        # Initialization of moment vectors
        self.m_w = [np.zeros_like(l.W) for l in layers if hasattr(l, 'W')]
        self.v_w = [np.zeros_like(l.W) for l in layers if hasattr(l, 'W')]
        self.m_b = [np.zeros_like(l.b) for l in layers if hasattr(l, 'b')]
        self.v_b = [np.zeros_like(l.b) for l in layers if hasattr(l, 'b')]

    def adam_step(self, layer_idx, dW, db, lr=0.001):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Adaptive weight updates
        self.m_w[layer_idx] = beta1 * self.m_w[layer_idx] + (1 - beta1) * dW
        self.v_w[layer_idx] = beta2 * self.v_w[layer_idx] + (1 - beta2) * (dW**2)
        mw_h = self.m_w[layer_idx] / (1 - beta1**self.t)
        vw_h = self.v_w[layer_idx] / (1 - beta2**self.t)
        
        return lr * mw_h / (np.sqrt(vw_h) + eps)