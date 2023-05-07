import numpy as np

from wichi import Tensor
from wichi.nn import Module


class Neuron(Module):
    def __init__(self, num_input):
        self.weight = Tensor(np.random.rand(num_input, 1))
        self.bias = Tensor(np.random.rand(1, 1))

    def __call__(self, x):
        # x has dim (B, num_input)
        act = x @ self.weight + self.bias  # (B, 1)
        out = act.tanh()
        return out

    def parameters(self):
        return self.weight, self.bias
