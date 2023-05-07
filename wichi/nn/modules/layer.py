import numpy as np

from wichi import Tensor
from wichi.nn import Module


class Layer(Module):
    def __init__(self, num_input, num_output, name='Layer', bias=True):
        self.name = name
        self.weight = Tensor(np.random.rand(num_output, num_input), label=f'{name}_weight')
        self.bias = Tensor(np.random.rand(1, num_output), label=f'{name}_bias') if bias else None

    def __call__(self, x):
        # x has dim (B, num_input)
        act = x @ self.weight.T  # (B, num_output)
        if self.bias is not None:
            B, num_input = x.shape
            act += Tensor(np.ones((B, 1), dtype=self.bias.dtype)) @ self.bias  # Broadcast the bias array with a Tensor so it will backprop
        out = act.tanh()
        return out

    def parameters(self):
        return self.weight, self.bias
