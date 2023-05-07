import numpy as np

from wichi import Tensor
from wichi.nn import Module, Layer


class MLP(Module):
    def __init__(self, dim, name='MLP', activation_fn='tanh'):
        self.name = name
        self.layers = [Layer(dim[i], dim[i+1], name=f'Layer{i}', activation_fn=activation_fn) for i in range(len(dim)-1)]

    def __call__(self, x):
        # x has dim (B, num_input)
        x = x if isinstance(x, Tensor) else Tensor(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
