import numpy as np

from wichi import Tensor
from wichi.nn import Module


class Layer(Module):
    def __init__(self, num_input, num_output, name='Layer', bias=True, activation_fn='tanh'):
        self.name = name
        self.weight = Tensor(np.random.rand(num_output, num_input) / np.sqrt(num_output * num_input), label=f'{name}.weight')
        self.bias = Tensor(np.random.rand(1, num_output) / np.sqrt(num_output), label=f'{name}.bias') if bias else None
        self.activation_fn = activation_fn
        assert activation_fn in ['tanh', 'relu'], f'Unknown activation function, {activation_fn}. Must be one of {{tanh, relu}}.'

    def __call__(self, x):
        # x has dim (B, num_input)
        act = x @ self.weight.T  # (B, num_output)
        act.label = f'{self.name}_dot_product'
        if self.bias is not None:
            if len(x.shape) == 1:
                B = 1
            else:
                B, num_input = x.shape
            act += Tensor(np.ones((B, 1), dtype=self.bias.dtype), label='Ones') @ self.bias  # Broadcast the bias array with a Tensor so it will backprop
        if self.activation_fn == 'tanh':
            out = act.tanh()
        elif self.activation_fn == 'relu':
            out = act.relu()
        out.label = f'{self.name}_output'
        return out

    def parameters(self):
        return self.weight, self.bias
