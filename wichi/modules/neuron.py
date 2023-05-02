import random
from wichi import Value, Module


class Neuron(Module):
    def __init__(self, num_input):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_input)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((w*x for w, x in zip(self.w, x)), start=self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
