from wichi.nn import Module, Layer


class MLP(Module):
    def __init__(self, dim, **kwargs):
        assert len(dim) > 1, 'dim must have length >= 2'
        self.layers = [Layer(dim[i], dim[i+1], **kwargs) for i in range(len(dim)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
