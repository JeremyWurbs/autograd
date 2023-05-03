from wichi import Neuron, Module


class Layer(Module):
    def __init__(self, num_input, num_output, **kwargs):
        self.neurons = [Neuron(num_input, **kwargs) for _ in range(num_output)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
