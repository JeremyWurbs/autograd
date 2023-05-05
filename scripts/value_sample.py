from wichi import Value
from wichi.utils import draw_dot


class Neuron(object):
    def __init__(self, w1, w2, b):
        self.w1 = Value(w1, label='w1')
        self.w2 = Value(w2, label='w2')
        self.b = Value(b, label='b')

    def forward(self, x):
        x1, x2 = x
        _1 = self.w1 * x1; _1.label = '_1'
        _2 = self.w2 * x2; _2.label = '_2'
        _3 = _1 + _2; _3.label = '_3'
        dp = _3 + self.b; dp.label = 'dp'
        out = dp.tanh(); out.label = 'tanh'
        return out

    def __call__(self, x):
        return self.forward(x)


neuron = Neuron(w1=-3., w2=1., b=6.7)
x1 = Value(2., label='x1')
x2 = Value(0., label='x2')

y = neuron(x=(x1, x2)); y.label = 'y'
draw_dot(y).render()  # note that all the gradients will be zero


# Backpropagation
y.backward()
draw_dot(y).render()  # note that all the gradients will be filled in
