import numpy as np


class Tensor(object):
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape, dtype='float64')
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor('{self.label}'\ndata={self.data}, \ngrad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            self.grad += out.grad @ other.data.transpose()
            other.grad += self.data.transpose() @ out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), '__pow__() only supports ints and floats'
        out = Tensor(self.data**other, _children=(self, ), _op=f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1) * out.grad)
        out._backward = _backward

        return out

    def transpose(self):
        out = Tensor(self.data.transpose(), _children=(self, ), _op='.T')

        def _backward():
            self.grad += out.grad.transpose()
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Tensor(np.exp(self.data), _children=(self, ), _op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Tensor(t, _children=(self, ), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.where(self.data > 0., self.data, 0.), _children=(self, ), _op='ReLU')

        def _backward():
            self.grad += np.where(out.data > 0., 1., 0.) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = list()
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones(self.data.shape)
        for node in reversed(topo):
            node._backward()

    @property
    def shape(self):
        return self.data.shape
