import numpy as np


class Tensor(object):
    def __init__(self, data, _children=(), _op='', label='', dtype='float32'):
        self.data = np.array(data, dtype=dtype)
        self.grad = None
        self._grad_fn = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

        self.zero_grad()
        if self.is_scalar:
            self.data = self.data.reshape((1, 1))
            self.grad = self.grad.reshape((1, 1))

    def __repr__(self):
        return f"Tensor('{self.label}'\ndata={self.data}, \ngrad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _grad_fn():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._grad_fn = _grad_fn

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _grad_fn():
            if self.is_scalar:
                self.grad += np.sum((other.data * out.grad))
            else:
                self.grad += other.data * out.grad
            if other.is_scalar:
                other.grad += np.sum((self.data * out.grad))
            else:
                other.grad += self.data * out.grad
        out._grad_fn = _grad_fn

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _grad_fn():
            self.grad += out.grad @ other.data.transpose()
            other.grad += self.data.transpose() @ out.grad
        out._grad_fn = _grad_fn

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

        def _grad_fn():
            self.grad += (other * self.data**(other-1) * out.grad)
        out._grad_fn = _grad_fn

        return out

    def transpose(self):
        out = Tensor(self.data.transpose(), _children=(self, ), _op='.T')

        def _grad_fn():
            self.grad += out.grad.transpose()
        out._grad_fn = _grad_fn

        return out

    @property
    def T(self):
        return self.transpose()

    def exp(self):
        x = self.data
        out = Tensor(np.exp(self.data), _children=(self, ), _op='exp')

        def _grad_fn():
            self.grad += out.data * out.grad
        out._grad_fn = _grad_fn

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Tensor(t, _children=(self, ), _op='tanh')

        def _grad_fn():
            self.grad += (1 - t**2) * out.grad
        out._grad_fn = _grad_fn

        return out

    def relu(self):
        out = Tensor(np.where(self.data > 0., self.data, 0.), _children=(self, ), _op='ReLU')

        def _grad_fn():
            self.grad += np.where(out.data > 0., 1., 0.) * out.grad
        out._grad_fn = _grad_fn

        return out

    def sum(self, dim=None):
        s = np.sum(self.data, axis=dim)
        out = Tensor(s, _children=(self, ), _op=f".sum(dim={dim if dim is not None else 'all'})")

        def _grad_fn():
            self.grad += np.ones(self.data.shape) * out.grad
        out._grad_fn = _grad_fn

        return out

    def numel(self):
        return self.data.size

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape, dtype='float32')

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
            node._grad_fn()

    @property
    def shape(self):
        if self.numel() == 1:
            return (1, 1)
        else:
            return self.data.shape

    @property
    def grad_shape(self):
        if self.grad.size == 1:
            return (1, 1)
        else:
            return self.grad.shape

    @property
    def is_scalar(self):
        if self.numel() == 1:
            return True
        else:
            return False

    @property
    def dtype(self):
        return self.data.dtype
