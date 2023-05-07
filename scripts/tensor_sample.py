from wichi import Tensor
from wichi.utils import draw_dot


U = Tensor([[1.0, -1.5, 3.0],
            [2.5, -3.0, 2.0],
            [-1.0, 2.5, -1.5],
            [1.5, -0.5, 2.0]],
           label='U')

V = Tensor([[1.0, -1.0],
            [2.5, -3.0],
            [-0.5, 1.5]],
           label='V')

T = U * U
W = U @ V; W.label = 'W'

Y = W @ Tensor([[0.5], [-1.5]])

Z = Y.relu()

draw_dot(Z).render()


