from wichi.nn import Neuron, Layer, MLP
from wichi.utils import draw_dot


neuron = Neuron(num_input=2)

x = [2., 3.]
y = neuron(x)
print(f'y: {y}')


layer = Layer(num_input=5, num_output=3)
y = layer([-3., 2., 4., -1., 2.5])
print(f'y: {y}')


mlp = MLP(dim=[5, 10, 30, 10, 1])
y = mlp([-3., 1., 5., -6., 3.])
print(f'y: {y}')

draw_dot(y).render()
y.backward()
draw_dot(y).render()
