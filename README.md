# Wichi

We follow Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd) 
package (following PyTorch conventions) to create Wichi— a simple
reference implementation of an autograd engine.

# Installation

If you wish to draw network graphs, first install graphviz.

# Installation

In order to create png images of the resulting network graphs, install graphviz
(used by `wichi.utils.graphing`):

```commandline
apt install graphviz
```

Then install the package, either through a wheel or just installing the 
dependendies. I.e.

```commandline
git clone https://github.com/JeremyWurbs/autograd.git && cd autograd
```

Followed by one of the following:

```commandline
pip install -r requirements.txt
```

OR

```commandline
python setup.py bdist_wheel
pip install dist/autograd-1.0.0-py3-none-any.whl
```

# Basic Usage

Wichi is built around the Value class, which maintains a piece of `data` as well as 
a `grad` (gradient) to be computed locally on any backward pass. All common operations
(summation, multiplication, exponentiation, etc.) are supported, with a few additional
advanced ops as well (ReLU, Tanh), which may be used to create standard perceptions.

You may create a network, run a forward and backward pass, and plot the resulting 
network graph with the following:

```python
from wichi import Value, draw_dot

x1 = Value(2.0, label='x1')
x2 = Value(-1.0, label='x2')
w1 = Value(0.5, label='w1')
w2 = Value(0.75, label='w2')
y = Value(0, label='y')

y_hat = w1*x1 + w2*x2; y_hat.label = 'y_hat'
loss = (y - y_hat) ** 2; loss.label = 'loss'

loss.backward()  # computes gradients
draw_dot(loss).render()
```

Which will yield the following diagram:

![Rendered DiGraph](./resources/Digraph.gv.svg)

In addition, there is also the Modules module, which provides Neuron, Layer and
MLP classes to help create simple neural networks.

```python
from wichi import MLP, draw_dot

mlp = MLP(dim=[3, 4, 2])

x = [2.5, 0.5, -1.0]
y = [-1.0, 1.0]

y_hat = MLP(x)
loss = sum([(y - y_hat)**2 for y, y_hat in zip(y, y_hat)]) / 2

loss.backward()  # computes gradients
draw_dot(loss).render()
```

which will yield the following diagram:

![Rendered MLP DiGraph](./resources/MLP_Digraph.gv.svg)

# Training

In order to train a network using Wichi, simply update any parameters according to
(some proportion of) their gradient. In pseudo-code:

```python
from wichi import MLP, DataModule  # E.g. MNIST

param = {'num_input': 784, 
         'hidden_dim_1': 20,
         'hidden_dim_2': 20,
         'num_output': 10,
         'max_epochs': 3,
         'error_func': MeanSquaredError(),
         'lr': 0.01}

mlp = MLP(dim=[num_input, hidden_dim_1, hidden_dim_2, num_output])
data = DataModule() 

for epoch in max_epochs:
    for x, y in data.next_training_batch():
        y_hat = [mlp(x) for x in x]
        loss = (error_func(y, y_hat) for y, y_hat in zip(y, y_hat))
        
        loss.zero_grad()
        loss.backward()
        
        for p in mlp.parameters():
            p -= lr * p.grad
        
        print(f'loss: {loss}')
```

For an explicit training sample, refer to the 
[mnist_training.py](./scripts/mnist_training.py)
sample, which initializes a Wichi network with the same weights as a torch model,
and then trains both side by side, showing that the Wichi autograd results exactly
match those from PyTorch.

# Testing

You can run the given unit tests, printing the resulting output, with

```commandline
pytest -s ./tests
```

which should output something similar to the following,

``` 
======================================= test session starts =========================================
platform linux -- Python 3.8.0, pytest-7.3.1, pluggy-1.0.0
rootdir: /home/jeremy/projects/autograd
collected 2 items                                                                                          

tests/test_value.py ..                                                                         [100%]
======================================== 2 passed in 1.37s ==========================================
```

