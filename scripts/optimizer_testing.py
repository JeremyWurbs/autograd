import torch
import pytest

import wichi
from wichi.datasets import MNIST
from wichi.nn import MLP
from wichi.utils import Logger, Metrics


param = {
    'max_epoch': 1,
    'num_train': 50000,
    'batch_size': 16,
    'H': 14,
    'W': 14,
    'hidden_dim': 80,
    'lr': 1e-1,
}
assert param['num_train'] % param['batch_size'] == 0

mnist = MNIST(num_train=param['num_train'], batch_size=param['batch_size'], H=param['H'], W=param['W'])
logger = Logger()
metrics = {'train': Metrics(), 'val': Metrics()}

# Create a torch model
mlp_torch = torch.nn.Sequential(torch.nn.Linear(param['H']*param['W'], param['hidden_dim']),
                                torch.nn.Tanh(),
                                torch.nn.Linear(param['hidden_dim'], 10),
                                torch.nn.Tanh(),)
for p in mlp_torch.parameters():
    p.requires_grad = True

# Create a wichi model and copy the initial weights over from the torch model
mlp_wichi = MLP(dim=[param['H']*param['W'], param['hidden_dim'], 10])
for layer_idx, layer in enumerate(mlp_wichi.layers):
    layer.weight = wichi.Tensor(mlp_torch.get_submodule(f'{2*layer_idx}').weight.detach())
    layer.bias = wichi.Tensor(mlp_torch.get_submodule(f'{2*layer_idx}').bias.detach())
    layer.bias.data = layer.bias.data.reshape((1, layer.bias.numel()))

optimizer_wichi = wichi.optim.SGD(mlp_wichi.parameters(), lr=param['lr'], momentum=0.9, dampening=0.1)
optimizer_torch = torch.optim.SGD(mlp_torch.parameters(), lr=param['lr'], momentum=0.9, dampening=0.1)

# Training loop
batches_per_epoch = param['num_train'] // param['batch_size']
for epoch in range(param['max_epoch']):
    for b in range(batches_per_epoch):
        x, y = mnist.next_train_batch(one_hot=True)

        # Forward pass
        preds_wichi = mlp_wichi(x)
        preds_torch = mlp_torch(x)

        metrics['train'].log_prediction(preds_wichi.data, y)
        logger.log_scalar('train accuracy', value=metrics['train'].accuracy, step=epoch * batches_per_epoch + b)

        # Compute loss
        loss_wichi = ((y - preds_wichi) ** 2).sum() / preds_wichi.numel()
        loss_torch = torch.sum((y - preds_torch) ** 2) / preds_torch.numel()

        # Compute gradients
        optimizer_wichi.zero_grad()
        loss_wichi.backward()

        optimizer_torch.zero_grad()
        loss_torch.backward()

        logger.log_scalar('train loss', value=loss_wichi.data, step=epoch * batches_per_epoch + b)

        # Update parameters
        optimizer_wichi.step()
        optimizer_torch.step()

        # Assert the two models are training identically
        assert torch.allclose(torch.tensor(preds_wichi.data), preds_torch, rtol=1e-3, atol=1e-5)
        assert loss_wichi.data == pytest.approx(loss_torch.data.item(), 1e-5)

        print(f"step: {epoch * batches_per_epoch + b}/{param['max_epoch'] * batches_per_epoch}, loss: {loss_wichi.data}, rolling accuracy: {metrics['train'].accuracy}")

        if (b + 1) % 5 == 0:  # Validation
            x, y = mnist.next_val_batch(one_hot=True, batch_size=60000 - param['num_train'])
            preds_wichi = mlp_wichi(x)
            loss_wichi = ((y - preds_wichi) ** 2).sum() / preds_wichi.numel()
            metrics['val'].log_prediction(preds_wichi.data, y)
            logger.log_scalar('val loss', value=loss_wichi.data, step=epoch * batches_per_epoch + b)
            logger.log_scalar('val accuracy', value=metrics['val'].accuracy, step=epoch * batches_per_epoch + b)

            metrics['train'].reset()
            metrics['val'].reset()

        if (b + 1) % 625 == 0:  # Plotting
            logger.plot(['train loss', 'val loss'], title='MSE')
            logger.plot(['train accuracy', 'val accuracy'], title='Accuracy')

logger.pause_plots(0)
