import torch
import pytest

from wichi.datasets import MNIST
from wichi.nn import MLP
from wichi.utils import Logger, Metrics
import wichi.utils.conversions as converter


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
mlp_torch = torch.nn.Sequential(torch.nn.Linear(param['H']*param['W'], param['hidden_dim'], dtype=torch.double),
                                torch.nn.Tanh(),
                                torch.nn.Linear(param['hidden_dim'], 10, dtype=torch.double),
                                torch.nn.Tanh(),)
for p in mlp_torch.parameters():
    p.requires_grad = True

# Create a wichi model and copy the initial weights over from the torch model
mlp_wichi = MLP(dim=[param['H']*param['W'], param['hidden_dim'], 10])
for layer_idx, layer in enumerate(mlp_wichi.layers):
    torch_weights = mlp_torch.get_submodule(f'{2*layer_idx}').weight
    torch_biases = mlp_torch.get_submodule(f'{2*layer_idx}').bias
    for neuron_idx, neuron in enumerate(layer.neurons):
        neuron.b.data = torch_biases[neuron_idx].data.item()
        for w_idx, w in enumerate(neuron.w):
            w.data = torch_weights[neuron_idx, w_idx].data.item()

# Training loop
batches_per_epoch = param['num_train'] // param['batch_size']
for epoch in range(param['max_epoch']):
    for b in range(batches_per_epoch):
        x, y = mnist.next_train_batch(one_hot=True)

        # Forward pass
        preds_wichi = [mlp_wichi([v.data.item() for v in x[i, :]]) for i in range(param['batch_size'])]
        preds_torch = mlp_torch(x)

        metrics['train'].log_prediction(converter.value_mat_to_torch_tensor(preds_wichi), y)
        logger.log_scalar('train accuracy', value=metrics['train'].accuracy, step=epoch * batches_per_epoch + b)

        # Compute loss
        loss_wichi = sum([sum([(y - p) ** 2 for p, y in zip(preds_wichi[i], y[i].data.numpy())]) for i in range(param['batch_size'])]) / (10*param['batch_size']) # TODO: this appears wrong
        loss_torch = torch.sum((y - preds_torch) ** 2) / preds_torch.numel()

        # Compute gradients
        mlp_wichi.zero_grad()
        loss_wichi.backward()

        mlp_torch.zero_grad()
        loss_torch.backward()

        logger.log_scalar('train loss', value=loss_wichi.data, step=epoch * batches_per_epoch + b)

        # Update parameters
        for p in mlp_wichi.parameters():
            p.data -= param['lr'] * p.grad

        with torch.no_grad():
            for p in mlp_torch.parameters():
                p_new = p - param['lr'] * p.grad
                p.copy_(p_new)

        # Assert the two models are training identically
        assert torch.allclose(converter.value_mat_to_torch_tensor(preds_wichi), preds_torch)
        assert loss_wichi.data == pytest.approx(loss_torch.data.item(), 1e-5)

        print(f"step: {epoch*batches_per_epoch + b}/{param['max_epoch']*batches_per_epoch}, loss: {loss_wichi.data:.4f}, rolling accuracy: {metrics['train'].accuracy:.4f}")

        if (b+1) % 5 == 0:  # Validation-- use only the torch model for speed
            x, y = mnist.next_val_batch(one_hot=True, batch_size=60000-param['num_train'])
            preds_torch = mlp_torch(x)
            loss_torch = torch.sum((y - preds_torch) ** 2) / preds_torch.numel()
            metrics['val'].log_prediction(preds_torch, y)
            logger.log_scalar('val loss', value=loss_torch.data.item(), step=epoch * batches_per_epoch + b)
            logger.log_scalar('val accuracy', value=metrics['val'].accuracy, step=epoch * batches_per_epoch + b)

            metrics['train'].reset()
            metrics['val'].reset()

        if (b+1) % 25 == 0:  # Plotting
            logger.plot(['train loss', 'val loss'], title='MSE')
            logger.plot(['train accuracy', 'val accuracy'], title='Accuracy')

    import matplotlib.pyplot as plt
    plt.pause(0)
