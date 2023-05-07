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
    'hidden_dims': [80],
    'lr': 1e-1,
    'optimizer': [{'momentum': 0, 'dampening': 0},
                  {'momentum': 0.5, 'dampening': 0},
                  {'momentum': 0.9, 'dampening': 0},
                  {'momentum': 0.9, 'dampening': 0.5}]
}
assert param['num_train'] % param['batch_size'] == 0

mnist = MNIST(num_train=param['num_train'], batch_size=param['batch_size'], H=param['H'], W=param['W'])
logger = {'train loss': Logger(), 'train accuracy': Logger(), 'val loss': Logger(), 'val accuracy': Logger()}
metrics = list()
for i in range(len(param['optimizer'])):
    metrics.append({'train': Metrics(), 'val': Metrics()})

# Create a number of identical wichi models, which will each get their own optimizer
models = list()
for i in range(len(param['optimizer'])):
    models.append(MLP(dim=[param['H']*param['W'], *param['hidden_dims'], 10], activation_fn='relu'))

for model in models[1:]:
    for layer_idx, layer in enumerate(model.layers):
        layer.weight.data = models[0].layers[layer_idx].weight.data.copy()
        layer.bias.data = models[0].layers[layer_idx].bias.data.copy()

optimizers = [wichi.optim.SGD(model.parameters(), lr=param['lr'], **opt_param) for model, opt_param in zip(models, param['optimizer'])]

# Training loop
batches_per_epoch = param['num_train'] // param['batch_size']
for epoch in range(param['max_epoch']):
    for b in range(batches_per_epoch):
        x, y = mnist.next_train_batch(one_hot=True)

        # Forward pass
        preds = [model(x) for model in models]

        [metrics['train'].log_prediction(preds.data, y) for metrics, preds in zip(metrics, preds)]
        for idx in range(len(models)):
            logger['train accuracy'].log_scalar(f"{param['optimizer'][idx]}", value=metrics[idx]['train'].accuracy, step=epoch * batches_per_epoch + b)

        # Compute loss
        losses = [((y.data - preds_) ** 2).sum() / preds_.numel() for preds_ in preds]

        # Compute gradients
        [optimizer.zero_grad() for optimizer in optimizers]
        [loss.backward() for loss in losses]

        for idx in range(len(models)):
            logger['train loss'].log_scalar(f"{param['optimizer'][idx]}", value=losses[idx].data, step=epoch * batches_per_epoch + b)

        # Update parameters
        [optimizer.step() for optimizer in optimizers]

        print(f"step: {epoch * batches_per_epoch + b}/{param['max_epoch'] * batches_per_epoch}, "
              f"losses: {[loss.data.item() for loss in losses]}, accuracies: {[metrics['train'].accuracy.item() for metrics in metrics]}")

        if (b + 1) % 25 == 0:  # Validation
            x, y = mnist.next_val_batch(one_hot=True, batch_size=60000 - param['num_train'])
            preds = [model(x) for model in models]
            losses = [((y.data - preds_) ** 2).sum() / preds_.numel() for preds_ in preds]
            [metrics['val'].log_prediction(preds.data, y) for metrics, preds in zip(metrics, preds)]
            for idx in range(len(models)):
                logger['val loss'].log_scalar(f"{param['optimizer'][idx]}", value=losses[idx].data.item(), step=epoch * batches_per_epoch + b)
            for idx, model in enumerate(metrics):
                logger['val accuracy'].log_scalar(f"{param['optimizer'][idx]}", value=metrics[idx]['val'].accuracy, step=epoch * batches_per_epoch + b)

            [metrics['train'].reset() for metrics in metrics]
            [metrics['val'].reset() for metrics in metrics]

        if (b + 1) % 625 == 0:  # Plotting
            logger['train loss'].plot([f"{param['optimizer'][i]}" for i in range(len(models))], title='Train Loss')
            logger['train accuracy'].plot([f"{param['optimizer'][i]}" for i in range(len(models))], title='Train Accuracy')
            logger['val loss'].plot([f"{param['optimizer'][i]}" for i in range(len(models))], title='Val Loss')
            logger['val accuracy'].plot([f"{param['optimizer'][i]}" for i in range(len(models))], title='Val Accuracy')

logger['train loss'].pause_plots(0)
