import numpy as np
import torch
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

from wichi import MNIST, MLP, Logger


DEBUG = False
DEBUG_DISPLAY_IMAGE = False
DEBUG_TRAIN_TORCH = True
MAX_EPOCH = 1000
NUM_TRAIN = 16
BATCH_SIZE = 16
H = 8  # orig = 28
W = 8  # orig = 28
NUM_HIDDEN = 20
LR = 1e-1
assert NUM_TRAIN % BATCH_SIZE == 0

batch_per_epoch = NUM_TRAIN // BATCH_SIZE
resize = Resize((H, W), antialias=False)

mnist = MNIST(num_train=NUM_TRAIN)
logger = Logger()

mlp = MLP(dim=[H*W, NUM_HIDDEN, 10], init_norm=np.sqrt(H*W*NUM_HIDDEN*10))
mlp_torch = torch.nn.Sequential(torch.nn.Linear(H*W, NUM_HIDDEN),
                                torch.nn.Tanh(),
                                torch.nn.Linear(NUM_HIDDEN, 10),
                                torch.nn.Tanh(),)
for p in mlp_torch:
    p.requires_grad = True
loss_torch = None

for layer_idx, layer in enumerate(mlp.layers):
    torch_weights = mlp_torch.get_submodule(f'{2*layer_idx}').weight
    torch_biases = mlp_torch.get_submodule(f'{2*layer_idx}').bias
    for neuron_idx, neuron in enumerate(layer.neurons):
        neuron.b.data = torch_biases[neuron_idx].data.item()
        for w_idx, w in enumerate(neuron.w):
            w.data = torch_weights[neuron_idx, w_idx].data.item()

fig, ax = plt.subplots()
print(f'MLP parameters: {len(mlp.parameters())}')
for epoch in range(MAX_EPOCH):
    for b in range(batch_per_epoch):
        labels = list()
        preds = list()
        preds_torch = torch.empty((0, 10))
        num_correct_torch = 0
        for i in range(BATCH_SIZE):
            # Get the next input
            x, y_ = mnist.train[b*BATCH_SIZE + i]
            x = resize(x)
            x = [x.data.item() for x in x.flatten()]
            y = [0. for _ in range(10)]
            y[y_] = 1.

            # Forward pass
            preds.append(mlp(x))
            labels.append(y)

            if DEBUG:
                print(f"preds ({y_}): {[f'{p.data:.4f}' for p in preds[-1]]}")

            if DEBUG_TRAIN_TORCH:
                preds_torch = torch.concat((preds_torch, mlp_torch(torch.Tensor(x)).view(1, 10)), dim=0)
                pred_torch = torch.argmax(preds_torch[-1, :])
                if pred_torch == y_:
                    num_correct_torch += 1

            if DEBUG_DISPLAY_IMAGE:
                img = np.array(x).reshape((H, W))
                plt.imshow(img)
                plt.show()
                print('')

        num_correct = 0
        for p, l in zip(preds, labels):
            p_ = [p.data for p in p]
            if np.argmax(p_) == np.argmax(l):
                num_correct += 1

        # Backward pass
        loss = list()
        for i in range(len(preds)):
            loss.append(sum([(p - y) ** 2 for p, y in zip(preds[i], labels[i])]) / len(preds[i]))
        batch_loss = sum(loss) / len(loss)
        logger.log_scalar('train loss', value=batch_loss.data, step=epoch*batch_per_epoch + b)

        mlp.zero_grad()
        batch_loss.backward()  # Computes gradients for all parameters for the entire batch

        if DEBUG_TRAIN_TORCH:
            labels_torch = torch.Tensor(labels)
            loss_torch = torch.sum((labels_torch - preds_torch)**2) / preds_torch.numel()

            mlp_torch.zero_grad()
            loss_torch.backward()

            with torch.no_grad():
                for p in mlp_torch.parameters():
                    p_new = p - LR * p.grad
                    p.copy_(p_new)

        # Apply gradients to update parameters
        for p in mlp.parameters():
            p.data -= LR * p.grad

        global_step = epoch*batch_per_epoch + b + 1
        if global_step % 1 == 0:
            if DEBUG_TRAIN_TORCH:
                print(f'epoch: {epoch}/{MAX_EPOCH - 1}, step: {b}/{batch_per_epoch - 1}, loss: {batch_loss.data:.6f}, torch loss: {loss_torch:.6f}, LR: {LR}, correct: {num_correct}/{BATCH_SIZE}, torch correct: {num_correct_torch}/{BATCH_SIZE}')
            else:
                print(f'epoch: {epoch}/{MAX_EPOCH-1}, step: {b}/{batch_per_epoch-1}, loss: {batch_loss.data:.4f}, LR: {LR}, correct: {num_correct}/{BATCH_SIZE}')

        if global_step % 5 == 0:
            steps = [t['step'] for t in logger.get_log('train loss')]
            losses = [t['value'] for t in logger.get_log('train loss')]
            ax.plot(steps, losses)
            ax.set_title('Train Loss')
            plt.draw()
            plt.pause(0.1)

        if global_step % 100 == 0:
            LR /= 2
