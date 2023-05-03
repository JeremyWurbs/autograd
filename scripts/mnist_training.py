import numpy as np
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

from wichi import MNIST, MLP, Logger


DEBUG = False
MAX_EPOCH = 1000
NUM_TRAIN = 16
BATCH_SIZE = 16
H = 8  # orig = 28
W = 8  # orig = 28
NUM_HIDDEN = 20
LR = 1e-1
assert NUM_TRAIN % BATCH_SIZE == 0

batch_per_epoch = NUM_TRAIN // BATCH_SIZE
resize = Resize((H, W))

mnist = MNIST(num_train=NUM_TRAIN)
logger = Logger()

mlp = MLP(dim=[H*W, NUM_HIDDEN, 10])
fig, ax = plt.subplots()
print(f'MLP parameters: {len(mlp.parameters())}')
for epoch in range(MAX_EPOCH):
    for b in range(batch_per_epoch):
        labels = list()
        preds = list()
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

        num_correct = 0
        for p, l in zip(preds, labels):
            p_ = [p.data for p in p]
            if np.argmax(p_) == np.argmax(l):
                num_correct += 1

        # Backward pass
        loss = [sum([e**2 for e in [p-y for p, y in zip(pred, y)]]) / len(pred) for pred, label in zip(preds, labels)]
        batch_loss = sum([l for l in loss]) / len(loss)
        logger.log_scalar('train loss', value=batch_loss.data, step=epoch*batch_per_epoch + b)

        mlp.zero_grad()
        batch_loss.backward()  # Computes gradients for all parameters for the entire batch

        # Apply gradients to update parameters
        for p in mlp.parameters():
            p.data -= LR * p.grad

        if b % 1 == 0:
            print(f'epoch: {epoch}/{MAX_EPOCH-1}, step: {b}/{batch_per_epoch-1}, loss: {batch_loss.data:.4f}, correct: {num_correct}/{BATCH_SIZE}')

        if b % 5 == 0:
            steps = [t['step'] for t in logger.get_log('train loss')]
            losses = [t['value'] for t in logger.get_log('train loss')]
            ax.plot(steps, losses)
            ax.set_title('Train Loss')
            plt.draw()
            plt.pause(0.01)
