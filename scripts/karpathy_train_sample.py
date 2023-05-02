from wichi import MLP, draw_dot

xs = [
    [2.0, 3.0, -1.0],
    [3.0,  -1.0, 0.5],
    [0.5, 1.0,  1.0],
    [1.0,  1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]


mlp = MLP(dim=[3, 8, 8, 1])
preds = [mlp(x) for x in xs]

print(f'preds: {preds}')


def mse(ys, preds):
    assert len(ys) == len(preds)
    loss = sum([(y-pred)**2 for y, pred in zip(ys, preds)]) / len(ys)
    return loss


loss = mse(ys, preds)
print(f'loss:  {loss}')
loss.backward()

draw_dot(loss).render()

print(f'Parameters: {mlp.parameters()}')
print(f'Num Parameters: {len(mlp.parameters())}')

# Training
lr = 0.005
for i in range(20000):
    # Forward pass
    preds = [mlp(x) for x in xs]

    # Backward pass
    mlp.zero_grad()
    loss = mse(ys, preds)
    loss.backward()

    # Update
    for p in mlp.parameters():
        p.data -= lr * p.grad

    # Print loss
    if i % 1000 == 0:
        print(f'({i}) loss: {loss}')
    if i % 5000 == 0:
        i /= 10

# Final predictions
print(f'Final predictions: {[mlp(x) for x in xs]}')
