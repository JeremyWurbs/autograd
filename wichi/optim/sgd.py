class SGD(object):
    def __init__(self, param, lr):
        self.param = param
        self.lr = lr

    def zero_grad(self):
        for p in self.param:
            p.zero_grad()

    def step(self):
        for p in self.param:
            p.data -= self.lr * p.grad
