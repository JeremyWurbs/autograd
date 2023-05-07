class SGD(object):
    def __init__(self, param, lr, momentum=0, dampening=0):
        self.param = param
        self.lr = lr
        self.momentum = momentum if momentum != 0 else None
        self.dampening = dampening if momentum != 0 else None
        self.velocity = dict() if momentum != 0 else None
        self.t = 0

    def zero_grad(self):
        for p in self.param:
            p.zero_grad()

    def step(self):
        self.t += 1
        for p in self.param:
            if self.momentum is not None:
                if p not in self.velocity:
                    self.velocity[p] = p.grad
                else:
                    self.velocity[p] = self.momentum * self.velocity[p] + (1-self.dampening) * p.grad
                p.data -= self.lr * self.velocity[p]
            else:
                p.data -= self.lr * p.grad
