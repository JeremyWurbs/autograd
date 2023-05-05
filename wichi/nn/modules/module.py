class Module(object):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return list()
