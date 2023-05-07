class Module(object):
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return list()
