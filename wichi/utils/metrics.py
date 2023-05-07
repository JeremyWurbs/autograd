import torch


class Metrics(object):
    def __init__(self, name=None, num_classes=10, confusion_decay=None):
        self.name = name
        self.num_classes = num_classes
        self.confusion_mat = None
        self.confusion_decay = confusion_decay

        self.reset()

    def reset(self):
        self.confusion_mat = torch.zeros((self.num_classes, self.num_classes))

    def log_prediction(self, pred, labels):
        pred = pred if isinstance(pred, torch.Tensor) else torch.Tensor(pred)
        labels = labels if isinstance(labels, torch.Tensor) else torch.Tensor(labels)
        p = torch.argmax(pred, dim=1).int()
        l = torch.argmax(labels, dim=1).int()

        if self.confusion_decay is None:
            for p, l in zip(p, l):
                self.confusion_mat[p, l] += 1
        else:
            cur_batch = torch.zeros((self.num_classes, self.num_classes))
            for p, l in zip(p, l):
                cur_batch[p, l] += 1

            self.confusion_mat *= self.confusion_decay
            self.confusion_mat += cur_batch

    @property
    def accuracy(self):
        return torch.sum(torch.diag(self.confusion_mat)) / torch.sum(self.confusion_mat)
