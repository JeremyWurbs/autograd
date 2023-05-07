import os

import torch.utils.data
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size=32, num_train=55000, H=28, W=28, dtype=torch.float32):
        assert num_train <= 60000, 'There are only 60000 total training samples available'
        super().__init__()
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = 60000 - num_train
        self.H = H
        self.W = W
        self.dtype = dtype

        self._train_idx = 0
        self._val_idx = 0
        self._test_idx = 0

        # download datasets
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=False)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Resize((H, W), antialias=False)])
        train, val = random_split(MNIST(os.getcwd(), train=True, transform=transform), [self.num_train, self.num_val])
        test = MNIST(os.getcwd(), train=False, transform=transform)

        self.train = self.parse_data(train, self.num_train, 'training', dtype=dtype)
        self.val = self.parse_data(val, self.num_val, 'validation', dtype=dtype)
        self.test = self.parse_data(test, 10000, 'test', dtype=dtype)

    def parse_data(self, dataset, total, name, dtype=torch.float32):
        parsed_data = dict()
        parsed_data['datasets'] = torch.empty((total, self.H*self.W), dtype=dtype)
        parsed_data['labels'] = torch.empty((total, 1), dtype=torch.int32)
        for i, (x, y) in tqdm(enumerate(dataset), total=total, desc=f'Parsing {name} datasets'):
            parsed_data['datasets'][i, :] = x.view(1, self.H*self.W)
            parsed_data['labels'][i] = torch.tensor(y).view(1)
        return parsed_data

    def _next_batch(self, data, labels, idx, batch_size=None, one_hot=False):
        num_samples = data.shape[0]
        if batch_size is None:
            batch_size = self.batch_size
        if idx + batch_size >= num_samples:
            batch_data = torch.concat((data[idx:],
                                 data[:idx + batch_size - num_samples]))
            batch_labels = torch.concat((labels[idx:],
                                   labels[:idx + batch_size - num_samples]))
            idx += batch_size - num_samples
        else:
            batch_data = data[idx:idx + batch_size]
            batch_labels = labels[idx:idx + batch_size]
            idx += batch_size

        if one_hot:
            batch_labels = F.one_hot(batch_labels.long(), num_classes=10).squeeze(dim=1).type(self.dtype)

        return batch_data, batch_labels, idx

    def next_train_batch(self, batch_size=None, one_hot=False):
        data, labels, self._train_idx = self._next_batch(self.train_data, self.train_labels, self._train_idx, batch_size=batch_size, one_hot=one_hot)
        return data, labels

    def next_val_batch(self, batch_size=None, one_hot=False):
        data, labels, self._val_idx = self._next_batch(self.val_data, self.val_labels, self._val_idx, batch_size=batch_size, one_hot=one_hot)
        return data, labels

    def next_test_batch(self, batch_size=None, one_hot=False):
        data, labels, self._test_idx = self._next_batch(self.test_data, self.test_labels, self._test_idx, batch_size=batch_size, one_hot=one_hot)
        return data, labels

    @property
    def train_data(self):
        return self.train['datasets']

    @property
    def train_labels(self):
        return self.train['labels']

    @property
    def val_data(self):
        return self.val['datasets']

    @property
    def val_labels(self):
        return self.val['labels']

    @property
    def test_data(self):
        return self.test['datasets']

    @property
    def test_labels(self):
        return self.test['labels']
