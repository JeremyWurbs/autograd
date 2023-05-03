import os

import torch.utils.data
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, num_train=55000):
        assert num_train <= 60000, 'There are only 60000 total training samples available'
        super().__init__()
        self.num_train = num_train
        self.num_val = 60000 - num_train

        # download data
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=False)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        self.train, self.val = random_split(MNIST(os.getcwd(), train=True, transform=transform), [self.num_train, self.num_val])
        self.test = MNIST(os.getcwd(), train=False, transform=transform)
