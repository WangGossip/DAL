import numpy as np
import torch

from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

# *获取训练、测试集，根据数据集不同分别处理
def get_dataset(name, data_path):
    if name == 'MNIST' :
        return get_MNIST(data_path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(data_path)
    elif name == 'SVHN' :
        return get_SVHN(data_path)
    elif name == 'CIFAR10':
        return get_CIFAR10(data_path)
    elif name[:9] == 'CIFAR10IM':
        return get_CIFAR10IM(name, data_path)

def get_MNIST(data_path):
    raw_tr = datasets.MNIST(data_path, train=True, download=False)
    raw_te = datasets.MNIST(data_path, train=False, download=False)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(data_path):
    raw_tr = datasets.FashionMNIST(data_path, train=True, download=True)
    raw_te = datasets.FashionMNIST(data_path, train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(data_path):
    data_tr = datasets.SVHN(data_path, split='train', download=True)
    data_te = datasets.SVHN(data_path, split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(data_path):
    raw_tr = datasets.CIFAR10(data_path, train=True, download=False)
    raw_te = datasets.CIFAR10(data_path, train=False, download=False)
    X_tr = raw_tr.data
    Y_tr = torch.tensor(raw_tr.targets)
    X_te = raw_te.data
    Y_te = torch.tensor(raw_te.targets)
    return X_tr, Y_tr, X_te, Y_te

# *处理不平衡数据集，测试集不变
def get_CIFAR10IM(name, data_path):
    imb_class_counts_list = [
        [2000, 5000, 2000, 5000, 2000, 5000, 2000, 5000, 2000, 5000],
        [1000, 5000, 1000, 5000, 1000, 5000, 1000, 5000, 1000, 5000],
        [500, 5000, 500, 5000, 500, 5000, 500, 5000, 500, 5000],
        [100, 5000, 100, 5000, 100, 5000, 100, 5000, 100, 5000]
    ]
    id_imb = int(name[9])
    imb_class_counts=imb_class_counts_list[id_imb]
    raw_tr = datasets.CIFAR10(data_path, train=True, download=False)
    raw_te = datasets.CIFAR10(data_path, train=False, download=False)
    targets = np.array(raw_tr.targets)
    X_te = raw_te.data
    Y_te = torch.tensor(raw_te.targets)    

    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)
    class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
    imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
    imb_class_idx = np.hstack(imb_class_idx)
    X_tr = raw_tr.data[imb_class_idx]
    Y_tr = torch.tensor(targets[imb_class_idx])
    return X_tr, Y_tr, X_te, Y_te

# *重载data类，编写合适的dataloader
def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)