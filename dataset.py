import numpy as np
import torch

from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

# *获取训练、测试集，根据数据集不同分别处理
def get_dataset(name, data_path):
    if name == 'MNIST' :
        raw_tr = datasets.MNIST(data_path, train=True, download=True)
        raw_te = datasets.MNIST(data_path, train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te