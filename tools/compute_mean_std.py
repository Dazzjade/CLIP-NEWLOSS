import torch
import clip
from torch.utils.data import Dataset, DataLoader
import clip.model_copy as pmt
from PIL import Image
from torch import nn, optim
from torchvision import transforms, utils
import os
import numpy as np
from dataset.CAER import CAER

dataset = CAER(target = 'number')
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
mean = torch.zeros(3)
std = torch.zeros(3)
for X, _ in loader:
    for d in range(3):
        mean[d] += X[:, d, :, :].mean()
        std[d] += X[:, d, :, :].std()
        
mean.div_(len(dataset))
std.div_(len(dataset))
print(list(mean.numpy()), list(std.numpy()))

