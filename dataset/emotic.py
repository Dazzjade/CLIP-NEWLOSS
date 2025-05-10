from __future__ import annotations
import os
import os.path as osp
from typing import Literal, Optional
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset 
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
    
class Emotic(Dataset):
    def __init__(
        self,
        data_dir: str = '/home/dazzy/emotic-master',
        split: Literal['train', 'val', 'test'] = 'train',
        target: Literal['test', 'train'] = 'train',
    ):
        super().__init__()
        self.RESIZE_SIZE = 224
        self.CROP_SIZE = 224
        
        self.data_dir = data_dir
        self.split = split
        self.target = target
        # load annotations
        annotation_dir = osp.join(data_dir, 'emotic_pre')
        self.index = pd.read_csv(osp.join(annotation_dir, f'{split}.csv'))
        self.text = self.index['Categorical_Labels']
        self.cat_labels: np.ndarray = np.load(osp.join(annotation_dir, f'{split}_cat_arr.npy'))
        self.cont_labels: np.ndarray = np.load(osp.join(annotation_dir, f'{split}_cont_arr.npy'))
        
        # clip-style preprocesser
        self.preprocesser = T.Compose([
            T.Resize(size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=self.CROP_SIZE),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i):
        sample_info = self.index.loc[i]
        
        # process img and generate mask
        img_file_path = osp.join(self.data_dir, 'emotic', sample_info['Folder'], sample_info['Filename'])
        img = Image.open(img_file_path).convert('RGB')
        img = F.resize(img, size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)
        img = F.center_crop(img, self.CROP_SIZE)
        img = F.normalize(
            tensor=F.to_tensor(img),
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ).float()
        
        if self.target == 'train':
            emotions = eval(self.text[i])
            target = "people on the picture feel "
            for emotion in emotions:
                target += emotion
                target += ", "
            return {'img':img, 'text':target, 'label':self.cat_labels[i]}
        
        else:
            target = self.cat_labels[i]
            target = torch.from_numpy(target).float()
            return img, target