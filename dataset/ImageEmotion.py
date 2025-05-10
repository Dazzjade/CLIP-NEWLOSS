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

EMOTION_CLASS_NAMES = [
    'amusement',
    'anger',
    'awe',
    'contentment',
    'disgust',
    'excitement',
    'fear',
    'sadness'
]

class ImageEmotion(Dataset):
    def __init__(
        self,
        data_dir: str = '/home/dazzy/Downloads/emotion_dataset',
        split: Literal['train', 'test'] = 'train',
        target: Literal['test', 'train'] = 'train',
    ):
        super().__init__()
        self.RESIZE_SIZE = 224
        self.CROP_SIZE = 224
        
        self.data_dir = data_dir
        self.split = split
        self.target = target
        
        # load annotations
        annotation_dir = osp.join(data_dir, 'label')
        self.index = pd.read_csv(osp.join(annotation_dir, f'{split}.csv'))
        if target == 'test':
            self.index['emotion'] = self.index['emotion'].apply(lambda x: EMOTION_CLASS_NAMES.index(x))
        
        self.preprocesser = T.Compose([
            T.Resize(size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=self.CROP_SIZE),
            T.ToTensor(),
            T.Normalize((0.44197568, 0.41516715, 0.39168137), (0.2502798, 0.24239047, 0.24234003)),
        ])
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i):
        sample_info = self.index.loc[i]
        
        # process img and generate mask
        img_file_path = sample_info['file_link']
        img = Image.open(img_file_path).convert('RGB')
        img = F.resize(img, size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)
        img = F.center_crop(img, self.CROP_SIZE)
        img = F.normalize(
            tensor=F.to_tensor(img),
            mean=[0.44197568, 0.41516715, 0.39168137],
            std=[0.2502798, 0.24239047, 0.24234003]
        ).float()
        
        if self.target == 'train':
            target = 'the picture makes people feel' + sample_info['emotion']
            return {'img':img, 'text':target, 'label':EMOTION_CLASS_NAMES.index(sample_info['emotion'])}
        elif self.target == 'test':
            target = sample_info['emotion']
            return img, target


