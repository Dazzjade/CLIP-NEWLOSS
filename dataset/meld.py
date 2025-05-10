from __future__ import annotations
import os
import os.path as osp
from re import L
from typing import Literal, Optional
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import orjson
import re


EMOTION_CLASS_NAMES = [
    'neutral',
    'surprise',
    'fear',
    'sadness',
    'joy',
    'disgust',
    'anger'
]

SENTIMENT_CLASS_NAMES = [
    'neutral',
    'positive',
    'negative'
]

logger = logging.getLogger()


# BUG train index 1165, no dir 'train_splits/dia125_utt3'
# BUG dev index 1084, no dir 'dev_splits/dia110_utt7'
# NOTE a lot of clip only have several frames
# num_frames statistics. max: 984, min: 2, mean: 75, std: 58
# n_sample with all / 32 / 64 frames: 
#   [train] 9988 / 8059 / 4454
#   [dev]   1108 /  892 / 496
#   [test]  2610 / 2160 / 1227
    
class MELD(Dataset):
    def __init__(
        self,
        data_dir: str = '/home/dazzy/meld',
        split: Literal['train', 'dev', 'test'] = 'train',
        target: Literal['train', 'test'] = 'train',
    ):
        assert split in ['train', 'dev', 'test']
        
        super().__init__()
        # constants
        self.RESIZE_SIZE = 256
        self.CROP_SIZE = 224
        self.BBOX_TO_MASK_THRESHOLD = 0.5
        
        self.data_dir = data_dir
        self.split = split
        self.target = target
        
        self.index: pd.DataFrame
        self.all_human_boxes: dict
        self._create_index()
        
        # clip-style preprocesser
        self.preprocesser = T.Compose([
            T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
    def _create_index(self):
        # load .csv data
        if self.split in ['train', 'dev', 'test']:
            annotation_file_path = osp.join(self.data_dir ,f'{self.split}_sent_emo.csv')
            self.index = pd.read_csv(annotation_file_path)
        else:
            raise NotImplementedError        
        
        logger.info(f'Index of {self.split} set created, {self.index.shape[0]} samples in total.')
        
        
    def __len__(self):
        return self.index.shape[0]
    
    
    def __getitem__(self, i):
        dialogue_id = self.index.loc[i, 'Dialogue_ID']
        utterance_id = self.index.loc[i, 'Utterance_ID']
        clip_id = f'dia{dialogue_id}_utt{utterance_id}.jpg'
        frame_path = osp.join(self.data_dir, f'{self.split}_splits', clip_id)
        
        # words = re.split(r"\s|,|\.",self.index.loc[i, 'Utterance'])
        # if len(words) > 71:
        #     new_i = np.random.randint(self.index.shape[0])
        #     return self.__getitem__(new_i)

        # load video frames, human bboxes and process them
        raw_frame = Image.open(frame_path).convert('RGB')
        resized_frame = F.resize(raw_frame, size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)
        frame = F.center_crop(resized_frame, self.CROP_SIZE)
        frame = F.normalize(
                tensor=F.to_tensor(frame),
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
                ).float()
        
        # load text
        if self.target == 'train':
            emotion = self.index.loc[i, 'Emotion']
            target = f"people in the picture feel {emotion}"
            return {'img':frame, 'text':target, 'label': EMOTION_CLASS_NAMES.index(emotion)}
            
        elif self.target == 'test':
            emotion = self.index.loc[i, 'Emotion']
            return frame, EMOTION_CLASS_NAMES.index(emotion)
        
        
if __name__ == '__main__':
    pass