from __future__ import annotations
import os
import os.path as osp
from re import L
from typing import Literal, Optional
import logging
import cv2
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import orjson
EMOTION_CLASS_NAMES = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]
all_samples = []
for emotion in EMOTION_CLASS_NAMES:
    frame_dir = os.path.join('/home/dazzy/Downloads/CAER-S/test',emotion)
    for link in os.listdir(frame_dir):
        all_samples.append((emotion, os.path.join(frame_dir, link)))


random.shuffle(all_samples)
random.shuffle(all_samples)
random.shuffle(all_samples)
random.shuffle(all_samples)
random.shuffle(all_samples)

for emotion, link in all_samples:
    label = {'file_link': [link], 'emotion': [emotion]}
    label = pd.DataFrame(label)
    label.to_csv("/home/dazzy/Downloads/CAER-S/label/test.csv", mode='a', index=False, header=False)


