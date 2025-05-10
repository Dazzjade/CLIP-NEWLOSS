from tools.evaluation import ZeroShotClassifier
import torch
print(torch.__version__)
import clip
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from dataset.meld import MELD
from dataset.ImageEmotion import ImageEmotion
from dataset.CAER import CAER
import clip.model_copy as pmt
import time
BATCH_SIZE = 100
EMOTION_CLASS_NAMES_CAER = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]
EMOTION_CLASS_NAMES_ImageEmotion = [
    'amusement',
    'anger',
    'awe',
    'contentment',
    'disgust',
    'excitement',
    'fear',
    'sadness'
]
EMOTION_CLASS_NAMES_MELD = [
    'neutral',
    'surprise',
    'fear',
    'sadness',
    'joy',
    'disgust',
    'anger'
]
CLASS_NAMES = [EMOTION_CLASS_NAMES_CAER, EMOTION_CLASS_NAMES_MELD, EMOTION_CLASS_NAMES_ImageEmotion]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model, preprocess = clip.load('ViT-B/32', device)
model = torch.load("/home/dazzy/CLIP/models/CLIP-SMP_newloss_TEST.pt")


dataset_caer = CAER(split='test', target='test')
dataset_meld = MELD(split='test', target='test')
dataset_ImageEmotion = ImageEmotion(split='test', target='test')

dataloader_caer = ('caer', DataLoader(dataset_caer, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
dataloader_meld = ('meld', DataLoader(dataset_meld, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
dataloader_ImageEmotion = ('ImageEmotion', DataLoader(dataset_ImageEmotion, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
loaders = [dataloader_caer, dataloader_meld, dataloader_ImageEmotion]


for i in range(3):
    name, loader = loaders[i]
    class_names = CLASS_NAMES[i]
    evaluator = ZeroShotClassifier(model, loader, class_names, device)
    acc = evaluator.eval()
    print(f"{name}: {acc}%")
    