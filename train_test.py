import torch
import clip
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from dataset.emotic import Emotic
from dataset.meld import MELD
from dataset.ImageEmotion import ImageEmotion
from dataset.CAER import CAER
import clip.model_copy as pmt
import time


def cross_entropy(preds, loss_number, loss_score):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = -torch.from_numpy(loss_score).float().to(device) * log_softmax(preds)
    # loss = -torch.diag(log_softmax(preds))#在这里我发现了一个非常有趣的现象，我阴差阳错地忘记加负号，于是进行了反向训练，结果效果更好，原因在于文本标签模版都是一样的，在负样本中，比起单个的正样本，有着更多的“正样本文本标签”
    return loss.sum() / loss_number

def cross_entropy_modify(preds, loss_score):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = -torch.from_numpy(loss_score).float().to(device) * log_softmax(preds)
    # loss = -torch.diag(log_softmax(preds))#在这里我发现了一个非常有趣的现象，我阴差阳错地忘记加负号，于是进行了反向训练，结果效果更好，原因在于文本标签模版都是一样的，在负样本中，比起单个的正样本，有着更多的“正样本文本标签”
    return loss.sum() / loss.shape[0]

def negative_cross_entropy(preds):
    log_softmax = nn.LogSoftmax(dim=-1)
    # loss = -torch.from_numpy(loss_score).float().to(device) * log_softmax(preds)
    loss = torch.diag(log_softmax(preds))#在这里我发现了一个非常有趣的现象，我阴差阳错地忘记加负号，于是进行了反向训练，结果效果更好，原因在于文本标签模版都是一样的，在负样本中，比起单个的正样本，有着更多的“正样本文本标签”
    return loss.mean()

def positive_cross_entropy(preds):
    log_softmax = nn.LogSoftmax(dim=-1)
    # loss = -torch.from_numpy(loss_score).float().to(device) * log_softmax(preds)
    loss = -torch.diag(log_softmax(preds))#在这里我发现了一个非常有趣的现象，我阴差阳错地忘记加负号，于是进行了反向训练，结果效果更好，原因在于文本标签模版都是一样的，在负样本中，比起单个的正样本，有着更多的“正样本文本标签”
    return loss.mean()


def get_loss_score(name: str, labels):
    loss_score = np.zeros((BATCH_SIZE, BATCH_SIZE))
    for i in range(BATCH_SIZE):
        for j in range(i, BATCH_SIZE):
            if name == 'emotic':
                if np.count_nonzero(labels[i]) > np.count_nonzero(labels[j]):
                    scale = np.count_nonzero(labels[i])
                else:
                    scale = np.count_nonzero(labels[j])
                    
                loss_score[i,j] = labels[i] @ labels[j].T / scale
                loss_score[j,i] = loss_score[i, j]
                
            else:
                loss_score[i,j] = labels[i] == labels[j]
                loss_score[j,i] = loss_score[i,j]
                
    return np.count_nonzero(loss_score), loss_score
        

EPOCH = 3
BATCH_SIZE = 50
LOSS = []
LR = 1e-4
WEIGHT_DECAY = 0.002

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

raw_model, preprocess = clip.load('ViT-B/32', device)
state_dict = raw_model.state_dict()
model = pmt.build_model(state_dict).to(device)
model.float()
model.lock()

dataset_caer = CAER(split='train', target='train')
dataset_emotic = Emotic(split='train', target='train')
# dataset_meld = MELD(split='train', target='train')
dataset_ImageEmotion = ImageEmotion(split='train', target='train')

dataloader_caer = ('caer', DataLoader(dataset_caer, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
dataloader_emotic = ('emotic', DataLoader(dataset_emotic, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
# dataloader_meld = ('meld', DataLoader(dataset_meld, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
dataloader_ImageEmotion = ('ImageEmotion', DataLoader(dataset_ImageEmotion, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))

loaders = [dataloader_caer, dataloader_emotic, dataloader_ImageEmotion]

optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-8,
                       weight_decay=WEIGHT_DECAY)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

grad_scaler = torch.amp.GradScaler()

time_now = time.time()
for epoch in range(EPOCH):
    loss_epoch = []
    for name, loader in loaders:
        loss_dataset = []
        for i, batch in enumerate(loader):
            data = batch
            data_images = data['img'].to(device)
            data_texts = data['text']
            
            images = data_images
            texts = clip.tokenize(data_texts).to(device)

            logits_per_image, logits_per_text = model(images, texts)
            optimizer.zero_grad()


            # new loss
            data_labels = data['label']
            # loss_number, loss_score = get_loss_score(name, data_labels)
            # total_loss = (cross_entropy(logits_per_image, loss_number, loss_score) + cross_entropy(logits_per_text, loss_number, loss_score)) / 2

            # modify loss
            loss_number, loss_score = get_loss_score(name, data_labels)
            total_loss = (cross_entropy_modify(logits_per_image, loss_score) + cross_entropy_modify(logits_per_text, loss_score)) / 2


            # negative loss
            # total_loss = (negative_cross_entropy(logits_per_image) + negative_cross_entropy(logits_per_text)) / 2

            # positive loss
            # total_loss = (positive_cross_entropy(logits_per_image) + positive_cross_entropy(logits_per_text)) / 2

            # ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            # total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            # total_loss.backward()
            # optimizer.step()
            
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            
            print(f"[{epoch}]-[{name}]-[{i}]: {total_loss.item()}")
            loss_dataset.append(total_loss.item())
            
        loss_epoch.append(np.mean(np.array(loss_dataset)))
        
    LOSS.append(loss_epoch)
    print("time is")
    print(time.time()-time_now)
    
torch.save(model, f'./models/CLIP-SMP_newloss_TEST.pt')

print("the loss of every dataset in every epoch:")
print("dataset  caer  meld  emotic  ImageEmotion")
for i in range(EPOCH):
    print(f"epoch_{i}  {LOSS[i]}")