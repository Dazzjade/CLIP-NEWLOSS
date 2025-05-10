import torch
import clip
from torch.utils.data import DataLoader
from torch import nn
import os.path as osp
import sys
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset.emotic import Emotic
from dataset.meld import MELD
from dataset.ImageEmotion import ImageEmotion
from dataset.CAER import CAER
import clip.model_copy as pmt
import wandb
import yaml
from tools.logger import setup_logger
from tools.args import ARG
from tools.lr_schedeler import LRWarmupScheduler
from tools.evaluation import LinearProbClassifier


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


def cross_entropy(preds, loss_number, loss_score, arg):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = -torch.from_numpy(loss_score).float().to(arg.DEVICE) * log_softmax(preds)
    # loss = -torch.diag(log_softmax(preds))#在这里我发现了一个非常有趣的现象，我阴差阳错地忘记加负号，于是进行了反向训练，结果效果更好，原因在于文本标签模版都是一样的，在负样本中，比起单个的正样本，有着更多的“正样本文本标签”
    return loss.sum() / loss_number

def get_loss_score(name: str, labels, arg):
    loss_score = np.zeros((arg.BATCH_SIZE, arg.BATCH_SIZE))
    for i in range(arg.BATCH_SIZE):
        for j in range(i, arg.BATCH_SIZE):
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
        
        
arg = ARG()
LOSS = []

### set logger
logger = setup_logger(
    name=None,
    color=True,
    output_dir=arg.LOG_DIR,
    exp_id=arg.EXP_ID
)


### load model
raw_model, preprocess = clip.load('ViT-B/32', arg.DEVICE)
state_dict = raw_model.state_dict()
model = pmt.build_model(state_dict).to(arg.DEVICE)

if arg.LOCK:
    model.lock()
if arg.LOCK_TEXT:
    model.lock_text()
model.train()
logger.info('Model created')


###load dataset
dataset_caer = CAER(split='train', target='train')
dataset_emotic = Emotic(split='train', target='train')
# dataset_meld = MELD(split='train', target='train')
dataset_ImageEmotion = ImageEmotion(split='train', target='train')

dataloader_caer = ('caer', DataLoader(dataset_caer, batch_size=arg.BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
dataloader_emotic = ('emotic', DataLoader(dataset_emotic, batch_size=arg.BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
# dataloader_meld = ('meld', DataLoader(dataset_meld, batch_size=arg.BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))
dataloader_ImageEmotion = ('ImageEmotion', DataLoader(dataset_ImageEmotion, batch_size=arg.BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True))

loaders = [dataloader_caer, dataloader_emotic, dataloader_ImageEmotion]

logger.info('datasets are loaded')

for name, loader in loaders:
    arg.EPOCH_LEN += len(loader)


##set evaluator
eva_train_dataset = MELD(split='train',target='test')
eva_test_dataset = MELD(split='test',target='test')
eva_train_dataloader = DataLoader(eva_train_dataset, batch_size=arg.BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True)
eva_test_dataloader = DataLoader(eva_test_dataset, batch_size=arg.BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True)
evaluator = LinearProbClassifier(model, [eva_train_dataloader, eva_test_dataloader], 'amp', 'cuda')


### copy config
with open(osp.join('/home/dazzy/CLIP/Arg_config', f'args_{arg.EXP_ID}.yaml'), mode='w') as f:
    yaml.safe_dump(vars(arg), f)
logger.info('config is copied')
    
    
### set optimizer and lr_scheduler
is_image_params = lambda n, p: 'visual' in n
is_text_params = lambda n, p: 'visual' not in n and 'logit_scale' not in n and "Prompt" not in n
is_gb_params = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n
is_Prompt_params = lambda n, p: "Prompt" in n

all_named_parameters = list(model.named_parameters())
image_gb_params = [
    p for n, p in all_named_parameters if is_gb_params(n, p) and is_image_params(n, p) and p.requires_grad
]
image_rest_params = [
    p for n, p in all_named_parameters if not is_gb_params(n, p) and is_image_params(n, p) and p.requires_grad
]
text_gb_params = [
    p for n, p in all_named_parameters if is_gb_params(n, p) and is_text_params(n, p) and p.requires_grad
]
text_rest_params = [
    p for n, p in all_named_parameters if not is_gb_params(n, p) and is_text_params(n, p) and p.requires_grad
]
Prompt_params = [
    p for n, p in all_named_parameters if "Prompt" in n and p.requires_grad
]
logit_scale_params = [
    p for n, p in all_named_parameters if 'logit_scale' in n and p.requires_grad
]

param_groups_for_optimizer = [
    {'params': image_gb_params, 'lr': arg.LR_GB, 'weight_decay': 0.},
    {'params': image_rest_params, 'lr': arg.LR_REST, 'weight_decay': arg.WEIGHT_DECAY},
    {'params': text_gb_params, 'lr': arg.LR_GB, 'weight_decay': 0.},
    {'params': text_rest_params, 'lr': arg.LR_REST, 'weight_decay': arg.WEIGHT_DECAY},
    {'params': Prompt_params, 'lr': arg.LR_GB, 'weight_decay': arg.WEIGHT_DECAY},
    {'params': logit_scale_params, 'lr': arg.LR_REST, 'weight_decay': 0.}
]

optimizer = torch.optim.AdamW(
    param_groups_for_optimizer,
    betas=(arg.BETA1, arg.BETA2),
    eps = arg.EPS
)

wrapped_lr_scheduler = CosineAnnealingLR(optimizer, T_max=arg.EPOCH, eta_min=arg.LR_MIN)

lr_scheduler = LRWarmupScheduler(
    wrapped_lr_scheduler,
    by_epoch=True,
    epoch_len=arg.EPOCH_LEN,
    warmup_t=arg.WARMUP_T,
    warmup_by_epoch=arg.WARMUP_BY_EPOCH,
    warmup_mode=arg.WARMUP_MODE,
    warmup_init_factor=arg.WARMUP_INIT_FACTOR
)
logger.info('Optimizer and scheduler are set up')

grad_scaler = torch.amp.GradScaler()
autocast = torch.amp.autocast

# set up wandb
wandb.init(
    name = f"experiment{arg.EXP_ID}",
    project = "CLIP_NEWLOSS",
    config = vars(arg)
)
wandb.watch(model, log='all')


# train
for epoch in range(arg.EPOCH):
    loss_epoch = []
    for name, loader in loaders:
        loss_dataset = []
        for i, batch in enumerate(loader):
            data = batch
            data_images = data['img'].to(arg.DEVICE)
            data_texts = data['text']
            data_labels = data['label']
            loss_number, loss_score = get_loss_score(name, data_labels, arg)
            images = data_images
            texts = clip.tokenize(data_texts).to(arg.DEVICE)

            with autocast('cuda'): 
                logits_per_image, logits_per_text = model(images, texts)
                total_loss = (cross_entropy(logits_per_image, loss_number, loss_score, arg) + cross_entropy(logits_per_text, loss_number, loss_score, arg)) / 2
            
            # total_loss.backward()
            # optimizer.step()
            
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            optimizer.zero_grad()
            lr_scheduler.iter_step()
            
            # wandb log parameters' lr
            param_groups = optimizer.param_groups
            base_lrs = [param_group['lr'] for param_group in param_groups]
            dict_params = {
                'image_gb_params': base_lrs[0],
                'image_rest_params': base_lrs[1],
                'text_gb_params': base_lrs[2],
                'text_rest_params': base_lrs[3],
                'Prompt_params': base_lrs[4],
                'logit_scale_params': base_lrs[5]
            }
            wandb.log(dict_params)
            
            logger.info(f"[{epoch}]-[{name}]-[{i}]: {total_loss.item()}")
            loss_dataset.append(total_loss.item())
            
        loss_epoch.append(np.mean(np.array(loss_dataset)))
        # wandb log loss
        loss_dict = {
            f'{name}_loss':loss_epoch[-1]
        }
        wandb.log(loss_dict)
    
    lr_scheduler.epoch_step()    
    LOSS.append(loss_epoch)
    # evaluate
    acc = evaluator.eval()
    logger.info(f"accuracy of epoch {epoch} is {acc}")
    loss_dict = {
        'evaluation_accuracy':acc
    }
    wandb.log(loss_dict)
    
torch.save(model, f'./models/CLIP-NEWLOSS_experiment{arg.EXP_ID}.pt')

logger.info("the loss of every dataset in every epoch:")
logger.info("dataset  meld  emotic  ImageEmotion")
for i in range(len(loaders)):
    logger.info(f"epoch_{i}  {LOSS[i]}")
    
                