import argparse
from dataclasses import dataclass
import torch


@dataclass
class ARG(argparse.Namespace):
    EXP_ID: int = 999
    EPOCH: int = 25
    BATCH_SIZE: int = 50
    LR_GB:float = 5e-5
    LR_REST:float = 1e-8
    LR_MIN:float = 1e-10
    WEIGHT_DECAY:float = 0.1
    BETA1:float = 0.98
    BETA2:float = 0.9
    EPS:float = 1e-6
    EPOCH_LEN: int = 0
    WARMUP_T: int = 1900
    WARMUP_MODE:str = 'auto'
    WARMUP_INIT_FACTOR:float = 1e-5
    WARMUP_BY_EPOCH:bool = False
    LOG_DIR:str = '/home/dazzy/CLIP/Log'
    DEVICE:str = "cuda" if torch.cuda.is_available() else "cpu"
    LOCK:bool = True
    LOCK_TEXT:bool = False