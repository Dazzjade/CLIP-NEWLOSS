import logging
from contextlib import suppress
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import abc
from typing import Callable, Optional
import argparse
import clip

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import top_k_accuracy_score, average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

TEMPLATES = ["people on the photo feel ",
             "a photo makes people feel ",
             "a photo of people feeling "
             ]


class EvaluatorBase(abc.ABC):
    @abc.abstractmethod
    def eval(self) -> dict:
        raise NotImplementedError


class ZeroShotClassifier(EvaluatorBase):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        class_names: list[str],
        device: str | torch.device = 'cuda',
    ):

        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device
        self.model = model

        with torch.no_grad():
            zeroshot_weights = []
            for class_name in class_names:
                texts = [template + class_name for template in TEMPLATES]  # format with class
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            self.classifier = torch.stack(zeroshot_weights, dim=1).to(self.device, non_blocking=True)


    def get_logits(self, dataloader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
        all_logits = []
        all_targets = []
        tdataloader = tqdm(dataloader, desc=f'zero-shot', unit_scale=dataloader.batch_size, leave=False)
        with torch.no_grad():
            for i, (visual_inputs, targets) in enumerate(tdataloader):
                visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                features = self.model.encode_image(visual_inputs)
                features = F.normalize(features, dim=-1)
                logits = 100. * features @ self.classifier

                all_logits.append(logits)
                all_targets.append(targets)


            all_logits = torch.cat(all_logits).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()
        return all_logits, all_targets

    def eval(self) -> dict[str, float]:
        self.model.eval()
        all_logits, all_targets = self.get_logits(self.dataloader)
        top1 = top_k_accuracy_score(all_targets, all_logits, k=1) * 100.
        return top1



class LinearProbClassifier(EvaluatorBase):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: list[torch.utils.data.DataLoader],
        precision: str = 'amp',
        device: str | torch.device = 'cuda',
    ):
        assert len(dataloaders) == 2, f'2 dataloaders are required for training and validation but got {len(dataloaders)=}'
        assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'

        self.model = model
        # override by args if provided

        self.train_dataloader, self.val_dataloader = dataloaders
        self.clf = LogisticRegression(
            random_state=1,
            max_iter=2000,
            C=3.16,
            solver='sag'
        )


    def get_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
        autocast = torch.amp.autocast if self.precision == 'amp' else suppress
        feature_extractor = self.model.encode_image 
        all_features = []
        all_targets = []
        with torch.no_grad():
            for i, (visual_inputs, targets) in enumerate(tdataloader):
                visual_inputs = visual_inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast('cuda'):
                    features = feature_extractor(visual_inputs)
                    features = F.normalize(features, dim=-1)
                all_features.append(features)
                all_targets.append(targets)

            all_features = torch.cat(all_features).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()
        return all_features, all_targets


    def eval(self) -> dict[str, float]:
        self.model.eval()
        train_features, train_targets = self.get_features(self.train_dataloader, split='train')
        val_features, val_targets = self.get_features(self.val_dataloader, split='val')
        if self.debug_mode is True:
            return {'top1': -1, 'top5': -1}
        self.clf.fit(train_features, train_targets)
        probs = self.clf.predict_proba(val_features)
        top1 = top_k_accuracy_score(val_targets, probs, k=1) * 100.
        return top1


# class MulticlassLinearClassifier(EvaluatorBase):
#     def __init__(
#         self,
#         model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
#         dataloaders: list[torch.utils.data.DataLoader],
#         input_type: str = 'video',
#         precision: str = 'amp',
#         device: str | torch.device = 'cuda',
#         debug_mode: bool = False,
#         args: Optional[argparse.Namespace] = None
#     ):
#         assert len(dataloaders) == 2, f'2 dataloaders are required for training and validation but got {len(dataloaders)=}'
#         assert precision in ['amp', 'fp32'], f'precision must be one of [amp, fp32], got {precision=}'
#         assert input_type in ['video', 'image'], f'`input_type` must be one of ["video", "image"] but got {input_type=}'

#         self.model = unwrap_model(model)
#         self.train_dataloader, self.val_dataloader = dataloaders
#         self.input_type = input_type
#         # override by args if provided
#         if args is not None:
#             self.precision = args.precision
#             self.device = args.device
#             self.debug_mode = args.debug_mode
#         else:
#             self.precision = precision
#             self.device = device
#             self.debug_mode = debug_mode

#         self.base_clf = LogisticRegression(
#             random_state=1,
#             max_iter=2000,
#             C=3.16,
#             solver='sag',
#             class_weight=None
#         )
#         self.clf = OneVsRestClassifier(self.base_clf, n_jobs=8)


#     def get_features(self, dataloader: torch.utils.data.DataLoader, split: str = 'train') -> tuple[np.ndarray, np.ndarray]:
#         tdataloader = tqdm(dataloader, desc=f'linear-eval ({split})', unit_scale=dataloader.batch_size, leave=False)
#         autocast = torch.cuda.amp.autocast if self.precision == 'amp' else suppress
#         feature_extractor = self.model.encode_image if self.input_type == 'image' else self.model.encode_video
#         all_features = []
#         all_targets = []
#         with torch.no_grad():
#             for i, batch in enumerate(tdataloader):
#                 if len(batch) == 2:
#                     visual_inputs, targets = batch
#                     visual_inputs = visual_inputs.to(self.device, non_blocking=True)
#                     targets = targets.to(self.device, non_blocking=True)

#                     with autocast():
#                         features = feature_extractor(visual_inputs)
#                         features = F.normalize(features, dim=-1)
#                     all_features.append(features)
#                     all_targets.append(targets)
                    
#                 elif len(batch) == 3:
#                     visual_inputs, visual_masks, targets = batch
#                     visual_inputs = visual_inputs.to(self.device, non_blocking=True)
#                     targets = targets.to(self.device, non_blocking=True)
#                     visual_masks = visual_masks.to(self.device, non_blocking=True)
#                     with autocast():
#                         features = feature_extractor(visual_inputs, visual_masks)
#                         features = F.normalize(features, dim=-1)
#                     all_features.append(features)
#                     all_targets.append(targets)
                    

#                 # debug mode
#                 if self.debug_mode is True and i == 1:
#                     break

#             all_features = torch.cat(all_features).cpu().numpy()
#             all_targets = torch.cat(all_targets).cpu().numpy()
#         return all_features, all_targets


#     def eval(self) -> dict[str, float]:
#         self.model.eval()
#         train_features, train_targets = self.get_features(self.train_dataloader, split='train')
#         val_features, val_targets = self.get_features(self.val_dataloader, split='val')
#         if self.debug_mode is True:
#             return {'mAP': -1, 'auc': -1}
#         self.clf.fit(train_features, train_targets)
#         probs = self.clf.predict_proba(val_features)
#         mAP = average_precision_score(val_targets, probs) * 100.
#         auc = roc_auc_score(val_targets, probs) * 100.
#         mAP, auc = np.round(mAP, 2), np.round(auc, 2)
#         return {'mAP': mAP, 'auc': auc}
    
    
#     def predict(self):
#         self.model.eval()
#         train_features, train_targets = self.get_features(self.train_dataloader, split='train')
#         val_features, val_targets = self.get_features(self.val_dataloader, split='val')
#         self.clf.fit(train_features, train_targets)
#         probs = self.clf.predict_proba(val_features)
#         return {'probs': probs, 'targets': val_targets}
