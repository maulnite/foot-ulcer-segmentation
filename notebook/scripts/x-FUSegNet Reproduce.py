# Generated from: x-FUSegNet Reproduce (1).ipynb
# Converted at: 2026-10-29T20:45:40.023Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # x-FUSegNet — Reproduksi `xfusegnet_all.py`
# ### Segmentasi Diabetic Foot Ulcer (DFU) · EfficientNet-B7 + Parallel scSE (P-scSE) · Ensemble 5-Fold
# 
# > **Atribusi (wajib dicantumkan bila notebook ini disebarkan / dipublikasikan)**
# > * **Kode acuan & arsitektur:** Mrinal Kanti Dhar, Taiyu Zhang, Yash Patel, Sandeep Gopalakrishnan, Zeyun Yu —
# >   *FUSegNet: A Deep Convolutional Neural Network for Foot Ulcer Segmentation* ([arXiv:2305.02961](https://arxiv.org/abs/2305.02961)).
# >   Repositori: <https://github.com/mrinal054/FUSegNet> (`xfusegnet_all.py`, `segmentation_models_pytorch/decoders/unet/decoder.py`,
# >   `segmentation_models_pytorch/base/modules.py`, `segmentation_models_pytorch/utils/*`).
# > * **Fondasi:** *Segmentation Models PyTorch* — Pavel Iakubovskii (qubvel), <https://github.com/qubvel/segmentation_models.pytorch>.
# > * **Lisensi:** repositori penulis memakai **MIT License**. Lisensi tersebut mewajibkan *copyright notice* dan *permission notice*


# ## 🧰 Bagian 1 — Setup & Import


# ==============================================================================
# 1. IMPORT  (identik dengan xfusegnet_all.py, + tambahan yang dibutuhkan notebook)
# ==============================================================================
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import KFold
from copy import deepcopy
import pickle

# ---- tambahan untuk notebook / lapisan kompatibilitas ----
import re
import sys
import json
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from functools import partial
from typing import Optional
from torch.nn.modules.loss import _Loss
from tqdm import tqdm
import pandas as pd
from PIL import Image

print('torch          :', torch.__version__)
print('smp            :', smp.__version__)
print('albumentations :', A.__version__)
print('opencv         :', cv2.__version__)
print('numpy          :', np.__version__)
print('CUDA tersedia  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))

# ==============================================================================
# 1b. LAPISAN KOMPATIBILITAS `smp.utils`
#     Disalin dari repo penulis: segmentation_models_pytorch/utils/{base,functional,losses,metrics,meter,train}.py
#
#   Kenapa disalin?  xfusegnet_all.py memakai `losses.FocalLoss()` dan `base.SumOfLosses`.
#   `FocalLoss` pada smp.utils adalah TAMBAHAN penulis (smp resmi tidak punya), sedangkan smp.utils
#   sendiri berstatus deprecated. Salinan ini menjamin perilaku loss / metrik / epoch-runner identik.
# ==============================================================================

# ------------------------------------ base.py ---------------------------------
class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass


class Loss(BaseObject):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        name = "{} + {}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    def __init__(self, loss, multiplier):
        # resolve name
        if len(loss.__name__.split("+")) > 1:
            name = "{} * ({})".format(multiplier, loss.__name__)
        else:
            name = "{} * {}".format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)


# ---------------------------------- functional.py -----------------------------
def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Intersection over Union between ground truth and prediction."""
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """F-score (beta=1 -> Dice) between ground truth and prediction."""
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)
    return score


smp_F = SimpleNamespace(iou=iou, jaccard=iou, f_score=f_score)


class Activation(nn.Module):
    """Versi ringkas dari smp.base.modules.Activation (hanya yang dipakai di sini)."""
    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        else:
            raise ValueError(f"Activation tidak didukung pada salinan ini: {name}")

    def forward(self, x):
        return self.activation(x)


# ----------------------------- focal loss (functional) ------------------------
def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary focal loss between target and output logits (smp/losses/_functional.py)."""
    target = target.type(output.type())

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"


# ------------------------------------ losses.py -------------------------------
class DiceLoss(Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - smp_F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class FocalLoss(_Loss, Loss):
    def __init__(
        self,
        mode: str = 'binary',
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss (tambahan penulis pada smp.utils.losses)."""
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss


# ------------------------------------ metrics.py ------------------------------
class IoU(Metric):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return smp_F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(Metric):
    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return smp_F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


# ------------------------------------ meter.py --------------------------------
class AverageValueMeter:
    def __init__(self):
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


# ------------------------------------ train.py --------------------------------
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


# ---- namespace agar baris-baris di kode asli (losses.X, base.X, ...) tetap sama ----
base        = SimpleNamespace(BaseObject=BaseObject, Metric=Metric, Loss=Loss,
                              SumOfLosses=SumOfLosses, MultipliedLoss=MultipliedLoss)
losses      = SimpleNamespace(DiceLoss=DiceLoss, FocalLoss=FocalLoss)
smp_metrics = SimpleNamespace(IoU=IoU, Fscore=Fscore)
smp_train   = SimpleNamespace(TrainEpoch=TrainEpoch, ValidEpoch=ValidEpoch)

print('✅ Lapisan kompatibilitas smp.utils siap: base, losses (Dice, Focal), smp_metrics (IoU, Fscore), smp_train')

# ## 🗂️ Bagian 2 — Dataloader, Augmentasi, Split Dataset & Parameter


# ==============================================================================
# 2a. DATALOADER  (identik dengan xfusegnet_all.py)
# ==============================================================================
class Dataset(BaseDataset):
    """Reference: https://github.com/qubvel/segmentation_models.pytorch

    Args:
        list_IDs (list): List of image names with extension
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            list_IDs,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)    # ----------------- pay attention ------------------ #
        mask = mask/255.0   # converting mask to (0 and 1) # ----------------- pay attention ------------------ #
        mask = np.expand_dims(mask, axis=-1)  # adding channel axis # ----------------- pay attention ------------------ #

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

# ==============================================================================
# 2b. AUGMENTATION  (identik dengan xfusegnet_all.py)
# ==============================================================================
def get_training_augmentation():
    train_transform = [

        A.OneOf(
            [
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.4),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=1, border_mode=0), # scale only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0), # rotate only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0), # shift only
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0), # affine transform
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Perspective(p=1),
                A.GaussNoise(p=1),
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.2,
        ),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.2,
        ),

    ]

    return A.Compose(train_transform, p=0.9) # 90% augmentation probability


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(512, 512)
    ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    if x.ndim == 2:  # [NOTEBOOK] guard: beberapa versi cv2/albumentations membuang channel tunggal pada mask
        x = x[..., None]
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

# ==============================================================================
# 2c. SPLIT DATASET  (struktur path mengikuti notebook (fix1) / proyek 'ganta')
#
#   notebook/                     <- notebook ini berada di sini  (BASE_DIR = "..")
#   data/raw/<DATASET_TYPE>/train/{images,labels}
#   data/raw/<DATASET_TYPE>/test/{images,labels}
#   output/{checkpoints,predictions,results}
# ==============================================================================
BASE_DIR = ".."

# Pilihan: 'fuseg_2021' (MICCAI FUSeg 2021, 512x512)  atau  'chronic_wound' (224x224)
DATASET_TYPE = "fuseg_2021"

DATA_DIR     = os.path.join(BASE_DIR, "data")
DATASET_ROOT = os.path.join(DATA_DIR, "raw", DATASET_TYPE)
TRAIN_DIR    = os.path.join(DATASET_ROOT, "train")
TEST_DIR     = os.path.join(DATASET_ROOT, "validation")

OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output_fuseg_2021")
CHECKPOINT_DIR  = os.path.join(OUTPUT_BASE_DIR, "checkpoints")
PREDICTIONS_DIR = os.path.join(OUTPUT_BASE_DIR, "predictions")
RESULTS_DIR     = os.path.join(OUTPUT_BASE_DIR, "results")
for _d in (CHECKPOINT_DIR, PREDICTIONS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --- nama-nama variabel persis seperti di xfusegnet_all.py ---
#%% Load dataset
x_train_dir = x_valid_dir = os.path.join(TRAIN_DIR, 'images')
y_train_dir = y_valid_dir = os.path.join(TRAIN_DIR, 'labels')

x_test_dir = os.path.join(TEST_DIR, 'images')
y_test_dir = os.path.join(TEST_DIR, 'labels')

# --- [NOTEBOOK] pengecekan path supaya pesan error jelas ---
for _p in (x_train_dir, y_train_dir, x_test_dir, y_test_dir):
    if not os.path.isdir(_p):
        _raw = os.path.join(DATA_DIR, "raw")
        _avail = os.listdir(_raw) if os.path.isdir(_raw) else "(folder data/raw tidak ada)"
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {os.path.abspath(_p)}\n"
            f"CWD saat ini      : {os.getcwd()}\n"
            f"Dataset tersedia di {os.path.abspath(_raw)}: {_avail}\n"
            f"-> Periksa BASE_DIR / DATASET_TYPE di sel ini."
        )

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def list_images(folder):
    # [NOTEBOOK] diurutkan + hanya file gambar (hindari .ipynb_checkpoints dsb.)
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))

names = list_images(x_train_dir)

random.seed(200) # seed for random number generator

random.shuffle(names) # shuffle names

list_IDs_kfold = names
list_IDs_test = list_images(x_test_dir)

print('No. of training images:', len(list_IDs_kfold))
print('No. of test images:', len(list_IDs_test))

# --- [NOTEBOOK] setiap citra harus punya mask dengan NAMA FILE YANG SAMA (asumsi Dataset class) ---
for _imgs, _lbl_dir, _tag in ((list_IDs_kfold, y_train_dir, 'train'), (list_IDs_test, y_test_dir, 'test')):
    _missing = [n for n in _imgs if not os.path.exists(os.path.join(_lbl_dir, n))]
    if _missing:
        raise FileNotFoundError(f"{len(_missing)} mask {_tag} tidak ditemukan di {_lbl_dir} (contoh: {_missing[:3]})")

# --- [NOTEBOOK] cek nilai mask: kode asli membagi dengan 255, jadi mask harus bernilai {0, 255} ---
_m = cv2.imread(os.path.join(y_train_dir, list_IDs_kfold[0]), 0)
_i = cv2.imread(os.path.join(x_train_dir, list_IDs_kfold[0]))
print('Contoh gambar   :', list_IDs_kfold[0], '| shape', _i.shape)
print('Nilai unik mask :', np.unique(_m)[:10], '| shape', _m.shape)
if _m.max() <= 1:
    print('⚠️  Mask bernilai {0,1}. Kode asli melakukan mask/255.0 -> mask menjadi ~0 semua! '
          'Ubah label ke {0,255} atau sesuaikan Dataset.__getitem__.')
if _i.shape[0] % 32 or _i.shape[1] % 32:
    print('⚠️  Ukuran gambar bukan kelipatan 32. Kode asli tidak melakukan resize/padding.')

# ==============================================================================
# 2d. PARAMETERS  (identik dengan xfusegnet_all.py)
# ==============================================================================
#%% Parameters
BASE_MODEL = 'xFUSegNet'
ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 2
n_classes = 1
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001 # learning rate
EPOCHS = 200
WEIGHT_DECAY = 1e-5
SAVE_WEIGHTS_ONLY = True
TO_CATEGORICAL = False
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = False
PERIOD = None # periodically save checkpoints
RAW_PREDICTION = False # if true, then stores raw predictions (i.e. before applying threshold)
PATIENCE = 30 # for early stopping
EARLY_STOP = True

# ------------------------------------------------------------------------------
# [NOTEBOOK] Mode eksekusi.
#   None                      -> training BARU (lalu validation + test)
#   'xFUSegNet_efficientnet-b7_YYYY-mm-dd_HH-MM-SS'  (nama folder di output/checkpoints)
#                             -> LEWATI training, langsung test memakai checkpoint tsb
# ------------------------------------------------------------------------------
RESUME_MODEL_NAME = None

#%% Create a unique model name
if RESUME_MODEL_NAME is None:
    model_name = BASE_MODEL + '_' + ENCODER + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
else:
    model_name = RESUME_MODEL_NAME
RUN_TRAINING = RESUME_MODEL_NAME is None
print(model_name)
print('RUN_TRAINING:', RUN_TRAINING, '| DEVICE:', DEVICE)

# ## 🧠 Bagian 3 — Build Model (P-scSE), Loss & Metrik
# 
# Di kode asli, model dibuat dengan `smp.Unet(..., decoder_attention_type='pscse')` memakai **smp versi modifikasi penulis** (v0.3.0).
# Pada smp resmi (lingkungan ini: 0.6.0) opsi `'pscse'` tidak ada, sehingga **decoder P-scSE disalin verbatim** dari
# `decoders/unet/decoder.py` dan `base/modules.py` milik penulis, lalu dipasang ke `smp.Unet` (encoder EfficientNet-B7 + segmentation head tetap dari smp).
# Nama submodul / kunci `state_dict` (`decoder.blocks.N.conv1`, `attention1`, `attention2`, ...) sama dengan versi penulis.


# ==============================================================================
# 3a. BUILD MODEL
# ==============================================================================
#%%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#%% Checkpoint directory
checkpoint_loc = os.path.join(CHECKPOINT_DIR, model_name)

# Create checkpoint directory if does not exist
if not os.path.exists(checkpoint_loc): os.makedirs(checkpoint_loc)

#%% Helper function: save a model

def save(model_path, epoch, model_state_dict, optimizer_state_dict):

    state = {
        'epoch': epoch + 1,
        'state_dict': deepcopy(model_state_dict),
        'optimizer': deepcopy(optimizer_state_dict),
        }

    torch.save(state, model_path)


#%% ---- P-scSE decoder: salinan verbatim dari repo penulis ----
# (segmentation_models_pytorch/base/modules.py  &  decoders/unet/decoder.py)

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


# Original SCSE module with addition type only
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


# Modified SCSE module supporing different strategies
class SCSEModule0(nn.Module):
    def __init__(self, in_channels, reduction=16, strategy='addition'):
        super().__init__()
        self.strategy = strategy
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            # nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        self.if_concat = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):

        xc = x * self.cSE(x) # cSE attention
        xs = x * self.sSE(x) # sSE attention

        if self.strategy == 'addition':
            x = xc + xs

        elif self.strategy == 'maxout':
            x = torch.maximum(xc, xs)

        elif self.strategy == 'concat':
            x = torch.cat([xc, xs], dim=1)
            x = self.if_concat(x)

        elif self.strategy == 'multiplication':
            x = xc * xs

        elif self.strategy == 'average':
            x = torch.stack((xc, xs), dim=0)
            x = torch.mean(x, dim=0)

        elif self.strategy == 'all-average':
            x1 = xc + xs # addition
            x2 = torch.maximum(xc, xs) # maxout
            x3 = self.if_concat(torch.cat([xc, xs], dim=1)) # concatenation
            x4 = xc * xs # multiplication

            # Take average of above mentioned strategies
            x = torch.stack((x1, x2, x3, x4), dim=0)
            x = torch.mean(x, dim=0)

        else:
            raise ValueError("Wrong keyword for attention strategy. Choose from [addition, maxout, concat, multiplication, average, all-average]")

        return x


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


md = SimpleNamespace(Conv2dReLU=Conv2dReLU, SCSEModule0=SCSEModule0, Attention=Attention)  # padanan `md.` di decoder.py


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()

        self.attention_type = attention_type

        self.conv1 = md.Conv2dReLU(
            # 2 * (in_channels + skip_channels), # -------->> use for concatenation <<----------
            in_channels + skip_channels, # -------->> use for addition <<----------
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # P-scSE
        if self.attention_type == 'pscse':
            self.attention1 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='maxout')
            self.attention2 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='addition')

        # scSE
        elif self.attention_type == 'scse':
            self.attention1 = md.Attention("scse", in_channels=in_channels + skip_channels)
            self.attention2 = md.Attention("scse", in_channels=out_channels)

        # Single attention (as per the original paper -> attention is applied at the end)
        elif self.attention_type in ('maxout', 'additive', 'concat', 'multiplication', 'average', 'all-average'):
            self.attention = md.SCSEModule0(in_channels=out_channels, strategy=self.attention_type)

        else:
            raise ValueError("Wrong keyword for attention. Choose either of - pscse, scse, \
                             maxout, additive, concat, multiplication, average, all-average")

    def forward(self, x, skip=None, strategy='add'):

        # For P-scSE
        if self.attention_type == 'pscse':
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
                x1 = self.attention1(x) # 1st attention
                x2 = self.attention2(x) # 2nd attention

                if strategy == 'cat': x3 = torch.cat([x1, x2], dim=1) # concatenate two attentions
                elif strategy == 'add': x3 = x1 + x2

            else:
                x2 = self.attention2(x) # 2nd attention
                if strategy == 'cat': x3 = torch.cat([x, x2], dim=1) # concatenate x and x2
                elif strategy == 'add': x3 = x + x2

            x3 = self.conv1(x3)

            return x3

        # For scSE
        elif self.attention_type == 'scse':
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
                x = self.attention1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.attention2(x)
            return x

        # Single attention (as per the original paper -> attention is applied at the end)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.attention(x)
            return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        # [NOTEBOOK] smp 0.3.x memanggil decoder(*features), smp >= 0.4 memanggil decoder(features) -> dukung keduanya
        if len(features) == 1 and isinstance(features[0], (list, tuple)):
            features = features[0]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


# initialization.py (penulis / smp)
def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def build_model(encoder_weights=ENCODER_WEIGHTS):
    """Padanan:  smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=n_classes,
                          activation=ACTIVATION, decoder_attention_type='pscse')   (smp modifikasi penulis)"""
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=encoder_weights,
        classes=n_classes,
        activation=ACTIVATION,
        decoder_attention_type=None,      # smp resmi tidak mengenal 'pscse' -> decoder diganti di bawah
    )
    model.decoder = UnetDecoder(
        encoder_channels=model.encoder.out_channels,
        decoder_channels=(256, 128, 64, 32, 16),
        n_blocks=5,
        use_batchnorm=True,
        attention_type='pscse',
        center=False,
    )
    initialize_decoder(model.decoder)     # sama dengan SegmentationModel.initialize() pada smp penulis
    return model

print('✅ build_model() siap (EfficientNet-B7 encoder + decoder P-scSE)')

# ==============================================================================
# 3b. LOSS AND METRICS  (identik dengan xfusegnet_all.py)
# ==============================================================================
#%% Loss and metrics
# Loss function
dice_loss = losses.DiceLoss()
focal_loss = losses.FocalLoss()

total_loss = base.SumOfLosses(dice_loss, focal_loss)

# Metrics
metrics = [
    smp_metrics.IoU(threshold=0.5),
    smp_metrics.Fscore(threshold=0.5),
]
print('Loss    :', total_loss.__name__)
print('Metrics :', [m.__name__ for m in metrics])

# ==============================================================================
# 3c. SMOKE TEST (OPSIONAL tetapi disarankan) — memastikan semua komponen jalan SEBELUM training panjang
#     Tidak mengubah state training. Memakai encoder_weights=None (tanpa unduhan) dan 2 sampel saja.
# ==============================================================================
_sm_model = build_model(encoder_weights=None).to(DEVICE)
_n_params = sum(p.numel() for p in _sm_model.parameters()) / 1e6
print(f'Jumlah parameter model : {_n_params:.2f} M')

_pre = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
_ds = Dataset(list_IDs_kfold[:4], x_train_dir, y_train_dir,
              augmentation=get_training_augmentation(), preprocessing=get_preprocessing(_pre))
_batch = [_ds[i] for i in range(2)]
_x = torch.from_numpy(np.stack([b[0] for b in _batch]))
_y = torch.from_numpy(np.stack([b[1] for b in _batch]))
print('batch image', tuple(_x.shape), _x.dtype, '| batch mask', tuple(_y.shape), _y.dtype,
      '| nilai mask unik:', torch.unique(_y).tolist())
assert _x.ndim == 4 and _x.shape[1] == 3 and _y.shape[1] == 1, 'bentuk batch tidak sesuai'

_sm_model.train()
_out = _sm_model(_x.to(DEVICE))
print('output model', tuple(_out.shape), '| min/max', float(_out.min()), float(_out.max()))
assert _out.shape == _y.shape, 'output model dan mask tidak sama bentuknya'
assert 0.0 <= float(_out.min()) and float(_out.max()) <= 1.0, 'output bukan sigmoid [0,1]'

_loss = total_loss(_out, _y.to(DEVICE))
_scores = {m.__name__: float(m(_out, _y.to(DEVICE))) for m in metrics}
print('loss', float(_loss), '| metrik', _scores)
assert torch.isfinite(_loss), 'loss tidak finite (NaN/Inf)'

_loss.backward()
_blk = _sm_model.decoder.blocks[0]
print('grad conv1 ada :', _blk.conv1[0].weight.grad is not None,
      '| grad conv2 ada (di kode asli memang TIDAK dipakai pada mode pscse):', _blk.conv2[0].weight.grad is not None)
assert _blk.conv1[0].weight.grad is not None

print('Contoh kunci state_dict decoder:', [k for k in _sm_model.state_dict().keys() if k.startswith('decoder.blocks.0')][:6])

del _sm_model, _ds, _batch, _x, _y, _out, _loss, _blk
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print('✅ Smoke test lulus — aman untuk lanjut ke training.')

# ## 🚀 Bagian 4 — Training & Validation (5-Fold Cross Validation)
# 
# Isi sel di bawah adalah loop `for current_split ...` dari `xfusegnet_all.py` apa adanya.
# Setiap epoch menjalankan `TrainEpoch` lalu `ValidEpoch` (itulah tahap **validation**); checkpoint terbaik disimpan sebagai
# `best_model_1.pth … best_model_5.pth` pada `../output/checkpoints/<model_name>/`.
# 
# > Baris bertanda `# [TAMBAHAN]` bukan bagian kode asli (menyimpan histori fold, menampilkan plot, membebaskan memori GPU).
# > Jika `RUN_TRAINING = False` (karena `RESUME_MODEL_NAME` diisi), sel ini otomatis dilewati.


# ==============================================================================
# 4. TRAIN MODELS (5-fold)   — identik dengan xfusegnet_all.py
# ==============================================================================
#%%
n_split = 5 # no. of split

store_best_model_epoch = []

store_best_val_loss = []

store_kfold_best_log = []

fold_histories = {}  # [TAMBAHAN] kurva tiap fold, dipakai Bagian 5

#%% Train models
kf = KFold(n_splits=n_split, shuffle=False) # already shuffled

if not RUN_TRAINING:
    print(f'RUN_TRAINING=False -> training dilewati. Checkpoint dipakai dari: {checkpoint_loc}')
else:
  for current_split, (train_idx, val_idx) in enumerate(kf.split(list_IDs_kfold)):

    print(f'****************** current_split: {current_split} ***********************')

    # create segmentation model with pretrained encoder
    model = build_model()   # padanan: smp.Unet(..., decoder_attention_type='pscse')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
    ])

    # Learning rate scheduler
    _sched_kwargs = dict(factor=0.1, mode='min', patience=10, min_lr=0.00001)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, **_sched_kwargs)
    except TypeError:  # torch versi baru yang sudah menghapus argumen `verbose`
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **_sched_kwargs)

    # Get training and validation names
    list_IDs_train = [list_IDs_kfold[idx] for idx in train_idx]
    list_IDs_val = [list_IDs_kfold[idx] for idx in val_idx]

    # Prepare dataloader
    train_dataset = Dataset(
        list_IDs_train,
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        list_IDs_val,
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    """## Training"""

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp_train.TrainEpoch(
        model,
        loss=total_loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp_train.ValidEpoch(
        model,
        loss=total_loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for N epochs
    best_viou = 0.0
    best_vloss = 1_000_000.
    save_model = False
    cnt_patience = 0

    store_train_loss, store_val_loss = [], []
    store_train_iou, store_val_iou = [], []
    store_train_dice, store_val_dice = [], []

    for epoch in range(EPOCHS):

        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Store losses and metrics
        train_loss_key = list(train_logs.keys())[0] # first key is for loss
        val_loss_key = list(valid_logs.keys())[0] # first key is for loss

        store_train_loss.append(train_logs[train_loss_key])
        store_val_loss.append(valid_logs[val_loss_key])
        store_train_iou.append(train_logs["iou_score"])
        store_val_iou.append(valid_logs["iou_score"])
        store_train_dice.append(train_logs["fscore"])
        store_val_dice.append(valid_logs["fscore"])

        # Track best performance, and save the model's state
        if  best_vloss > valid_logs[val_loss_key]:
            best_vloss = valid_logs[val_loss_key]
            if SAVE_BEST_MODEL: print(f'Validation loss reduced. Saving the model at epoch: {epoch:04d}')
            cnt_patience = 0 # reset patience
            best_model_epoch = epoch
            save_model = True

        # Compare iou score
        elif best_viou < valid_logs['iou_score']:
            best_viou = valid_logs['iou_score']
            if SAVE_BEST_MODEL: print(f'Validation IoU increased. Saving the model at epoch: {epoch:04d}.')
            cnt_patience = 0 # reset patience
            best_model_epoch = epoch
            save_model = True

        else: cnt_patience += 1

        # Learning rate scheduler
        scheduler.step(valid_logs[sorted(valid_logs.keys())[0]]) # monitor validation loss

        # Save the model
        if SAVE_BEST_MODEL and save_model:
            save(os.path.join(checkpoint_loc, 'best_model_' + str(current_split+1) + '.pth'),
                 epoch+1, model.state_dict(), optimizer.state_dict())
            save_model = False

        # Early stopping
        if EARLY_STOP and cnt_patience >= PATIENCE:
          print(f"Early stopping at epoch: {epoch:04d} for current split {current_split:02d}")
          break

        # Periodic checkpoint save
        if not SAVE_BEST_MODEL and PERIOD is not None:
          if (epoch+1) % PERIOD == 0:
            save(os.path.join(checkpoint_loc, str(current_split+1) + f"_cp-{epoch+1:04d}.pth"),
                 epoch+1, model.state_dict(), optimizer.state_dict())
            print(f'Checkpoint saved for epoch {epoch:04d} when current split is {current_split:02d}')

    if not EARLY_STOP and SAVE_LAST_MODEL:
        print('Saving last model')
        save(os.path.join(checkpoint_loc, 'last_model_' + str(current_split+1) + '.pth'),
             epoch+1, model.state_dict(), optimizer.state_dict())

    # sorted(valid_logs.keys())
    store_kfold_best_log.append([current_split, min(store_val_loss), best_model_epoch])

    """## Plotting """
    fig, ax = plt.subplots(3,1, figsize=(7, 14))

    ax[0].plot(store_train_loss, 'r')
    ax[0].plot(store_val_loss, 'b')
    ax[0].set_title('Loss curve')
    ax[0].legend(['training', 'validation'])

    ax[1].plot(store_train_iou, 'r')
    ax[1].plot(store_val_iou, 'b')
    ax[1].set_title('IoU curve')
    ax[1].legend(['training', 'validation'])

    ax[2].plot(store_train_iou, 'r')
    ax[2].plot(store_val_iou, 'b')
    ax[2].set_title('Dice curve')
    ax[2].legend(['training', 'validation'])

    fig.tight_layout()

    # plt.show()

    save_fig_dir = os.path.join(RESULTS_DIR, "plots")
    if not os.path.exists(save_fig_dir): os.makedirs(save_fig_dir)

    fig.savefig(os.path.join(save_fig_dir, model_name + '_' + str(current_split+1) + '.png'))

    # ---------------- [TAMBAHAN] bukan bagian kode asli ----------------
    plt.show()
    fold_histories[current_split] = dict(
        train_loss=store_train_loss, val_loss=store_val_loss,
        train_iou=store_train_iou, val_iou=store_val_iou,
        train_dice=store_train_dice, val_dice=store_val_dice,
        best_epoch=best_model_epoch,
    )
    with open(os.path.join(checkpoint_loc, 'fold_histories.pickle'), 'wb') as f:
        pickle.dump(fold_histories, f)
    del model, optimizer, scheduler, train_epoch, valid_epoch, train_loader, valid_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # -------------------------------------------------------------------

  #%% Save store_kfold_best_log
  with open(os.path.join(checkpoint_loc, model_name + '.pickle'), 'wb') as f:
    pickle.dump(store_kfold_best_log, f)

# ## 📉 Bagian 5 — Ringkasan Validasi (tambahan, hanya membaca hasil training)
# Menampilkan *best validation loss* & epoch terbaik tiap fold dari `<model_name>.pickle`, serta kurva Loss / IoU / **Dice** (yang benar) tiap fold
# dari `fold_histories.pickle` (hanya ada bila training dijalankan memakai notebook ini).


# ==============================================================================
# 5. RINGKASAN VALIDASI  (tambahan)
# ==============================================================================
_log_path = os.path.join(checkpoint_loc, model_name + '.pickle')
if not os.path.exists(_log_path):
    print('⚠️ Belum ada log k-fold:', _log_path, '-> jalankan training dulu atau isi RESUME_MODEL_NAME.')
else:
    with open(_log_path, 'rb') as f:
        _kfold_log = pickle.load(f)
    df_val = pd.DataFrame(_kfold_log, columns=['fold_idx(0-based)', 'min_val_loss', 'best_epoch(0-based)'])
    df_val['checkpoint'] = [f'best_model_{int(i)+1}.pth' for i in df_val['fold_idx(0-based)']]
    print(df_val.to_string(index=False))
    print(f"\nRata-rata min val loss : {df_val['min_val_loss'].mean():.4f}  |  std : {df_val['min_val_loss'].std(ddof=0):.4f}")
    df_val.to_csv(os.path.join(RESULTS_DIR, model_name + '_kfold_summary.csv'), index=False)

    _hist_path = os.path.join(checkpoint_loc, 'fold_histories.pickle')
    if os.path.exists(_hist_path):
        with open(_hist_path, 'rb') as f:
            _hist = pickle.load(f)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        for k, h in sorted(_hist.items()):
            c = plt.cm.tab10(k % 10)
            for ax, key, ttl in zip(axes, ('loss', 'iou', 'dice'), ('Loss', 'IoU', 'Dice (F-score)')):
                ax.plot(h['train_' + key], color=c, ls='--', alpha=0.6)
                ax.plot(h['val_' + key], color=c, label=f'fold {k+1} (val)')
                ax.set_title(ttl + '  (putus-putus = train, tebal = validation)')
                ax.set_xlabel('epoch'); ax.grid(True, ls=':', alpha=0.5)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, model_name + '_val_curves.png'), dpi=150)
        plt.show()
    else:
        print('ℹ️ fold_histories.pickle tidak ada (training tidak dijalankan lewat notebook ini) -> kurva dilewati.')

# ## 🔬 Bagian 6 — Test: Inferensi Ensemble (Average & Voting)
# Mengikuti bagian `INFERENCE` pada `xfusegnet_all.py`: lima checkpoint `best_model_1..5.pth` memprediksi seluruh citra test, lalu
# digabung dengan **rata-rata probabilitas** (`_avg`) dan **majority voting** (`_voting`), threshold 0.5.
# Hasil PNG disimpan di `../output/predictions/<model_name>_avg` dan `..._voting`.


# ==============================================================================
# 6a. INFERENCE — siapkan log, preprocessing, dataloader test  (identik dengan xfusegnet_all.py)
# ==============================================================================
'=================================== INFERENCE ================================='
#%% Inference
RAW_PREDICTION = False # if true, then stores raw predictions (i.e. before applying threshold)
save_pred = True
verbose = False
threshold = 0.5
ep = 1e-6

# Get checkpoint directory
# model_name = '>>>>>>>>>>>>>>>>Give name<<<<<<<<<<<<<<<<<<<<<<'   (di notebook: atur RESUME_MODEL_NAME)
checkpoint_loc = os.path.join(CHECKPOINT_DIR, model_name)

with open(os.path.join(checkpoint_loc, model_name + '.pickle'), 'rb') as f:
    store_kfold_best_log = pickle.load(f)

get_val_loss = []
get_epochs = []
for row in store_kfold_best_log:
    get_val_loss.append(row[1])
    get_epochs.append(row[2])

min_val_loss_idx = np.argmin(get_val_loss)
max_val_loss_idx = np.argmax(get_val_loss)

max_epoch = np.max(get_epochs)
avg_epoch = int(np.mean(get_epochs))

ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'

# Model names
models = [
    'best_model_1.pth',
    'best_model_2.pth',
    'best_model_3.pth',
    'best_model_4.pth',
    'best_model_5.pth',
    ]

# [NOTEBOOK] semua checkpoint harus ada sebelum inferensi dimulai
_absent = [m for m in models if not os.path.exists(os.path.join(checkpoint_loc, m))]
if _absent:
    raise FileNotFoundError(f'Checkpoint tidak lengkap di {checkpoint_loc}: {_absent}')

def load_checkpoint(path):
    # [NOTEBOOK] PyTorch >= 2.6: default weights_only=True; checkpoint ini memuat state optimizer -> butuh False
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # torch < 1.13 tidak mengenal `weights_only`
        return torch.load(path, map_location=DEVICE)

'===================================== VOTING/AVG ========================================='
#%% Prediction based on voting
store_pred = []

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Test dataloader
test_dataset = Dataset(
    list_IDs_test,
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=2)
print('Siap: ', len(list_IDs_test), 'citra test |', len(models), 'model')

# ==============================================================================
# 6b. PREDIKSI TIAP MODEL FOLD  (identik dengan xfusegnet_all.py)
# ==============================================================================
print('\n')
for m_name in models:

    print(f"Generating prediction for {m_name}")

    # create segmentation model with pretrained encoder
    model = build_model()   # padanan: smp.Unet(..., decoder_attention_type='pscse')

    model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
    ])

    # Load model
    checkpoint = load_checkpoint(os.path.join(checkpoint_loc, m_name))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    iter_test_dataloader = iter(test_dataloader)

    raw_pred = []

    for i in range(len(list_IDs_test)):

        name = os.path.splitext(list_IDs_test[i])[0] # remove extension

        image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

        # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])

        pr_mask = model.predict(image.to(DEVICE)) # Move image tensor to gpu

        # Move to CPU and convert to numpy
        pred = pr_mask.squeeze().cpu().numpy()

        # Save raw prediction
        raw_pred.append(pred)

    store_pred.append(raw_pred)

    # [TAMBAHAN] bebaskan memori GPU sebelum model berikutnya
    del model, optimizer, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

raw_preds_all = list(store_pred)  # [TAMBAHAN] salinan referensi untuk analisis per-fold (Bagian 7)
print('\nSelesai. Prediksi mentah terkumpul untuk', len(store_pred), 'model x', len(list_IDs_test), 'citra.')

# ==============================================================================
# 6c. AVERAGE  (identik dengan xfusegnet_all.py)
# ==============================================================================
'--------------------------------- Average ------------------------------------'
#%% Average
print('\n\nGenerating output taking average of all raw predictions')
# Save directory
save_dir_pred = os.path.join(PREDICTIONS_DIR, model_name + '_avg')
if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)

iter_test_dataloader = iter(test_dataloader)

for i in range(len(list_IDs_test)):

    name = os.path.splitext(list_IDs_test[i])[0] # remove extension

    image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

    # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])

    temp_store = []
    for j in range(len(store_pred)): # iterate over all models
        curr_pred = store_pred[j][i] # get jth model's prediction for the current image
        temp_store.append(curr_pred)

    temp_store = np.array(temp_store)

    temp_store = np.mean(temp_store, axis=0) # take mean along axis=0

    pred = (temp_store >= threshold) * 1

    gt_mask = gt_mask.squeeze().cpu().numpy()

    # Save prediction as png
    if save_pred:
        output_im = Image.fromarray((np.squeeze(pred)*255 ).astype(np.uint8))
        output_im.save(os.path.join(save_dir_pred, list_IDs_test[i]))

print('✅ Tersimpan di:', os.path.abspath(save_dir_pred))

# ==============================================================================
# 6d. VOTING  (identik dengan xfusegnet_all.py)
# ==============================================================================
'--------------------------------- voting ------------------------------------'
#%% Voting
print('\n\nGenerating output based on voting.')

# Note: voting is easier if there are odd no. of models.
# So, if there are even no. of models, we will remove that k-fold model which has max val loss
if len(store_pred) % 2 == 0:
    print(f"Found even no. of models for voting. So, removing best_model_{max_val_loss_idx} from k-fold models")
    # store_pred.remove(store_pred[max_val_loss_idx + 1]) # adding one because store_pred has prediction for last_model
                                            # which is not in k-fold.
    store_pred_temp = []
    for pc in range(len(store_pred)):
        if pc != max_val_loss_idx + 1:
            store_pred_temp.append(store_pred[pc])
    del store_pred
    store_pred = store_pred_temp
    del store_pred_temp

threshold_sum = int(np.ceil(len(store_pred)/2)) # considering odd numbers of models
# Note: If no. of models = 5, then threshold_sum = 3.
# So, consider for binary segmentation,
# if sum of the values of a pixel located at (x,y) position for all 5 prediction is 3, it means
# that three predictions voted 1 and two predictions voted 0. So, 1 is winner.
# Finally, the value of that pixel at (x,y) will be set to 1.

# Save directory
save_dir_pred = os.path.join(PREDICTIONS_DIR, model_name + '_voting')
if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)

iter_test_dataloader = iter(test_dataloader)

for i in range(len(list_IDs_test)):

    name = os.path.splitext(list_IDs_test[i])[0] # remove extension

    image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

    # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])

    temp_store = []
    for j in range(len(store_pred)): # iterate over all models
        curr_pred = store_pred[j][i] # get jth model's prediction for the current image
        curr_pred = (curr_pred >= threshold) * 1 # converting to binary image
        temp_store.append(curr_pred)

    temp_store = np.array(temp_store)

    temp_store = np.sum(temp_store, axis=0) # take sum along axis=0

    pred = (temp_store >= threshold_sum) * 1

    gt_mask = gt_mask.squeeze().cpu().numpy()

    # Save prediction as png
    if save_pred:
        output_im = Image.fromarray((np.squeeze(pred)*255 ).astype(np.uint8))
        output_im.save(os.path.join(save_dir_pred, list_IDs_test[i]))

print('✅ Tersimpan di:', os.path.abspath(save_dir_pred))

# ## 📊 Bagian 7 — Analysis (tambahan, hanya membaca hasil)
# Kode asli hanya **menyimpan** prediksi dan tidak menghitung skor. Bagian ini menghitung metrik pada set test dengan membaca PNG prediksi
# (`_avg`, `_voting`) dan *ground truth*, plus metrik tiap model fold (bila prediksi mentah masih ada di memori kernel).
# * **Data-based**: TP/FP/FN dijumlahkan di seluruh piksel dataset. **Image-based**: skor per citra lalu dirata-rata.
# * Dice = 2TP / (2TP+FP+FN) · IoU = TP / (TP+FP+FN).


# ==============================================================================
# 7a. METRIK TEST  (tambahan)
# ==============================================================================
def load_gt(name):
    m = cv2.imread(os.path.join(y_test_dir, name), 0)
    return ((m / 255.0) >= 0.5).astype(np.uint8)          # sama dengan perlakuan mask di Dataset

def load_pred_png(folder, name):
    p = cv2.imread(os.path.join(folder, name), 0)
    return (p > 127).astype(np.uint8)

def confusion(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    return tp, fp, fn

def scores_from_counts(tp, fp, fn, eps=1e-7):
    return dict(
        dice=(2*tp + eps) / (2*tp + fp + fn + eps),
        iou=(tp + eps) / (tp + fp + fn + eps),
        precision=(tp + eps) / (tp + fp + eps),
        recall=(tp + eps) / (tp + fn + eps),
    )

gts = [load_gt(n) for n in list_IDs_test]

def evaluate(get_pred):
    TP = FP = FN = 0
    per_dice, per_iou = [], []
    for i, n in enumerate(list_IDs_test):
        pr, gt = get_pred(i, n), gts[i]
        assert pr.shape == gt.shape, f'shape prediksi {pr.shape} != GT {gt.shape} untuk {n}'
        tp, fp, fn = confusion(pr, gt)
        TP += tp; FP += fp; FN += fn
        s = scores_from_counts(tp, fp, fn)
        per_dice.append(s['dice']); per_iou.append(s['iou'])
    d = scores_from_counts(TP, FP, FN)
    return {
        'Data Dice (%)': d['dice']*100, 'Data IoU (%)': d['iou']*100,
        'Data Prec (%)': d['precision']*100, 'Data Rec (%)': d['recall']*100,
        'Image Dice (%)': float(np.mean(per_dice))*100, 'Image IoU (%)': float(np.mean(per_iou))*100,
    }

dir_avg  = os.path.join(PREDICTIONS_DIR, model_name + '_avg')
dir_vote = os.path.join(PREDICTIONS_DIR, model_name + '_voting')

rows = []
if 'raw_preds_all' in globals():
    for j in range(len(raw_preds_all)):
        r = evaluate(lambda i, n, j=j: (raw_preds_all[j][i] >= threshold).astype(np.uint8))
        rows.append({'Model': f'FUSegNet fold {j+1} (single)', **r})
else:
    print('ℹ️ Prediksi mentah per-fold tidak ada di memori (kernel di-restart) -> baris single-model dilewati.')

rows.append({'Model': 'x-FUSegNet (Average)', **evaluate(lambda i, n: load_pred_png(dir_avg, n))})
rows.append({'Model': 'x-FUSegNet (Voting)',  **evaluate(lambda i, n: load_pred_png(dir_vote, n))})

df_test = pd.DataFrame(rows)
df_test.to_csv(os.path.join(RESULTS_DIR, model_name + '_test_metrics.csv'), index=False)

print(f'Model: {model_name} | Dataset: {DATASET_TYPE} | Citra test: {len(list_IDs_test)}\n')
print(df_test.round(2).to_string(index=False))
print('\nTersimpan:', os.path.abspath(os.path.join(RESULTS_DIR, model_name + '_test_metrics.csv')))

# ==============================================================================
# 7b. ERROR MAP & KONTUR  (tambahan)   TP = hijau · FP = merah · FN = biru
# ==============================================================================
def error_map(pred, gt):
    em = np.zeros((*gt.shape, 3), dtype=np.uint8)
    em[(pred == 1) & (gt == 1)] = (0, 255, 0)     # TP
    em[(pred == 1) & (gt == 0)] = (255, 0, 0)     # FP
    em[(pred == 0) & (gt == 1)] = (0, 0, 255)     # FN
    return em

def with_contours(img, pred, gt):
    canvas = img.copy()
    cg, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cp, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cg, -1, (255, 0, 0), 2)   # merah = GT
    cv2.drawContours(canvas, cp, -1, (0, 255, 0), 2)   # hijau = prediksi
    return canvas

N_VIS = min(4, len(list_IDs_test))
idxs = np.linspace(0, len(list_IDs_test) - 1, N_VIS, dtype=int)

fig, axes = plt.subplots(N_VIS, 4, figsize=(16, 4 * N_VIS))
axes = np.atleast_2d(axes)
for r, i in enumerate(idxs):
    n = list_IDs_test[i]
    img = cv2.cvtColor(cv2.imread(os.path.join(x_test_dir, n)), cv2.COLOR_BGR2RGB)
    pr = load_pred_png(dir_avg, n)
    gt = gts[i]
    tp, fp, fn = confusion(pr, gt)
    s = scores_from_counts(tp, fp, fn)
    axes[r, 0].imshow(img);                     axes[r, 0].set_title(n[:24], fontsize=9)
    axes[r, 1].imshow(gt, cmap='gray');         axes[r, 1].set_title('Ground truth', fontsize=9)
    axes[r, 2].imshow(error_map(pr, gt));       axes[r, 2].set_title(f"Error map (TP=G, FP=R, FN=B) | Dice {s['dice']*100:.1f}%", fontsize=9)
    axes[r, 3].imshow(with_contours(img, pr, gt)); axes[r, 3].set_title('Kontur: merah=GT, hijau=Avg', fontsize=9)
    for a in axes[r]: a.axis('off')
fig.suptitle(f'x-FUSegNet (Average) — {model_name}', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, model_name + '_error_maps.png'), dpi=150)
plt.show()

# ==============================================================================
# 7c. RINGKASAN RUN + REFERENSI PENULIS  (tambahan)
# ==============================================================================
# Angka rujukan dari README repositori penulis (https://github.com/mrinal054/FUSegNet):
#   - Chronic Wound dataset : FUSegNet (model tunggal)  data-based Dice 92.70 %
#   - MICCAI FUSeg 2021     : x-FUSegNet (ensemble)     Dice 89.23 % (dievaluasi di server challenge)
# Catatan: skor di atas dihitung pada set test resmi; split test lokal Anda bisa saja berbeda, jadi bandingkan secara kasar.
_ref = {'chronic_wound': ('FUSegNet (single)', 92.70), 'fuseg_2021': ('x-FUSegNet (ensemble)', 89.23)}
_best = df_test[df_test['Model'].str.contains('Average|Voting')]['Data Dice (%)'].max()
print(f'Dice data-based terbaik ensemble (notebook ini) : {_best:.2f} %')
if DATASET_TYPE in _ref:
    print(f'Rujukan README penulis untuk {DATASET_TYPE:<14}        : {_ref[DATASET_TYPE][1]:.2f} %  [{_ref[DATASET_TYPE][0]}]')

run_info = dict(
    model_name=model_name, dataset_type=DATASET_TYPE, n_train=len(list_IDs_kfold), n_test=len(list_IDs_test),
    encoder=ENCODER, encoder_weights=ENCODER_WEIGHTS, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS, patience=PATIENCE, threshold=float(threshold), n_splits=n_split,
    torch=torch.__version__, smp=smp.__version__, albumentations=A.__version__,
    resumed_from_existing_checkpoints=not RUN_TRAINING,
)
with open(os.path.join(RESULTS_DIR, model_name + '_run_info.json'), 'w') as f:
    json.dump(run_info, f, indent=2)
print('run_info tersimpan di', os.path.abspath(os.path.join(RESULTS_DIR, model_name + '_run_info.json')))