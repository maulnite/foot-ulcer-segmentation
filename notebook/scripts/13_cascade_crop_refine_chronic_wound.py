# Generated from: 13_cascade_crop_refine_chronic_wound.ipynb
# Converted at: 2026-11-14T00:54:11.231Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # M3: Cascade Crop-Refine (coarse lalu refiner pada crop yang di-zoom)
# 
# Dasar: FUSegNet (Dhar et al.) | dataset: chronic_wound 224x224 | protokol: train dipecah 85/15 per ID foto (seed 42), evaluasi pada folder `test` (278 citra)
# 
# **Hipotesis (M3):** luka kecil lebih baik disegmentasi jika dilihat dari dekat. Pola ini dipakai pemenang DFUC 2022, HRDecoder, dan DCSNet.
# 
# **Desain (2 tahap, dilatih berurutan di notebook yang sama):**
# 
# 1. **Stage 1 (coarse):** FUSegNet baseline pada citra penuh.
# 2. **Stage 2 (refiner):** FUSegNet kedua pada **crop yang di-zoom** (di-resize ke `CROP_SIZE`) di sekitar luka.
#    Training: crop di sekitar bbox GT dengan margin acak, plus peluang `P_RANDOM_CROP` crop acak agar refiner belajar menolak proposal yang salah.
# 3. **Inferensi:** piksel stage 1 dengan prob >= `PROP_THR` (lebih rendah dari 0.5, supaya luka yang terlewat masih jadi kandidat) dikelompokkan jadi komponen terhubung
#    (maks `MAX_COMPONENTS`), tiap komponen di-crop, di-refine, ditempel kembali. Di area crop, prediksi refiner menggantikan prediksi coarse; di luar crop tetap coarse.
# 
# Evaluasi mencetak **dua baris dari model stage 1 yang sama**: `cascade_coarse` dan `cascade`. Selisihnya murni efek refiner, bebas dari noise antar-run.
# 
# > Ini **probe**, belum klaim novelty. Catatan: chronic_wound sudah berupa crop di sekitar luka, jadi efek zoom di sana kemungkinan lebih kecil daripada di fuseg_2021 (citra penuh).


# ## Catatan desain dan asumsi (baca dulu)
# 
# **Sesuai instruksi:** protokol training = `fusegnet_all.py` (split tunggal 85/15, seed 42, Adam lr 1e-4, wd 1e-5,
# ReduceLROnPlateau factor 0.1 patience 10, batch 2, 200 epoch, early stop 30). Dataset **chronic_wound** (224x224): `train` (831) dipecah 85/15 menjadi train/validation **per ID foto** (bukan per file), dan folder `test` (278 citra) dipakai sebagai set evaluasi. Catatan: paper memakai 200 citra test, jadi angka tidak persis sebanding dengan Tabel 2 paper.
# Logika checkpoint **sudah diubah** (loss dan IoU dicek independen, simpan jika salah satunya membaik).
# 
# **Hal yang tidak ditentukan di paper/kode sehingga saya pilih sendiri** (semuanya ada sebagai konstanta di sel konfigurasi):
# 
# 1. Arsitektur direkonstruksi dari paper (Gambar 1-2, Persamaan 3): `upsample -> concat skip -> P-scSE -> Conv3x3-BN-ReLU`,
#    satu konvolusi per tahap, *shorted* P-scSE di Decoder 0. Jumlah parameter dicetak di bawah (paper: 64,90 M).
# 2. Reduction ratio SE = 16 (default smp; tidak disebut paper).
# 3. Focal Loss baseline: `gamma=2`, tanpa `alpha` (implementasi `losses.FocalLoss` di fork penulis tidak tersedia).
# 4. Pada LG-P-scSE, varian *shorted* di Decoder 0 menjadi `F_add + x + F_local`.
# 5. Pada CL-DPSC, channel proyeksi 1x1 = 16 per fitur encoder, residual = identity.
# 6. Focal Tversky dihitung *batch-pooled* (sama seperti Dice baseline), `eps = 1`.
# 7. Boundary F1 memakai toleransi 2 piksel. "Luka kecil" = luas GT > 0 dan < 1% dari citra.
# 8. Metrik data-based dihitung dari TP/FP/FN murni. Kode asli menambahkan `len(mask)` ke TP jika prediksi identik dengan GT
#    (terjadi pada citra tanpa luka), yang menggelembungkan Dice data-based.
# 9. Nilai *val loss* **tidak bisa dibandingkan antar notebook** karena fungsi lossnya berbeda. Bandingkan lewat metrik (Dice, IoU, dst).
# 


# ## Catatan M3 (pilihan desain yang tidak ditentukan oleh paper mana pun)
# 
# 1. Refiner **hanya menerima RGB crop** (tanpa channel mask coarse), supaya encoder pretrained 3-channel tidak perlu diubah.
# 2. Crop berbentuk persegi, sisi minimal 25% sisi citra (`MIN_BOX_FRAC`) agar luka sangat kecil tetap punya konteks.
# 3. Validasi stage 2 memakai crop deterministik dari bbox GT (jadi mengukur kualitas refiner dengan proposal yang bagus, cenderung optimis). Checkpoint dipilih dengan logika loss/IoU independen.
# 4. Proposal di inferensi: `PROP_THR = 0.3`, komponen < 4 piksel diabaikan, maksimal 5 komponen per citra.
# 5. Refiner memakai arsitektur, loss (Dice + Focal), dan hyperparameter yang sama dengan stage 1.
# 6. Waktu training sekitar dua kali baseline, dan inferensi lebih lambat karena satu forward tambahan per kandidat.
# 7. Citra yang tidak punya kandidat (prob stage 1 < `PROP_THR` di semua piksel) tidak di-refine, sehingga luka yang benar-benar terlewat total tetap terlewat.
# 


# ## 1. Import


# !pip install torch torchvision efficientnet_pytorch albumentations opencv-python scipy pandas matplotlib tqdm
import os, math, random, time, warnings, ssl
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as BaseDataset
import albumentations as A
from scipy import ndimage as ndi
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from efficientnet_pytorch import EfficientNet
from tqdm.auto import tqdm

ssl._create_default_https_context = ssl._create_unverified_context   # seperti kode asli (unduh bobot ImageNet)
warnings.filterwarnings("ignore")
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())


# ## 2. Konfigurasi


# ==============================================================================
# KONFIGURASI EKSPERIMEN
# ==============================================================================
EXP_NAME = "cascade"

# --- saklar novelty ---
USE_LG = False      # Novelty 1: Local-Global P-scSE
USE_CL = False      # Novelty 2: Cross-Level Detail-Preserving Skip (ke Decoder 0)
USE_BW = False      # Novelty 3: Boundary-Weighted Focal Tversky Loss
USE_RDS = False    # Probe M1: Raw Detail Skip (PixelUnshuffle dari input mentah -> decoder)
USE_FREQ = False  # M1: Frequency Detail Skip (sub-band frekuensi tinggi dari input mentah -> decoder)
UP_MODE = 'nearest'   # M2: upsampling decoder: 'nearest' (asli) | 'bilinear' (kontrol) | 'guided' (offset dipelajari)
USE_SA = False      # M4: Size-Aware Tversky (butuh USE_BW = True)

# --- parameter asli FUSegNet (fusegnet_all.py) ---
ENCODER         = 'efficientnet-b7'
PRETRAINED      = True            # bobot ImageNet
BATCH_SIZE      = 2
LR              = 1e-4
WEIGHT_DECAY    = 1e-5
EPOCHS          = 200
PATIENCE        = 30              # early stopping
EARLY_STOP      = True
SCHED_FACTOR    = 0.1
SCHED_PATIENCE  = 10
SCHED_MIN_LR    = 1e-5
VAL_FRACTION    = 0.15
SEED            = 42
NUM_WORKERS     = 4
THRESHOLD       = 0.5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- parameter modul (lihat catatan desain) ---
SE_REDUCTION    = 16
DECODER_CHANNELS = (256, 128, 64, 32, 16)

# LG-P-scSE
LG_LOCAL_KERNEL = 3               # depthwise separable conv 3x3
LG_FUSION       = 'sum'           # 'sum' (F_add + F_max + F_local)  |  'learnable' (bobot skalar yang dipelajari)

# CL-DPSC
CL_PROJ_CH      = 16              # channel proyeksi 1x1 untuk tiap fitur encoder

# RDS (Raw Detail Skip)
RDS_PROJ_CH     = 16              # channel proyeksi 1x1 per skala
# {faktor PixelUnshuffle: indeks blok decoder}. Resolusi blok: 4 = Decoder 0 (1/1), 3 = Decoder 1 (1/2), 2 = Decoder 2 (1/4)
RDS_INJECT      = {1: 4, 2: 3, 4: 2}

# M1: Frequency Detail Skip
FREQ_PROJ_CH    = 16
# {level: indeks blok decoder}. level 0 = Laplacian resolusi penuh, 1 = Haar 1/2, 2 = Haar 1/4
FREQ_INJECT     = {0: 4, 1: 3, 2: 2}

# M2: Guided offset upsampling
UP_MAX_OFFSET   = 1.0             # batas geser sampling (satuan piksel resolusi rendah)

# M4: Size-Aware Tversky (alpha = 1 - beta per citra)
SA_BETA_MIN     = 0.5             # luka besar -> setara Dice
SA_BETA_MAX     = 0.8             # luka kecil -> hukuman FN lebih berat
SA_AREA_LO      = 0.5             # luas luka (% citra) di bawah ini -> beta = SA_BETA_MAX
SA_AREA_HI      = 10.0            # luas luka (% citra) di atas ini  -> beta = SA_BETA_MIN

# loss baseline (Dice + Focal)
FOCAL_GAMMA     = 2.0
FOCAL_ALPHA     = None

# BW-FTL
FT_ALPHA        = 0.3
FT_BETA         = 0.7
FT_GAMMA        = 0.75
FT_EPS          = 1.0
FT_PER_SAMPLE   = False           # False = batch-pooled (sama seperti Dice baseline)
LAMBDA_BOUNDARY = 0.01

# evaluasi
BOUNDARY_TOL_PX = 2
SMALL_AREA_PCT  = 1.0             # luka kecil: 0 < luas GT (%) < SMALL_AREA_PCT


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEED)
print(f'EXP={EXP_NAME} | LG={USE_LG} CL={USE_CL} BW={USE_BW} RDS={USE_RDS} FREQ={USE_FREQ} UP={UP_MODE} SA={USE_SA} | device={DEVICE}')


# ==============================================================================
# KONFIGURASI CASCADE (stage 1 = FUSegNet baseline pada citra penuh, stage 2 = refiner pada crop yang di-zoom)
# ==============================================================================
CROP_SIZE        = 224     # ukuran input refiner (crop di-resize ke sini)
CROP_MARGIN      = (0.3, 0.8)        # training: margin acak di sekeliling bbox GT (fraksi sisi bbox)
CROP_MARGIN_EVAL = 0.5               # validasi/test: margin tetap
MIN_BOX_FRAC     = 0.25              # sisi crop minimal = 25% sisi citra (konteks untuk luka sangat kecil)
P_RANDOM_CROP    = 0.2               # training: peluang crop acak (bukan di sekitar luka) -> refiner belajar menolak proposal salah
PROP_THR         = 0.3               # stage 1: piksel dengan prob >= PROP_THR menjadi kandidat luka
MAX_COMPONENTS   = 5                 # maksimum kandidat yang di-refine per citra
MIN_COMPONENT_PX = 4                 # kandidat lebih kecil dari ini diabaikan


# ## 3. Path dan split dataset


# ==============================================================================
# DATASET (struktur path mengikuti proyek 'ganta')
#   notebook/                     <- notebook ini berada di sini  (BASE_DIR = "..")
#   data/raw/<DATASET_TYPE>/train/{images,labels}
#   data/raw/<DATASET_TYPE>/test/{images,labels}         <- dipakai sebagai set evaluasi (278 citra)
#   output_chronic_wound/{checkpoints,predictions,results}
# ==============================================================================
BASE_DIR = ".."
DATASET_TYPE = "chronic_wound"   # 224x224

DATA_DIR     = os.path.join(BASE_DIR, "data")
DATASET_ROOT = os.path.join(DATA_DIR, "raw", DATASET_TYPE)
TRAIN_DIR    = os.path.join(DATASET_ROOT, "train")
TEST_DIR     = os.path.join(DATASET_ROOT, "test")

OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output_chronic_wound")
CHECKPOINT_DIR  = os.path.join(OUTPUT_BASE_DIR, "checkpoints")
PREDICTIONS_DIR = os.path.join(OUTPUT_BASE_DIR, "predictions")
RESULTS_DIR     = os.path.join(OUTPUT_BASE_DIR, "results")
for _d in (CHECKPOINT_DIR, PREDICTIONS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# nama unik per run
model_name      = f"{EXP_NAME}_{DATASET_TYPE}_{ENCODER}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
checkpoint_loc  = os.path.join(CHECKPOINT_DIR, model_name)
save_dir_pred   = os.path.join(PREDICTIONS_DIR, model_name)
os.makedirs(checkpoint_loc, exist_ok=True)
os.makedirs(save_dir_pred, exist_ok=True)
print('Run:', model_name)

#%% Load dataset
x_train_dir = x_valid_dir = os.path.join(TRAIN_DIR, 'images')
y_train_dir = y_valid_dir = os.path.join(TRAIN_DIR, 'labels')
x_test_dir = os.path.join(TEST_DIR, 'images')
y_test_dir = os.path.join(TEST_DIR, 'labels')

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
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))

# split 85/15 (seed 42) PER ID FOTO. Nama file = <id>_<n>.png; potongan dari foto yang sama
# harus masuk ke sisi yang sama supaya tidak bocor antara train dan validation.
names = list_images(x_train_dir)
groups = defaultdict(list)
for n_ in names:
    groups[os.path.splitext(n_)[0].rsplit('_', 1)[0]].append(n_)
group_ids = sorted(groups)
random.seed(42)
random.shuffle(group_ids)
n_val_target = int(len(names) * VAL_FRACTION)
list_IDs_val, list_IDs_train = [], []
for gid in group_ids:
    (list_IDs_val if len(list_IDs_val) < n_val_target else list_IDs_train).extend(groups[gid])
list_IDs_test = list_images(x_test_dir)
set_seed(SEED)
_gv = {os.path.splitext(n_)[0].rsplit('_', 1)[0] for n_ in list_IDs_val}
_gt = {os.path.splitext(n_)[0].rsplit('_', 1)[0] for n_ in list_IDs_train}
assert not (_gv & _gt), 'Ada ID foto yang muncul di train dan validation!'
print(f'ID foto unik: {len(group_ids)} | train {len(_gt)} ID | validation {len(_gv)} ID | tumpang tindih: 0')

print('No. of training images  :', len(list_IDs_train))
print('No. of validation images:', len(list_IDs_val))
print('No. of test images      :', len(list_IDs_test), '(folder test)')

# setiap citra harus punya mask dengan NAMA FILE YANG SAMA
for _imgs, _lbl_dir, _tag in ((names, y_train_dir, 'train'), (list_IDs_test, y_test_dir, 'test')):
    _missing = [n for n in _imgs if not os.path.exists(os.path.join(_lbl_dir, n))]
    if _missing:
        raise FileNotFoundError(f"{len(_missing)} mask {_tag} tidak ditemukan di {_lbl_dir} (contoh: {_missing[:3]})")

# kode asli membagi mask dengan 255 -> mask harus bernilai {0, 255}
_m = cv2.imread(os.path.join(y_train_dir, names[0]), 0)
_i = cv2.imread(os.path.join(x_train_dir, names[0]))
print('Contoh gambar   :', names[0], '| shape', _i.shape)
print('Nilai unik mask :', np.unique(_m)[:10], '| shape', _m.shape)
if _m.max() <= 1:
    print('PERINGATAN: mask bernilai {0,1}. Kode asli melakukan mask/255.0 -> mask menjadi ~0 semua! '
          'Ubah label ke {0,255} atau sesuaikan Dataset.__getitem__.')


# ## 4. Dataset dan augmentasi


# ==============================================================================
# DATASET + AUGMENTASI  (augmentasi identik dengan fusegnet_all.py / Tabel 1 paper)
# ==============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def signed_distance_map(mask_bin):
    # Kervadec et al.: negatif di dalam objek, positif di luar. Citra tanpa luka -> nol.
    pos = mask_bin.astype(bool)
    if (not pos.any()) or pos.all():
        return np.zeros(pos.shape, dtype=np.float32)
    neg = ~pos
    dist = ndi.distance_transform_edt(neg) * neg - (ndi.distance_transform_edt(pos) - 1) * pos
    return dist.astype(np.float32)


class Dataset(BaseDataset):
    def __init__(self, list_IDs, images_dir, masks_dir, augmentation=None, with_dist=False):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps  = [os.path.join(masks_dir, i) for i in self.ids]
        self.augmentation = augmentation
        self.with_dist = with_dist      # True hanya jika USE_BW (butuh distance map)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0).astype(np.float32) / 255.0   # {0,1}

        if self.augmentation:
            s = self.augmentation(image=image, mask=mask)
            image, mask = s['image'], s['mask']

        # preprocessing EfficientNet (imagenet): /255, mean/std, HWC -> CHW
        image = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image = image.transpose(2, 0, 1).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask[None].astype(np.float32)                                    # (1,H,W)

        if self.with_dist:
            dist = signed_distance_map(mask[0] > 0.5)[None]                     # dihitung SETELAH augmentasi
        else:
            dist = np.zeros((1, 1, 1), dtype=np.float32)
        return image, mask, dist

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        A.OneOf([A.HorizontalFlip(p=0.8), A.VerticalFlip(p=0.4)], p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,  shift_limit=0,   p=1, border_mode=0),  # scale
            A.ShiftScaleRotate(scale_limit=0,   rotate_limit=30, shift_limit=0,   p=1, border_mode=0),  # rotate
            A.ShiftScaleRotate(scale_limit=0,   rotate_limit=0,  shift_limit=0.1, p=1, border_mode=0),  # shift
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),  # affine
        ], p=0.9),
        A.OneOf([
            A.Perspective(p=1), A.GaussNoise(p=1), A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.2),
    ]
    return A.Compose(train_transform, p=0.9)


def get_validation_augmentation():
    return A.Compose([])


# ==============================================================================
# DATASET STAGE 2: crop di sekitar luka -> di-resize ke CROP_SIZE (zoom)
# ==============================================================================
def expand_box(x0, y0, x1, y1, margin, H, W, min_frac):
    # kotak persegi di sekeliling bbox (x0,y0,x1,y1) dengan margin relatif, sisi minimal min_frac*min(H,W), digeser agar tetap di dalam citra
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = max((x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin), min_frac * min(H, W))
    s = int(round(min(s, min(H, W))))
    nx0 = int(round(cx - s / 2.0))
    ny0 = int(round(cy - s / 2.0))
    nx0 = min(max(nx0, 0), W - s)
    ny0 = min(max(ny0, 0), H - s)
    return nx0, ny0, nx0 + s, ny0 + s


class CropDataset(BaseDataset):
    # train=True : crop di sekitar bbox GT dengan margin acak (peluang P_RANDOM_CROP: crop acak). Augmentasi diterapkan SEBELUM crop.
    # train=False: crop deterministik (margin CROP_MARGIN_EVAL; citra tanpa luka -> crop tengah 50%).
    def __init__(self, list_IDs, images_dir, masks_dir, augmentation=None, train=True):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps = [os.path.join(masks_dir, i) for i in self.ids]
        self.augmentation = augmentation
        self.train = train

    def _pick_box(self, mask, H, W):
        ys, xs = np.where(mask > 0.5)
        if (self.train and random.random() < P_RANDOM_CROP) or len(xs) == 0:
            if self.train:
                s = int(random.uniform(MIN_BOX_FRAC, 0.8) * min(H, W))
                x0 = random.randint(0, W - s)
                y0 = random.randint(0, H - s)
            else:
                s = int(0.5 * min(H, W))
                x0, y0 = (W - s) // 2, (H - s) // 2
            return x0, y0, x0 + s, y0 + s
        margin = random.uniform(CROP_MARGIN[0], CROP_MARGIN[1]) if self.train else CROP_MARGIN_EVAL
        return expand_box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, margin, H, W, MIN_BOX_FRAC)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0).astype(np.float32) / 255.0
        if self.augmentation:
            sm = self.augmentation(image=image, mask=mask)
            image, mask = sm['image'], sm['mask']
        if mask.ndim == 3:
            mask = mask[..., 0]
        H, W = mask.shape
        x0, y0, x1, y1 = self._pick_box(mask, H, W)
        img_c = cv2.resize(np.ascontiguousarray(image[y0:y1, x0:x1]), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        msk_c = cv2.resize(np.ascontiguousarray(mask[y0:y1, x0:x1]), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        img_c = (img_c.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return img_c.transpose(2, 0, 1).astype(np.float32), msk_c[None].astype(np.float32), np.zeros((1, 1, 1), dtype=np.float32)

    def __len__(self):
        return len(self.ids)


# ## 5. Model


# ==============================================================================
# MODEL: FUSegNet (EfficientNet encoder + decoder U-Net dengan P-scSE) + novelty
# ==============================================================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------------- Encoder (gaya smp) ---------------------------------
class EfficientNetEncoder(nn.Module):
    # fitur: [identity, stem(1/2), 1/4, 1/8, 1/16, 1/32] persis seperti smp.encoders efficientnet
    def __init__(self, name='efficientnet-b7', pretrained=True):
        super().__init__()
        net = EfficientNet.from_pretrained(name) if pretrained else EfficientNet.from_name(name)
        for attr in ('_fc', '_conv_head', '_bn1', '_avg_pooling', '_dropout'):   # bagian klasifikasi tidak dipakai
            if hasattr(net, attr):
                delattr(net, attr)
        self.net = net

        # titik potong stage dicari otomatis dari blok ber-stride 2 (tidak bergantung ukuran varian b0..b7)
        strides = []
        for b in net._blocks:
            s = b._block_args.stride
            strides.append(s[0] if isinstance(s, (list, tuple)) else s)
        down_idx = [i for i, s in enumerate(strides) if s == 2]      # blok yang menurunkan resolusi
        self.split = down_idx[1:4]                                   # awal stage 1/8, 1/16, 1/32

        self.eval()
        with torch.no_grad():
            self.out_channels = [3] + [f.shape[1] for f in self.forward(torch.zeros(1, 3, 64, 64))[1:]]
        self.train()

    def forward(self, x):
        net = self.net
        s0, s1, s2 = self.split
        stages = [
            nn.Identity(),
            nn.Sequential(net._conv_stem, net._bn0, net._swish),
            net._blocks[:s0], net._blocks[s0:s1], net._blocks[s1:s2], net._blocks[s2:],
        ]
        drop_rate = net._global_params.drop_connect_rate
        block_number, feats = 0., []
        for i, st in enumerate(stages):
            if i < 2:
                x = st(x)
            else:
                for blk in st:
                    dc = drop_rate * block_number / len(net._blocks) if drop_rate else None
                    block_number += 1.
                    x = blk(x, dc)
            feats.append(x)
        return feats


# ------------------------------ Modul atensi -------------------------------------
class SCSE(nn.Module):
    # mengembalikan (keluaran cSE, keluaran sSE) supaya bisa diagregasi additive / max-out
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 1)
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, ch, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x), x * self.sse(x)


class LocalAttention(nn.Module):
    # Cabang lokal LG-P-scSE (tanpa GAP):  A = sigmoid(PW(DW(F))),  F_local = F * A
    def __init__(self, ch, k=3):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.pw(self.dw(x)))


class PscSE(nn.Module):
    # P-scSE asli : out = F_add + F_max         (F_add = cSE+sSE, F_max = max(cSE,sSE); dua modul scSE terpisah)
    # shorted     : out = F_add + x             (max-out di-bypass, dipakai di decoder teratas)
    # LG-P-scSE   : + F_local                   (use_local=True)
    def __init__(self, ch, reduction=16, shorted=False, use_local=False, local_kernel=3, fusion='sum'):
        super().__init__()
        self.shorted = shorted
        self.add_branch = SCSE(ch, reduction)
        self.max_branch = None if shorted else SCSE(ch, reduction)
        self.local = LocalAttention(ch, local_kernel) if use_local else None
        n_branch = 2 + (1 if use_local else 0)
        self.w = nn.Parameter(torch.ones(n_branch)) if (use_local and fusion == 'learnable') else None

    def forward(self, x):
        c, s = self.add_branch(x)
        f_add = c + s
        if self.shorted:
            f_max = x
        else:
            c2, s2 = self.max_branch(x)
            f_max = torch.maximum(c2, s2)
        branches = [f_add, f_max]
        if self.local is not None:
            branches.append(self.local(x))
        if self.w is not None:
            branches = [wi * b for wi, b in zip(self.w, branches)]
        out = branches[0]
        for b in branches[1:]:
            out = out + b
        return out


# ------------------------------ Cross-Level Skip ---------------------------------
class CrossLevelFusion(nn.Module):
    # 1x1 proyeksi -> upsample -> concat -> Conv3x3-BN-ReLU -> + residual (fitur decoder)
    def __init__(self, dec_ch, enc_chs, proj_ch=16):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, proj_ch, 1, bias=False) for c in enc_chs])
        self.block = ConvBNReLU(dec_ch + proj_ch * len(enc_chs), dec_ch)

    def forward(self, x, enc_feats):
        ps = [F.interpolate(p(f), size=x.shape[-2:], mode='bilinear', align_corners=False)
              for p, f in zip(self.proj, enc_feats)]
        return x + self.block(torch.cat([x] + ps, dim=1))


# ------------------------------ Raw Detail Skip (RDS) ----------------------------
class RawDetailPath(nn.Module):
    # Space-to-depth (PixelUnshuffle, LOSSLESS) pada input mentah -> proyeksi 1x1 (+BN+ReLU).
    # Faktor f menghasilkan fitur pada resolusi 1/f dengan 3*f*f channel sebelum proyeksi.
    def __init__(self, in_ch, factors, proj_ch=16):
        super().__init__()
        self.factors = tuple(sorted(set(factors)))
        self.unshuffle = nn.ModuleDict({str(f): nn.PixelUnshuffle(f) for f in self.factors})
        self.proj = nn.ModuleDict({str(f): ConvBNReLU(in_ch * f * f, proj_ch, k=1) for f in self.factors})

    def forward(self, x):
        m = max(self.factors)
        ph, pw = (-x.shape[-2]) % m, (-x.shape[-1]) % m
        if ph or pw:                                   # PixelUnshuffle butuh H,W kelipatan faktor
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return {f: self.proj[str(f)](self.unshuffle[str(f)](x)) for f in self.factors}


# ------------------------------ Frequency Detail Skip (M1) -----------------------
class FrequencyDetailPath(nn.Module):
    # Sub-band frekuensi tinggi dari input mentah dengan filter TETAP (tanpa bobot), lalu proyeksi 1x1 yang dipelajari.
    #   level 0: Laplacian high-pass resolusi penuh   x - Gaussian5x5(x)                  -> 3 channel
    #   level 1: Haar DWT level-1, sub-band LH/HL/HH  (resolusi 1/2)                      -> 9 channel
    #   level 2: Haar DWT level-2 dari LL level-1, sub-band LH/HL/HH (resolusi 1/4)       -> 9 channel
    def __init__(self, in_ch, levels, proj_ch=16):
        super().__init__()
        assert all(l in (0, 1, 2) for l in levels), 'FREQ_INJECT: level harus 0, 1, atau 2'
        self.in_ch = in_ch
        self.levels = tuple(sorted(set(levels)))
        g = torch.tensor([1., 4., 6., 4., 1.])
        g = (g[:, None] * g[None, :]) / 256.0
        self.register_buffer('gauss', g.expand(in_ch, 1, 5, 5).contiguous())
        h = 0.5 * torch.tensor([[[1., 1.], [1., 1.]],       # LL
                                [[1., 1.], [-1., -1.]],     # LH
                                [[1., -1.], [1., -1.]],     # HL
                                [[1., -1.], [-1., 1.]]])    # HH
        self.register_buffer('haar', h.unsqueeze(1))        # (4, 1, 2, 2)
        n_ch = {0: in_ch, 1: 3 * in_ch, 2: 3 * in_ch}
        self.proj = nn.ModuleDict({str(l): ConvBNReLU(n_ch[l], proj_ch, k=1) for l in self.levels})

    def _dwt(self, x):
        B, C, H, W = x.shape
        y = F.conv2d(x.reshape(B * C, 1, H, W), self.haar, stride=2)     # (B*C, 4, H/2, W/2)
        y = y.reshape(B, C, 4, H // 2, W // 2)
        return y[:, :, 0], y[:, :, 1:].reshape(B, C * 3, H // 2, W // 2)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        out = {}
        if 0 in self.levels:
            blur = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), self.gauss, groups=self.in_ch)
            out[0] = self.proj['0'](x - blur)
        ll1, hf1 = self._dwt(x)
        if 1 in self.levels:
            out[1] = self.proj['1'](hf1)
        if 2 in self.levels:
            _, hf2 = self._dwt(ll1)
            out[2] = self.proj['2'](hf2)
        return out


# ------------------------------ Guided offset upsampling (M2) --------------------
class GuidedOffsetUpsample(nn.Module):
    # Bilinear upsampling + pergeseran titik sampling yang DIPELAJARI dari (fitur low-res yang di-upsample, fitur guide resolusi tinggi).
    # Inisialisasi offset = 0 -> awalnya identik dengan bilinear. Ini formulasi sederhana buatan sendiri (terinspirasi DySample/SAPA), bukan kode aslinya.
    def __init__(self, in_ch, guide_ch, max_offset=1.0):
        super().__init__()
        self.max_offset = max_offset
        self.offset = nn.Conv2d(in_ch + guide_ch, 2, 3, padding=1)

    def forward(self, x, guide, size):
        h, w = x.shape[-2:]
        H, W = size
        u = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        o = torch.tanh(self.offset(torch.cat([u, guide], dim=1))) * self.max_offset    # (B, 2, H, W), satuan piksel low-res
        ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H * 2 - 1
        xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack([gx, gy], dim=-1).unsqueeze(0)                              # (1, H, W, 2) urutan (x, y)
        off = torch.stack([o[:, 0] * (2.0 / w), o[:, 1] * (2.0 / h)], dim=-1)          # (B, H, W, 2) koordinat ternormalisasi
        return F.grid_sample(x, base + off, mode='bilinear', padding_mode='border', align_corners=False)


# ---------------------------------- Decoder --------------------------------------
class DecoderBlock(nn.Module):
    # upsample -> concat skip -> [concat detail RDS/FREQ] -> [CL fusion] -> P-scSE -> Conv3x3-BN-ReLU
    def __init__(self, in_ch, skip_ch, out_ch, shorted=False, use_lg=False, cl_enc_chs=None,
                 extra_ch=0, up_mode='nearest', guide_ch=0):
        super().__init__()
        cat_ch = in_ch + skip_ch + extra_ch
        self.up_mode = up_mode
        self.up = GuidedOffsetUpsample(in_ch, guide_ch, UP_MAX_OFFSET) if up_mode == 'guided' else None
        self.cl = CrossLevelFusion(cat_ch, cl_enc_chs, CL_PROJ_CH) if cl_enc_chs else None
        self.att = PscSE(cat_ch, SE_REDUCTION, shorted=shorted, use_local=use_lg,
                         local_kernel=LG_LOCAL_KERNEL, fusion=LG_FUSION)
        self.conv = ConvBNReLU(cat_ch, out_ch)

    def forward(self, x, skip, size, cl_feats=None, extra_feats=None, guide=None):
        if self.up is not None:
            x = self.up(x, guide, size)
        elif self.up_mode == 'bilinear':
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if extra_feats:
            for r in extra_feats:
                if r.shape[-2:] != x.shape[-2:]:
                    r = F.interpolate(r, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, r], dim=1)
        if self.cl is not None:
            x = self.cl(x, cl_feats)
        return self.conv(self.att(x))


class FUSegNetDecoder(nn.Module):
    def __init__(self, enc_channels, decoder_channels, use_lg=False, use_cl=False, extra_ch=None, up_mode='nearest'):
        super().__init__()
        enc = list(enc_channels[1:])[::-1]                     # [1/32, 1/16, 1/8, 1/4, 1/2]
        in_chs = [enc[0]] + list(decoder_channels[:-1])
        skip_chs = enc[1:] + [0]                               # tahap terakhir tanpa skip
        n = len(decoder_channels)
        cl_chs = [enc_channels[1], enc_channels[2]]            # fitur stride-2 dan stride-4
        extra_ch = extra_ch or {}
        self.blocks = nn.ModuleList([
            DecoderBlock(ic, sc, oc, shorted=(i == n - 1), use_lg=use_lg,
                         cl_enc_chs=(cl_chs if (use_cl and i == n - 1) else None),
                         extra_ch=extra_ch.get(i, 0), up_mode=up_mode,
                         guide_ch=(sc if sc > 0 else enc_channels[0]))
            for i, (ic, sc, oc) in enumerate(zip(in_chs, skip_chs, decoder_channels))
        ])

    def forward(self, feats, extra_by_block=None):
        extra_by_block = extra_by_block or {}
        input_hw = feats[0].shape[-2:]
        f_rev = feats[1:][::-1]
        x, skips = f_rev[0], f_rev[1:]
        n = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            size = skip.shape[-2:] if skip is not None else input_hw
            cl_feats = [feats[1], feats[2]] if i == n - 1 else None
            guide = skip if skip is not None else feats[0]     # guide resolusi tinggi (blok terakhir: citra input)
            x = blk(x, skip, size, cl_feats, extra_by_block.get(i), guide)
        return x


class FUSegNet(nn.Module):
    # keluaran = logit (sigmoid diterapkan di loss / saat prediksi)
    def __init__(self, encoder_name, pretrained, decoder_channels, use_lg, use_cl,
                 use_rds=False, use_freq=False, up_mode='nearest'):
        super().__init__()
        self.encoder = EfficientNetEncoder(encoder_name, pretrained)
        n_blocks = len(decoder_channels)
        extra_ch = {}

        self.rds_inject = dict(RDS_INJECT) if use_rds else {}
        for f, bi in self.rds_inject.items():
            assert 0 <= bi < n_blocks, f'RDS_INJECT: indeks blok {bi} di luar jangkauan'
            extra_ch[bi] = extra_ch.get(bi, 0) + RDS_PROJ_CH
        self.raw = RawDetailPath(3, list(self.rds_inject), RDS_PROJ_CH) if use_rds else None

        self.freq_inject = dict(FREQ_INJECT) if use_freq else {}
        for l, bi in self.freq_inject.items():
            assert 0 <= bi < n_blocks, f'FREQ_INJECT: indeks blok {bi} di luar jangkauan'
            extra_ch[bi] = extra_ch.get(bi, 0) + FREQ_PROJ_CH
        self.freq = FrequencyDetailPath(3, list(self.freq_inject), FREQ_PROJ_CH) if use_freq else None

        assert up_mode in ('nearest', 'bilinear', 'guided'), "UP_MODE harus 'nearest', 'bilinear', atau 'guided'"
        self.decoder = FUSegNetDecoder(self.encoder.out_channels, decoder_channels, use_lg, use_cl,
                                       extra_ch=extra_ch, up_mode=up_mode)
        self.head = nn.Conv2d(decoder_channels[-1], 1, 3, padding=1)
        self._init_decoder()

    def _init_decoder(self):
        mods = list(self.decoder.modules())
        for extra in (self.raw, self.freq):
            if extra is not None:
                mods += list(extra.modules())
        for m in mods:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        for m in self.decoder.modules():                       # offset awal = 0 -> identik dengan bilinear
            if isinstance(m, GuidedOffsetUpsample):
                nn.init.zeros_(m.offset.weight); nn.init.zeros_(m.offset.bias)
        nn.init.xavier_uniform_(self.head.weight); nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        extra_by_block = {}
        if self.raw is not None:
            r = self.raw(x)
            for f, bi in self.rds_inject.items():
                extra_by_block.setdefault(bi, []).append(r[f])
        if self.freq is not None:
            q = self.freq(x)
            for l, bi in self.freq_inject.items():
                extra_by_block.setdefault(bi, []).append(q[l])
        return self.head(self.decoder(self.encoder(x), extra_by_block or None))


# ## 6. Loss dan metrik


# ==============================================================================
# LOSS + METRIK
# ==============================================================================
def _sum(t, per_sample):
    return t.sum(dim=(1, 2, 3)) if per_sample else t.sum()


def dice_loss(logits, target, eps=1.0):
    # setara smp.utils.losses.DiceLoss (batch-pooled, eps=1)
    p = torch.sigmoid(logits)
    tp = (p * target).sum(); fp = p.sum() - tp; fn = target.sum() - tp
    return 1 - (2 * tp + eps) / (2 * tp + fp + fn + eps)


def focal_loss(logits, target, gamma=2.0, alpha=None):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.exp(-bce)
    loss = (1 - pt) ** gamma * bce
    if alpha is not None:
        loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.mean()


def size_aware_beta(target):
    # beta per citra: SA_BETA_MAX untuk luka kecil, turun ke SA_BETA_MIN untuk luka besar
    # (interpolasi linear pada log10(luas luka dalam %)). Citra tanpa luka -> SA_BETA_MIN (alpha terbesar, FP paling dihukum).
    area = target.mean(dim=(1, 2, 3)) * 100.0
    lo, hi = math.log10(SA_AREA_LO), math.log10(SA_AREA_HI)
    t = ((torch.log10(area.clamp(min=1e-6)) - lo) / (hi - lo)).clamp(0.0, 1.0)
    beta = SA_BETA_MAX - (SA_BETA_MAX - SA_BETA_MIN) * t
    return torch.where(area > 0, beta, torch.full_like(beta, SA_BETA_MIN))


def focal_tversky_loss(logits, target, alpha=0.3, beta=0.7, gamma=0.75, eps=1.0, per_sample=False, size_aware=False):
    # TI = (TP+eps) / (TP + alpha*FP + beta*FN + eps);  L = (1 - TI)^gamma
    p = torch.sigmoid(logits)
    if size_aware:                       # beta (dan alpha = 1 - beta) berbeda per citra -> wajib per-sample
        per_sample = True
        beta = size_aware_beta(target).detach()
        alpha = 1.0 - beta
    tp = _sum(p * target, per_sample)
    fp = _sum(p * (1 - target), per_sample)
    fn = _sum((1 - p) * target, per_sample)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti).clamp(min=1e-7) ** gamma).mean()


def boundary_loss(logits, dist):
    # Kervadec et al. (versi sigmoid biner): mean( p * signed_distance_GT )
    return (torch.sigmoid(logits) * dist).mean()


def compute_loss(logits, target, dist):
    if USE_BW:
        ft = focal_tversky_loss(logits, target, FT_ALPHA, FT_BETA, FT_GAMMA, FT_EPS, FT_PER_SAMPLE, size_aware=USE_SA)
        bd = boundary_loss(logits, dist)
        total = ft + LAMBDA_BOUNDARY * bd
        comps = {'focal_tversky': ft.item(), 'boundary_raw': bd.item(), 'boundary_weighted': LAMBDA_BOUNDARY * bd.item()}
        if USE_SA:
            comps['beta_mean'] = size_aware_beta(target).mean().item()
        return total, comps
    dl = dice_loss(logits, target)
    fl = focal_loss(logits, target, FOCAL_GAMMA, FOCAL_ALPHA)
    return dl + fl, {'dice_loss': dl.item(), 'focal_loss': fl.item()}


@torch.no_grad()
def batch_metrics(logits, target, thr=0.5, eps=1e-7):
    # setara smp.utils.metrics.IoU / Fscore (threshold 0.5, batch-pooled)
    pr = (torch.sigmoid(logits) > thr).float()
    tp = (pr * target).sum(); fp = pr.sum() - tp; fn = target.sum() - tp
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fn + fp + eps)
    return iou.item(), dice.item()


def run_epoch(model, loader, optimizer=None, desc=''):
    train = optimizer is not None
    model.train(train)
    sums, n = defaultdict(float), 0
    for x, y, d in tqdm(loader, desc=desc, leave=False):
        x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss, comps = compute_loss(logits, y, d)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        iou, dice = batch_metrics(logits, y, THRESHOLD)
        bs = x.shape[0]
        sums['loss'] += loss.item() * bs; sums['iou'] += iou * bs; sums['dice'] += dice * bs
        for k, v in comps.items():
            sums[k] += v * bs
        n += bs
    return {k: v / n for k, v in sums.items()}


# ## 7. Training 2 tahap (coarse lalu refiner)


# ==============================================================================
# TRAINING 2 TAHAP. Logika checkpoint baru (loss dan IoU dicek independen) dipakai di kedua tahap.
# ==============================================================================
def train_model(model, train_loader, valid_loader, ckpt_path, tag):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHED_FACTOR, patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR)
    history = defaultdict(list)
    best_vloss, best_viou, best_epoch, cnt_patience = float('inf'), 0.0, -1, 0
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, desc=f'{tag} train {epoch}')
        va = run_epoch(model, valid_loader, None, desc=f'{tag} valid {epoch}')
        for k, v in tr.items(): history['train_' + k].append(v)
        for k, v in va.items(): history['val_' + k].append(v)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        loss_membaik = va['loss'] < best_vloss
        iou_membaik = va['iou'] > best_viou
        if loss_membaik:
            best_vloss = va['loss']
        if iou_membaik:
            best_viou = va['iou']
        if loss_membaik or iou_membaik:
            torch.save({'epoch': epoch + 1, 'state_dict': deepcopy(model.state_dict())}, ckpt_path)
            best_epoch, cnt_patience, flag = epoch, 0, ' | SAVED'
        else:
            cnt_patience += 1
            flag = ''
        scheduler.step(va['loss'])
        print(f"[{tag}] Epoch {epoch:03d} [{time.time()-t0:5.0f}s] lr={history['lr'][-1]:.1e} | "
              f"train loss={tr['loss']:.4f} iou={tr['iou']:.4f} | val loss={va['loss']:.4f} iou={va['iou']:.4f} dice={va['dice']:.4f}{flag}")
        if EARLY_STOP and cnt_patience >= PATIENCE:
            print(f'[{tag}] Early stopping di epoch {epoch:04d} (best epoch: {best_epoch})')
            break
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])          # pakai checkpoint terbaik
    return history, best_epoch


# ---------------- STAGE 1: coarse (citra penuh) ----------------
set_seed(SEED)
model1 = FUSegNet(ENCODER, PRETRAINED, DECODER_CHANNELS, False, False).to(DEVICE)
train_loader1 = DataLoader(Dataset(list_IDs_train, x_train_dir, y_train_dir, augmentation=get_training_augmentation()),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader1 = DataLoader(Dataset(list_IDs_val, x_valid_dir, y_valid_dir, augmentation=get_validation_augmentation()),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
n_params1 = sum(p.numel() for p in model1.parameters())
print(f'Stage 1 params: {n_params1/1e6:.2f} M')
history1, best_epoch1 = train_model(model1, train_loader1, valid_loader1, os.path.join(checkpoint_loc, 'stage1_coarse.pth'), 'S1')
pd.DataFrame(history1).to_csv(os.path.join(RESULTS_DIR, model_name + '_history_stage1.csv'), index=False)

# ---------------- STAGE 2: refiner (crop yang di-zoom) ----------------
set_seed(SEED)
model2 = FUSegNet(ENCODER, PRETRAINED, DECODER_CHANNELS, False, False).to(DEVICE)
train_loader2 = DataLoader(CropDataset(list_IDs_train, x_train_dir, y_train_dir, augmentation=get_training_augmentation(), train=True),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader2 = DataLoader(CropDataset(list_IDs_val, x_valid_dir, y_valid_dir, augmentation=None, train=False),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
n_params2 = sum(p.numel() for p in model2.parameters())
print(f'Stage 2 params: {n_params2/1e6:.2f} M | total cascade: {(n_params1 + n_params2)/1e6:.2f} M')
history2, best_epoch2 = train_model(model2, train_loader2, valid_loader2, os.path.join(checkpoint_loc, 'stage2_refiner.pth'), 'S2')
pd.DataFrame(history2).to_csv(os.path.join(RESULTS_DIR, model_name + '_history_stage2.csv'), index=False)
print('Selesai. Best epoch stage 1:', best_epoch1, '| stage 2:', best_epoch2)


# ## 8. Kurva training


# ==============================================================================
# KURVA TRAINING (stage 1 dan stage 2)
# ==============================================================================
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for r, (hist, tag, be) in enumerate(((history1, 'Stage 1 (coarse)', best_epoch1), (history2, 'Stage 2 (refiner)', best_epoch2))):
    for c, key in enumerate(('loss', 'iou', 'dice')):
        ax[r, c].plot(hist['train_' + key], 'r'); ax[r, c].plot(hist['val_' + key], 'b')
        ax[r, c].axvline(be, color='gray', ls='--', lw=0.8)
        ax[r, c].set_title(f'{tag}: {key}'); ax[r, c].legend(['training', 'validation', 'best epoch'])
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, model_name + '_curves.png'), dpi=120)
plt.show()


# ## 9. Evaluasi: coarse vs cascade


# ==============================================================================
# EVALUASI: coarse saja (stage 1) vs cascade (stage 1 + refiner), pada set evaluasi yang sama
# ==============================================================================
ep = 1e-6


def boundary_of(mask_bool):
    m8 = mask_bool.astype(np.uint8)
    return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))).astype(bool)


def boundary_f1(gt, pred, tol=2):
    g, p = boundary_of(gt), boundary_of(pred)
    if g.sum() == 0 and p.sum() == 0: return 1.0
    if g.sum() == 0 or p.sum() == 0:  return 0.0
    dg = ndi.distance_transform_edt(~g)
    dp = ndi.distance_transform_edt(~p)
    prec = (dg[p] <= tol).mean()
    rec = (dp[g] <= tol).mean()
    return 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))


def per_image_row(name, gt, pred):
    tp = int((gt & pred).sum());  fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum()); tn = int(gt.size - tp - fp - fn)
    if gt.sum() == 0 and pred.sum() == 0:      # tanpa luka & prediksi kosong -> sempurna
        p_, r_, d_, iou_ = 100., 100., 100., 100.
    else:
        p_ = tp / (tp + fp + ep) * 100
        r_ = tp / (tp + fn + ep) * 100
        d_ = 2 * tp / (2 * tp + fp + fn) * 100
        iou_ = tp / (tp + fp + fn + ep) * 100
    return dict(Name=name, GT_area_pct=gt.mean() * 100,
                Accuracy=(tp + tn) / gt.size * 100, Specificity=tn / (tn + fp + ep) * 100,
                iou=iou_, Precision=p_, Recall=r_, Dice=d_,
                BoundaryF1=boundary_f1(gt, pred, BOUNDARY_TOL_PX) * 100,
                tp=tp, fp=fp, fn=fn, tn=tn)


def pooled(sub):
    tp, fp, fn, tn = (sub[c].sum() for c in ('tp', 'fp', 'fn', 'tn'))
    return dict(Accuracy=(tp + tn) / (tp + tn + fp + fn) * 100, Specificity=tn / (tn + fp + ep) * 100,
                iou=tp / (tp + fp + fn + ep) * 100, Precision=tp / (tp + fp + ep) * 100,
                Recall=tp / (tp + fn + ep) * 100, Dice=2 * tp / (2 * tp + fp + fn + ep) * 100)


@torch.no_grad()
def cascade_predict(m1, m2, image_t):
    # image_t: (1,3,H,W) ternormalisasi, di DEVICE. Mengembalikan (prob coarse, prob cascade, jumlah kandidat yang di-refine)
    prob = torch.sigmoid(m1(image_t))[0, 0].cpu().numpy()
    H, W = prob.shape
    prop = (prob >= PROP_THR).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(prop, connectivity=8)
    comps = [(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX]
    comps = sorted(comps, reverse=True)[:MAX_COMPONENTS]
    if not comps:
        return prob, prob, 0
    img = np.ascontiguousarray(image_t[0].cpu().numpy().transpose(1, 2, 0))        # HWC float32 (ternormalisasi)
    refined = np.zeros_like(prob)
    covered = np.zeros(prob.shape, dtype=bool)
    for _, k in comps:
        bx, by = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
        bw, bh = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
        x0, y0, x1, y1 = expand_box(bx, by, bx + bw, by + bh, CROP_MARGIN_EVAL, H, W, MIN_BOX_FRAC)
        crop = cv2.resize(np.ascontiguousarray(img[y0:y1, x0:x1]), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        ct = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)).astype(np.float32))[None].to(DEVICE)
        p_c = torch.sigmoid(m2(ct))[0, 0].cpu().numpy()
        p_c = cv2.resize(p_c, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        refined[y0:y1, x0:x1] = np.maximum(refined[y0:y1, x0:x1], p_c)
        covered[y0:y1, x0:x1] = True
    return prob, np.where(covered, refined, prob), len(comps)


test_dataset = Dataset(list_IDs_test, x_test_dir, y_test_dir, augmentation=get_validation_augmentation())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

model1.eval(); model2.eval()
rows_c, rows_f, n_refined = [], [], 0
for i, (x, y, _) in enumerate(tqdm(test_loader, desc='cascade')):
    name = os.path.splitext(list_IDs_test[i])[0]
    gt = y[0, 0].numpy() > 0.5
    prob, final, n_comp = cascade_predict(model1, model2, x.to(DEVICE))
    pred_c, pred_f = prob >= THRESHOLD, final >= THRESHOLD
    Image.fromarray((pred_f * 255).astype(np.uint8)).save(os.path.join(save_dir_pred, name + '.png'))
    rows_c.append(per_image_row(name, gt, pred_c))
    rows_f.append(per_image_row(name, gt, pred_f))
    n_refined += int(n_comp > 0)

df_c, df_f = pd.DataFrame(rows_c), pd.DataFrame(rows_f)
df_c.to_csv(os.path.join(RESULTS_DIR, model_name + '_per_image_coarse.csv'), index=False)
df_f.to_csv(os.path.join(RESULTS_DIR, model_name + '_per_image_cascade.csv'), index=False)
print(f'Citra dengan >=1 kandidat yang di-refine: {n_refined} / {len(df_f)}')


def make_summary(tag, df, params_M, best_epoch, epochs_run):
    small = df[(df.GT_area_pct > 0) & (df.GT_area_pct < SMALL_AREA_PCT)]
    db = pooled(df)
    return dict(
        exp=tag, LG=False, CL=False, BW=False, model_name=model_name, params_M=round(params_M, 2),
        best_epoch=best_epoch, epochs_run=epochs_run,
        img_dice=df.Dice.mean(), img_iou=df.iou.mean(), img_precision=df.Precision.mean(), img_recall=df.Recall.mean(),
        img_boundaryF1=df.BoundaryF1.mean(),
        data_dice=db['Dice'], data_iou=db['iou'], data_precision=db['Precision'], data_recall=db['Recall'],
        small_n=len(small),
        small_dice_img=small.Dice.mean() if len(small) else np.nan,
        small_dice_data=pooled(small)['Dice'] if len(small) else np.nan,
        small_recall_img=small.Recall.mean() if len(small) else np.nan,
        small_boundaryF1=small.BoundaryF1.mean() if len(small) else np.nan)


sum_c = make_summary('cascade_coarse', df_c, n_params1 / 1e6, best_epoch1, len(history1['lr']))
sum_f = make_summary('cascade', df_f, (n_params1 + n_params2) / 1e6, best_epoch2, len(history2['lr']))
cmp = pd.DataFrame([sum_c, sum_f]).set_index('exp')[['img_dice', 'img_iou', 'img_precision', 'img_recall', 'img_boundaryF1',
                                                    'data_dice', 'small_n', 'small_dice_img', 'small_recall_img', 'small_boundaryF1']]
print('\n=== COARSE vs CASCADE (model stage 1 yang sama) ===')
print(cmp.round(2).to_string())


# ## 10. Ringkasan


# ==============================================================================
# RINGKASAN -> ditambahkan ke results/summary_all_experiments.csv (dua baris: cascade_coarse dan cascade)
# ==============================================================================
summary_path = os.path.join(RESULTS_DIR, 'summary_all_experiments.csv')
new_rows = pd.DataFrame([sum_c, sum_f])
if os.path.exists(summary_path):
    new_rows = pd.concat([pd.read_csv(summary_path), new_rows], ignore_index=True)
new_rows.to_csv(summary_path, index=False)
cmp