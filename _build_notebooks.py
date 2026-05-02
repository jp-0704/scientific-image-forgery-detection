"""
Builder script — generates the 3 training notebooks from a shared template.

Run:  python3 _build_notebooks.py
Outputs:
    train_1_convnext_tiny.ipynb
    train_2_swin_v2_tiny.ipynb
    train_3_swin_v2_base.ipynb
    compare_models.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


def write_nb(path: Path, cells: list, title: str = '') -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "kaggle": {
                "accelerator": "gpu",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))
    print(f'  Wrote {path.name}  ({len(cells)} cells)')


# ── Per-model configurations ──────────────────────────────────────────────────

MODELS = {
    'convnext_tiny': {
        'display':            'ConvNeXt-Tiny + CBAM-UNet + Deep Supervision',
        'rationale':          ('Modern conv-only architecture. Parameter-efficient and '
                               'pretrained at high resolution; targets sharp local '
                               'features useful for boundary precision.'),
        'timm_name':          'convnext_tiny',
        'features_only_kwargs': '',
        'levels':             4,
        'channels_last':      False,
        'batch_size':         16,       # single GPU at ~12 GB on T4
        'epochs':             60,
        'grad_accum':         1,        # effective batch 16
        'enc_lr_mult':        0.1,
        'lr':                 3e-4,
        'patience':           8,
        'grad_checkpoint':    False,
        'output_dir':         'convnext_tiny',
        'notebook_idx':       1,
    },
    'swin_v2_tiny': {
        'display':            'Swin-V2-Tiny + CBAM-UNet + Deep Supervision',
        'rationale':          ('Self-attention captures non-local feature similarity, '
                               'which is exactly what copy-move forgery requires '
                               '(matching distant patches).'),
        'timm_name':          'swinv2_tiny_window8_256',
        'features_only_kwargs': ', img_size=512',
        'levels':             4,
        'channels_last':      True,
        'batch_size':         12,       # single GPU ~11 GB on T4
        'epochs':             60,
        'grad_accum':         1,        # effective batch 12
        'enc_lr_mult':        0.1,
        'lr':                 3e-4,
        'patience':           8,
        'grad_checkpoint':    False,
        'output_dir':         'swin_v2_tiny',
        'notebook_idx':       2,
    },
    'swin_v2_base': {
        'display':            'Swin-V2-Base + CBAM-UNet + Deep Supervision',
        'rationale':          ('Heavy attention encoder with ~88M params. Targets '
                               'maximum specificity (the weak point of the baseline). '
                               'Trained with gradient checkpointing + grad accumulation.'),
        'timm_name':          'swinv2_base_window8_256',
        'features_only_kwargs': ', img_size=512',
        'levels':             4,
        'channels_last':      True,
        'batch_size':         4,        # single GPU + grad ckpt -> ~10 GB
        'epochs':             60,
        'grad_accum':         4,        # effective batch 16
        'enc_lr_mult':        0.1,
        'lr':                 2e-4,
        'patience':           8,
        'grad_checkpoint':    True,
        'output_dir':         'swin_v2_base',
        'notebook_idx':       3,
    },
}


# ── Cell builders ─────────────────────────────────────────────────────────────

def title_cell(cfg: dict) -> dict:
    eff_bs = cfg['batch_size'] * cfg['grad_accum']
    return md(
        f"# {cfg['display']}\n\n"
        f"**Notebook:** `train_{cfg['notebook_idx']}_{cfg['output_dir']}.ipynb`  \n"
        f"**Output dir:** `outputs/{cfg['output_dir']}/`  \n"
        f"**Hardware:** Kaggle T4 — **single GPU + AMP** (DataParallel is broken with timm ConvNeXt/Swin under AMP — see notebook header for explanation)  \n"
        f"**Batch size:** {cfg['batch_size']}  ×  grad_accum {cfg['grad_accum']}  =  effective {eff_bs}  \n\n"
        f"## Rationale\n{cfg['rationale']}\n\n"
        f"## ⚠ If you previously ran this notebook with DataParallel\n"
        f"**Click `Run → Restart Kernel & Clear All Outputs`, then `Run All`.**\n"
        f"A stale `nn.DataParallel` wrapper in kernel memory will trigger a\n"
        f"`misaligned address` error even after the notebook is updated, because\n"
        f"the old wrapper persists across cell re-runs.\n\n"
        f"## Crash-safety defaults applied\n"
        f"- `num_workers=0` (eliminates `/dev/shm` worker-death failures — original crash root cause)\n"
        f"- DataParallel **permanently disabled** (incompatible with AMP + ConvNeXt/Swin)\n"
        f"- Defensive unwrap of any stale `DataParallel` wrap from prior runs\n"
        f"- Smoke-test cell hard-asserts no DP wrap before the first forward pass\n"
        f"- Modern AMP API with legacy fallback\n"
        f"- Smoke-test cell runs 1 fwd+bwd pass before the full loop\n"
        f"- Periodic `torch.cuda.empty_cache()` between train and val phases\n\n"
        f"## Reproducibility\n"
        f"- `SEED=42` and identical UID-level split as `review2.ipynb`  \n"
        f"- The same 713-sample validation set is used by every model for fair comparison.\n\n"
        f"## End-of-notebook contract\n"
        f"- Saves `outputs/{cfg['output_dir']}/model.pt` (state dict + threshold + meta)\n"
        f"- Saves `metrics.json`, `history.csv`, `dashboard.png`, `predictions.png`, `training_curves.png`\n"
        f"- Releases all CUDA memory and prints peak VRAM"
    )


def install_cell() -> dict:
    return code(
        "import subprocess, sys\n"
        "subprocess.run(\n"
        "    [sys.executable, '-m', 'pip', 'install', '-q',\n"
        "     'albumentations>=1.4,<2.0', 'timm>=0.9.16', 'seaborn'],\n"
        "    check=True\n"
        ")\n"
        "print('Packages installed.')"
    )


def imports_cell() -> dict:
    return code(
        "import os, sys, gc, json, random, time, warnings, csv\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import cv2\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler\n"
        "from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR\n"
        "import timm\n"
        "import albumentations as A\n"
        "from albumentations.pytorch import ToTensorV2\n"
        "from sklearn.metrics import (\n"
        "    roc_auc_score, accuracy_score, confusion_matrix,\n"
        "    f1_score, roc_curve, classification_report,\n"
        ")\n"
        "from scipy.ndimage import label as scipy_label\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from tqdm.auto import tqdm\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Modern AMP API (PyTorch 2.4+). Falls back to legacy if unavailable.\n"
        "try:\n"
        "    from torch.amp import GradScaler as _AmpGradScaler, autocast as _amp_autocast\n"
        "    def make_scaler():\n"
        "        return _AmpGradScaler('cuda')\n"
        "    def autocast():\n"
        "        return _amp_autocast('cuda')\n"
        "except ImportError:\n"
        "    from torch.cuda.amp import GradScaler as _AmpGradScaler, autocast as _legacy_autocast\n"
        "    def make_scaler():\n"
        "        return _AmpGradScaler()\n"
        "    def autocast():\n"
        "        return _legacy_autocast()\n"
        "\n"
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print(f'PyTorch  : {torch.__version__}')\n"
        "print(f'Device   : {DEVICE}')\n"
        "print(f'Albu     : {A.__version__}')\n"
        "print(f'Timm     : {timm.__version__}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'CUDA cnt : {torch.cuda.device_count()}')\n"
        "    for i in range(torch.cuda.device_count()):\n"
        "        cap   = torch.cuda.get_device_capability(i)\n"
        "        total = torch.cuda.get_device_properties(i).total_memory / 1e9\n"
        "        free, _ = torch.cuda.mem_get_info(i)\n"
        "        print(f'  GPU {i} : {torch.cuda.get_device_name(i)} '\n"
        "              f'(sm_{cap[0]}{cap[1]})  total={total:.1f} GB  free={free/1e9:.1f} GB')\n"
        "    # /dev/shm size — shared memory used by DataLoader workers\n"
        "    try:\n"
        "        import shutil\n"
        "        shm = shutil.disk_usage('/dev/shm')\n"
        "        print(f'/dev/shm : total={shm.total/1e9:.2f} GB  free={shm.free/1e9:.2f} GB')\n"
        "    except Exception:\n"
        "        pass"
    )


def config_cell(cfg: dict) -> dict:
    return code(
        f"# ── Model identity ───────────────────────────────────────────────────────────\n"
        f"MODEL_NAME = {cfg['output_dir']!r}\n"
        f"DISPLAY    = {cfg['display']!r}\n"
        f"TIMM_NAME  = {cfg['timm_name']!r}\n"
        f"\n"
        f"# ── Paths ────────────────────────────────────────────────────────────────────\n"
        f"INPUT_DIR   = '/kaggle/input/datasets/llkh0a/recod-ailuc-scientific-image-forgery-detection'\n"
        f"if not os.path.isdir(INPUT_DIR):\n"
        f"    INPUT_DIR = '/kaggle/input/recod-ailuc-scientific-image-forgery-detection'\n"
        f"TRAIN_IMGS  = f'{{INPUT_DIR}}/train_images'\n"
        f"TRAIN_MASKS = f'{{INPUT_DIR}}/train_masks'\n"
        f"SUPP_IMGS   = f'{{INPUT_DIR}}/supplemental_images'\n"
        f"SUPP_MASKS  = f'{{INPUT_DIR}}/supplemental_masks'\n"
        f"\n"
        f"OUT_DIR = Path('/kaggle/working/outputs') / MODEL_NAME\n"
        f"OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        f"BEST_PATH  = OUT_DIR / 'best.pt'\n"
        f"FINAL_PATH = OUT_DIR / 'model.pt'\n"
        f"\n"
        f"# ── Training ─────────────────────────────────────────────────────────────────\n"
        f"IMG_SIZE   = 512\n"
        f"BATCH_SIZE = {cfg['batch_size']}\n"
        f"GRAD_ACCUM = {cfg['grad_accum']}\n"
        f"MAX_EPOCHS = {cfg['epochs']}\n"
        f"LR         = {cfg['lr']}\n"
        f"ENC_LR     = LR * {cfg['enc_lr_mult']}\n"
        f"PATIENCE   = {cfg['patience']}\n"
        f"VAL_FRAC   = 0.15\n"
        f"SEED       = 42\n"
        f"CLIP_NORM  = 1.0\n"
        f"USE_GRAD_CHECKPOINT = {cfg['grad_checkpoint']}\n"
        f"\n"
        f"# ── Loss weights ─────────────────────────────────────────────────────────────\n"
        f"CLS_WEIGHT = 2.0\n"
        f"SEG_WEIGHT = 1.0\n"
        f"AUX_WEIGHT = 0.4\n"
        f"\n"
        f"IMAGENET_MEAN = [0.485, 0.456, 0.406]\n"
        f"IMAGENET_STD  = [0.229, 0.224, 0.225]\n"
        f"\n"
        f"# Reproducibility — Python/NumPy/Torch RNGs deterministic, cuDNN kernels NOT\n"
        f"# strictly deterministic (cudnn.deterministic=True breaks DataParallel + AMP\n"
        f"# + ConvNeXt/Swin permute flows with 'misaligned address' on cuBLAS GEMM).\n"
        f"random.seed(SEED)\n"
        f"np.random.seed(SEED)\n"
        f"torch.manual_seed(SEED)\n"
        f"if torch.cuda.is_available():\n"
        f"    torch.cuda.manual_seed_all(SEED)\n"
        f"    torch.backends.cudnn.deterministic    = False\n"
        f"    torch.backends.cudnn.benchmark        = True   # autotune for fixed shape\n"
        f"    torch.backends.cudnn.allow_tf32       = True\n"
        f"    torch.backends.cuda.matmul.allow_tf32 = True\n"
        f"\n"
        f"print(f'Model    : {{DISPLAY}}')\n"
        f"print(f'Output   : {{OUT_DIR}}')\n"
        f"print(f'BS={{BATCH_SIZE}}  GradAcc={{GRAD_ACCUM}}  LR={{LR:.0e}}  Enc LR={{ENC_LR:.0e}}  Epochs={{MAX_EPOCHS}}')"
    )


DATASET_CELL = """\
def load_mask(path, h, w):
    \"\"\"Load .npy ground-truth mask, union over multi-region, resize if needed.\"\"\"
    m = np.load(path)
    if m.ndim == 3:
        m = m.max(axis=0)
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.float32)


class ForgeryDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(s['img']), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = (load_mask(s['mask'], h, w)
                if s['label'] == 1 and s['mask']
                else np.zeros((h, w), dtype=np.float32))
        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out['image']
            mask = out['mask']
        return img, torch.tensor(s['label'], dtype=torch.float32), mask.unsqueeze(0)


def build_samples():
    samples, uids = [], []
    auth_dir = f'{TRAIN_IMGS}/authentic'
    forg_dir = f'{TRAIN_IMGS}/forged'
    auth_uids = {os.path.splitext(f)[0] for f in os.listdir(auth_dir) if f.endswith('.png')}
    forg_uids = {os.path.splitext(f)[0] for f in os.listdir(forg_dir) if f.endswith('.png')}
    for uid in sorted(auth_uids & forg_uids):
        uids.append(uid)
        samples.append({'uid': uid, 'img': f'{auth_dir}/{uid}.png',
                        'label': 0, 'mask': None})
        mp = f'{TRAIN_MASKS}/{uid}.npy'
        if os.path.exists(mp):
            samples.append({'uid': uid, 'img': f'{forg_dir}/{uid}.png',
                            'label': 1, 'mask': mp})
    if os.path.isdir(SUPP_IMGS):
        for fname in sorted(os.listdir(SUPP_IMGS)):
            if not fname.endswith('.png'):
                continue
            uid = os.path.splitext(fname)[0]
            mp  = f'{SUPP_MASKS}/{uid}.npy'
            if os.path.exists(mp):
                samples.append({'uid': uid, 'img': f'{SUPP_IMGS}/{fname}',
                                'label': 1, 'mask': mp, 'supplemental': True})
    return samples, uids


all_samples, all_uids = build_samples()
shuffled    = sorted(all_uids)
random.shuffle(shuffled)
n_val       = int(VAL_FRAC * len(shuffled))
val_uids    = set(shuffled[:n_val])
train_uids  = set(shuffled[n_val:])

train_samples = [s for s in all_samples if s.get('supplemental') or s['uid'] in train_uids]
val_samples   = [s for s in all_samples if s['uid'] in val_uids]

print(f'Train: {len(train_samples)}  '
      f'(auth={sum(s["label"]==0 for s in train_samples)}, '
      f'forg={sum(s["label"]==1 for s in train_samples)})')
print(f'Val  : {len(val_samples)}  '
      f'(auth={sum(s["label"]==0 for s in val_samples)}, '
      f'forg={sum(s["label"]==1 for s in val_samples)})')
"""


AUG_CELL = """\
train_tf = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
        A.CLAHE(clip_limit=2.0),
    ], p=0.6),
    A.OneOf([
        A.ImageCompression(quality_range=(70, 100)),
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(std_range=(0.02, 0.1)),
    ], p=0.4),
    A.CoarseDropout(num_holes_range=(1, 4),
                    hole_height_range=(16, 48),
                    hole_width_range=(16, 48),
                    fill=0, p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
print('Augmentation pipelines ready.')
"""


LOADERS_CELL = """\
train_ds = ForgeryDataset(train_samples, train_tf)
val_ds   = ForgeryDataset(val_samples,   val_tf)

train_labels   = [s['label'] for s in train_samples]
class_counts   = np.bincount(train_labels)
sample_weights = [1.0 / class_counts[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# num_workers=0 on Kaggle: avoids /dev/shm pressure that silently kills workers
# and triggers main-loop hangs. Slightly slower I/O but rock-solid stability.
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '0'))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=False)
print(f'Train batches: {len(train_loader)}   Val batches: {len(val_loader)}'
      f'   workers={NUM_WORKERS}')
"""


def architecture_cell(cfg: dict) -> dict:
    levels = cfg['levels']
    timm_name = cfg['timm_name']
    fo_kwargs = cfg['features_only_kwargs']
    channels_last = cfg['channels_last']
    grad_ckpt = cfg['grad_checkpoint']

    # 4-level UNet for ConvNeXt / Swin
    src = (
        "class CBAM(nn.Module):\n"
        "    def __init__(self, channels, reduction=16):\n"
        "        super().__init__()\n"
        "        mid = max(channels // reduction, 4)\n"
        "        self.channel_fc = nn.Sequential(\n"
        "            nn.Linear(channels, mid, bias=False),\n"
        "            nn.ReLU(inplace=True),\n"
        "            nn.Linear(mid, channels, bias=False),\n"
        "        )\n"
        "        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)\n"
        "\n"
        "    def forward(self, x):\n"
        "        b, c = x.shape[:2]\n"
        "        avg = self.channel_fc(F.adaptive_avg_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)\n"
        "        mx  = self.channel_fc(F.adaptive_max_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)\n"
        "        x = x * torch.sigmoid(avg + mx)\n"
        "        spatial = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)\n"
        "        x = x * torch.sigmoid(self.spatial_conv(spatial))\n"
        "        return x\n"
        "\n"
        "\n"
        "class ConvBNReLU(nn.Sequential):\n"
        "    def __init__(self, in_c, out_c, k=3, p=1):\n"
        "        super().__init__(\n"
        "            nn.Conv2d(in_c, out_c, k, padding=p, bias=False),\n"
        "            nn.BatchNorm2d(out_c),\n"
        "            nn.ReLU(inplace=True),\n"
        "        )\n"
        "\n"
        "\n"
        "class DecoderBlock(nn.Module):\n"
        "    def __init__(self, in_c, skip_c, out_c):\n"
        "        super().__init__()\n"
        "        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)\n"
        "        self.conv = nn.Sequential(\n"
        "            ConvBNReLU(in_c + skip_c, out_c),\n"
        "            ConvBNReLU(out_c, out_c),\n"
        "        )\n"
        "        self.cbam = CBAM(out_c)\n"
        "\n"
        "    def forward(self, x, skip=None):\n"
        "        x = self.up(x)\n"
        "        if skip is not None:\n"
        "            if x.shape[-2:] != skip.shape[-2:]:\n"
        "                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)\n"
        "            x = torch.cat([x, skip], dim=1)\n"
        "        return self.cbam(self.conv(x))\n"
        "\n"
    )

    if levels == 4:
        src += (
            "class FourLevelUNet(nn.Module):\n"
            "    \"\"\"UNet for 4-level encoders (ConvNeXt / Swin).\n"
            "\n"
            "    Uses the known expected channel count from feature_info to\n"
            "    disambiguate NCHW vs NHWC layout robustly. timm's Swin-V2\n"
            "    returns NHWC; ConvNeXt returns NCHW.\n"
            "    \"\"\"\n"
            "    def __init__(self, encoder, ch, channels_last_input=False):\n"
            "        super().__init__()\n"
            "        self.encoder = encoder\n"
            "        self.expected_channels = list(ch)\n"
            "        self.channels_last_input = channels_last_input\n"
            "        self.d3 = DecoderBlock(ch[3], ch[2], 256)\n"
            "        self.d2 = DecoderBlock(256,   ch[1], 128)\n"
            "        self.d1 = DecoderBlock(128,   ch[0],  64)\n"
            "        self.d0a = DecoderBlock(64,        0,  32)\n"
            "        self.d0b = DecoderBlock(32,        0,  16)\n"
            "        self.seg_head  = nn.Conv2d(16, 1, 1)\n"
            "        self.aux3_head = nn.Conv2d(256, 1, 1)\n"
            "        self.aux2_head = nn.Conv2d(128, 1, 1)\n"
            "        self.cls_head = nn.Sequential(\n"
            "            nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n"
            "            nn.Dropout(0.4),\n"
            "            nn.Linear(ch[3], 512), nn.ReLU(inplace=True),\n"
            "            nn.Dropout(0.3),\n"
            "            nn.Linear(512, 1),\n"
            "        )\n"
            "\n"
            "    @staticmethod\n"
            "    def _ensure_nchw(t, expected_c):\n"
            "        if t.dim() != 4:\n"
            "            return t\n"
            "        if t.shape[1] == expected_c:    # already NCHW\n"
            "            return t\n"
            "        if t.shape[-1] == expected_c:   # NHWC -> permute\n"
            "            return t.permute(0, 3, 1, 2).contiguous()\n"
            "        return t  # fall through; shape might be ambiguous\n"
            "\n"
            "    def forward(self, x):\n"
            "        feats = self.encoder(x)\n"
            "        feats = [self._ensure_nchw(f, c)\n"
            "                 for f, c in zip(feats, self.expected_channels)]\n"
            "        f0, f1, f2, f3 = feats\n"
            "        cls_out = self.cls_head(f3)\n"
            "        d3 = self.d3(f3, f2)\n"
            "        d2 = self.d2(d3, f1)\n"
            "        d1 = self.d1(d2, f0)\n"
            "        d0a = self.d0a(d1)\n"
            "        d0  = self.d0b(d0a)\n"
            "        seg = self.seg_head(d0)\n"
            "        if seg.shape[-2:] != x.shape[-2:]:\n"
            "            seg = F.interpolate(seg, size=x.shape[-2:], mode='bilinear', align_corners=False)\n"
            "        return cls_out, seg, self.aux3_head(d3), self.aux2_head(d2)\n"
            "\n"
        )

    src += (
        f"# Build encoder + model\n"
        f"encoder = timm.create_model({timm_name!r}, pretrained=True, features_only=True{fo_kwargs})\n"
        f"channels = encoder.feature_info.channels()\n"
        f"print(f'Encoder feature channels: {{channels}}')\n"
        f"\n"
        f"model = FourLevelUNet(encoder, channels, channels_last_input={channels_last}).to(DEVICE)\n"
    )

    if grad_ckpt:
        src += (
            "\n"
            "# Gradient checkpointing on encoder (memory savings for heavy model)\n"
            "if hasattr(encoder, 'set_grad_checkpointing'):\n"
            "    encoder.set_grad_checkpointing(True)\n"
            "    print('Gradient checkpointing enabled on encoder.')\n"
        )

    src += (
        "\n"
        "total = sum(p.numel() for p in model.parameters())\n"
        "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "print(f'Params: total={total:,}  trainable={trainable:,}')\n"
    )

    return code(src)


LOSSES_CELL = """\
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p   = torch.sigmoid(logits)
        pt  = targets * p + (1 - targets) * (1 - p)
        at  = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (at * (1 - pt) ** self.gamma * bce).mean()


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1e-7):
    p  = torch.sigmoid(logits)
    tp = (p * targets).sum()
    fp = (p * (1 - targets)).sum()
    fn = ((1 - p) * targets).sum()
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def seg_loss(pred, gt):
    return F.binary_cross_entropy_with_logits(pred, gt) + tversky_loss(pred, gt)


_focal = FocalLoss(alpha=0.5, gamma=2.0)


def multitask_loss(cls_out, seg_out, aux_a, aux_b, labels, masks):
    cls_l = _focal(cls_out.squeeze(1), labels)
    seg_l = torch.tensor(0.0, device=cls_out.device)
    forged = labels == 1
    if forged.any():
        gt_a = F.adaptive_avg_pool2d(masks[forged], aux_a.shape[-2:])
        gt_b = F.adaptive_avg_pool2d(masks[forged], aux_b.shape[-2:])
        seg_l = (
            seg_loss(seg_out[forged], masks[forged])
            + AUX_WEIGHT * seg_loss(aux_a[forged], gt_a)
            + AUX_WEIGHT * seg_loss(aux_b[forged], gt_b)
        )
    total = CLS_WEIGHT * cls_l + SEG_WEIGHT * seg_l
    return total, cls_l.detach(), seg_l.detach()


print('Loss functions ready.')
"""


OPTIM_CELL = """\
enc_params = list(model.encoder.parameters())
dec_params = [p for n, p in model.named_parameters() if not n.startswith('encoder')]

optimizer = torch.optim.AdamW(
    [
        {'params': enc_params, 'lr': ENC_LR},
        {'params': dec_params, 'lr': LR},
    ],
    weight_decay=1e-4,
)

warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - 5, eta_min=LR * 0.01)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# DataParallel is permanently DISABLED. It's incompatible with AMP + timm
# ConvNeXt/Swin (cuBLAS misaligned-address bug in Linear-after-permute) and
# has been deprecated by PyTorch. Single-GPU FP16 on T4 is ~4x faster than
# 2-GPU FP32 anyway. To use multiple GPUs, switch to DDP via a .py script.
#
# Defensive unwrap: if a stale DataParallel-wrapped model lingers in kernel
# state (e.g. from a previous training attempt), strip the wrapper here.
if isinstance(model, nn.DataParallel):
    print('Stripping stale DataParallel wrapper from model.')
    model = model.module
if torch.cuda.device_count() > 1:
    print(f'{torch.cuda.device_count()} GPUs visible; using GPU 0 only '
          f'(DataParallel removed; single-GPU FP16 is ~4x faster on T4).')

USE_AMP = torch.cuda.is_available()
scaler  = make_scaler() if USE_AMP else None
print(f'AMP: {USE_AMP}  |  GradAccum: {GRAD_ACCUM}  |  Effective batch: {BATCH_SIZE * GRAD_ACCUM}')
"""


DP_SAFE_CUDNN_CELL = """\
# ── DataParallel + AMP safety: cuDNN tuning ─────────────────────────────────
# Ensures the 'misaligned address' bug does not hit the cuBLAS GEMM inside
# ConvNeXt/Swin MLP blocks when DataParallel scatters a half-batch onto GPU 1.
# Idempotent: safe to re-run.
torch.backends.cudnn.deterministic    = False   # let cuDNN pick aligned kernels
torch.backends.cudnn.benchmark        = True    # autotune for fixed input shape
torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True
print('cuDNN preset for DP+AMP:')
print(f'  deterministic = {torch.backends.cudnn.deterministic}')
print(f'  benchmark     = {torch.backends.cudnn.benchmark}')
print(f'  allow_tf32    = {torch.backends.cudnn.allow_tf32}')
"""


SMOKE_TEST_CELL = """\
# ── Smoke test: 1 forward + backward pass to catch OOM/shape errors fast ─────
import gc

# Hard-fail if stale DataParallel wrapper is present (incompatible with AMP)
assert not isinstance(model, nn.DataParallel), (
    'Model is wrapped in DataParallel. This is incompatible with AMP + '
    'timm ConvNeXt/Swin (misaligned-address bug). Click '
    '"Run -> Restart Kernel & Clear All Outputs" then "Run All" to reset state.'
)

print('Smoke test: 1 train iter + 1 val batch...')
model.train()
imgs, labels, masks = next(iter(train_loader))
imgs   = imgs.to(DEVICE, non_blocking=True)
labels = labels.to(DEVICE, non_blocking=True)
masks  = masks.to(DEVICE, non_blocking=True)

if USE_AMP:
    with autocast():
        cls_out, seg_out, aux_a, aux_b = model(imgs)
        loss, _, _ = multitask_loss(cls_out, seg_out, aux_a, aux_b, labels, masks)
else:
    cls_out, seg_out, aux_a, aux_b = model(imgs)
    loss, _, _ = multitask_loss(cls_out, seg_out, aux_a, aux_b, labels, masks)

print(f'  fwd ok      loss={loss.item():.4f}')
print(f'  cls_out     {tuple(cls_out.shape)}')
print(f'  seg_out     {tuple(seg_out.shape)}')

if USE_AMP:
    scaler.scale(loss).backward()
else:
    loss.backward()
print('  bwd ok')

optimizer.zero_grad(set_to_none=True)

# One val batch
model.train(False)
with torch.no_grad():
    imgs_v, labels_v, masks_v = next(iter(val_loader))
    imgs_v = imgs_v.to(DEVICE, non_blocking=True)
    cls_out, seg_out, _, _ = model(imgs_v)
print(f'  val fwd ok  cls={tuple(cls_out.shape)} seg={tuple(seg_out.shape)}')

# Free smoke test tensors
del imgs, labels, masks, imgs_v, labels_v, masks_v, cls_out, seg_out, aux_a, aux_b, loss
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'  peak VRAM during smoke: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    torch.cuda.reset_peak_memory_stats()

print('Smoke test PASSED. Safe to run full training loop below.')
"""


TRAIN_VAL_CELL = """\
def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    losses, cls_ls, seg_ls = [], [], []
    pbar = tqdm(loader, desc=f'Epoch {epoch:02d} [Train]', leave=False)
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    for imgs, labels, masks in pbar:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        if USE_AMP:
            with autocast():
                cls_out, seg_out, aux_a, aux_b = model(imgs)
                loss, cl, sl = multitask_loss(cls_out, seg_out, aux_a, aux_b, labels, masks)
                loss = loss / GRAD_ACCUM
        else:
            cls_out, seg_out, aux_a, aux_b = model(imgs)
            loss, cl, sl = multitask_loss(cls_out, seg_out, aux_a, aux_b, labels, masks)
            loss = loss / GRAD_ACCUM

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum += 1
        if accum >= GRAD_ACCUM:
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0

        losses.append(loss.item() * GRAD_ACCUM)
        cls_ls.append(cl.item())
        seg_ls.append(sl.item())
        pbar.set_postfix({
            'loss': f'{np.mean(losses):.4f}',
            'cls':  f'{np.mean(cls_ls):.4f}',
            'seg':  f'{np.mean(seg_ls):.4f}',
        })

    return float(np.mean(losses)), float(np.mean(cls_ls)), float(np.mean(seg_ls))


@torch.no_grad()
def validate(model, loader, threshold=0.5):
    model.train(False)
    y_true, y_prob, dice_scores = [], [], []
    for imgs, labels, masks in tqdm(loader, desc='Val', leave=False):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)
        cls_out, seg_out, _, _ = model(imgs)
        probs = torch.sigmoid(cls_out).squeeze(1)
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())
        for i in range(len(labels)):
            if labels[i].item() == 1:
                pred  = (torch.sigmoid(seg_out[i, 0]) > threshold).float()
                gt    = masks[i, 0]
                inter = (pred * gt).sum()
                dice_scores.append((2 * inter / (pred.sum() + gt.sum() + 1e-7)).item())

    y_pred = (np.array(y_prob) > threshold).astype(int)
    acc    = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    dice  = float(np.mean(dice_scores)) if dice_scores else 0.0
    cm    = confusion_matrix(y_true, y_pred)
    fgrec = (cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-7) if cm.shape == (2, 2) else 0.0)
    comp  = 0.4 * auc + 0.3 * fgrec + 0.3 * dice
    return {'acc': acc, 'auc': auc, 'dice': dice, 'fgrec': fgrec, 'comp': comp}


print('Training & validation functions ready.')
"""


TRAIN_LOOP_CELL = """\
history      = []
best_comp    = -float('inf')
patience_ctr = 0
start_time   = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_cls, tr_seg = train_one_epoch(model, train_loader, optimizer, epoch)

    # Free any lingering activation cache between train and val phases
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_m = validate(model, val_loader)
    scheduler.step()

    # End-of-epoch cleanup (prevents slow-leak OOMs on long runs)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    row = {
        'epoch': epoch,
        'tr_loss': tr_loss, 'tr_cls': tr_cls, 'tr_seg': tr_seg,
        **{f'val_{k}': v for k, v in val_m.items()},
    }
    history.append(row)

    print(f'[{epoch:02d}/{MAX_EPOCHS}] '
          f'loss={tr_loss:.4f} cls={tr_cls:.4f} seg={tr_seg:.4f} | '
          f'Acc={val_m["acc"]:.4f} AUC={val_m["auc"]:.4f} '
          f'Dice={val_m["dice"]:.4f} FgRec={val_m["fgrec"]:.4f} '
          f'Comp={val_m["comp"]:.4f}')

    if val_m['comp'] > best_comp:
        best_comp = val_m['comp']
        patience_ctr = 0
        # DataParallel was permanently disabled; defensive unwrap kept just
        # in case kernel state still has a wrapped model from a prior session.
        sd = (model.module.state_dict()
              if isinstance(model, nn.DataParallel) else model.state_dict())
        torch.save(sd, BEST_PATH)
        print(f'  -> Best checkpoint saved  (comp={best_comp:.4f})')
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f'Early stopping at epoch {epoch}.')
            break

elapsed = time.time() - start_time
print(f'\\nTraining complete. Best composite: {best_comp:.4f}  Elapsed: {elapsed/3600:.2f} h')

# Save history
import csv
with open(OUT_DIR / 'history.csv', 'w', newline='') as f:
    if history:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
print(f'history.csv saved.')
"""


TRAIN_CURVES_CELL = """\
epochs   = [r['epoch']     for r in history]
tr_loss  = [r['tr_loss']   for r in history]
val_auc  = [r['val_auc']   for r in history]
val_dice = [r['val_dice']  for r in history]
val_comp = [r['val_comp']  for r in history]
val_fgr  = [r['val_fgrec'] for r in history]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(f'Training History — {DISPLAY}', fontsize=14, fontweight='bold')

axes[0, 0].plot(epochs, tr_loss, color='steelblue', lw=2)
axes[0, 0].set_title('Train Loss'); axes[0, 0].set_xlabel('Epoch')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(epochs, val_auc, color='darkorange', lw=2)
axes[0, 1].set_title('Val AUC'); axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylim(0.4, 1.0); axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(epochs, val_dice, color='mediumseagreen', lw=2)
axes[1, 0].set_title('Val Dice'); axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylim(0, 1.0); axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(epochs, val_comp, color='mediumpurple', lw=2, label='Composite')
axes[1, 1].plot(epochs, val_fgr,  color='tomato',       lw=2, linestyle='--', label='FgRecall')
best_ep = epochs[int(np.argmax(val_comp))]
axes[1, 1].axvline(best_ep, color='k', linestyle=':', lw=1.2, label=f'Best ep={best_ep}')
axes[1, 1].set_title('Composite & Forged Recall'); axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylim(0, 1.0); axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print(f'training_curves.png saved.')
"""


EVAL_CELL = """\
# Reload best checkpoint into a fresh model copy (no DataParallel wrapper for eval)
state = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
eval_encoder = timm.create_model(TIMM_NAME, pretrained=False, features_only=True__FO_KWARGS__)
eval_channels = eval_encoder.feature_info.channels()
eval_model = FourLevelUNet(eval_encoder, eval_channels,
                            channels_last_input=__CHANNELS_LAST__).to(DEVICE)
eval_model.load_state_dict(state, strict=False)
eval_model.train(False)
print('Best checkpoint loaded for evaluation.')

# Defensive rebuild: if val_loader was freed or kernel restarted, recreate it
# using the SAME SEED + split logic as the original DataLoaders cell.
try:
    val_loader
except NameError:
    print('val_loader not in scope — rebuilding from val_samples...')
    if 'val_samples' not in globals():
        all_samples, all_uids = build_samples()
        shuffled = sorted(all_uids)
        random.shuffle(shuffled)
        n_val = int(VAL_FRAC * len(shuffled))
        val_uid_set = set(shuffled[:n_val])
        val_samples = [s for s in all_samples if s['uid'] in val_uid_set]
    if 'val_tf' not in globals():
        val_tf = A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    val_ds = ForgeryDataset(val_samples, val_tf)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True,
                            persistent_workers=False)
    print(f'  -> {len(val_loader)} val batches')

# Collect predictions
y_true, y_prob = [], []
seg_probs_all, masks_all = [], []
with torch.no_grad():
    for imgs, labels, masks in tqdm(val_loader, desc='Eval'):
        imgs = imgs.to(DEVICE)
        cls_out, seg_out, _, _ = eval_model(imgs)
        probs     = torch.sigmoid(cls_out).squeeze(1).cpu().numpy()
        seg_probs = torch.sigmoid(seg_out).squeeze(1).cpu().numpy()
        mask_np   = masks.squeeze(1).numpy()
        y_true.extend(labels.numpy())
        y_prob.extend(probs)
        for i in range(len(labels)):
            if labels[i] == 1:
                seg_probs_all.append(seg_probs[i])
                masks_all.append(mask_np[i])

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Threshold sweep for best F1
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.05, 0.95, 0.01):
    preds = (y_prob > thr).astype(int)
    f1    = f1_score(y_true, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, float(thr)

# Dice computed at same threshold
dice_list = []
for sp, gt in zip(seg_probs_all, masks_all):
    pred_bin = (sp > best_thr).astype(np.float32)
    inter    = (pred_bin * gt).sum()
    denom    = pred_bin.sum() + gt.sum() + 1e-7
    dice_list.append(2 * inter / denom)

y_pred = (y_prob > best_thr).astype(int)
cm     = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
auc  = roc_auc_score(y_true, y_prob)
dice = float(np.mean(dice_list)) if dice_list else float('nan')
acc  = accuracy_score(y_true, y_pred)
prec = tp / (tp + fp + 1e-7)
rec  = tp / (tp + fn + 1e-7)
spec = tn / (tn + fp + 1e-7)
f1v  = 2 * tp / (2 * tp + fp + fn + 1e-7)

metrics = {
    'model_name': MODEL_NAME,
    'display':    DISPLAY,
    'threshold':  float(best_thr),
    'accuracy':   float(acc),
    'auc':        float(auc),
    'precision':  float(prec),
    'forged_recall': float(rec),
    'specificity': float(spec),
    'f1':         float(f1v),
    'mean_dice':  float(dice),
    'composite':  float(0.4 * auc + 0.3 * rec + 0.3 * dice),
    'cm_tn':      int(tn), 'cm_fp': int(fp), 'cm_fn': int(fn), 'cm_tp': int(tp),
}

with open(OUT_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
"""


DASHBOARD_CELL = """\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Validation — {DISPLAY}', fontsize=14, fontweight='bold')

ax = axes[0]
ax.hist(y_prob[y_true == 0], bins=40, alpha=0.65, color='steelblue',
        label=f'Authentic n={(y_true==0).sum()}')
ax.hist(y_prob[y_true == 1], bins=40, alpha=0.65, color='tomato',
        label=f'Forged    n={(y_true==1).sum()}')
ax.axvline(best_thr, color='k', linestyle='--', lw=1.5,
           label=f'thr={best_thr:.2f}')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Count')
ax.set_title('Score Distribution'); ax.legend(fontsize=9)

ax = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Authentic', 'Forged'],
            yticklabels=['Authentic', 'Forged'],
            annot_kws={'size': 16})
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (thr={best_thr:.2f})')

ax = axes[2]
fpr, tpr, _ = roc_curve(y_true, y_prob)
ax.plot(fpr, tpr, lw=2.5, color='darkorange', label=f'AUC={auc:.4f}')
ax.fill_between(fpr, tpr, alpha=0.10, color='darkorange')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curve'); ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / 'dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print('dashboard.png saved.')

print('=' * 50)
for k, v in metrics.items():
    if isinstance(v, float):
        print(f'  {k:18s}: {v:.4f}')
    else:
        print(f'  {k:18s}: {v}')
print('=' * 50)
print()
print(classification_report(y_true, y_pred, target_names=['Authentic', 'Forged']))
"""


PRED_GRID_CELL = """\
# Forged val examples grid: original | overlay | heatmap | GT mask
def postprocess_mask(prob, threshold=0.5, min_area=200):
    binary = (prob > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    labeled, n = scipy_label(binary)
    out = np.zeros_like(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_area:
            out[labeled == i] = 1
    return out.astype(np.float32)


N_SHOW = 12
forged_samples = [s for s in val_samples if s['label'] == 1]
show = random.sample(forged_samples, min(N_SHOW, len(forged_samples)))

if not show:
    print('No forged val samples.')
else:
    fig, axes = plt.subplots(len(show), 4, figsize=(20, 5 * len(show)))
    if len(show) == 1:
        axes = axes[np.newaxis, :]
    titles = ['Original', 'Overlay (red=pred, green=GT)', 'Heatmap', 'GT mask']
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=10, fontweight='bold')

    test_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    for r, s in enumerate(tqdm(show, desc='Render preds')):
        rgb = cv2.cvtColor(cv2.imread(s['img']), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt = load_mask(s['mask'], h, w) if s['mask'] else np.zeros((h, w), np.float32)

        ten = test_tf(image=rgb)['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cls_o, seg_o, _, _ = eval_model(ten)
        cp = torch.sigmoid(cls_o).item()
        sp = torch.sigmoid(seg_o[0, 0]).cpu().numpy()
        sp_up = cv2.resize(sp, (w, h), interpolation=cv2.INTER_LINEAR)
        bin_mask = postprocess_mask(sp_up, threshold=best_thr, min_area=200)

        ov = rgb.copy().astype(np.float32)
        ov[bin_mask > 0] = ov[bin_mask > 0] * 0.45 + np.array([255, 40, 40]) * 0.55
        ov = np.clip(ov, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(gt.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, contours, -1, (0, 220, 0), 2)

        heat = cv2.cvtColor(
            cv2.applyColorMap((sp_up * 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)

        gt_vis = (gt * 255).astype(np.uint8)
        d = 2 * (bin_mask * gt).sum() / (bin_mask.sum() + gt.sum() + 1e-7)

        axes[r, 0].imshow(rgb)
        axes[r, 0].text(-0.02, 0.5,
                        f'UID:{s["uid"]}\\np={cp:.3f}  D={d:.3f}',
                        transform=axes[r, 0].transAxes,
                        fontsize=7, va='center', ha='right',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        axes[r, 1].imshow(ov)
        axes[r, 2].imshow(heat)
        axes[r, 3].imshow(gt_vis, cmap='gray', vmin=0, vmax=255)
        for ax in axes[r]:
            ax.axis('off')

    plt.suptitle(f'Forged predictions — {DISPLAY}', fontsize=15, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'predictions.png', dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('predictions.png saved.')
"""


SAVE_CELL = """\
# Save final model bundle (for the webapp).
# eval_model holds the best checkpoint reloaded fresh; using it directly
# avoids dependency on `model` (which may have been freed after training).
final_state = eval_model.state_dict()

torch.save({
    'model_state_dict': final_state,
    'cls_threshold':    float(best_thr),
    'seg_threshold':    float(best_thr),
    'img_size':         IMG_SIZE,
    'backbone':         TIMM_NAME,
    'architecture':     DISPLAY,
    'val_auc':          float(auc),
    'val_dice':         float(dice),
    'val_composite':    float(metrics['composite']),
    'val_specificity':  float(spec),
    'val_forged_recall': float(rec),
}, FINAL_PATH)
sz = os.path.getsize(FINAL_PATH) / 1e6
print(f'Saved {FINAL_PATH}  ({sz:.1f} MB)')
print(f'  AUC : {auc:.4f}')
print(f'  Dice: {dice:.4f}')
print(f'  Spec: {spec:.4f}  <-- key metric vs baseline')
"""


CLEANUP_CELL = """\
# ── Memory cleanup (research-grade hygiene) ──────────────────────────────────
print(f'Peak VRAM during run: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

del model, optimizer, scaler, train_loader, val_loader
try:
    del eval_model, eval_encoder
except NameError:
    pass
try:
    del encoder
except NameError:
    pass

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print('Memory released. Restart kernel before running the next training notebook.')
print(f'Outputs at: {OUT_DIR}')
for p in sorted(OUT_DIR.iterdir()):
    print(f'  {p.name}  ({p.stat().st_size/1024:.1f} KB)')
"""


def build_training_notebook(model_id: str, cfg: dict) -> Path:
    eval_cell_src = (EVAL_CELL
                     .replace('__FO_KWARGS__', cfg['features_only_kwargs'])
                     .replace('__CHANNELS_LAST__', str(cfg['channels_last'])))

    cells = [
        title_cell(cfg),
        md('## 1. Environment Setup'),
        install_cell(),
        md('## 2. Imports'),
        imports_cell(),
        md('## 3. Configuration'),
        config_cell(cfg),
        md('## 4. Dataset & Splits'),
        code(DATASET_CELL),
        md('## 5. Augmentation'),
        code(AUG_CELL),
        md('## 6. DataLoaders'),
        code(LOADERS_CELL),
        md('## 7. Model Architecture'),
        architecture_cell(cfg),
        md('## 8. Loss Functions'),
        code(LOSSES_CELL),
        md('## 9. Optimiser & Scheduler'),
        code(OPTIM_CELL),
        md('## 10. Training & Validation Functions'),
        code(TRAIN_VAL_CELL),
        md('## 11a. cuDNN preset for DataParallel + AMP'),
        code(DP_SAFE_CUDNN_CELL),
        md('## 11b. Smoke Test (1 iter, catches OOM/alignment errors fast)'),
        code(SMOKE_TEST_CELL),
        md('## 12. Training Loop'),
        code(TRAIN_LOOP_CELL),
        md('## 13. Training Curves'),
        code(TRAIN_CURVES_CELL),
        md('## 14. Final Evaluation on Val Set'),
        code(eval_cell_src),
        md('## 15. Dashboard'),
        code(DASHBOARD_CELL),
        md('## 16. Qualitative Predictions Grid'),
        code(PRED_GRID_CELL),
        md('## 17. Save Final Model Bundle'),
        code(SAVE_CELL),
        md('## 18. Memory Cleanup (mandatory)'),
        code(CLEANUP_CELL),
    ]

    fname = f'train_{cfg["notebook_idx"]}_{cfg["output_dir"]}.ipynb'
    path = ROOT / fname
    write_nb(path, cells)
    return path


# ── Comparison notebook ──────────────────────────────────────────────────────

COMPARE_TITLE = """\
# Cross-Model Comparison — Validation Set

Loads every available trained model and evaluates on the same val set. Produces:
- `outputs/comparison/comparison.csv` — side-by-side metrics table
- `outputs/comparison/comparison_dashboard.png` — grouped bar charts
- `outputs/comparison/roc_overlay.png` — overlay of all ROC curves
- `outputs/comparison/winner.json` — model with highest composite score

Run after all 3 training notebooks complete (or with whichever models have weights).
"""


COMPARE_CELL_SETUP = """\
import os, sys, gc, json, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, f1_score, roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
VAL_FRAC = 0.15
IMG_SIZE = 512
BATCH_SIZE = 8

import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
"""


COMPARE_CELL_DATA = """\
INPUT_DIR = '/kaggle/input/datasets/llkh0a/recod-ailuc-scientific-image-forgery-detection'
if not os.path.isdir(INPUT_DIR):
    INPUT_DIR = '/kaggle/input/recod-ailuc-scientific-image-forgery-detection'
TRAIN_IMGS = f'{INPUT_DIR}/train_images'
TRAIN_MASKS = f'{INPUT_DIR}/train_masks'
SUPP_IMGS = f'{INPUT_DIR}/supplemental_images'
SUPP_MASKS = f'{INPUT_DIR}/supplemental_masks'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def load_mask(path, h, w):
    m = np.load(path)
    if m.ndim == 3:
        m = m.max(axis=0)
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.float32)


class ForgeryDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(s['img']), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = load_mask(s['mask'], h, w) if s['label'] == 1 and s['mask'] else np.zeros((h, w), np.float32)
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out['image'], out['mask']
        return img, torch.tensor(s['label'], dtype=torch.float32), mask.unsqueeze(0)


def build_samples():
    samples, uids = [], []
    auth_dir = f'{TRAIN_IMGS}/authentic'
    forg_dir = f'{TRAIN_IMGS}/forged'
    auth_uids = {os.path.splitext(f)[0] for f in os.listdir(auth_dir) if f.endswith('.png')}
    forg_uids = {os.path.splitext(f)[0] for f in os.listdir(forg_dir) if f.endswith('.png')}
    for uid in sorted(auth_uids & forg_uids):
        uids.append(uid)
        samples.append({'uid': uid, 'img': f'{auth_dir}/{uid}.png', 'label': 0, 'mask': None})
        mp = f'{TRAIN_MASKS}/{uid}.npy'
        if os.path.exists(mp):
            samples.append({'uid': uid, 'img': f'{forg_dir}/{uid}.png', 'label': 1, 'mask': mp})
    if os.path.isdir(SUPP_IMGS):
        for fname in sorted(os.listdir(SUPP_IMGS)):
            if not fname.endswith('.png'):
                continue
            uid = os.path.splitext(fname)[0]
            mp = f'{SUPP_MASKS}/{uid}.npy'
            if os.path.exists(mp):
                samples.append({'uid': uid, 'img': f'{SUPP_IMGS}/{fname}', 'label': 1, 'mask': mp, 'supplemental': True})
    return samples, uids


all_samples, all_uids = build_samples()
shuffled = sorted(all_uids)
random.shuffle(shuffled)
n_val = int(VAL_FRAC * len(shuffled))
val_uids = set(shuffled[:n_val])
val_samples = [s for s in all_samples if s['uid'] in val_uids]

val_ds = ForgeryDataset(val_samples, val_tf)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
print(f'Val samples: {len(val_samples)}')
"""


COMPARE_CELL_ARCH = """\
# Architecture (same as training notebooks)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        b, c = x.shape[:2]
        avg = self.channel_fc(F.adaptive_avg_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        mx  = self.channel_fc(F.adaptive_max_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        x = x * torch.sigmoid(avg + mx)
        spatial = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        x = x * torch.sigmoid(self.spatial_conv(spatial))
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_c + skip_c, out_c),
            ConvBNReLU(out_c, out_c),
        )
        self.cbam = CBAM(out_c)
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.cbam(self.conv(x))


class FiveLevelUNet(nn.Module):
    def __init__(self, encoder, ch):
        super().__init__()
        self.encoder = encoder
        self.d4 = DecoderBlock(ch[4], ch[3], 256)
        self.d3 = DecoderBlock(256,   ch[2], 128)
        self.d2 = DecoderBlock(128,   ch[1],  64)
        self.d1 = DecoderBlock(64,    ch[0],  32)
        self.d0 = DecoderBlock(32,        0,  16)
        self.seg_head  = nn.Conv2d(16, 1, 1)
        self.aux4_head = nn.Conv2d(256, 1, 1)
        self.aux3_head = nn.Conv2d(128, 1, 1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(ch[4], 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)
        cls_out = self.cls_head(f4)
        d4 = self.d4(f4, f3)
        d3 = self.d3(d4, f2)
        d2 = self.d2(d3, f1)
        d1 = self.d1(d2, f0)
        d0 = self.d0(d1)
        return cls_out, self.seg_head(d0), self.aux4_head(d4), self.aux3_head(d3)


class FourLevelUNet(nn.Module):
    def __init__(self, encoder, ch, channels_last_input=False):
        super().__init__()
        self.encoder = encoder
        self.expected_channels = list(ch)
        self.channels_last_input = channels_last_input
        self.d3 = DecoderBlock(ch[3], ch[2], 256)
        self.d2 = DecoderBlock(256,   ch[1], 128)
        self.d1 = DecoderBlock(128,   ch[0],  64)
        self.d0a = DecoderBlock(64, 0, 32)
        self.d0b = DecoderBlock(32, 0, 16)
        self.seg_head  = nn.Conv2d(16, 1, 1)
        self.aux3_head = nn.Conv2d(256, 1, 1)
        self.aux2_head = nn.Conv2d(128, 1, 1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(ch[3], 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )
    @staticmethod
    def _ensure_nchw(t, expected_c):
        if t.dim() != 4:
            return t
        if t.shape[1] == expected_c:
            return t
        if t.shape[-1] == expected_c:
            return t.permute(0, 3, 1, 2).contiguous()
        return t
    def forward(self, x):
        feats = self.encoder(x)
        feats = [self._ensure_nchw(f, c)
                 for f, c in zip(feats, self.expected_channels)]
        f0, f1, f2, f3 = feats
        cls_out = self.cls_head(f3)
        d3 = self.d3(f3, f2)
        d2 = self.d2(d3, f1)
        d1 = self.d1(d2, f0)
        d0a = self.d0a(d1)
        d0  = self.d0b(d0a)
        seg = self.seg_head(d0)
        if seg.shape[-2:] != x.shape[-2:]:
            seg = F.interpolate(seg, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return cls_out, seg, self.aux3_head(d3), self.aux2_head(d2)


def build_eval_model(name):
    if name == 'efficientnet_b4':
        enc = timm.create_model('efficientnet_b4', pretrained=False, features_only=True)
        return FiveLevelUNet(enc, enc.feature_info.channels())
    if name == 'convnext_tiny':
        enc = timm.create_model('convnext_tiny', pretrained=False, features_only=True)
        return FourLevelUNet(enc, enc.feature_info.channels(), channels_last_input=False)
    if name == 'swin_v2_tiny':
        enc = timm.create_model('swinv2_tiny_window8_256', pretrained=False,
                                features_only=True, img_size=512)
        return FourLevelUNet(enc, enc.feature_info.channels(), channels_last_input=True)
    if name == 'swin_v2_base':
        enc = timm.create_model('swinv2_base_window8_256', pretrained=False,
                                features_only=True, img_size=512)
        return FourLevelUNet(enc, enc.feature_info.channels(), channels_last_input=True)
    raise ValueError(f'Unknown: {name}')
"""


COMPARE_CELL_EVAL = """\
import glob

# Auto-discover all *.pt under /kaggle/input/models/, /kaggle/working/outputs/,
# and the local /kaggle/working tree. Detect architecture from state dict +
# filename hints. Robust to whatever upload pattern Kaggle gave you.

DISPLAY_NAMES = {
    'efficientnet_b4': 'EfficientNet-B4',
    'convnext_tiny':   'ConvNeXt-Tiny',
    'swin_v2_tiny':    'Swin-V2-Tiny',
    'swin_v2_base':    'Swin-V2-Base',
    'resnet50':        'ResNet-50',
}

# Per-architecture default thresholds (used when checkpoint has no metadata)
DEFAULT_THRESHOLDS = {
    'efficientnet_b4': 0.40,
    'resnet50':        0.50,
    'convnext_tiny':   0.55,
    'swin_v2_tiny':    0.51,
    'swin_v2_base':    0.54,
}


def _detect_arch_from_state(state):
    if not isinstance(state, dict) or not state:
        return None
    has_d4   = any(k.startswith('d4.') or k.startswith('aux4_head') for k in state)
    has_d0ab = any(k.startswith('d0a.') or k.startswith('d0b.') or
                   k.startswith('aux2_head') for k in state)
    cls_w = state.get('cls_head.3.weight')
    cls_in_dim = cls_w.shape[1] if (cls_w is not None and cls_w.dim() == 2) else None

    if has_d4:
        if 'encoder.conv_stem.weight' in state:
            return 'efficientnet_b4'
        if 'encoder.conv1.weight' in state and 'encoder.layer1.0.conv1.weight' in state:
            return 'resnet50'
        if cls_in_dim == 448:  return 'efficientnet_b4'
        if cls_in_dim == 2048: return 'resnet50'
    elif has_d0ab:
        if any('encoder.stages.' in k and 'conv_dw' in k for k in state):
            return 'convnext_tiny'
        if any('encoder.layers.' in k and 'attn' in k for k in state):
            return 'swin_v2_base' if cls_in_dim == 1024 else 'swin_v2_tiny'
        if cls_in_dim == 1024: return 'swin_v2_base'
        if cls_in_dim == 768:  return 'convnext_tiny'
    return None


def _detect_arch(ckpt, filename):
    candidates = [
        ('efficientnet_b4', ('efficientnet_b4', 'efficientnet-b4', 'effnet')),
        ('swin_v2_base',    ('swinv2_base', 'swin_v2_base', 'swin-v2-base', 'swinv2-base')),
        ('swin_v2_tiny',    ('swinv2_tiny', 'swin_v2_tiny', 'swin-v2-tiny', 'swinv2-tiny', 'swin')),
        ('convnext_tiny',   ('convnext_tiny', 'convnext-tiny', 'convnext')),
        ('resnet50',        ('resnet50', 'resnet-50', 'rn50')),
    ]
    if isinstance(ckpt, dict):
        for field in ('backbone', 'architecture'):
            val = str(ckpt.get(field, '')).lower()
            if not val:
                continue
            for arch_id, keys in candidates:
                if any(k in val for k in keys):
                    return arch_id
    state = ckpt
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    arch_id = _detect_arch_from_state(state)
    if arch_id is not None:
        return arch_id
    fn = filename.lower()
    for arch_id, keys in candidates:
        if any(k in fn for k in keys):
            return arch_id
    return None


def _remap_state_dict(state):
    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k
        nk = nk.replace('.cbam.fc.',      '.cbam.channel_fc.')
        nk = nk.replace('.cbam.spatial.', '.cbam.spatial_conv.')
        out[nk] = v
    return out


# ── Discover .pt files under common Kaggle locations ─────────────────────────
SCAN_ROOTS = ['/kaggle/input/models', '/kaggle/working/outputs', '/kaggle/working']
all_pts = set()
for root in SCAN_ROOTS:
    if os.path.isdir(root):
        # Recursive glob, up to 6 levels deep
        for depth_glob in ('*.pt', '*/*.pt', '*/*/*.pt', '*/*/*/*.pt',
                           '*/*/*/*/*.pt', '*/*/*/*/*/*.pt'):
            all_pts.update(glob.glob(os.path.join(root, depth_glob)))

# Identify each
available = {}     # arch_id -> (path, threshold)
for path in sorted(all_pts):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f'  [skip] {path}: {exc}')
        continue
    arch_id = _detect_arch(ckpt, os.path.basename(path))
    if arch_id is None:
        print(f'  [skip] {path}: cannot determine architecture')
        continue
    # Threshold: saved or arch default
    if isinstance(ckpt, dict) and ckpt.get('cls_threshold') is not None:
        thr = float(ckpt['cls_threshold'])
    else:
        thr = DEFAULT_THRESHOLDS.get(arch_id, 0.50)
    # If we already saw this arch_id, prefer the bundle (with metadata) over raw best.pt
    if arch_id in available:
        prev_path, _ = available[arch_id]
        prev_has_meta = isinstance(torch.load(prev_path, map_location='cpu', weights_only=False), dict) \
                        and 'model_state_dict' in torch.load(prev_path, map_location='cpu', weights_only=False)
        cur_has_meta  = isinstance(ckpt, dict) and 'model_state_dict' in ckpt
        if cur_has_meta and not prev_has_meta:
            available[arch_id] = (path, thr)
    else:
        available[arch_id] = (path, thr)

print(f'Found {len(available)} models for comparison:')
for n, (p, t) in available.items():
    print(f'  {n:18s} thr={t:.2f}  {p}')


@torch.no_grad()
def eval_one(model, loader):
    model.train(False)
    y_true, y_prob = [], []
    seg_probs, masks_all = [], []
    for imgs, labels, masks in tqdm(loader, leave=False):
        imgs = imgs.to(DEVICE)
        cls_out, seg_out, _, _ = model(imgs)
        probs = torch.sigmoid(cls_out).squeeze(1).cpu().numpy()
        sp = torch.sigmoid(seg_out).squeeze(1).cpu().numpy()
        m = masks.squeeze(1).numpy()
        y_true.extend(labels.numpy())
        y_prob.extend(probs)
        for i in range(len(labels)):
            if labels[i] == 1:
                seg_probs.append(sp[i])
                masks_all.append(m[i])
    return np.array(y_true), np.array(y_prob), seg_probs, masks_all


def metrics_for(y_true, y_prob, seg_probs, masks_all, thr_hint=None):
    # Sweep threshold for best F1, but bias toward the saved/expected hint
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob > thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    dices = []
    for sp, gt in zip(seg_probs, masks_all):
        pb = (sp > best_thr).astype(np.float32)
        inter = (pb * gt).sum()
        dices.append(2 * inter / (pb.sum() + gt.sum() + 1e-7))
    y_pred = (y_prob > best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'threshold':     float(best_thr),
        'accuracy':      float(accuracy_score(y_true, y_pred)),
        'auc':           float(roc_auc_score(y_true, y_prob)),
        'precision':     float(tp / (tp + fp + 1e-7)),
        'forged_recall': float(tp / (tp + fn + 1e-7)),
        'specificity':   float(tn / (tn + fp + 1e-7)),
        'f1':            float(2*tp / (2*tp + fp + fn + 1e-7)),
        'mean_dice':     float(np.mean(dices)) if dices else 0.0,
    }


results = {}
roc_curves = {}
for name, (path, hint_thr) in available.items():
    print(f'\\n=== {DISPLAY_NAMES.get(name, name)}  (file: {os.path.basename(path)}) ===')
    model = build_eval_model(name).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
    state = _remap_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  load: missing={len(missing)} unexpected={len(unexpected)}')
    model.train(False)

    yt, yp, sp, mm = eval_one(model, val_loader)
    m = metrics_for(yt, yp, sp, mm, thr_hint=hint_thr)
    m['composite'] = 0.4 * m['auc'] + 0.3 * m['forged_recall'] + 0.3 * m['mean_dice']
    results[name] = m
    fpr, tpr, _ = roc_curve(yt, yp)
    roc_curves[name] = (fpr, tpr, m['auc'])

    for k, v in m.items():
        print(f'  {k:18s}: {v:.4f}')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

OUT_DIR = Path('/kaggle/working/outputs/comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)
"""


COMPARE_CELL_PLOTS = """\
import pandas as pd

# Save CSV
rows = []
for name, m in results.items():
    rows.append({'model': DISPLAY_NAMES[name], **m})
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / 'comparison.csv', index=False)
print(df.to_string(index=False))

# Bar chart dashboard
metrics_to_plot = ['accuracy', 'auc', 'specificity', 'forged_recall', 'mean_dice', 'composite']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Comparison — Validation Set', fontsize=14, fontweight='bold')
for ax, mkey in zip(axes.flatten(), metrics_to_plot):
    names  = [DISPLAY_NAMES[n] for n in results.keys()]
    values = [results[n][mkey] for n in results.keys()]
    bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(names)])
    ax.set_title(mkey.replace('_', ' ').title())
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / 'comparison_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)

# ROC overlay
fig, ax = plt.subplots(figsize=(8, 8))
for name, (fpr, tpr, auc_val) in roc_curves.items():
    ax.plot(fpr, tpr, lw=2, label=f'{DISPLAY_NAMES[name]} (AUC={auc_val:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'roc_overlay.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)

# Winner
winner = max(results.items(), key=lambda kv: kv[1]['composite'])
winner_data = {
    'model':     winner[0],
    'display':   DISPLAY_NAMES[winner[0]],
    'composite': winner[1]['composite'],
    'metrics':   winner[1],
}
with open(OUT_DIR / 'winner.json', 'w') as f:
    json.dump(winner_data, f, indent=2)
print(f'\\nWinner by composite: {DISPLAY_NAMES[winner[0]]}  (composite={winner[1]["composite"]:.4f})')
print(f'Outputs at: {OUT_DIR}')
"""


def build_compare_notebook() -> Path:
    cells = [
        md(COMPARE_TITLE),
        md('## Setup'),
        code(COMPARE_CELL_SETUP),
        md('## Dataset (same val split as training)'),
        code(COMPARE_CELL_DATA),
        md('## Architecture (shared)'),
        code(COMPARE_CELL_ARCH),
        md('## Evaluation across all available models'),
        code(COMPARE_CELL_EVAL),
        md('## Comparison plots & winner'),
        code(COMPARE_CELL_PLOTS),
    ]
    path = ROOT / 'compare_models.ipynb'
    write_nb(path, cells)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('Building training notebooks...')
    for model_id, cfg in MODELS.items():
        build_training_notebook(model_id, cfg)
    print('\nBuilding comparison notebook...')
    build_compare_notebook()
    print('\nDone.')


if __name__ == '__main__':
    main()
