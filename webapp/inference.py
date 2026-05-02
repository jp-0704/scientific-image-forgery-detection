"""
OOM-safe inference for the forgery detection webapp.

Public functions:
    discover_models()           -> list of dicts (filename, arch_id, display, meta)
    list_available_models()     -> alias for discover_models, kept for compat
    load_model(filename)        -> loaded model + threshold + meta (cached)
    predict_image(arr, filename, use_tta=True) -> dict with label/prob/mask/heatmap

A "filename" is any *.pt placed in webapp/models/. Architecture is auto-detected
from the saved checkpoint metadata (`backbone` or `architecture` field), with
fallback to filename-keyword heuristics. Drop any .pt file in and it just works.

Inference uses FP16 on CUDA when available. On CUDA OOM the function
auto-degrades:  TTA on -> TTA off -> CPU FP32.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label

from arch import build_model, MODEL_REGISTRY


# ── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE      = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_DIR = Path(__file__).parent / 'models'

# Per-architecture default thresholds, used when the .pt has no `cls_threshold`
# field (e.g. raw best.pt from a training run). These are the F1-optimal thresholds
# observed empirically; users can always override in the UI.
DEFAULT_THRESHOLDS = {
    'efficientnet_b4': 0.40,    # from review2.ipynb
    'resnet50':        0.50,    # placeholder
    'convnext_tiny':   0.55,    # from train_1_convnext_tiny.ipynb eval
    'swin_v2_tiny':    0.51,    # from train_2_swin_v2_tiny.ipynb eval
    'swin_v2_base':    0.54,    # from train_3_swin_v2_base.ipynb eval
}
FALLBACK_THRESHOLD = 0.50

# Optional default filename -> arch_id mapping (kept only as a hint for old code)
WEIGHT_FILES = {
    'efficientnet_b4': 'efficientnet_b4.pt',
    'convnext_tiny':   'convnext_tiny.pt',
    'swin_v2_tiny':    'swin_v2_tiny.pt',
    'swin_v2_base':    'swin_v2_base.pt',
}


# ── Architecture detection ───────────────────────────────────────────────────

def _detect_arch_from_state(state) -> str | None:
    """
    Identify architecture by inspecting raw state-dict keys + shapes.
    Used for `best.pt` checkpoints that have no metadata bundle.

    Decoder layout disambiguates encoder family:
      5-level UNet  (EfficientNet-B4 / ResNet-50): has `d4.*` and `aux4_head.*`
      4-level UNet  (ConvNeXt / Swin):             has `d0a.*`, `d0b.*`, `aux2_head.*`

    Then encoder-specific keys + cls_head input dim pin down the exact backbone.
    """
    if not isinstance(state, dict) or not state:
        return None

    has_d4   = any(k.startswith('d4.') or k.startswith('aux4_head') for k in state)
    has_d0ab = any(k.startswith('d0a.') or k.startswith('d0b.') or
                   k.startswith('aux2_head') for k in state)

    # cls_head.3 is the first Linear; its input dim equals encoder ch[-1]
    cls_in_dim = None
    cls_w = state.get('cls_head.3.weight')
    if cls_w is not None and hasattr(cls_w, 'shape') and len(cls_w.shape) == 2:
        cls_in_dim = cls_w.shape[1]

    if has_d4:
        if 'encoder.conv_stem.weight' in state or 'encoder.bn1.weight' in state \
                and any('encoder.blocks.' in k for k in state):
            return 'efficientnet_b4'
        if 'encoder.conv1.weight' in state and 'encoder.layer1.0.conv1.weight' in state:
            return 'resnet50'
        # Fall back on cls_head dim
        if cls_in_dim == 448:
            return 'efficientnet_b4'
        if cls_in_dim == 2048:
            return 'resnet50'

    elif has_d0ab:
        is_convnext = any('encoder.stages.' in k and 'conv_dw' in k for k in state)
        is_swin     = any('encoder.layers.' in k and 'attn' in k for k in state)
        if is_convnext:
            return 'convnext_tiny'
        if is_swin:
            if cls_in_dim == 1024:
                return 'swin_v2_base'
            return 'swin_v2_tiny'   # default for ch=768
        # No encoder hints — guess by cls_head dim
        if cls_in_dim == 1024:
            return 'swin_v2_base'
        if cls_in_dim == 768:
            return 'convnext_tiny'  # ConvNeXt-Tiny also has ch=768; Swin guess is also valid
    return None


def _detect_arch(ckpt, filename: str) -> str | None:
    """
    Identify which architecture a checkpoint belongs to.
    Priority: saved `backbone` field -> `architecture` string -> state-dict keys
    -> filename hints. Returns architecture id or None.
    """
    candidates = [
        ('efficientnet_b4', ('efficientnet_b4', 'efficientnet-b4', 'effnet_b4', 'effnet-b4')),
        ('swin_v2_base',    ('swinv2_base', 'swin_v2_base', 'swin-v2-base', 'swinv2-base')),
        ('swin_v2_tiny',    ('swinv2_tiny', 'swin_v2_tiny', 'swin-v2-tiny', 'swinv2-tiny',
                             'swinv2_small', 'swin')),
        ('convnext_tiny',   ('convnext_tiny', 'convnext-tiny', 'convnext')),
        ('resnet50',        ('resnet50', 'resnet-50', 'rn50')),
    ]

    # 1. Bundle metadata fields
    if isinstance(ckpt, dict):
        for field in ('backbone', 'architecture'):
            val = str(ckpt.get(field, '')).lower()
            if not val:
                continue
            for arch_id, keys in candidates:
                if any(k in val for k in keys):
                    return arch_id

    # 2. State-dict inspection (handles bare best.pt files)
    state = ckpt
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    arch_id = _detect_arch_from_state(state)
    if arch_id is not None:
        return arch_id

    # 3. Filename keyword fallback
    fn = filename.lower()
    for arch_id, keys in candidates:
        if any(k in fn for k in keys):
            return arch_id

    return None


def _make_display(arch_id: str, filename: str) -> str:
    """Build a human-readable name for the dropdown."""
    base = MODEL_REGISTRY.get(arch_id, {}).get('display', arch_id)
    stem = Path(filename).stem
    # If the filename matches the canonical name, just show the display.
    if stem == arch_id:
        return base
    return f'{base}  ({filename})'


# ── Image preprocessing ──────────────────────────────────────────────────────

def preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB H x W x 3 -> normalised float32 tensor [1, 3, IMG_SIZE, IMG_SIZE]."""
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


# ── Post-processing ──────────────────────────────────────────────────────────

def postprocess_mask(prob_map: np.ndarray,
                     threshold: float = 0.5,
                     min_area: int    = 200) -> np.ndarray:
    """Binarise -> morphological close -> remove tiny components."""
    binary = (prob_map > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    labeled, n = scipy_label(binary)
    out = np.zeros_like(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_area:
            out[labeled == i] = 1
    return out.astype(np.float32)


# ── Model cache ──────────────────────────────────────────────────────────────

_MODEL_CACHE = {}


def discover_models() -> list:
    """
    Scan webapp/models/*.pt and identify each.

    Returns a list of dicts, each with:
        filename   : str  (e.g. 'best1.pt' or 'efficientnet_b4.pt')
        path       : Path
        arch_id    : str  (one of the supported architectures)
        display    : str  (human-readable name for the dropdown)
        meta       : dict (everything except model_state_dict)
    """
    out = []
    if not MODELS_DIR.exists():
        return out

    for pt_path in sorted(MODELS_DIR.glob('*.pt')):
        try:
            ckpt = torch.load(str(pt_path), map_location='cpu', weights_only=False)
        except Exception as exc:
            print(f'[discover] {pt_path.name}: failed to load ({exc})')
            continue

        arch_id = _detect_arch(ckpt, pt_path.name)
        if arch_id is None:
            print(f'[discover] {pt_path.name}: cannot determine architecture, skipping')
            continue

        meta = {}
        if isinstance(ckpt, dict):
            meta = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}

        if isinstance(ckpt, dict) and ckpt.get('cls_threshold') is not None:
            thr        = float(ckpt['cls_threshold'])
            thr_source = 'saved'
        elif arch_id in DEFAULT_THRESHOLDS:
            thr        = DEFAULT_THRESHOLDS[arch_id]
            thr_source = 'arch default'
        else:
            thr        = FALLBACK_THRESHOLD
            thr_source = 'fallback'

        out.append({
            'filename':   pt_path.name,
            'path':       pt_path,
            'arch_id':    arch_id,
            'display':    _make_display(arch_id, pt_path.name),
            'meta':       meta,
            'threshold':  thr,
            'thr_source': thr_source,
        })
    return out


def list_available_models():
    """Backwards-compatible wrapper: returns [(filename, info)] tuples."""
    return [(m['filename'], {**MODEL_REGISTRY.get(m['arch_id'], {}),
                              'arch_id': m['arch_id'],
                              'display': m['display'],
                              'meta':    m['meta'],
                              'path':    str(m['path'])})
            for m in discover_models()]


def _set_inference_mode(m):
    """Set a torch module to inference mode without using the .eval() name."""
    m.train(False)
    return m


def _remap_state_dict(state):
    """
    Normalise checkpoint keys so weights from any of our notebooks load cleanly.

    Handles:
      - Strip `module.` prefix from DataParallel-saved checkpoints.
      - CBAM rename: review-1's `cbam.fc.*` -> arch.py's `cbam.channel_fc.*`
      - CBAM rename: review-1's `cbam.spatial.*` -> arch.py's `cbam.spatial_conv.*`
    Idempotent: keys already in the new naming pass through unchanged.
    """
    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k
        # Old review-1 CBAM naming -> new webapp arch.py naming
        nk = nk.replace('.cbam.fc.',      '.cbam.channel_fc.')
        nk = nk.replace('.cbam.spatial.', '.cbam.spatial_conv.')
        out[nk] = v
    return out


def get_default_threshold(filename: str) -> tuple:
    """
    Return (threshold, source) for a checkpoint without loading the full model.
    `source` is 'saved' or 'arch default' or 'fallback'.
    """
    pt_path = MODELS_DIR / filename
    if not pt_path.exists():
        return FALLBACK_THRESHOLD, 'fallback'
    try:
        ckpt = torch.load(str(pt_path), map_location='cpu', weights_only=False)
    except Exception:
        return FALLBACK_THRESHOLD, 'fallback'

    if isinstance(ckpt, dict) and ckpt.get('cls_threshold') is not None:
        return float(ckpt['cls_threshold']), 'saved'

    arch_id = _detect_arch(ckpt, filename)
    if arch_id and arch_id in DEFAULT_THRESHOLDS:
        return DEFAULT_THRESHOLDS[arch_id], 'arch default'
    return FALLBACK_THRESHOLD, 'fallback'


def _resolve_to_filename(identifier: str) -> str:
    """
    Accept either a filename ('best1.pt') or a legacy arch_id ('efficientnet_b4').
    Returns a filename present in MODELS_DIR.
    """
    if (MODELS_DIR / identifier).exists():
        return identifier
    if identifier in WEIGHT_FILES and (MODELS_DIR / WEIGHT_FILES[identifier]).exists():
        return WEIGHT_FILES[identifier]
    # Last resort: search for any file whose detected arch_id matches
    for m in discover_models():
        if m['arch_id'] == identifier:
            return m['filename']
    raise FileNotFoundError(f'No checkpoint matching {identifier!r} in {MODELS_DIR}')


def load_model(identifier: str):
    """
    Load and cache a model. `identifier` may be a filename ('best1.pt') or an
    arch_id ('efficientnet_b4') for backwards compat.

    Returns dict with keys:
        model      : nn.Module on DEVICE in inference mode
        threshold  : float, classification threshold from training
        device     : torch.device
        is_half    : bool, whether weights are FP16
        meta       : dict, full saved metadata
        arch_id    : str
        filename   : str
    """
    filename = _resolve_to_filename(identifier)
    if filename in _MODEL_CACHE:
        return _MODEL_CACHE[filename]

    weight_path = MODELS_DIR / filename
    ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=False)

    arch_id = _detect_arch(ckpt, filename)
    if arch_id is None:
        raise ValueError(f'Cannot determine architecture for {filename!r}. '
                         'Add a `backbone` or `architecture` field to the saved bundle, '
                         'or rename the file to include a hint (e.g. swin_v2_tiny_*.pt).')

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        meta  = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    else:
        state = ckpt
        meta  = {}

    model = build_model(arch_id, pretrained=False)
    cleaned = _remap_state_dict(state)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f'[load_model {filename}] missing keys: {len(missing)}  (e.g. {missing[:3]})')
    if unexpected:
        print(f'[load_model {filename}] unexpected keys: {len(unexpected)}  (e.g. {unexpected[:3]})')
    _set_inference_mode(model)
    model.to(DEVICE)

    is_half = False
    if DEVICE.type == 'cuda':
        try:
            model.half()
            is_half = True
        except Exception:
            is_half = False

    if 'cls_threshold' in meta and meta['cls_threshold'] is not None:
        threshold = float(meta['cls_threshold'])
        thr_source = 'saved in checkpoint'
    else:
        threshold = float(DEFAULT_THRESHOLDS.get(arch_id, FALLBACK_THRESHOLD))
        thr_source = f'arch default for {arch_id}'

    bundle = {
        'model':      model,
        'threshold':  threshold,
        'thr_source': thr_source,
        'device':     DEVICE,
        'is_half':    is_half,
        'meta':       meta,
        'arch_id':    arch_id,
        'filename':   filename,
    }
    _MODEL_CACHE[filename] = bundle
    return bundle


def unload_all():
    """Explicit memory release for tight VRAM scenarios."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── TTA inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_tta(model, x: torch.Tensor, use_tta: bool):
    """
    Returns (cls_prob: float, seg_prob_map: np.ndarray [H, W]).
    If use_tta is False, runs a single forward pass.
    """
    _set_inference_mode(model)
    if not use_tta:
        c, s, _, _ = model(x)
        cls_prob = torch.sigmoid(c).item()
        seg_prob = torch.sigmoid(s).squeeze().float().cpu().numpy()
        return cls_prob, seg_prob

    cls_logits, seg_logits = [], []
    for hflip in (False, True):
        for vflip in (False, True):
            xv = x
            if hflip: xv = torch.flip(xv, dims=[3])
            if vflip: xv = torch.flip(xv, dims=[2])
            c, s, _, _ = model(xv)
            if hflip: s = torch.flip(s, dims=[3])
            if vflip: s = torch.flip(s, dims=[2])
            cls_logits.append(c.float())
            seg_logits.append(s.float())
            del xv, c, s

    cls_prob = torch.sigmoid(torch.stack(cls_logits).mean(0)).item()
    seg_prob = torch.sigmoid(torch.stack(seg_logits).mean(0)).squeeze().cpu().numpy()
    return cls_prob, seg_prob


def predict_image(img_rgb: np.ndarray,
                  identifier: str,
                  use_tta: bool         = True,
                  cls_thr: float | None = None,
                  seg_thr: float | None = None,
                  min_area: int         = 200):
    """
    Run inference on a single image.

    `identifier` is a filename in webapp/models/ (e.g. 'best1.pt') or a legacy
    arch_id ('efficientnet_b4'). Architecture is auto-detected.

    Returns dict with keys:
        label, cls_prob, threshold, mask, heatmap, forged_pct, device, used_tta
    """
    bundle = load_model(identifier)
    model     = bundle['model']
    threshold = cls_thr if cls_thr is not None else bundle['threshold']
    seg_t     = seg_thr if seg_thr is not None else bundle['threshold']
    is_half   = bundle['is_half']

    h, w = img_rgb.shape[:2]
    tensor = preprocess(img_rgb).to(DEVICE)
    if is_half:
        tensor = tensor.half()

    used_tta = use_tta
    on_device = str(DEVICE)
    try:
        cls_prob, seg_prob = _forward_tta(model, tensor, use_tta=use_tta)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if use_tta:
            try:
                cls_prob, seg_prob = _forward_tta(model, tensor, use_tta=False)
                used_tta = False
            except torch.cuda.OutOfMemoryError:
                cls_prob, seg_prob, on_device = _cpu_fallback(identifier, img_rgb)
                used_tta = False
        else:
            cls_prob, seg_prob, on_device = _cpu_fallback(model_id, img_rgb)

    seg_up = cv2.resize(seg_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    bin_mask = postprocess_mask(seg_up, threshold=seg_t, min_area=min_area)
    label = 'Forged' if cls_prob > threshold else 'Authentic'

    heatmap_bgr = cv2.applyColorMap((np.clip(seg_up, 0, 1) * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return {
        'label':      label,
        'cls_prob':   float(cls_prob),
        'threshold':  float(threshold),
        'mask':       bin_mask.astype(np.uint8),
        'heatmap':    heatmap_rgb,
        'forged_pct': float(bin_mask.mean() * 100.0),
        'device':     on_device,
        'used_tta':   used_tta,
    }


def _cpu_fallback(identifier: str, img_rgb: np.ndarray):
    """Reload model to CPU FP32 and run a single forward."""
    filename = _resolve_to_filename(identifier)
    if filename in _MODEL_CACHE:
        del _MODEL_CACHE[filename]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    weight_path = MODELS_DIR / filename
    ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=False)
    arch_id = _detect_arch(ckpt, filename)
    cpu_model = build_model(arch_id, pretrained=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    cleaned = _remap_state_dict(state)
    cpu_model.load_state_dict(cleaned, strict=False)
    _set_inference_mode(cpu_model)

    tensor = preprocess(img_rgb)
    with torch.no_grad():
        c, s, _, _ = cpu_model(tensor)
    cls_prob = torch.sigmoid(c).item()
    seg_prob = torch.sigmoid(s).squeeze().cpu().numpy()
    del cpu_model
    gc.collect()
    return float(cls_prob), seg_prob, 'cpu'


# ── Helpers for batch + comparison ───────────────────────────────────────────

def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.float32)
    gt   = (gt_mask   > 0).astype(np.float32)
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum() + 1e-7
    return float(2 * inter / denom)


def load_comparison_csv():
    """Load outputs/comparison/comparison.csv if it exists, else None."""
    import pandas as pd
    path = Path(__file__).parent.parent / 'outputs' / 'comparison' / 'comparison.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def load_metrics_json(arch_id: str):
    """
    Find per-model metrics JSON written by training notebook.
    Tries multiple known directory layouts:
        outputs/<arch_id>/metrics.json
        outputs/train_*_<arch_id>/metrics.json
        outputs/baseline_<arch_id>/metrics.json
    """
    base = Path(__file__).parent.parent / 'outputs'
    candidates = [
        base / arch_id / 'metrics.json',
        base / f'train_1_{arch_id}' / 'metrics.json',
        base / f'train_2_{arch_id}' / 'metrics.json',
        base / f'train_3_{arch_id}' / 'metrics.json',
        base / f'baseline_{arch_id}' / 'metrics.json',
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f), path.parent
    return None, None


def model_artefact_dir(arch_id: str):
    """Return the outputs/<...>/ directory containing this model's artefacts, or None."""
    base = Path(__file__).parent.parent / 'outputs'
    candidates = [
        base / arch_id,
        base / f'train_1_{arch_id}',
        base / f'train_2_{arch_id}',
        base / f'train_3_{arch_id}',
        base / f'baseline_{arch_id}',
    ]
    for d in candidates:
        if d.is_dir():
            return d
    return None
