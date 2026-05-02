"""
Shared model architectures for the forgery detection webapp.

All models share:
    - CBAM (channel + spatial attention) primitive
    - DecoderBlock pattern: upsample -> concat skip -> 2 ConvBNReLU -> CBAM
    - Output signature: (cls_logit, seg_logit, aux_d4_logit, aux_d3_logit)

Encoders differ in feature-map count:
    - EfficientNet-B4 / ResNet-50 : 5 levels  (f0 .. f4)
    - ConvNeXt / Swin-V2          : 4 levels  (f0 .. f3)

The 4-level decoder appends one extra plain decoder block at the end
to recover the final 4x upsample to native resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────

class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = self.channel_fc(F.adaptive_avg_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        mx  = self.channel_fc(F.adaptive_max_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        x = x * torch.sigmoid(avg + mx)
        spatial = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        x = x * torch.sigmoid(self.spatial_conv(spatial))
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int = 3, p: int = 1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample x2 -> concat skip -> 2x ConvBNReLU -> CBAM."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_c + skip_c, out_c),
            ConvBNReLU(out_c, out_c),
        )
        self.cbam = CBAM(out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.cbam(self.conv(x))


# ── 5-level encoder (EfficientNet, ResNet) ───────────────────────────────────

class FiveLevelUNet(nn.Module):
    """
    UNet wrapper for 5-level encoders (EfficientNet-B4, ResNet-50).
    Encoder must return list [f0, f1, f2, f3, f4] via timm features_only.
    """

    def __init__(self, encoder, ch: list):
        super().__init__()
        self.encoder = encoder

        self.d4 = DecoderBlock(ch[4], ch[3], 256)
        self.d3 = DecoderBlock(256,   ch[2], 128)
        self.d2 = DecoderBlock(128,   ch[1],  64)
        self.d1 = DecoderBlock( 64,   ch[0],  32)
        self.d0 = DecoderBlock( 32,       0,  16)

        self.seg_head  = nn.Conv2d(16,  1, 1)
        self.aux4_head = nn.Conv2d(256, 1, 1)
        self.aux3_head = nn.Conv2d(128, 1, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(ch[4], 512),
            nn.ReLU(inplace=True),
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


# ── 4-level encoder (ConvNeXt, Swin) ─────────────────────────────────────────

class FourLevelUNet(nn.Module):
    """
    UNet wrapper for 4-level encoders (ConvNeXt-Tiny, Swin-V2-{Tiny,Base}).
    Encoder must return list [f0, f1, f2, f3] via timm features_only.
    Channel sizes typical:
        ConvNeXt-Tiny  : [ 96, 192, 384,  768]
        Swin-V2-Tiny   : [ 96, 192, 384,  768]
        Swin-V2-Base   : [128, 256, 512, 1024]
    Strides: f0 @ /4, f1 @ /8, f2 @ /16, f3 @ /32 (input is /1).
    """

    def __init__(self, encoder, ch: list, channels_last_input: bool = False):
        super().__init__()
        self.encoder = encoder
        self.expected_channels = list(ch)        # known correct channel counts
        self.channels_last_input = channels_last_input

        self.d3 = DecoderBlock(ch[3], ch[2], 256)   # /32 -> /16
        self.d2 = DecoderBlock(256,   ch[1], 128)   # /16 -> /8
        self.d1 = DecoderBlock(128,   ch[0],  64)   #  /8 -> /4
        self.d0a = DecoderBlock( 64,      0,  32)   #  /4 -> /2
        self.d0b = DecoderBlock( 32,      0,  16)   #  /2 -> /1

        self.seg_head  = nn.Conv2d(16,  1, 1)
        self.aux3_head = nn.Conv2d(256, 1, 1)
        self.aux2_head = nn.Conv2d(128, 1, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(ch[3], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    @staticmethod
    def _ensure_nchw(t: torch.Tensor, expected_c: int) -> torch.Tensor:
        """Convert to NCHW using the known expected channel count.
        Robust to spatial dims being larger or smaller than C
        (the previous shape-only heuristic broke for Swin-V2 at img_size=512)."""
        if t.dim() != 4:
            return t
        if t.shape[1] == expected_c:
            return t                                     # already NCHW
        if t.shape[-1] == expected_c:
            return t.permute(0, 3, 1, 2).contiguous()    # NHWC -> NCHW
        return t

    def forward(self, x):
        feats = self.encoder(x)
        feats = [self._ensure_nchw(f, c)
                 for f, c in zip(feats, self.expected_channels)]
        f0, f1, f2, f3 = feats

        cls_out = self.cls_head(f3)
        d3 = self.d3(f3, f2)     # 256 ch @ /16
        d2 = self.d2(d3, f1)     # 128 ch @ /8
        d1 = self.d1(d2, f0)     #  64 ch @ /4
        d0a = self.d0a(d1)       #  32 ch @ /2
        d0  = self.d0b(d0a)      #  16 ch @ /1

        # Match seg_head output to original input size if any rounding mismatch.
        seg_logit = self.seg_head(d0)
        if seg_logit.shape[-2:] != x.shape[-2:]:
            seg_logit = F.interpolate(seg_logit, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return cls_out, seg_logit, self.aux3_head(d3), self.aux2_head(d2)


# ── Public factory ───────────────────────────────────────────────────────────

def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """
    Build a model by short name. Returns a module with forward signature:
        cls_logit  [B, 1]
        seg_logit  [B, 1, H, W]
        aux_logit_a, aux_logit_b   (deep supervision; ignored at inference)
    """
    import timm

    name = name.lower()

    if name in ('efficientnet_b4', 'effnet_b4', 'efficientnet-b4'):
        enc = timm.create_model('efficientnet_b4', pretrained=pretrained,
                                features_only=True)
        ch = enc.feature_info.channels()
        return FiveLevelUNet(enc, ch)

    if name in ('resnet50', 'resnet-50', 'rn50'):
        enc = timm.create_model('resnet50', pretrained=pretrained,
                                features_only=True)
        ch = enc.feature_info.channels()
        return FiveLevelUNet(enc, ch)

    if name in ('convnext_tiny', 'convnext-tiny'):
        enc = timm.create_model('convnext_tiny', pretrained=pretrained,
                                features_only=True)
        ch = enc.feature_info.channels()
        assert len(ch) == 4, f'expected 4 levels, got {len(ch)}'
        return FourLevelUNet(enc, ch, channels_last_input=False)

    if name in ('swin_v2_tiny', 'swinv2_tiny'):
        enc = timm.create_model('swinv2_tiny_window8_256', pretrained=pretrained,
                                features_only=True, img_size=512)
        ch = enc.feature_info.channels()
        assert len(ch) == 4, f'expected 4 levels, got {len(ch)}'
        return FourLevelUNet(enc, ch, channels_last_input=True)

    if name in ('swin_v2_base', 'swinv2_base'):
        enc = timm.create_model('swinv2_base_window8_256', pretrained=pretrained,
                                features_only=True, img_size=512)
        ch = enc.feature_info.channels()
        assert len(ch) == 4, f'expected 4 levels, got {len(ch)}'
        return FourLevelUNet(enc, ch, channels_last_input=True)

    raise ValueError(f'Unknown model name: {name!r}')


MODEL_REGISTRY: dict = {
    'efficientnet_b4': {
        'display':     'EfficientNet-B4 + CBAM-UNet',
        'levels':      5,
        'description': 'Pretrained ImageNet baseline; sensitive but low specificity.',
    },
    'convnext_tiny': {
        'display':     'ConvNeXt-Tiny + CBAM-UNet',
        'levels':      4,
        'description': 'Modern conv-only architecture; strong fine-detail features.',
    },
    'swin_v2_tiny': {
        'display':     'Swin-V2-Tiny + CBAM-UNet',
        'levels':      4,
        'description': 'Self-attention captures non-local copy-move similarity.',
    },
    'swin_v2_base': {
        'display':     'Swin-V2-Base + CBAM-UNet',
        'levels':      4,
        'description': 'Heavy attention model targeting maximum specificity.',
    },
}
