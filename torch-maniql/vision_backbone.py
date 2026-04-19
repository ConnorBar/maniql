"""PyTorch-native vision utilities + R3M-backed ResNet feature extractor.

This project trains IQL end-to-end in PyTorch. Image observations are stored
raw (uint8, HWC, [0..255]) in the dataset and are preprocessed on GPU here.

Key requirements:
- **No JAX/Flax** dependencies/imports.
- The backbone must remain **trainable** (no freezing) so gradients from IQL
  losses update the visual encoder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RESNET_OUT_DIM: Dict[str, int] = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

FORCE_FIELD_DIM = 420
FORCE_GRID_SHAPE = (14, 10, 3)

# ---------------------------------------------------------------------------
#  Preprocessing helpers
# ---------------------------------------------------------------------------

def r3m_preprocess_bhwc(images: torch.Tensor) -> torch.Tensor:
    """Convert BHWC images to normalized BCHW for ImageNet-pretrained ResNets.

    - If input is uint8, assumes [0..255].
    - If input is float, assumes values are already in a reasonable scale for
      normalization (this repo's dataset uses uint8 for real images; float inputs
      are mainly used for synthetic modalities like force-fields).
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected (B,H,W,3) images, got {tuple(images.shape)}")
    x = images
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        if x.dtype != torch.float32:
            x = x.float()
    x = x.permute(0, 3, 1, 2).contiguous()  # BCHW
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def force_to_image(force: torch.Tensor) -> torch.Tensor:
    """(B,420) -> (B,224,224,3) float (unnormalized)."""
    if force.ndim != 2 or force.shape[-1] != FORCE_FIELD_DIM:
        raise ValueError(f"Expected (B,{FORCE_FIELD_DIM}) force, got {tuple(force.shape)}")
    b = force.shape[0]
    x = force.view(b, *FORCE_GRID_SHAPE)  # BHWC
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    x = (x - mean) / std
    # interpolate expects NCHW
    x = x.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


@dataclass(frozen=True)
class R3MLoadResult:
    arch: str
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]


def _strip_prefix(key: str) -> str:
    for p in ("module.", "convnet."):
        if key.startswith(p):
            return key[len(p):]
    return key


def load_r3m_resnet_weights(backbone: nn.Module, checkpoint_path: str) -> R3MLoadResult:
    """Load R3M convnet weights into a torchvision ResNet backbone.

    Accepts checkpoints that are either:
    - raw state_dict (convnet.* keys), or
    - dict with 'r3m' key (common in this repo).
    """
    path = os.path.expanduser(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["r3m"] if isinstance(ckpt, dict) and "r3m" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    cleaned = {_strip_prefix(k): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    msg = backbone.load_state_dict(cleaned, strict=False)
    return R3MLoadResult(
        arch=backbone.__class__.__name__,
        missing_keys=tuple(getattr(msg, "missing_keys", [])),
        unexpected_keys=tuple(getattr(msg, "unexpected_keys", [])),
    )


class ResNetBackbone(nn.Module):
    """ResNet feature extractor with optional R3M initialization (trainable)."""

    def __init__(self, arch: str = "resnet18", r3m_checkpoint: str | None = None):
        super().__init__()
        if arch not in RESNET_OUT_DIM:
            raise ValueError(f"Unknown arch {arch!r}; expected one of {sorted(RESNET_OUT_DIM)}")
        if arch == "resnet18":
            net = torchvision.models.resnet18(weights=None)
        elif arch == "resnet34":
            net = torchvision.models.resnet34(weights=None)
        else:
            net = torchvision.models.resnet50(weights=None)
        net.fc = nn.Identity()
        self.net = net
        self.out_dim = RESNET_OUT_DIM[arch]

        if r3m_checkpoint:
            res = load_r3m_resnet_weights(self.net, r3m_checkpoint)
            if res.unexpected_keys:
                # Most unexpected keys are language heads; safe to ignore.
                pass

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        # torchvision ResNet expects normalized BCHW float
        return self.net(x_bchw)
