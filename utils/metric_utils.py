"""Metric compatibility helpers."""

from __future__ import annotations

import torch

try:
    from lpipsPyTorch import lpips as _lpips_pytorch
except ImportError:
    _lpips_pytorch = None

_LPIPS_VGG = None


def lpips_vgg(image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
    """Compute VGG LPIPS for image tensors with shape [B, 3, H, W] in [0, 1]."""
    if _lpips_pytorch is not None:
        return _lpips_pytorch(image_a, image_b, net_type="vgg")

    import lpips

    global _LPIPS_VGG
    if _LPIPS_VGG is None:
        _LPIPS_VGG = lpips.LPIPS(net="vgg").cuda().eval()

    return _LPIPS_VGG(image_a * 2.0 - 1.0, image_b * 2.0 - 1.0).mean()
