"""Tensor-level image operations shared by training augmentation and the robustness benchmark.

All functions take / return float tensors (B, 3, H, W) in [0, 1].
"""
from __future__ import annotations

import io
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import functional as TF


# ---------------------------------------------------------------------------------- primitive ops
def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Re-encode every image as JPEG at `quality` (PIL encoder, 4:2:0 chroma, like most software)."""
    out = []
    for img in (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy():
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert("RGB")))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float().div(255).to(x.device)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = max(3, int(2 * round(3 * sigma) + 1))
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def gaussian_noise(x: torch.Tensor, std: float, generator: torch.Generator | None = None) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, device=x.device if generator is None else "cpu")
    return (x + std * noise.to(x.device)).clamp(0, 1)


def resize_down_up(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Downscale by `scale` then upscale back (destroys high-frequency content, like web re-sizing)."""
    h, w = x.shape[-2:]
    small = max(4, int(round(h * scale))), max(4, int(round(w * scale)))
    y = F.interpolate(x, size=small, mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def center_crop_resize(x: torch.Tensor, fraction: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    ch, cw = max(4, int(round(h * fraction))), max(4, int(round(w * fraction)))
    y = TF.center_crop(x, [ch, cw])
    return F.interpolate(y, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)


def brightness(x, factor):
    return TF.adjust_brightness(x, factor)


def contrast(x, factor):
    return TF.adjust_contrast(x, factor)


def saturation(x, factor):
    return TF.adjust_saturation(x, factor)


def hue(x, shift):
    return TF.adjust_hue(x, shift)


def gamma(x, g):
    return TF.adjust_gamma(x, g)


# ------------------------------------------------------------------------- robustness benchmark
# name -> (function, list of severities). Severity lists go from mild to strong.
PERTURBATIONS: dict[str, tuple[Callable, list]] = {
    "jpeg": (jpeg, [90, 75, 50, 30]),
    "resize": (resize_down_up, [0.875, 0.75, 0.5]),
    "crop": (center_crop_resize, [0.9, 0.8, 0.7]),
    "gaussian_noise": (gaussian_noise, [0.01, 0.02, 0.05]),
    "blur": (gaussian_blur, [0.5, 1.0, 1.5]),
    "brightness": (brightness, [0.7, 1.3]),
    "contrast": (contrast, [0.7, 1.3]),
    "saturation": (saturation, [0.5, 1.5]),
    "hue": (hue, [0.05, 0.1]),
}


def apply_perturbation(x: torch.Tensor, name: str, severity, seed: int = 0) -> torch.Tensor:
    fn, _ = PERTURBATIONS[name]
    if name == "gaussian_noise":
        return fn(x, severity, generator=torch.Generator().manual_seed(seed))
    return fn(x, severity)


# --------------------------------------------------------------------------- training augmentation
def random_hflip(x: torch.Tensor) -> torch.Tensor:
    flip = torch.rand(x.size(0), device=x.device) < 0.5
    x = x.clone()
    x[flip] = x[flip].flip(-1)
    return x


def random_translate(x: torch.Tensor, max_frac: float) -> torch.Tensor:
    """Random integer shift with reflect padding (equivalent to pad + random crop)."""
    if max_frac <= 0:
        return x
    b, _, h, w = x.shape
    p = max(1, int(round(h * max_frac)))
    padded = F.pad(x, (p, p, p, p), mode="reflect")
    dx = torch.randint(0, 2 * p + 1, (b,)).tolist()
    dy = torch.randint(0, 2 * p + 1, (b,)).tolist()
    return torch.stack([padded[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(b)])


def _robust_one(img: torch.Tensor) -> torch.Tensor:
    """One random degradation for a single image (1, 3, H, W). Ranges deliberately overlap but are
    not identical to the evaluation severities, so the benchmark is not a pure train/test match."""
    op = random.choice(["jpeg", "blur", "noise", "resize", "colour"])
    if op == "jpeg":
        return jpeg(img, random.randint(40, 95))
    if op == "blur":
        return gaussian_blur(img, random.uniform(0.3, 1.2))
    if op == "noise":
        return gaussian_noise(img, random.uniform(0.003, 0.03))
    if op == "resize":
        return resize_down_up(img, random.uniform(0.6, 0.95))
    img = brightness(img, random.uniform(0.8, 1.2))
    img = contrast(img, random.uniform(0.8, 1.2))
    return saturation(img, random.uniform(0.7, 1.3))


def build_train_augment(cfg_aug: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    hflip, translate = cfg_aug.get("hflip", True), float(cfg_aug.get("translate", 0.0))
    robust, robust_p = bool(cfg_aug.get("robust", False)), float(cfg_aug.get("robust_p", 0.5))

    def augment(x: torch.Tensor) -> torch.Tensor:
        if hflip:
            x = random_hflip(x)
        if translate > 0:
            x = random_translate(x, translate)
        if robust:
            x = x.clone()
            for i in range(x.size(0)):
                if random.random() < robust_p:
                    x[i:i + 1] = _robust_one(x[i:i + 1])
        return x

    return augment
