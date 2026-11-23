"""Explainability: Grad-CAM, Grad-CAM++, Integrated Gradients, attention rollout, frequency analysis,
and a deletion-based faithfulness test to compare saliency methods quantitatively.

Conventions: inputs are (1, 3, 32, 32) tensors in [0, 1]; each model returns one FAKE logit.
`target="fake"` explains evidence for FAKE, `target="real"` for REAL (negated logit), and
`target="pred"` explains whichever class the model predicts.
All heatmaps are returned as float numpy arrays of shape (32, 32) in [0, 1] (or [-1, 1] for signed maps).
"""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.common import Attention
from .transforms import gaussian_blur


def _sign_for_target(logit: torch.Tensor, target: str) -> float:
    if target == "fake":
        return 1.0
    if target == "real":
        return -1.0
    return 1.0 if logit.item() >= 0 else -1.0


def _normalise(cam: torch.Tensor) -> np.ndarray:
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).cpu().numpy()


# ----------------------------------------------------------------------------- Grad-CAM family
class _Hook:
    def __init__(self, module: nn.Module):
        self.acts = self.grads = None
        self.h1 = module.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self.h1.remove()


def grad_cam(model: nn.Module, x: torch.Tensor, layer: str | None = None, target: str = "pred",
             plus_plus: bool = False) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017) or Grad-CAM++ (Chattopadhay et al., 2018)."""
    layer = layer or getattr(model, "cam_layer", None)
    if not layer:
        raise ValueError(f"{type(model).__name__} has no convolutional cam_layer")
    model.eval()
    hook = _Hook(model.get_submodule(layer))
    try:
        x = x.clone().requires_grad_(True)
        logit = model(x)[0]
        score = _sign_for_target(logit, target) * logit
        model.zero_grad(set_to_none=True)
        score.backward()
        a, g = hook.acts.detach()[0], hook.grads.detach()[0]           # (C, h, w)
    finally:
        hook.remove()
    if plus_plus:
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2 * g2 + a.sum(dim=(1, 2), keepdim=True) * g3
        alpha = torch.where(g != 0, g2 / (denom + 1e-8), torch.zeros_like(g))
        weights = (alpha * F.relu(g)).sum(dim=(1, 2))
    else:
        weights = g.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * a).sum(0))
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return _normalise(cam)


# ---------------------------------------------------------------------- Integrated Gradients
def integrated_gradients(model: nn.Module, x: torch.Tensor, baseline: str = "blur", steps: int = 32,
                         target: str = "fake") -> tuple[np.ndarray, float]:
    """Integrated Gradients (Sundararajan et al., 2017).

    The default *blurred* baseline removes exactly the fine-grained high-frequency content in which
    generator artefacts live, so attributions answer "which pixels' fine detail makes this look FAKE?".
    Returns a signed (32, 32) map in [-1, 1] (positive = pushes toward FAKE when target='fake') and the
    completeness error |sum(attr) - (f(x) - f(baseline))| / |f(x) - f(baseline)| as a sanity check.
    """
    model.eval()
    base = gaussian_blur(x, 2.0) if baseline == "blur" else torch.zeros_like(x)
    alphas = (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps        # midpoint rule
    path = (base + alphas.view(-1, 1, 1, 1) * (x - base)).detach()
    with torch.no_grad():
        logit_x = model(x)[0]
    sign = _sign_for_target(logit_x, target)
    grads = []
    for chunk in path.split(16):
        chunk = chunk.detach().requires_grad_(True)
        out = sign * model(chunk).sum()
        grads.append(torch.autograd.grad(out, chunk)[0])
    attr = (x - base)[0] * torch.cat(grads).mean(0)                                       # (3, H, W)
    with torch.no_grad():
        delta = sign * (model(x)[0] - model(base)[0])
    completeness = float(abs(attr.sum() - delta) / (abs(delta) + 1e-8))
    m = attr.sum(0)
    return (m / (m.abs().max() + 1e-8)).detach().cpu().numpy(), completeness


# ---------------------------------------------------------------------------- attention rollout
def attention_rollout(model: nn.Module, x: torch.Tensor, head_fusion: str = "mean") -> tuple[np.ndarray, dict]:
    """Attention rollout (Abnar & Zuidema, 2020) from the [CLS] token to the spatial tokens.

    For SFHybrid the share of rolled-out attention that goes to the frequency token is also returned
    (`frequency_token_share`), a direct measure of how much the decision used spectral evidence."""
    attn_modules = [m for m in model.modules() if isinstance(m, Attention)]
    if not attn_modules:
        raise ValueError(f"{type(model).__name__} has no attention layers")
    model.eval()
    for m in attn_modules:
        m.keep_attention = True
    try:
        with torch.no_grad():
            model(x)
        mats = [m.last_attention[0] for m in attn_modules]                                # (heads, N, N)
    finally:
        for m in attn_modules:
            m.keep_attention, m.last_attention = False, None
    n = mats[0].shape[-1]
    rollout = torch.eye(n, device=mats[0].device)
    for a in mats:
        a = a.max(0).values if head_fusion == "max" else a.mean(0)
        a = 0.5 * a + 0.5 * torch.eye(n, device=a.device)
        a = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout
    prefix = int(getattr(model, "n_prefix_tokens", 1))
    cls_row = rollout[0]
    spatial = cls_row[prefix:]
    grid = int(round(spatial.numel() ** 0.5))
    heat = F.interpolate(spatial.view(1, 1, grid, grid), size=x.shape[-2:], mode="bilinear",
                         align_corners=False)[0, 0]
    info = {"grid": grid}
    if prefix == 2:  # [CLS, FREQ, ...]
        info["frequency_token_share"] = float(cls_row[1] / cls_row[1:].sum())
    return _normalise(heat), info


# ------------------------------------------------------------------------------- frequency view
def residual_spectrum(x: torch.Tensor) -> np.ndarray:
    """Centred log-magnitude spectrum of the grey-scale high-pass residual of one image (32, 32)."""
    from .models.hybrid import high_pass_residual
    grey = x[:, :1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    res = high_pass_residual(grey)[0, 0]
    spec = torch.fft.fftshift(torch.fft.fft2(res, norm="ortho"))
    return torch.log1p(spec.abs()).cpu().numpy()


def mean_residual_spectra(images_uint8: np.ndarray, labels: np.ndarray, batch: int = 2048) -> dict:
    """Average residual spectrum per class over a dataset (reference 'fingerprints')."""
    from .data import to_float_tensor
    sums = {0: np.zeros((32, 32)), 1: np.zeros((32, 32))}
    counts = {0: 0, 1: 0}
    for i in range(0, len(images_uint8), batch):
        x = to_float_tensor(images_uint8[i:i + batch])
        from .models.hybrid import high_pass_residual
        grey = x[:, :1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        spec = torch.log1p(torch.fft.fftshift(torch.fft.fft2(high_pass_residual(grey)[:, 0], norm="ortho"),
                                              dim=(-2, -1)).abs()).numpy()
        for lab in (0, 1):
            sel = labels[i:i + batch] == lab
            sums[lab] += spec[sel].sum(0)
            counts[lab] += int(sel.sum())
    return {"real": sums[0] / max(1, counts[0]), "fake": sums[1] / max(1, counts[1])}


def spectral_evidence(spectrum: np.ndarray, reference: dict) -> dict:
    """Model-independent cue: is this image's residual spectrum closer to the average FAKE or REAL one?

    Uses the projection of (spectrum - midpoint) on the (fake - real) direction; > 0 = FAKE-like."""
    real, fake = np.asarray(reference["real"]), np.asarray(reference["fake"])
    direction = (fake - real).ravel()
    mid = ((fake + real) / 2).ravel()
    score = float((spectrum.ravel() - mid) @ direction / (np.linalg.norm(direction) ** 2 + 1e-12))
    return {"score": score, "leans": "FAKE" if score > 0 else "REAL"}


# ---------------------------------------------------------------------- faithfulness (deletion)
@torch.no_grad()
def deletion_score(model: nn.Module, x: torch.Tensor, saliency: np.ndarray, steps: int = 8) -> float:
    """Deletion metric (Petsiuk et al., 2018), adapted: replace the most salient pixels first with their
    blurred value and track P(predicted class). Lower area under the curve = more faithful map."""
    model.eval()
    blurred = gaussian_blur(x, 2.0)
    p0 = torch.sigmoid(model(x)[0]).item()
    pred_fake = p0 >= 0.5
    order = np.argsort(-saliency.ravel())
    n = order.size
    batch = []
    for k in range(steps + 1):
        mask = np.zeros(n, dtype=bool)
        mask[order[: int(round(n * k / steps))]] = True
        m = torch.from_numpy(mask.reshape(saliency.shape)).to(x.device)
        batch.append(torch.where(m, blurred[0], x[0]))
    probs = torch.sigmoid(model(torch.stack(batch))).cpu().numpy()
    probs = probs if pred_fake else 1 - probs
    return float(np.trapezoid(probs, dx=1.0 / steps))


# ------------------------------------------------------------------------------------- rendering
def _cmap(name: str):
    import matplotlib
    return matplotlib.colormaps[name]


def to_png_b64(rgb: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def upscale(img_uint8: np.ndarray, size: int = 256) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(img_uint8).resize((size, size), Image.Resampling.NEAREST))


def overlay(image_uint8: np.ndarray, heat: np.ndarray, signed: bool = False, alpha: float = 0.55,
            size: int = 256) -> np.ndarray:
    """Blend a heatmap over the image. Unsigned: 'inferno'; signed [-1, 1]: blue (REAL) - red (FAKE)."""
    from PIL import Image
    if signed:
        colours = _cmap("RdBu_r")((heat + 1) / 2)[..., :3]
        weight = np.abs(heat)[..., None] * alpha
    else:
        colours = _cmap("inferno")(heat)[..., :3]
        weight = np.full(heat.shape + (1,), alpha)
    base = image_uint8.astype(np.float32) / 255
    blend = (1 - weight) * base + weight * colours
    blend = (np.clip(blend, 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(blend).resize((size, size), Image.Resampling.BICUBIC))


def colourise(values: np.ndarray, cmap: str = "magma", size: int = 256) -> np.ndarray:
    from PIL import Image
    v = (values - values.min()) / (np.ptp(values) + 1e-8)
    rgb = (_cmap(cmap)(v)[..., :3] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST))
