"""Inference service: loads a trained run directory and produces predictions + explanations.

A run directory (e.g. runs/cpu_experiment) contains
  checkpoints/<Model>.pt          self-describing checkpoints written by training
  results/ensemble.json           temperatures + ensemble parameters written by scripts/evaluate.py
  results/spectral_reference.npz  class-average residual spectra (frequency explanation)
  results/test_metrics.csv        held-out metrics shown in the dashboard
"""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from . import explain
from .ensemble import Ensemble, agreement, sigmoid
from .pipeline import load_checkpoint
from .utils import count_params, get_logger, load_json

log = get_logger("inference")

Image.MAX_IMAGE_PIXELS = 40_000_000          # decompression-bomb guard (~40 MP)
MODEL_DESCRIPTIONS = {
    "LightCNN": "Compact convolutional network trained from scratch (0.6 M parameters).",
    "EfficientNetB0": "ImageNet-pretrained EfficientNet-B0, fine-tuned (transfer learning).",
    "TinyViT": "Small Vision Transformer trained from scratch (self-attention over 4x4 patches).",
    "SFHybrid": "Spatial-frequency CNN-Transformer hybrid: fuses pixel features with an FFT spectrum token.",
    "LightCNN_robust": "LightCNN trained with JPEG/blur/noise/resize/colour augmentation for robustness.",
    "SFHybrid_robust": "SFHybrid trained with JPEG/blur/noise/resize/colour augmentation for robustness.",
    "SFHybrid_noFreq": "Ablation: SFHybrid without its frequency branch.",
}


class InvalidImageError(ValueError):
    """Raised for unreadable / unsupported uploads (mapped to HTTP 400/415 by the API)."""


def load_image_bytes(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImageError("Empty file")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Image.DecompressionBombError as e:
        raise InvalidImageError("Image is too large (decompression-bomb protection)") from e
    except Exception as e:
        raise InvalidImageError("File is not a readable image") from e
    return im


def preprocess(im: Image.Image, size: int = 32, match_cifake_jpeg: bool = True) -> tuple[np.ndarray, dict]:
    """PIL image -> (32, 32, 3) uint8 array matching the CIFAKE input distribution.

    Steps: apply EXIF orientation, flatten transparency onto white, centre-crop to a square, downscale
    with an anti-aliased filter, then (if resizing happened) re-encode as JPEG quality 75 - the exact
    storage format of every CIFAKE training image - so that compression statistics match training."""
    info = {"original_size": list(im.size), "original_mode": im.mode, "original_format": im.format}
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    w, h = im.size
    resized = (w, h) != (size, size)
    if resized:
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((size, size), Image.Resampling.LANCZOS)
        if match_cifake_jpeg:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=75)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
    info.update({"resized": resized, "cropped_to_square": resized and w != h,
                 "jpeg_reencoded": resized and match_cifake_jpeg})
    return np.asarray(im, dtype=np.uint8), info


def confidence_label(p_fake: float) -> tuple[str, str]:
    """Map the ensemble probability to a verdict and a plain-language confidence band."""
    if p_fake >= 0.9:
        return "FAKE", "very likely AI-generated"
    if p_fake >= 0.7:
        return "FAKE", "likely AI-generated"
    if p_fake > 0.3:
        return "UNCERTAIN", "inconclusive - the evidence is mixed"
    if p_fake > 0.1:
        return "REAL", "likely a real photograph"
    return "REAL", "very likely a real photograph"


def plain_summary(p_fake: float, agree: dict, members: dict, info: dict, spectral: dict | None) -> list[str]:
    verdict, band = confidence_label(p_fake)
    lines = [f"The ensemble estimates a {p_fake:.0%} probability that this image is AI-generated: {band}."]
    if agree["unanimous"]:
        lines.append(f"All {agree['n_models']} models agree, which makes the result more trustworthy.")
    else:
        lines.append(f"The models disagree ({agree['n_fake_votes']} say FAKE, {agree['n_real_votes']} say REAL); "
                     "treat the result with caution.")
    strongest = max(members.items(), key=lambda kv: abs(kv[1]["prob_calibrated"] - 0.5))
    lines.append(f"The most confident model is {strongest[0]} "
                 f"({strongest[1]['prob_calibrated']:.0%} AI-generated).")
    if spectral:
        lines.append(f"Independently of the neural networks, the fine-detail frequency pattern of this image "
                     f"looks more like the average {spectral['leans']} training image.")
    if info.get("resized"):
        w, h = info["original_size"]
        lines.append(f"Note: the detector works on 32x32 pixels; your {w}x{h} image was cropped and shrunk, "
                     "which removes much of the fine detail used for detection.")
    lines.append("Limitation: the models were trained only on CIFAKE (CIFAR-10 photos vs. Stable Diffusion v1.4 "
                 "images). Results for other generators, photo types or heavily edited images are unreliable. "
                 "Never use this output as sole evidence.")
    return lines


class Predictor:
    def __init__(self, run_dir: str | Path, device: str = "cpu", members: list[str] | None = None):
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self._lock = threading.Lock()                     # autograd hooks are not thread-safe
        ens_path = self.run_dir / "results" / "ensemble.json"
        ckpt_dir = self.run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"No checkpoints directory in {self.run_dir}")
        if ens_path.exists():
            self.ensemble = Ensemble.from_dict(load_json(ens_path))
        else:
            names = members or sorted(p.stem for p in ckpt_dir.glob("*.pt"))
            log.warning("No ensemble.json in %s; using uncalibrated equal weights", self.run_dir)
            self.ensemble = Ensemble(names, method="mean_calibrated")
        if members:
            self.ensemble = Ensemble(members, {m: self.ensemble.temperatures.get(m, 1.0) for m in members},
                                     method="mean_calibrated")
        self.models, self.ckpts = {}, {}
        for name in self.ensemble.members:
            model, ckpt = load_checkpoint(ckpt_dir / f"{name}.pt", self.device)
            self.models[name], self.ckpts[name] = model, ckpt
        ref = self.run_dir / "results" / "spectral_reference.npz"
        self.spectral_reference = dict(np.load(ref)) if ref.exists() else None
        metrics = self.run_dir / "results" / "test_metrics.csv"
        self.test_metrics = pd.read_csv(metrics, index_col=0) if metrics.exists() else None
        log.info("Predictor ready: %s (ensemble=%s)", list(self.models), self.ensemble.method)

    # ------------------------------------------------------------------ metadata
    def model_info(self) -> list[dict]:
        out = []
        for name, model in self.models.items():
            row = {"name": name, "arch": self.ckpts[name]["spec"]["arch"],
                   "description": MODEL_DESCRIPTIONS.get(name, ""), "params_M": count_params(model) / 1e6,
                   "temperature": self.ensemble.temperatures.get(name, 1.0),
                   "weight": self.ensemble.weights[self.ensemble.members.index(name)],
                   "best_epoch": self.ckpts[name].get("epoch"),
                   "explanations": self.available_explanations(name)}
            if self.test_metrics is not None and name in self.test_metrics.index:
                row["test_metrics"] = {k: float(v) for k, v in self.test_metrics.loc[name].items()
                                       if isinstance(v, (int, float, np.floating)) and np.isfinite(v)}
            out.append(row)
        return out

    def available_explanations(self, name: str) -> list[str]:
        model = self.models[name]
        ex = ["integrated_gradients"]
        if getattr(model, "cam_layer", None):
            ex = ["gradcam", "gradcam_pp"] + ex
        if any(isinstance(m, explain.Attention) for m in model.modules()):
            ex.append("attention_rollout")
        return ex

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def _logits(self, x: torch.Tensor) -> dict[str, float]:
        return {n: float(m(x)[0]) for n, m in self.models.items()}

    def predict(self, im: Image.Image, explain_models: list[str] | str | None = "all",
                ig_steps: int = 32) -> dict:
        t0 = time.perf_counter()
        img, info = preprocess(im)
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        with self._lock:
            logits = self._logits(x)
            names = self.ensemble.members
            z = np.array([[logits[n] for n in names]])
            cal = self.ensemble.calibrated_logits(z)[0]
            members = {n: {"logit": logits[n], "prob_raw": float(sigmoid(logits[n])),
                           "prob_calibrated": float(sigmoid(cal[i])),
                           "verdict": "FAKE" if cal[i] >= 0 else "REAL",
                           "weight": float(self.ensemble.weights[i])} for i, n in enumerate(names)}
            p = float(self.ensemble.predict_proba(z)[0])
            strategies = {m: float(self.ensemble.predict_proba(z, m)[0])
                          for m in ("mean_prob", "majority", "mean_calibrated", "weighted", "stacking")}
            agree = agreement([members[n]["prob_calibrated"] for n in names])
            spectrum = explain.residual_spectrum(x)
            spectral = (explain.spectral_evidence(spectrum, self.spectral_reference)
                        if self.spectral_reference is not None else None)
            explanations = self._explanations(x, img, explain_models, ig_steps)
        verdict, band = confidence_label(p)
        visuals = {"input": explain.to_png_b64(explain.upscale(img)),
                   "spectrum": explain.to_png_b64(explain.colourise(spectrum))}
        if self.spectral_reference is not None:
            diff = self.spectral_reference["fake"] - self.spectral_reference["real"]
            visuals["spectrum_reference_diff"] = explain.to_png_b64(explain.colourise(diff, "RdBu_r"))
        return {
            "verdict": verdict, "confidence_band": band, "prob_fake": p,
            "confidence": max(p, 1 - p), "ensemble_method": self.ensemble.method,
            "ensemble_strategies": strategies, "agreement": agree, "models": members,
            "spectral_evidence": spectral, "preprocessing": info,
            "summary": plain_summary(p, agree, members, info, spectral),
            "explanations": explanations, "visuals": visuals,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    def _explanations(self, x, img, which, ig_steps) -> dict:
        if not which:
            return {}
        names = list(self.models) if which == "all" else [n for n in which if n in self.models]
        out = {}
        for name in names:
            model, e = self.models[name], {}
            try:
                if getattr(model, "cam_layer", None):
                    e["gradcam"] = explain.to_png_b64(explain.overlay(img, explain.grad_cam(model, x)))
                    e["gradcam_pp"] = explain.to_png_b64(
                        explain.overlay(img, explain.grad_cam(model, x, plus_plus=True)))
                ig, completeness = explain.integrated_gradients(model, x, steps=ig_steps)
                e["integrated_gradients"] = explain.to_png_b64(explain.overlay(img, ig, signed=True))
                e["ig_completeness_error"] = completeness
                if "attention_rollout" in self.available_explanations(name):
                    heat, ainfo = explain.attention_rollout(model, x)
                    e["attention_rollout"] = explain.to_png_b64(explain.overlay(img, heat))
                    e.update({f"attention_{k}": v for k, v in ainfo.items()})
            except Exception:                                     # never fail a prediction on a visual
                log.exception("Explanation failed for %s", name)
                e["error"] = "explanation failed"
            out[name] = e
        return out
