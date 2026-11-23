"""YAML configuration with defaults, deep-merge and dotted command-line overrides.

Example override from the CLI:  --set data.train_per_class=2000 models.LightCNN.epochs=3
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "deterministic": False,
    "paths": {
        "data_root": None,          # folder containing train/ and test/; None = download with kagglehub
        "cache_dir": "data_cache",  # decoded uint8 arrays are cached here (fast loading)
        "output_dir": "runs/default",
    },
    "data": {
        "train_per_class": None,    # None = all remaining training images
        "val_per_class": 2500,      # early stopping / model selection
        "calib_per_class": 2500,    # temperature scaling + ensemble fitting (disjoint from val)
        "test_per_class": None,     # None = full official test split (10k per class)
        "batch_size": 128,
        "eval_batch_size": 512,
        "exclude_test_near_duplicates": False,
    },
    "augment": {
        "hflip": True,
        "translate": 0.125,         # max shift as a fraction of the image side (reflect padding)
        "robust": False,            # JPEG / blur / noise / resize / colour augmentation
        "robust_p": 0.5,
    },
    "training": {
        "patience": 4,
        "grad_clip": 1.0,
        "amp": True,                # mixed precision, only used on CUDA
    },
    "models": {
        "LightCNN": {"arch": "light_cnn", "epochs": 15, "lr": 2e-3, "weight_decay": 5e-4,
                     "warmup_epochs": 1, "label_smoothing": 0.0},
        "EfficientNetB0": {"arch": "efficientnet_b0", "img_size": 128, "pretrained": True, "epochs": 8,
                           "lr": 1e-3, "weight_decay": 1e-4, "warmup_epochs": 1, "freeze_epochs": 1,
                           "label_smoothing": 0.0, "batch_size": 128},
        "TinyViT": {"arch": "tiny_vit", "epochs": 20, "lr": 1e-3, "weight_decay": 0.05,
                    "warmup_epochs": 2, "label_smoothing": 0.1},
        "SFHybrid": {"arch": "sf_hybrid", "epochs": 15, "lr": 2e-3, "weight_decay": 5e-3,
                     "warmup_epochs": 1, "label_smoothing": 0.0, "use_freq": True, "use_transformer": True},
    },
    "evaluation": {
        "bootstrap": 1000,
        "robustness_per_class": 1000,
        "ensemble_members": None,   # None = every trained model
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "models":
            out[k] = deep_merge(out[k], v)
        elif k == "models" and isinstance(v, dict):
            # Models listed in a config file *replace* the default set, but each one inherits the
            # defaults of the same-named default model (so a config can just say `LightCNN: {epochs: 3}`).
            out[k] = {name: deep_merge(base.get("models", {}).get(name, {}), spec or {})
                      for name, spec in v.items()}
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(value)
    return cfg


def validate(cfg: dict) -> None:
    known_arch = {"light_cnn", "efficientnet_b0", "tiny_vit", "sf_hybrid"}
    if not cfg["models"]:
        raise ValueError("Config defines no models")
    for name, spec in cfg["models"].items():
        if spec.get("arch") not in known_arch:
            raise ValueError(f"Model {name}: unknown arch {spec.get('arch')!r}; choose from {sorted(known_arch)}")
        if spec.get("epochs", 0) < 1:
            raise ValueError(f"Model {name}: epochs must be >= 1")
    for key in ("val_per_class", "calib_per_class"):
        if not cfg["data"][key] or cfg["data"][key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    user = {}
    if path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = apply_overrides(deep_merge(DEFAULTS, user), overrides)
    validate(cfg)
    return cfg
