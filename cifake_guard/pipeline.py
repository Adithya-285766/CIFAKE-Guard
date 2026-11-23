"""Glue used by the train / evaluate scripts: data preparation and checkpoint loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from .data import SplitArrays, load_split, make_splits, resolve_data_root
from .models import build_model
from .utils import get_logger, load_json, save_json

log = get_logger("pipeline")


def prepare_data(cfg: dict, out_dir: Path) -> dict[str, SplitArrays]:
    """Load CIFAKE, create (or re-load) the deterministic splits and return them as arrays."""
    root = resolve_data_root(cfg["paths"]["data_root"])
    cache = cfg["paths"]["cache_dir"]
    train_full, test_full = load_split(root, "train", cache), load_split(root, "test", cache)
    split_file = Path(out_dir) / "results" / "splits.json"
    if split_file.exists():
        stored = load_json(split_file)
        splits = {k: np.asarray(v, dtype=np.int64) for k, v in stored["indices"].items()}
        log.info("Re-using stored splits from %s", split_file)
    else:
        exclude = None
        if cfg["data"].get("exclude_test_near_duplicates"):
            dup_file = Path(cache) / "test_near_duplicates.json"
            if not dup_file.exists():
                raise FileNotFoundError(f"{dup_file} missing: run scripts/audit_data.py first")
            exclude = np.asarray(load_json(dup_file)["test_indices"], dtype=np.int64)
            log.info("Excluding %d test images with near-duplicates in the training split", len(exclude))
        splits = make_splits(train_full.labels, test_full.labels, cfg["data"], cfg["seed"], exclude)
        save_json({"seed": cfg["seed"], "indices": {k: v.tolist() for k, v in splits.items()},
                   "test_files": [test_full.files[i] for i in splits["test"][:20]]}, split_file)
    data = {k: train_full.subset(splits[k]) for k in ("train", "val", "calib")}
    data["test"] = test_full.subset(splits["test"])
    log.info("Split sizes: %s", {k: len(v) for k, v in data.items()})
    return data


def save_config(cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Load a self-describing checkpoint -> (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["spec"], pretrained_override=False)   # weights come from the checkpoint
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_logits(model, images_uint8: np.ndarray, device, batch_size: int = 512, transform=None) -> np.ndarray:
    """Raw logits for an (N, 32, 32, 3) uint8 array; `transform` is an optional tensor perturbation."""
    from .data import to_float_tensor
    model.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        x = to_float_tensor(images_uint8[i:i + batch_size])
        if transform is not None:
            x = transform(x)
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()
