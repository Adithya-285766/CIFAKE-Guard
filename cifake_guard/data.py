"""CIFAKE data access: discovery, decoding to cached uint8 arrays, deterministic stratified splits,
and a fast in-memory batch loader.

Decoding 120k JPEGs once and caching them as a .npy array makes every later epoch I/O-free; this is
what makes CPU training on the *full* dataset practical.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from PIL import Image

from . import FAKE_LABEL, REAL_LABEL
from .utils import get_logger, load_json, save_json

log = get_logger("data")

KAGGLE_DATASET = "birdy654/cifake-real-and-ai-generated-synthetic-images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------------------- discovery
def find_split_root(base: Path, split: str) -> Path:
    """Locate the folder that contains REAL/ and FAKE/ for a split (robust to nesting)."""
    for p in [base / split, *base.rglob(split)]:
        if p.is_dir() and (p / "REAL").is_dir() and (p / "FAKE").is_dir():
            return p
    raise FileNotFoundError(f"Could not find '{split}/REAL' and '{split}/FAKE' under {base}")


def resolve_data_root(data_root: str | None) -> Path:
    if data_root:
        root = Path(data_root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"paths.data_root={root} does not exist")
        return root
    env = os.environ.get("CIFAKE_DATA_ROOT")
    if env:
        return Path(env)
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Set paths.data_root or install kagglehub to download CIFAKE") from e
    try:  # corporate proxies / antivirus HTTPS inspection (kept from the original notebook)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    log.info("Downloading / locating CIFAKE via kagglehub (%s)", KAGGLE_DATASET)
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def list_labelled_images(split_root: Path) -> tuple[list[str], np.ndarray]:
    """Sorted relative paths and labels (1 = FAKE) for a folder with REAL/ and FAKE/ sub-folders."""
    files, labels = [], []
    for cls, lab in (("REAL", REAL_LABEL), ("FAKE", FAKE_LABEL)):
        names = sorted(n for n in os.listdir(split_root / cls) if Path(n).suffix.lower() in IMAGE_EXTS)
        files += [f"{cls}/{n}" for n in names]
        labels += [lab] * len(names)
    return files, np.asarray(labels, dtype=np.int64)


def _decode(path: Path, size: int = 32) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def decode_folder(split_root: Path, files: list[str], workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        arrays = list(ex.map(lambda f: _decode(split_root / f), files, chunksize=256))
    return np.stack(arrays)


@dataclass
class SplitArrays:
    """Decoded images (N, 32, 32, 3) uint8, labels (N,) with 1 = FAKE, and relative file names."""
    images: np.ndarray
    labels: np.ndarray
    files: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx) -> "SplitArrays":
        idx = np.asarray(idx, dtype=np.int64)
        return SplitArrays(self.images[idx], self.labels[idx], [self.files[i] for i in idx])


def load_split(data_root: Path, split: str, cache_dir: str | Path = "data_cache") -> SplitArrays:
    """Decode (or load from cache) one official CIFAKE split."""
    split_root = find_split_root(data_root, split)
    files, labels = list_labelled_images(split_root)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    img_path, meta_path = cache / f"cifake_{split}_images.npy", cache / f"cifake_{split}_meta.json"
    if img_path.exists() and meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("files") == files:
            return SplitArrays(np.load(img_path, mmap_mode=None), labels, files)
        log.warning("Cache for split %s is stale; re-decoding", split)
    log.info("Decoding %d images from %s (one-off, cached afterwards)", len(files), split_root)
    images = decode_folder(split_root, files)
    np.save(img_path, images)
    save_json({"files": files, "root": str(split_root)}, meta_path)
    return SplitArrays(images, labels, files)


def load_image_folder(folder: str | Path) -> SplitArrays:
    """Load any REAL/ + FAKE/ folder (e.g. an unseen-generator benchmark), resized to 32x32."""
    folder = Path(folder)
    files, labels = list_labelled_images(folder)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}/REAL or {folder}/FAKE")
    return SplitArrays(decode_folder(folder, files), labels, files)


# ------------------------------------------------------------------------------------------- splits
def stratified_indices(labels, per_class, rng: np.random.Generator, exclude=None) -> np.ndarray:
    """Pick `per_class` indices for each class (None = all remaining), excluding `exclude`."""
    labels = np.asarray(labels)
    keep = np.ones(len(labels), dtype=bool)
    if exclude is not None and len(exclude):
        keep[np.asarray(exclude, dtype=int)] = False
    chosen = []
    for c in np.unique(labels):
        idx = np.where((labels == c) & keep)[0]
        rng.shuffle(idx)
        if per_class is not None and per_class > len(idx):
            raise ValueError(f"Requested {per_class} images of class {c} but only {len(idx)} available")
        chosen.append(idx if per_class is None else idx[:per_class])
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def make_splits(train_labels, test_labels, data_cfg: dict, seed: int,
                test_exclude: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Deterministic, disjoint train / val / calib indices (official train split) + test indices.

    * val   -> early stopping and checkpoint selection
    * calib -> temperature scaling and ensemble-weight fitting (never seen during training/selection)
    * test  -> reported numbers only
    """
    rng = np.random.default_rng(seed)
    val = stratified_indices(train_labels, data_cfg["val_per_class"], rng)
    calib = stratified_indices(train_labels, data_cfg["calib_per_class"], rng, exclude=val)
    train = stratified_indices(train_labels, data_cfg["train_per_class"], rng,
                               exclude=np.concatenate([val, calib]))
    test = stratified_indices(test_labels, data_cfg["test_per_class"], rng, exclude=test_exclude)
    splits = {"train": train, "val": val, "calib": calib, "test": test}
    assert not (set(train) & set(val)) and not (set(train) & set(calib)) and not (set(val) & set(calib))
    return splits


# ----------------------------------------------------------------------------------------- loading
def to_float_tensor(images_uint8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1]."""
    t = torch.as_tensor(images_uint8)
    return t.permute(0, 3, 1, 2).float().div_(255.0)


class BatchLoader:
    """Minimal in-memory loader over uint8 arrays; augmentation is applied per batch on tensors."""

    def __init__(self, data: SplitArrays, batch_size: int, shuffle: bool = False, drop_last: bool = False,
                 augment: Callable[[torch.Tensor], torch.Tensor] | None = None, seed: int = 0):
        self.data, self.batch_size, self.shuffle, self.drop_last = data, batch_size, shuffle, drop_last
        self.augment = augment
        self.generator = torch.Generator().manual_seed(seed)
        self._images = torch.from_numpy(np.ascontiguousarray(data.images))
        self._labels = torch.from_numpy(data.labels.astype(np.float32))

    def __len__(self) -> int:
        n = len(self.data)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.data)
        order = torch.randperm(n, generator=self.generator) if self.shuffle else torch.arange(n)
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            x = to_float_tensor(self._images[idx])
            if self.augment is not None:
                x = self.augment(x)
            yield x, self._labels[idx]
