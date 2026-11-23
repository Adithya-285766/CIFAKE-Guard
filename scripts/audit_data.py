"""Dataset audit for shortcut features and train/test leakage.

Checks
 1. file format / JPEG quantisation tables / file sizes per class (a format or compression difference
    between REAL and FAKE would be a trivial shortcut);
 2. exact pixel-level duplicates within and across splits and classes;
 3. near-duplicates between the test split and the training split (same class), found with an 8x8
    grey-scale cosine pre-filter and verified at full resolution (mean absolute pixel difference).

Writes  <out>/data_audit.json  and  <cache_dir>/test_near_duplicates.json (used by
`data.exclude_test_near_duplicates: true`).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from cifake_guard.config import load_config  # noqa: E402
from cifake_guard.data import find_split_root, load_split, resolve_data_root  # noqa: E402
from cifake_guard.utils import get_logger, save_json  # noqa: E402

log = get_logger("audit")


def format_stats(split_root: Path, n: int = 500) -> dict:
    out = {}
    for cls in ("REAL", "FAKE"):
        names = sorted(os.listdir(split_root / cls))[:n]
        sizes, fmts, qtabs, subs = [], set(), set(), set()
        for name in names:
            p = split_root / cls / name
            with Image.open(p) as im:
                fmts.add(im.format)
                if im.format == "JPEG":
                    from PIL import JpegImagePlugin
                    qtabs.add(hashlib.md5(str(im.quantization).encode()).hexdigest()[:8])
                    subs.add(JpegImagePlugin.get_sampling(im))
            sizes.append(p.stat().st_size)
        out[cls] = {"formats": sorted(fmts), "n_distinct_quant_tables": len(qtabs), "quant_table_ids": sorted(qtabs),
                    "chroma_subsampling": sorted(subs), "file_size_mean": float(np.mean(sizes)),
                    "file_size_std": float(np.std(sizes))}
    return out


def exact_duplicates(arrs: dict) -> dict:
    """arrs: name -> SplitArrays. Group identical pixel arrays across all splits/classes."""
    groups = defaultdict(list)
    for split, a in arrs.items():
        for i, img in enumerate(a.images):
            groups[hashlib.md5(img.tobytes()).hexdigest()].append((split, int(a.labels[i]), i))
    dup = [g for g in groups.values() if len(g) > 1]
    summary = defaultdict(int)
    for g in dup:
        kinds = sorted({(s, lab) for s, lab, _ in g})
        summary[" & ".join(f"{s}/{'FAKE' if lab else 'REAL'}" for s, lab in kinds)] += 1
    return {"n_duplicate_groups": len(dup), "by_location": dict(summary), "examples": dup[:10]}


def near_duplicates(train, test, cos_thr: float = 0.97, mae_thr: float = 8.0) -> dict:
    def thumbs(images):
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float().mean(1, keepdim=True)
        t = F.adaptive_avg_pool2d(x, 8).flatten(1)
        t = t - t.mean(1, keepdim=True)
        return F.normalize(t, dim=1)

    found, pairs = [], []
    for lab in (0, 1):
        tr_idx, te_idx = np.where(train.labels == lab)[0], np.where(test.labels == lab)[0]
        tr_t, te_t = thumbs(train.images[tr_idx]), thumbs(test.images[te_idx])
        for s in range(0, len(te_idx), 1000):
            sim = te_t[s:s + 1000] @ tr_t.T
            best, arg = sim.max(1)
            for k in torch.where(best > cos_thr)[0].tolist():
                ti, ri = te_idx[s + k], tr_idx[arg[k].item()]
                mae = np.abs(test.images[ti].astype(np.int16) - train.images[ri].astype(np.int16)).mean()
                if mae < mae_thr:
                    found.append(int(ti))
                    pairs.append({"test": test.files[ti], "train": train.files[ri], "mae": float(mae),
                                  "cos8x8": float(best[k])})
    return {"test_indices": sorted(set(found)), "pairs": pairs,
            "criteria": {"cos8x8_gt": cos_thr, "full_res_mae_lt": mae_thr}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cpu_experiment.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = resolve_data_root(cfg["paths"]["data_root"])
    cache = cfg["paths"]["cache_dir"]
    train, test = load_split(root, "train", cache), load_split(root, "test", cache)

    report = {"format": {s: format_stats(find_split_root(root, s)) for s in ("train", "test")}}
    log.info("Format check: %s", report["format"])
    report["exact_duplicates"] = exact_duplicates({"train": train, "test": test})
    log.info("Exact duplicate groups: %s", report["exact_duplicates"]["by_location"])
    nd = near_duplicates(train, test)
    n_real = sum(1 for p in nd["pairs"] if p["test"].startswith("REAL"))
    report["near_duplicates_test_vs_train"] = {
        "n_test_images": len(nd["test_indices"]), "n_real": n_real, "n_fake": len(nd["pairs"]) - n_real,
        "criteria": nd["criteria"], "examples": nd["pairs"][:25]}
    log.info("Test images with a near-duplicate in train: %d (REAL %d, FAKE %d)",
             len(nd["test_indices"]), n_real, len(nd["pairs"]) - n_real)
    save_json(report, Path(args.out) / "data_audit.json")
    save_json(nd, Path(cache) / "test_near_duplicates.json")
    log.info("Wrote %s", Path(args.out) / "data_audit.json")


if __name__ == "__main__":
    main()
