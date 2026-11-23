"""Train one or more detectors defined in a YAML config.

Examples
  python scripts/train.py --config configs/cpu_experiment.yaml --models LightCNN SFHybrid
  python scripts/train.py --config configs/full_gpu.yaml                      # all models, Colab GPU
  python scripts/train.py --config configs/smoke.yaml --set models.LightCNN.epochs=1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cifake_guard.config import load_config  # noqa: E402
from cifake_guard.data import BatchLoader  # noqa: E402
from cifake_guard.models import build_model  # noqa: E402
from cifake_guard.pipeline import prepare_data, save_config  # noqa: E402
from cifake_guard.training import train_model  # noqa: E402
from cifake_guard.transforms import build_train_augment  # noqa: E402
from cifake_guard.utils import add_file_log, get_device, get_logger, seed_everything  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="subset of model names from the config (default: all)")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. data.batch_size=64")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg["paths"]["output_dir"])
    add_file_log(out_dir / "logs" / "train.log")
    log = get_logger("train")
    if args.threads:
        torch.set_num_threads(args.threads)
    names = args.models or list(cfg["models"])
    unknown = set(names) - set(cfg["models"])
    if unknown:
        ap.error(f"unknown model(s) {sorted(unknown)}; config defines {list(cfg['models'])}")

    seed_everything(cfg["seed"], cfg.get("deterministic", False))
    device = get_device(cfg["device"])
    save_config(cfg, out_dir)
    data = prepare_data(cfg, out_dir)
    for name in names:
        spec = cfg["models"][name]
        # per-model augmentation override, e.g. `augment: {robust: true}` for a robustness-trained variant
        augment = build_train_augment({**cfg["augment"], **spec.get("augment", {})})
        seed_everything(cfg["seed"], cfg.get("deterministic", False))   # order-independent per-model init
        bs = int(spec.get("batch_size", cfg["data"]["batch_size"]))
        train_loader = BatchLoader(data["train"], bs, shuffle=True, drop_last=True, augment=augment, seed=cfg["seed"])
        val_loader = BatchLoader(data["val"], cfg["data"]["eval_batch_size"])
        model = build_model(spec)
        train_model(model, name, spec, train_loader, val_loader, device, out_dir, cfg["training"], cfg["seed"])
    log.info("Done. Checkpoints in %s", out_dir / "checkpoints")


if __name__ == "__main__":
    main()
