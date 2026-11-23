"""Seed-variance study: the same model trained with different seeds on the *same* data split.

  python scripts/seed_study.py --config configs/cpu_experiment.yaml --model LightCNN \
      --runs runs/cpu_experiment runs/seed_study/seed43 runs/seed_study/seed44

Reports test metrics per seed and mean +- std (uncalibrated probabilities, threshold 0.5), written to
runs/seed_study/<model>_seed_study.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cifake_guard.config import load_config  # noqa: E402
from cifake_guard.ensemble import sigmoid  # noqa: E402
from cifake_guard.metrics import compute_metrics  # noqa: E402
from cifake_guard.pipeline import load_checkpoint, predict_logits, prepare_data  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="LightCNN")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    data = prepare_data(cfg, Path(cfg["paths"]["output_dir"]))        # identical split for every run
    rows = {}
    for run in args.runs:
        model, ckpt = load_checkpoint(Path(run) / "checkpoints" / f"{args.model}.pt")
        p = sigmoid(predict_logits(model, data["test"].images, "cpu", cfg["data"]["eval_batch_size"]))
        m = compute_metrics(data["test"].labels, p)
        rows[f"seed {ckpt.get('seed')}"] = {k: m[k] for k in ("accuracy", "f1", "roc_auc", "pr_auc", "ece")}
    df = pd.DataFrame(rows).T
    df.loc["mean"], df.loc["std"] = df.mean(), df.iloc[:len(rows)].std(ddof=1)
    out = Path("runs/seed_study") / f"{args.model}_seed_study.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
