"""Shared training loop (kept from the original notebook, with fixes).

Fixes w.r.t. the notebook:
* crash if validation loss is never finite (`best_state` stayed None) -> explicit error;
* non-finite training loss now aborts with a clear message instead of silently corrupting weights;
* validation also reports ROC-AUC (threshold-free), not only accuracy at 0.5;
* checkpoints store the model spec, epoch and validation metrics, so they are self-describing and can
  be loaded by the evaluation scripts and the API without the training config.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score

from . import __version__
from .utils import count_params, get_logger

log = get_logger("training")


def warmup_cosine(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None,
              label_smoothing=0.0, grad_clip=1.0, amp=False):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_y = [], []
    use_amp = amp and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        target = y * (1 - label_smoothing) + 0.5 * label_smoothing if train else y
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.float(), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered; lower the learning rate or disable AMP")
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += ((logits.detach() > 0).float() == y).sum().item()
        n += bs
        if not train:
            all_probs.append(torch.sigmoid(logits.detach().float()).cpu())
            all_y.append(y.cpu())
    auc = float("nan")
    if all_probs:
        yy, pp = torch.cat(all_y).numpy(), torch.cat(all_probs).numpy()
        if len(np.unique(yy)) == 2:
            auc = roc_auc_score(yy, pp)
    return total_loss / n, correct / n, auc


def save_epoch_plots(history: dict, model_name: str, plots_dir: Path) -> None:
    """Overwrite <model>_loss.png and <model>_accuracy.png after every epoch (original behaviour)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    for metric, ylabel, fname in [("loss", "BCE loss", "loss"), ("acc", "Accuracy", "accuracy")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history[f"train_{metric}"], marker="o", label="train")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{model_name}: {ylabel} (epoch {len(history['train_loss'])})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{model_name}_{fname}.png", dpi=120)
        plt.close(fig)


def train_model(model: nn.Module, name: str, spec: dict, train_loader, val_loader, device: torch.device,
                out_dir: Path, training_cfg: dict, seed: int):
    """Train with AdamW + warmup/cosine, early stopping on validation loss; returns (model, history)."""
    out_dir = Path(out_dir)
    ckpt_dir, plots_dir, res_dir = out_dir / "checkpoints", out_dir / "plots", out_dir / "results"
    for d in (ckpt_dir, plots_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    epochs, lr = int(spec["epochs"]), float(spec["lr"])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    params = model.param_groups(lr) if hasattr(model, "param_groups") else model.parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(spec.get("weight_decay", 0.0)))
    steps = len(train_loader)
    scheduler = warmup_cosine(optimizer, epochs * steps, int(spec.get("warmup_epochs", 1)) * steps)
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    freeze_epochs = int(spec.get("freeze_epochs", 0))
    patience = int(training_cfg.get("patience", 4))

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "lr", "epoch_time_s"]}
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    log.info("Training %s | %.3f M params | %d epochs | %d steps/epoch | device=%s",
             name, count_params(model) / 1e6, epochs, steps, device)

    for epoch in range(1, epochs + 1):
        if freeze_epochs and hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(epoch > freeze_epochs)
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, device, criterion, optimizer, scheduler, scaler,
                                       float(spec.get("label_smoothing", 0.0)),
                                       float(training_cfg.get("grad_clip", 1.0)), amp)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader, device, criterion, amp=amp)
        dt = time.time() - t0
        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc, va_auc, optimizer.param_groups[-1]["lr"], dt]):
            history[k].append(v)
        save_epoch_plots(history, name, plots_dir)

        improved = va_loss < best_val
        if improved:
            best_val, best_epoch, bad = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "name": name, "spec": spec, "state_dict": best_state, "epoch": epoch,
                "val_loss": va_loss, "val_acc": va_acc, "val_auc": va_auc, "seed": seed,
                "temperature": 1.0, "package_version": __version__,
            }, ckpt_dir / f"{name}.pt")
        else:
            bad += 1
        log.info("[%s] epoch %02d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f auc %.4f | %.0fs %s",
                 name, epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, va_auc, dt, "*best*" if improved else "")
        if bad >= patience:
            log.info("[%s] early stopping: no val-loss improvement for %d epochs", name, patience)
            break

    if best_state is None:
        raise RuntimeError(f"{name}: validation loss never improved from +inf; training diverged")
    model.load_state_dict(best_state)
    hist_df = pd.DataFrame(history).rename_axis("epoch")
    hist_df.index += 1
    hist_df.to_csv(res_dir / f"{name}_history.csv")
    log.info("[%s] best epoch %d (val loss %.4f)", name, best_epoch, best_val)
    return model, history
