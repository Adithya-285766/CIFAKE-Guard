"""Evaluation metrics (positive class = FAKE), calibration error and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE with equal-width bins on the predicted probability of the positive class."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Return (mean predicted prob, observed frequency, count) per non-empty bin."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    conf, freq, cnt = [], [], []
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            conf.append(y_prob[mask].mean())
            freq.append(y_true[mask].mean())
            cnt.append(int(mask.sum()))
    return np.array(conf), np.array(freq), np.array(cnt)


def equal_error_rate(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def tpr_at_fpr(y_true, y_prob, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.interp(target_fpr, fpr, tpr))


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred) if both else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_TPR": recall_score(y_true, y_pred, zero_division=0),
        "specificity_TNR": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if both else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_prob) if both else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if both else float("nan"),
        "eer": equal_error_rate(y_true, y_prob) if both else float("nan"),
        "tpr@1%fpr": tpr_at_fpr(y_true, y_prob, 0.01) if both else float("nan"),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 threshold: float = 0.5) -> dict:
    """Percentile bootstrap 95% CIs for accuracy, F1 and ROC-AUC (resampling test images)."""
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {"accuracy": [], "f1": [], "roc_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        pred = (yp >= threshold).astype(int)
        stats["accuracy"].append((pred == yt).mean())
        stats["f1"].append(f1_score(yt, pred, zero_division=0))
        stats["roc_auc"].append(roc_auc_score(yt, yp))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in stats.items() if v}


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact-binomial McNemar test for a difference in error rates between two classifiers."""
    from scipy.stats import binomtest
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right, b_right = pred_a == y_true, pred_b == y_true
    n01 = int((a_right & ~b_right).sum())
    n10 = int((~a_right & b_right).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": float(p)}
