"""Probability calibration (temperature scaling) and ensembling strategies.

All fitting happens on the *calibration* split (disjoint from training, validation and test), so the
test numbers of calibrated models and of fitted ensembles are not optimistically biased.

Strategies
  mean_prob      soft voting on raw (uncalibrated) probabilities - the naive baseline
  majority       hard majority vote (ties -> FAKE); probability = fraction of FAKE votes
  mean_calibrated average of temperature-scaled probabilities
  weighted       convex combination of calibrated logits, weights fitted by minimising log loss
  stacking       logistic regression on calibrated logits (learned ensemble)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression

ENSEMBLE_METHODS = ("mean_prob", "majority", "mean_calibrated", "weighted", "stacking")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _nll(logits, y):
    z = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit_temperature(logits, y, bounds=(0.05, 20.0)) -> float:
    """Single scalar T minimising NLL of sigmoid(logit / T) (Guo et al., 2017)."""
    logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)
    res = minimize_scalar(lambda log_t: _nll(logits / np.exp(log_t), y),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def _softmax(w):
    e = np.exp(w - w.max())
    return e / e.sum()


class Ensemble:
    """Holds per-model temperatures and fitted combination parameters; serialisable to JSON."""

    def __init__(self, members: list[str], temperatures: dict[str, float] | None = None,
                 weights: list[float] | None = None, stack_coef: list[float] | None = None,
                 stack_intercept: float = 0.0, method: str = "weighted"):
        if method not in ENSEMBLE_METHODS:
            raise ValueError(f"Unknown ensemble method {method!r}")
        self.members = list(members)
        self.temperatures = temperatures or {m: 1.0 for m in members}
        self.weights = weights or [1.0 / len(members)] * len(members)
        self.stack_coef = stack_coef or [1.0] * len(members)
        self.stack_intercept = stack_intercept
        self.method = method

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, logits: np.ndarray, y: np.ndarray, members: list[str], method: str = "weighted") -> "Ensemble":
        """`logits`: (N, M) raw logits on the calibration split, columns ordered like `members`."""
        logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)
        temps = {m: fit_temperature(logits[:, j], y) for j, m in enumerate(members)}
        cal = logits / np.array([temps[m] for m in members])
        # weighted: softmax-parametrised convex weights on calibrated logits
        res = minimize(lambda w: _nll(cal @ _softmax(w), y), np.zeros(len(members)), method="L-BFGS-B")
        weights = _softmax(res.x).tolist()
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(cal, y.astype(int))
        return cls(members, temps, weights, lr.coef_[0].tolist(), float(lr.intercept_[0]), method)

    # --------------------------------------------------------------- inference
    def calibrated_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        return logits / np.array([self.temperatures[m] for m in self.members])

    def predict_proba(self, logits: np.ndarray, method: str | None = None) -> np.ndarray:
        method = method or self.method
        logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        if method == "mean_prob":
            return sigmoid(logits).mean(axis=1)
        if method == "majority":
            return (logits > 0).mean(axis=1)
        cal = self.calibrated_logits(logits)
        if method == "mean_calibrated":
            return sigmoid(cal).mean(axis=1)
        if method == "weighted":
            return sigmoid(cal @ np.asarray(self.weights))
        if method == "stacking":
            return sigmoid(cal @ np.asarray(self.stack_coef) + self.stack_intercept)
        raise ValueError(method)

    def to_dict(self) -> dict:
        return {"members": self.members, "temperatures": self.temperatures, "weights": self.weights,
                "stack_coef": self.stack_coef, "stack_intercept": self.stack_intercept, "method": self.method}

    @classmethod
    def from_dict(cls, d: dict) -> "Ensemble":
        return cls(d["members"], d.get("temperatures"), d.get("weights"), d.get("stack_coef"),
                   d.get("stack_intercept", 0.0), d.get("method", "weighted"))


def agreement(member_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Agreement statistics for one image given its per-model probabilities."""
    p = np.asarray(member_probs, dtype=np.float64)
    votes = p >= threshold
    n_fake = int(votes.sum())
    majority = max(n_fake, len(p) - n_fake)
    return {"n_models": int(len(p)), "n_fake_votes": n_fake, "n_real_votes": int(len(p) - n_fake),
            "agreement": majority / len(p), "unanimous": bool(majority == len(p)), "prob_std": float(p.std())}
