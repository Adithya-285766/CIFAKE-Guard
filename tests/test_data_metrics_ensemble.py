import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from cifake_guard.config import DEFAULTS, apply_overrides, deep_merge, load_config
from cifake_guard.data import BatchLoader, SplitArrays, make_splits, stratified_indices
from cifake_guard.ensemble import Ensemble, agreement, fit_temperature, sigmoid
from cifake_guard.metrics import bootstrap_ci, compute_metrics, expected_calibration_error, mcnemar_test


# ------------------------------------------------------------------------------ data
def test_splits_are_disjoint_stratified_and_deterministic():
    labels = np.array([0] * 500 + [1] * 500)
    cfg = {"val_per_class": 50, "calib_per_class": 40, "train_per_class": 100, "test_per_class": 30}
    a = make_splits(labels, labels, cfg, seed=1)
    b = make_splits(labels, labels, cfg, seed=1)
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert not set(a["train"]) & set(a["val"]) and not set(a["val"]) & set(a["calib"])
    assert not set(a["train"]) & set(a["calib"])
    assert np.bincount(labels[a["val"]]).tolist() == [50, 50]
    assert np.bincount(labels[a["calib"]]).tolist() == [40, 40]
    assert len(a["test"]) == 60


def test_split_excludes_test_duplicates():
    labels = np.array([0] * 10 + [1] * 10)
    s = make_splits(labels, labels, {"val_per_class": 2, "calib_per_class": 2, "train_per_class": None,
                                     "test_per_class": None}, 0, test_exclude=np.array([0, 1, 15]))
    assert not {0, 1, 15} & set(s["test"]) and len(s["test"]) == 17


def test_stratified_too_many_raises():
    with pytest.raises(ValueError):
        stratified_indices(np.array([0, 0, 1]), 5, np.random.default_rng(0))


def test_batch_loader_shapes_and_range():
    data = SplitArrays(np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8), np.arange(10) % 2,
                       [f"f{i}" for i in range(10)])
    batches = list(BatchLoader(data, 4))
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]
    x, y = batches[0]
    assert x.shape[1:] == (3, 32, 32) and 0 <= x.min() and x.max() <= 1 and y.dtype == torch.float32
    assert len(list(BatchLoader(data, 4, drop_last=True))) == 2


# ---------------------------------------------------------------------------- config
def test_config_merge_and_overrides():
    cfg = deep_merge(DEFAULTS, {"models": {"LightCNN": {"epochs": 3}}})
    assert list(cfg["models"]) == ["LightCNN"] and cfg["models"]["LightCNN"]["lr"] == 2e-3
    cfg = apply_overrides(cfg, ["data.batch_size=7", "models.LightCNN.epochs=2"])
    assert cfg["data"]["batch_size"] == 7 and cfg["models"]["LightCNN"]["epochs"] == 2


def test_config_files_are_valid():
    for f in ("configs/smoke.yaml", "configs/cpu_experiment.yaml", "configs/full_gpu.yaml"):
        load_config(f)


def test_bad_config_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("models: {X: {arch: resnet999, epochs: 1}}")
    with pytest.raises(ValueError):
        load_config(p)


# --------------------------------------------------------------------------- metrics
def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.random(500) * 0.5, 0, 1)
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p >= 0.5))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 500


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.0, 0.1, 0.9, 1.0]))
    assert m["accuracy"] == 1 and m["roc_auc"] == 1 and m["eer"] == 0 and m["tpr@1%fpr"] == 1


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, (p > 0.5).astype(float)) > 0.2


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.5 + rng.random(400) * 0.6, 0, 1)
    ci = bootstrap_ci(y, p, n_boot=200)
    acc = compute_metrics(y, p)["accuracy"]
    assert ci["accuracy"][0] <= acc <= ci["accuracy"][1]


def test_mcnemar_identical_predictions():
    y = np.array([0, 1, 1, 0])
    assert mcnemar_test(y, y, y)["p_value"] == 1.0


# --------------------------------------------------------------------- calibration
def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 20_000)
    y = (rng.random(20_000) < sigmoid(true_logits)).astype(int)
    t = fit_temperature(true_logits * 3, y)      # model 3x overconfident
    assert t == pytest.approx(3.0, rel=0.1)


def test_ensemble_fit_and_predict_and_roundtrip():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good = (2 * y - 1) * 3 + rng.normal(0, 1, 2000)
    noise = rng.normal(0, 1, 2000)
    Z = np.stack([good, noise], 1)
    ens = Ensemble.fit(Z, y, ["good", "noise"])
    assert ens.weights[0] > 0.8                                   # informative model dominates
    for method in ("mean_prob", "majority", "mean_calibrated", "weighted", "stacking"):
        p = ens.predict_proba(Z, method)
        assert p.shape == (2000,) and ((0 <= p) & (p <= 1)).all()
    assert accuracy_score(y, ens.predict_proba(Z, "weighted") >= 0.5) > 0.95
    again = Ensemble.from_dict(ens.to_dict())
    assert np.allclose(again.predict_proba(Z), ens.predict_proba(Z))


def test_agreement():
    a = agreement([0.9, 0.8, 0.2])
    assert a["n_fake_votes"] == 2 and not a["unanimous"] and a["agreement"] == pytest.approx(2 / 3)
    assert agreement([0.1, 0.2])["unanimous"]
