import io

import numpy as np
import pytest
import torch
from PIL import Image

from cifake_guard import explain
from cifake_guard.inference import confidence_label, preprocess
from cifake_guard.models import build_model
from cifake_guard.transforms import PERTURBATIONS, apply_perturbation, build_train_augment


@pytest.mark.parametrize("kind", list(PERTURBATIONS))
def test_perturbations_preserve_shape_and_range(kind, rand_batch):
    for sev in PERTURBATIONS[kind][1]:
        y = apply_perturbation(rand_batch, kind, sev)
        assert y.shape == rand_batch.shape and y.min() >= 0 and y.max() <= 1


def test_noise_is_seeded(rand_batch):
    a = apply_perturbation(rand_batch, "gaussian_noise", 0.05, seed=3)
    b = apply_perturbation(rand_batch, "gaussian_noise", 0.05, seed=3)
    assert torch.equal(a, b)


def test_jpeg_changes_image_but_q100_is_close(rand_batch):
    assert (apply_perturbation(rand_batch, "jpeg", 30) - rand_batch).abs().mean() > 0.02


def test_train_augment(rand_batch):
    aug = build_train_augment({"hflip": True, "translate": 0.125, "robust": True, "robust_p": 1.0})
    y = aug(rand_batch)
    assert y.shape == rand_batch.shape and y.min() >= 0 and y.max() <= 1


# ------------------------------------------------------------------------- explain
@pytest.fixture(scope="module")
def x():
    torch.manual_seed(0)
    return torch.rand(1, 3, 32, 32)


@pytest.mark.parametrize("arch", ["light_cnn", "sf_hybrid"])
@pytest.mark.parametrize("pp", [False, True])
def test_gradcam(arch, pp, x):
    m = build_model({"arch": arch}).eval()
    cam = explain.grad_cam(m, x, plus_plus=pp)
    assert cam.shape == (32, 32) and cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_gradcam_rejects_vit(x):
    with pytest.raises(ValueError):
        explain.grad_cam(build_model({"arch": "tiny_vit"}), x)


def test_integrated_gradients_completeness(x):
    m = build_model({"arch": "light_cnn"}).eval()
    attr, err = explain.integrated_gradients(m, x, steps=64)
    assert attr.shape == (32, 32) and np.abs(attr).max() <= 1 + 1e-6
    assert err < 0.05          # IG axiom: attributions sum to f(x) - f(baseline)


@pytest.mark.parametrize("arch,prefix", [("tiny_vit", 1), ("sf_hybrid", 2)])
def test_attention_rollout(arch, prefix, x):
    m = build_model({"arch": arch}).eval()
    heat, info = explain.attention_rollout(m, x)
    assert heat.shape == (32, 32) and info["grid"] == 8
    assert ("frequency_token_share" in info) == (prefix == 2)
    # attention capture must be switched off again afterwards
    assert all(not a.keep_attention for a in m.modules() if isinstance(a, explain.Attention))


def test_spectrum_and_deletion(x):
    spec = explain.residual_spectrum(x)
    assert spec.shape == (32, 32)
    ref = {"real": np.zeros((32, 32)), "fake": np.ones((32, 32))}
    assert explain.spectral_evidence(np.ones((32, 32)) * 2, ref)["leans"] == "FAKE"
    m = build_model({"arch": "light_cnn"}).eval()
    score = explain.deletion_score(m, x, np.random.rand(32, 32))
    assert 0 <= score <= 1


def test_overlay_and_png(x):
    img = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    for signed, heat in ((False, np.random.rand(32, 32)), (True, np.random.rand(32, 32) * 2 - 1)):
        rgb = explain.overlay(img, heat, signed=signed)
        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert len(explain.to_png_b64(img)) > 100


# ---------------------------------------------------------------------- preprocess
def test_preprocess_large_rgba_image():
    im = Image.new("RGBA", (300, 200), (255, 0, 0, 128))
    arr, info = preprocess(im)
    assert arr.shape == (32, 32, 3) and arr.dtype == np.uint8
    assert info["resized"] and info["cropped_to_square"] and info["jpeg_reencoded"]


def test_preprocess_native_size_untouched():
    src = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    arr, info = preprocess(Image.fromarray(src))
    assert np.array_equal(arr, src) and not info["resized"]


def test_preprocess_grayscale_and_palette():
    for mode in ("L", "P", "CMYK"):
        arr, _ = preprocess(Image.new(mode, (64, 64)))
        assert arr.shape == (32, 32, 3)


def test_confidence_bands():
    assert confidence_label(0.95)[0] == "FAKE"
    assert confidence_label(0.5)[0] == "UNCERTAIN"
    assert confidence_label(0.02)[0] == "REAL"
