import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifake_guard.ensemble import Ensemble  # noqa: E402
from cifake_guard.models import build_model  # noqa: E402
from cifake_guard.utils import save_json  # noqa: E402

SPECS = {
    "LightCNN": {"arch": "light_cnn"},
    "EfficientNetB0": {"arch": "efficientnet_b0", "img_size": 64, "pretrained": False},
    "TinyViT": {"arch": "tiny_vit", "depth": 2},
    "SFHybrid": {"arch": "sf_hybrid"},
}


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """A run directory with *untrained* checkpoints: exercises loading / inference / API code paths."""
    run = tmp_path_factory.mktemp("run")
    (run / "checkpoints").mkdir()
    (run / "results").mkdir()
    (run / "plots").mkdir()
    torch.manual_seed(0)
    for name, spec in SPECS.items():
        m = build_model(spec)
        torch.save({"name": name, "spec": spec, "state_dict": m.state_dict(), "epoch": 1, "val_loss": 0.69},
                   run / "checkpoints" / f"{name}.pt")
    ens = Ensemble(list(SPECS), {n: 1.0 for n in SPECS}, method="weighted")
    save_json(ens.to_dict(), run / "results" / "ensemble.json")
    rng = np.random.default_rng(0)
    np.savez(run / "results" / "spectral_reference.npz", real=rng.random((32, 32)), fake=rng.random((32, 32)))
    return run


@pytest.fixture()
def rand_batch():
    torch.manual_seed(0)
    return torch.rand(4, 3, 32, 32)
