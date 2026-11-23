import pytest
import torch

from cifake_guard.models import build_model
from cifake_guard.utils import count_params

from conftest import SPECS


@pytest.mark.parametrize("name", list(SPECS))
def test_forward_shape_and_backward(name, rand_batch):
    model = build_model(SPECS[name])
    out = model(rand_batch)
    assert out.shape == (4,)
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_param_counts_match_original_design():
    assert 0.55e6 < count_params(build_model({"arch": "light_cnn"})) < 0.65e6
    assert 1.7e6 < count_params(build_model({"arch": "tiny_vit"})) < 1.9e6


def test_sfhybrid_ablations_run(rand_batch):
    for use_freq in (True, False):
        for use_tr in (True, False):
            m = build_model({"arch": "sf_hybrid", "use_freq": use_freq, "use_transformer": use_tr})
            assert m(rand_batch).shape == (4,)


def test_efficientnet_frozen_backbone_keeps_bn_stats():
    """Regression test for the original notebook bug: BN statistics drifted while 'frozen'."""
    m = build_model({"arch": "efficientnet_b0", "img_size": 64, "pretrained": False})
    m.set_backbone_trainable(False)
    m.train()
    bn = next(mod for mod in m.features.modules() if isinstance(mod, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    m(torch.rand(8, 3, 32, 32))
    assert torch.equal(before, bn.running_mean)
    assert all(not p.requires_grad for p in m.features.parameters())
    m.set_backbone_trainable(True)
    m.train()
    m(torch.rand(8, 3, 32, 32))
    assert not torch.equal(before, bn.running_mean)


def test_eval_is_deterministic(rand_batch):
    m = build_model({"arch": "sf_hybrid"}).eval()
    with torch.no_grad():
        assert torch.allclose(m(rand_batch), m(rand_batch))
