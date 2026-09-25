from argparse import Namespace

import numpy as np
import pytest
import torch

from source.losses import MaskedDepthRegressionLoss
from source.models import ModelDeepLabV3Plus


@pytest.mark.parametrize('encoder', ['resnet18', 'resnet34'])
def test_joint_model_output_shapes(encoder):
    cfg = Namespace(model_encoder_name=encoder, pretrained=False)
    model = ModelDeepLabV3Plus(cfg, {'semseg': 19, 'depth': 1})
    out = model(torch.randn(2, 3, 64, 128))
    assert out['semseg'].shape == (2, 19, 64, 128)
    assert out['depth'].shape == (2, 1, 64, 128)
    assert (out['depth'] >= 0.1).all() and (out['depth'] <= 300).all()


@pytest.mark.parametrize('kind', ['l1', 'l2'])
def test_masked_depth_loss_matches_numpy(kind):
    rng = np.random.default_rng(0)
    pred = rng.uniform(1, 100, (2, 8, 8)).astype(np.float32)
    gt = rng.uniform(1, 100, (2, 8, 8)).astype(np.float32)
    gt[0, :2] = 0.0
    gt[1, 0, 0] = np.nan
    valid = np.isfinite(gt) & (gt > 0)
    diff = pred[valid] - gt[valid]
    expected = np.abs(diff).mean() if kind == 'l1' else (diff ** 2).mean()
    loss = MaskedDepthRegressionLoss(kind)(torch.from_numpy(pred).unsqueeze(1), torch.from_numpy(gt))
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_masked_depth_loss_all_invalid_is_zero_with_gradient():
    pred = torch.rand(1, 1, 4, 4, requires_grad=True)
    loss = MaskedDepthRegressionLoss('l1')(pred, torch.zeros(1, 4, 4))
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(pred.grad).all()
