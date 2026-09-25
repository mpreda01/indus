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


def test_visualization_image_keeps_colors_in_wandb(monkeypatch):
    """W&B renders float tensors as black images: the logged image must be 8-bit and keep its colors."""
    import wandb
    from source.utils.visualization import to_pil_image
    monkeypatch.setenv('WANDB_MODE', 'offline')
    vis = torch.zeros(3, 8, 8)
    vis[:, :, :4] = torch.tensor([128, 64, 128]).view(3, 1, 1) / 255       # Cityscapes "road" color
    logged = np.array(wandb.Image(to_pil_image(vis)).image)
    assert logged[2, 1].tolist() == [128, 64, 128]
    assert logged[2, 6].tolist() == [0, 0, 0]
