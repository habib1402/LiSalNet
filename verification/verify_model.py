#!/usr/bin/env python3
"""PyTorch checks to run in your environment before evaluating the revised model.

Checks dimensions, parameter counts, known legacy state migration, synthetic
eval-output equivalence after restoring the old terminal block, and backward
connectivity. No dataset accuracy or GPU efficiency is inferred from these tests.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models import LiSalNet
from models.MobileNetV2 import ConvBNReLU
from losses import DeepSupervisionLoss
from checkpoint_utils import load_checkpoint, load_model_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', type=Path, help='Optional actual original/revised model weights.')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(19)
    model = LiSalNet(pretrained=False).to(device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    assert sum(p.numel() for p in model.parameters()) == 3984607
    assert len(model.Backbone.features) == 18
    assert not any(key.startswith('Backbone.features.18.') for key in model.state_dict())
    assert model.DPM_3.conv_1x1.out_channels == 8
    assert model.MAN_5.cross_attention[2].out_channels == 320
    assert [p.channel_attention.mlp[0].out_channels for p in [model.PFU1, model.PFU3, model.PFU5]] == [8, 8, 20]
    assert [p.spatial_refinement.channel_reduce.out_channels for p in [model.PFU1, model.PFU3, model.PFU5]] == [8, 16, 80]
    assert [m.conv.out_channels for m in model.SFD.lateral_convs] == [16, 32, 320]
    assert model.SFD.fusion_modules[0].fusion[0].kernel_size == (3, 3)

    # Reconstruct only the original unused terminal block for a controlled
    # equivalence check. It is never added by production model code.
    original = copy.deepcopy(model)
    original.Backbone.features.add_module('18', ConvBNReLU(320, 1280, kernel_size=1).to(device))
    original.eval()
    assert sum(p.numel() for p in original.parameters()) == 4396767
    migrated = LiSalNet(pretrained=False).to(device).eval()
    assert len(load_model_state(migrated, {'state_dict': original.state_dict()})) == 6
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(migrated.state_dict()[key], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        for size in [256, 288, 320]:
            rgb, depth = torch.randn(1, 3, size, size, device=device), torch.randn(1, 1, size, size, device=device)
            features = model.Backbone(rgb)
            assert [tuple(x.shape[1:]) for x in features] == [
                (16, size // 2, size // 2), (24, size // 4, size // 4),
                (32, size // 8, size // 8), (96, size // 16, size // 16), (320, size // 32, size // 32)]
            before = original(rgb, depth)
            after = migrated(rgb, depth)
            assert tuple(after.shape) == (1, 3, size, size)
            assert torch.isfinite(after).all() and ((after >= 0) & (after <= 1)).all()
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    del original, migrated

    model.train()
    bn = model.DPM_3.dilated_conv_6.conv_block[0]
    count_before = bn.num_batches_tracked.item()
    rgb, depth = torch.randn(2, 3, 64, 64, device=device), torch.randn(2, 1, 64, 64, device=device)
    target = torch.randint(0, 2, (2, 1, 64, 64), device=device).float()
    prediction = model(rgb, depth)
    loss = DeepSupervisionLoss()(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert bn.num_batches_tracked.item() == count_before + 2
    print(json.dumps({'status': 'PASS', 'parameters': 3984607,
                      'tested_sizes': [256, 288, 320], 'legacy_eval_equivalence': 'exact',
                      'backward_connectivity': 'all registered parameters',
                      'checkpoint': str(args.checkpoint) if args.checkpoint else None}, indent=2))


if __name__ == '__main__':
    main()
