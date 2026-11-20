"""Section 3.7: equally weighted BCE + mini-batch Dice for P1, P3, P5."""
import torch.nn as nn
import torch.nn.functional as F


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


class DeepSupervisionLoss(nn.Module):
    def forward(self, inputs, target):
        if isinstance(target, tuple):
            target = target[0]
        if target.ndim == 4:
            target = target[:, 0, :, :]
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("Expected three probability maps [B, 3, H, W].")
        target = target.to(device=inputs.device, dtype=inputs.dtype)
        loss1 = BCEDiceLoss(inputs[:, 0, :, :], target)
        loss2 = BCEDiceLoss(inputs[:, 1, :, :], target)
        loss3 = BCEDiceLoss(inputs[:, 2, :, :], target)
        return loss1 + loss2 + loss3
