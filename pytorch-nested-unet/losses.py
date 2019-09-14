"""Define loss function."""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, target):
        bce = F.binary_cross_entropy_with_logits(inputs, target)
        smooth = 1e-5
        inputs = torch.sigmoid(inputs)
        num = target.size(0)
        inputs = inputs.view(num, -1)
        target = target.view(num, -1)
        intersection = (inputs * target)
        dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, target):
        inputs = inputs.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(inputs, target, per_image=True)

        return loss
