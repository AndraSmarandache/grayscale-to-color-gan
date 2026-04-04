import torch
from torch import nn


class NoGreyLoss(nn.Module):
    """
    No Grey Loss

    Penalizes predicted ab values that are too close to zero (achromatic).
    The generator tends to hedge toward grey in ambiguous regions because
    it minimizes expected L1 error over the distribution of plausible colors.
    This loss provides a soft lower bound on color saturation.

    Parameters
    ----------
    threshold : float
        Saturation threshold in normalized ab space. Pixels whose predicted
        ab magnitude is below this value contribute to the loss. Default: 0.05.
    """
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, ab_pred):
        """
        ab_pred : [B, 2, H, W] — predicted color channels (normalized to [-1, 1])
        """
        saturation = torch.norm(ab_pred, dim=1, keepdim=True)  # [B, 1, H, W]
        loss = torch.clamp(self.threshold - saturation, min=0.0)
        return loss.mean()
