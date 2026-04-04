import torch
from torch import nn


class SpatialColorAffinityLoss(nn.Module):
    """
    Spatial Color Affinity Loss — original contribution.

    Core idea: adjacent pixels with similar luminance (L) should also have similar
    color (ab). This enforces local color consistency and reduces artifacts like
    isolated color patches in uniform regions.

    Parameters
    ----------
    threshold : float
        Luminance difference threshold. Pixel pairs with |L_i - L_j| < threshold
        are considered similar and contribute to the loss. Default: 0.1.
    """
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, L, ab_pred, ab_target):
        """
        L        : [B, 1, H, W] — luminance channel
        ab_pred  : [B, 2, H, W] — predicted color channels
        ab_target: [B, 2, H, W] — ground truth color channels
        """
        # Horizontal luminance difference between adjacent pixels
        L_diff_h = torch.abs(L[:, :, :, :-1] - L[:, :, :, 1:])
        # Vertical luminance difference between adjacent pixels
        L_diff_v = torch.abs(L[:, :, :-1, :] - L[:, :, 1:, :])

        # Binary mask: 1 where adjacent pixels have similar luminance
        similar_h = (L_diff_h < self.threshold).float()
        similar_v = (L_diff_v < self.threshold).float()

        # Color difference between adjacent pixels (predicted)
        ab_diff_pred_h = torch.abs(ab_pred[:, :, :, :-1] - ab_pred[:, :, :, 1:]).mean(dim=1, keepdim=True)
        ab_diff_pred_v = torch.abs(ab_pred[:, :, :-1, :] - ab_pred[:, :, 1:, :]).mean(dim=1, keepdim=True)

        # Color difference between adjacent pixels (ground truth)
        ab_diff_target_h = torch.abs(ab_target[:, :, :, :-1] - ab_target[:, :, :, 1:]).mean(dim=1, keepdim=True)
        ab_diff_target_v = torch.abs(ab_target[:, :, :-1, :] - ab_target[:, :, 1:, :]).mean(dim=1, keepdim=True)

        # For similar-L pairs, penalize mismatch in color smoothness
        loss_h = (similar_h * torch.abs(ab_diff_pred_h - ab_diff_target_h)).mean()
        loss_v = (similar_v * torch.abs(ab_diff_pred_v - ab_diff_target_v)).mean()

        return loss_h + loss_v
