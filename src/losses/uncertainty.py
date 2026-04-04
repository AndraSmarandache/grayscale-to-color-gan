import torch
from torch import nn


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for uncertainty-aware colorization.

    The generator predicts both a mean (ab_mean) and a log-variance (log_var)
    per pixel. The loss has two terms:
        - reconstruction: (ab_gt - ab_mean)^2 * exp(-log_var)
          → large error is penalized less when the model says it's uncertain
        - regularization: log_var
          → penalizes the model for claiming everything is uncertain

    Together they force the model to be uncertain only where it genuinely
    cannot predict the correct color, and confident where it can.

    Parameters
    ----------
    reduction : str
        'mean' or 'sum'
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ab_mean, log_var, ab_target):
        # clamp log_var for numerical stability
        log_var = torch.clamp(log_var, -10.0, 10.0)
        loss = 0.5 * (log_var + (ab_target - ab_mean) ** 2 * torch.exp(-log_var))
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()
