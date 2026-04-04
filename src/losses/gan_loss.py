import torch
from torch import nn


class GANLoss(nn.Module):
    """
    Adversarial loss for both discriminator and generator.

    The discriminator outputs a tensor of patch logits. We compare them to a target map
    (all ones for real, all zeros for fake). The generator reuses this module with
    target_is_real=True on fake inputs to push D's scores up for generated images.

    Supports two modes:
        vanilla  — BCEWithLogitsLoss on raw logits (standard GAN)
        lsgan    — MSELoss treating scores as regression toward 0 or 1

    register_buffer keeps real_label / fake_label on the correct device without
    making them trainable parameters. expand_as broadcasts the scalar to every patch.

    Parameters
    ----------
    gan_mode : str
        'vanilla' or 'lsgan'
    real_label : float
        Target for real patches. Use 0.9 instead of 1.0 for one-sided label smoothing.
    fake_label : float
        Target for fake patches.
    """
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        return self.loss(preds, labels)
