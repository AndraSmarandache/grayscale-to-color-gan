import torch
from torch import nn


def init_weights(net, init='norm', gain=0.02):
    """
    Initialize network weights.

    Conv layers get small random weights (normal, xavier, or kaiming).
    BatchNorm layers are initialized with scale near 1 and bias 0 so they
    start close to identity on normalized activations.

    Note: do not call this on a pretrained encoder — it would overwrite
    the ImageNet weights. Apply only to the discriminator or randomly
    initialized parts of the generator.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    net.apply(init_func)
    return net


def init_model(model, device):
    """Move model to device and initialize weights."""
    model = model.to(device)
    model = init_weights(model)
    return model


class AverageMeter:
    """Tracks a running average — used to log loss values per epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    """Create one AverageMeter per tracked loss term."""
    return {
        'loss_D_fake':    AverageMeter(),
        'loss_D_real':    AverageMeter(),
        'loss_D':         AverageMeter(),
        'loss_G_GAN':     AverageMeter(),
        'loss_G_L1':      AverageMeter(),
        'loss_G_TV':      AverageMeter(),
        'loss_G_contrast': AverageMeter(),
        'loss_G':         AverageMeter(),
    }


def update_losses(model, loss_meter_dict, count):
    """Read loss attributes off the model and update the meters."""
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def log_results(loss_meter_dict):
    """Print current average for each tracked loss."""
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
