import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """Patch Discriminator for GAN

    Instead of classification with one scalar (real vs fake) for the whole image, the network outputs a grid of
    predictions: each spatial location summarizes whether a local image patch looks real or fake.

    This encourages sharp, locally consistent structure (edges, texture), making it a common choice in
    image-to-image models.

    The stack is a series of strided convolutions that increase channels while shrinking spatial size,
    then a 1-channel head without activation so training can use BCEWithLogitsLoss on logits directly.

    Parameters
    ----------
    input_c : int
        Number of input channels (e.g. 1 for L only, 3 for RGB, or L+ab stacked = 3 for conditional D).
    num_filters : int
        Base width; channel count grows as num_filters * 2**stage.
    n_down : int
        How many middle conv blocks to stack after the first conv (each doubles channels from the previous width).
    """
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()

        # First block: project input channels (3: L, a, b) up to num_filters. No norm on the very first layer
        model = [self.get_layers(input_c, num_filters, norm=False)]

        # n_down blocks: ni = num_filters * 2^i, nf = num_filters * 2^(i+1).
        # Stride 2 for all but the last block results in spatial downsampling most of the way
        # Last block uses stride 1 for refined features without shrinking further before the 1x1-style head.
        model += [
            self.get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]

        # Output: one logit per spatial location. No norm, no activation; logits for BCEWithLogits
        model += [
            self.get_layers(
                num_filters * 2**n_down,
                1,
                s=1,
                norm=False,
                act=False,
            )
        ]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        # Conv -> BatchNorm -> LeakyReLU block used throughout the discriminator
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
