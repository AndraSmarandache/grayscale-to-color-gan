import torch
from torch import nn
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet


def build_res_unet(n_input=1, n_output=2, size=256):
    """
    Build U-Net with ResNet18 backbone - Compatible with fastai 2.7+ and PyTorch 2.x
    Inputs: n_input - we input 1 channel (L from Lab - lightness)
            n_output - output are the a and b channels - the colors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CPU would result in a very slow training process!

    # Create ResNet18 model - should work on both versions of Colab
    try:
        # PyTorch 2.x syntax
        from torchvision.models import ResNet18_Weights
        model = resnet18(weights='DEFAULT') # ResNet18_Weights.DEFAULT is equivalent to ResNet18_Weights.IMAGENET1K_V1
    except:
        try:
            # PyTorch 1.x syntax
            model = resnet18(pretrained=True)
        except:
            # No pretrained weights
            model = resnet18(pretrained=False)
            print("Warning: Could not load pretrained weights!")

    # Modify first layer for grayscale input (1 channel instead of 3 - used in ResNet18)
    if n_input == 1:
        # Get the weights from the first layer
        old_conv = model.conv1
        # Create the new conv layer for 1 channel input by averaging the 3 RGB channels to get grayscale weights
        with torch.no_grad(): # disable gradient calculation
            new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Average the pretrained weights across RGB channels
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = new_conv

    body = create_body(model, cut=-2) # eliminate the classifier layers of ResNet18 - keep just feature detection
    net_G = DynamicUnet(body, n_output, (size, size)).to(device) # build U-Net and push it to GPU (if available)

    return net_G


class UnetBlock(nn.Module):
    """
    U-Net block with skip connections.

    Each block does:
        1. DOWN: convolve (optionally with ReLU, Norm) to reduce spatial size and go to 'ni' channels.
        2. Pass through submodule (another UnetBlock, or nothing if innermost).
        3. UP: transpose conv to go back to original spatial size and 'nf' channels.
        4. For non-outermost blocks: OUTPUT = concat(original_input, up_output) along channels.

    Parameters
    ----------
    nf : int
        Number of filters (channels) at the OUTPUT of this block (after the up path)
    ni : int
        Number of channels in the internal representation (after down path, before submodule)
    submodule : nn.Module, optional
        The inner block (next level of U-Net). None only for innermost block
    input_c : int, optional
        Number of input channels. Only set for the first (outermost) block; else defaults to nf
    dropout : bool
        If True, add Dropout(0.5) in the up path (only for middle blocks)
    innermost : bool
        If True, there is no submodule - just one down layer and one up layer (bottleneck)
    outermost : bool
        If True, this is the top block: input is image, output is ab; no skip concat in forward
    """
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf # input_c is given when the block is the first one, else we assume input channels are the same as output's

        # Encoder (down path)
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False) # spatial size divided by 2
        downrelu = nn.LeakyReLU(0.2, True) # keep small gradient for negative values for a stable encoder training
        downnorm = nn.BatchNorm2d(ni)

        # Decoder (up path)
        uprelu = nn.ReLU(True) # keep non-negative features for reconstruction
        upnorm = nn.BatchNorm2d(nf)

        if outermost: # top of U-Net
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1) # skip ni + up ni => ni * 2 channels for up input
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()] # output in [-1, 1] for the a, b channels
            model = down + [submodule] + up
        elif innermost: # bottleneck
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False) # no skip concatenation, just ni for up input
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up # no submodule
        else: # middle block
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False) # submodule returns concat(skip, submodule_output) => up input = ni * 2
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)] # optional dropout
            model = down + [submodule] + up

        self.model = nn.Sequential(*model) # sequential list of modules

    def forward(self, x):
        if self.outermost:
            return self.model(x) # no skip connection, keep only decoder output (predicted a, b channels)
        else:
            return torch.cat([x, self.model(x)], 1) # skip + decoder output for the level above


class ClassicUnet(nn.Module):
    """
    Classic U-Net without ResNet backbone

    Each block halves spatial size on the way down and doubles on the way up; skip connections concatenate encoder feature maps with decoder feature maps.

    Parameters
    ----------
    input_c : int
        Input channels (1 for grayscale L channel)
    output_c : int
        Output channels (2 for ab in LAB colorization)
    n_down : int
        Number of downsampling stages (depth of U-Net)
    num_filters : int
        Base number of filters
    """
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5): # middle blocks of 512 channels each, with dropout
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3): # blocks that reduce channel count (from 512 to 64)
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


def build_classic_unet(n_input=1, n_output=2, size=256):
    """Build classic U-Net without ResNet - all weights trained from scratch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = ClassicUnet(input_c=n_input, output_c=n_output, n_down=8, num_filters=64).to(device)
    return net_G


USE_RESNET = True

def build_generator(n_input=1, n_output=2, size=256):
    """Build generator - choose between ResNet or Classic U-Net"""
    if USE_RESNET:
        print("Using U-Net with ResNet18 backbone (pretrained)")
        return build_res_unet(n_input, n_output, size)
    else:
        print("Using Classic U-Net (no pretrained weights)")
        return build_classic_unet(n_input, n_output, size)
