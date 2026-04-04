import torch
from torch import nn, optim

from src.models.generator import build_res_unet, UncertaintyGenerator
from src.models.discriminator import PatchDiscriminator
from src.losses.gan_loss import GANLoss
from src.losses.auxiliary import total_variation_loss, contrast_loss
from src.losses.spatial_affinity import SpatialColorAffinityLoss
from src.losses.no_grey import NoGreyLoss
from src.losses.uncertainty import GaussianNLLLoss
from src.training.utils import init_model

SIZE = 256


class MainModel(nn.Module):
    """
    Full GAN model for image colorization.

    Wraps generator, discriminator, losses, and optimizers.
    Call setup_input(data) then optimize() on each training batch.

    Parameters
    ----------
    net_G : nn.Module or None
        Generator. If None, builds ResNet-18 U-Net.
        Pass an UncertaintyGenerator instance to enable uncertainty mode.
    use_uncertainty : bool
        If True, net_G must be an UncertaintyGenerator. The reconstruction
        loss switches from L1 to Gaussian NLL and the model tracks per-pixel
        uncertainty (self.uncertainty after each forward pass).
    lr_G, lr_D : float
        Adam learning rates.
    beta1, beta2 : float
        Adam betas.
    lambda_L1 : float
        Weight on reconstruction loss (L1 or NLL depending on use_uncertainty).
    lambda_TV : float
        Weight on Total Variation loss.
    lambda_contrast : float
        Weight on Contrast loss (0 = disabled).
    lambda_affinity : float
        Weight on Spatial Color Affinity loss (0 = disabled).
    lambda_nogrey : float
        Weight on No Grey loss (0 = disabled).
    """
    def __init__(self, net_G=None, use_uncertainty=False,
                 lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999,
                 lambda_L1=100., lambda_TV=1.0,
                 lambda_contrast=0.0, lambda_affinity=0.0, lambda_nogrey=0.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_uncertainty = use_uncertainty
        self.lambda_L1       = lambda_L1
        self.lambda_TV       = lambda_TV
        self.lambda_contrast = lambda_contrast
        self.lambda_affinity = lambda_affinity
        self.lambda_nogrey   = lambda_nogrey

        if net_G is None:
            base = init_model(build_res_unet(n_input=1, n_output=2, size=SIZE), self.device)
            if use_uncertainty:
                self.net_G = UncertaintyGenerator(base).to(self.device)
            else:
                self.net_G = base
        else:
            self.net_G = net_G.to(self.device)

        self.net_D         = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion  = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion   = nn.L1Loss()
        self.NLLcriterion  = GaussianNLLLoss()
        self.aff_criterion = SpatialColorAffinityLoss(threshold=0.1)
        self.ng_criterion  = NoGreyLoss(threshold=0.05)
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L  = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        if self.use_uncertainty:
            self.fake_color, self.fake_log_var = self.net_G(self.L)
            # uncertainty = mean std across a and b channels, shape [B, 1, H, W]
            self.uncertainty = torch.sqrt(torch.exp(
                torch.clamp(self.fake_log_var, -10, 10)
            ).mean(dim=1, keepdim=True))
        else:
            self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)

        self.loss_G_GAN = self.GANcriterion(fake_preds, True)

        if self.use_uncertainty:
            self.loss_G_L1 = self.NLLcriterion(
                self.fake_color, self.fake_log_var, self.ab) * self.lambda_L1
        else:
            self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1

        self.loss_G_TV = total_variation_loss(fake_image) * self.lambda_TV

        if self.lambda_contrast > 0:
            real_image = torch.cat([self.L, self.ab], dim=1)
            self.loss_G_contrast = contrast_loss(fake_image, real_image) * self.lambda_contrast
        else:
            self.loss_G_contrast = torch.tensor(0.0, device=self.device)

        if self.lambda_affinity > 0:
            self.loss_G_affinity = self.aff_criterion(
                self.L, self.fake_color, self.ab) * self.lambda_affinity
        else:
            self.loss_G_affinity = torch.tensor(0.0, device=self.device)

        if self.lambda_nogrey > 0:
            self.loss_G_nogrey = self.ng_criterion(self.fake_color) * self.lambda_nogrey
        else:
            self.loss_G_nogrey = torch.tensor(0.0, device=self.device)

        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 + self.loss_G_TV
                       + self.loss_G_contrast + self.loss_G_affinity + self.loss_G_nogrey)
        self.loss_G.backward()

    def optimize(self):
        self.forward()

        # only update D when it's not too dominant
        # (if loss_D_fake is already very low, D doesn't need more training)
        if not hasattr(self, 'loss_D_fake') or self.loss_D_fake.item() > 0.3:
            self.net_D.train()
            self.set_requires_grad(self.net_D, True)
            self.opt_D.zero_grad()
            self.backward_D()
            self.opt_D.step()
        else:
            # still need loss_D values for logging even if we skip the update
            with torch.no_grad():
                fake_image = torch.cat([self.L, self.fake_color], dim=1)
                fake_preds = self.net_D(fake_image.detach())
                self.loss_D_fake = self.GANcriterion(fake_preds, False)
                real_image = torch.cat([self.L, self.ab], dim=1)
                real_preds = self.net_D(real_image)
                self.loss_D_real = self.GANcriterion(real_preds, True)
                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
