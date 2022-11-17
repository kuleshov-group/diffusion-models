"""Implements the core diffusion algorithms."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx
from .gaussian import GaussianDiffusion

class LearnedGaussianDiffusion(GaussianDiffusion):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, noise_model, z_encoder_model, timesteps, img_shape,
        schedule='linear', device='cpu'
    ):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device
        )
        self.z_encoder_model = z_encoder_model
        self.z_shape = img_shape

    def z_sample(self, x0, noise=None):
        """Samples an auxiliary latent a."""
        if noise is None: 
            noise_shape = [x0.shape[0]] + self.z_shape
            noise = torch.randn(noise_shape, device=self.device)

        mu_z, logsigma_z = self.z_encoder_model(x0)
        z = mu_z + torch.exp(logsigma_z) * noise

        return z, mu_z, logsigma_z

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0).
        """
        # encode x0 into auxiliary encoder variable a
        if noise is None:  noise, _, _ = self.z_sample(x0)
        return super().q_sample(x0, t, noise)

    def loss_at_step_t(self, x0, t, loss_type="l1", noise=None):
        if noise is not None: raise NotImplementedError()

        # encode x0 into auxiliary encoder variable a
        z, mu_z, logsigma_z = self.z_sample(x0)
        x_noisy = self.q_sample(x0=x0, t=t, noise=z)
        predicted_noise = self.model(x_noisy, t)
        p_loss = self.p_loss_at_step_t(z, predicted_noise, loss_type)

        z_loss = (
            0.5*(-1+mu_z**2 + torch.exp(logsigma_z)**2 - 2*logsigma_z).mean(0)
        ).sum()
        loss = p_loss + z_loss

        return loss
