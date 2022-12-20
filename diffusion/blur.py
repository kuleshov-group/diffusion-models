"""Implements the core diffusion algorithms."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx
from .gaussian import GaussianDiffusion

class InfoMaxDiffusion(GaussianDiffusion):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, noise_model, a_encoder_model, timesteps, img_shape,
        schedule='linear', device='cpu', a_shape=[10]
    ):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device
        )
        self.a_encoder_model = a_encoder_model
        self.a_shape = a_shape

    def a_sample(self, x0, noise=None):
        """Samples an auxiliary latent a."""
        if noise is None: 
            noise_shape = [x0.shape[0]] + self.a_shape
            noise = torch.randn(noise_shape, device=self.device)

        mu_a, logsigma_a = self.a_encoder_model(x0)
        a = mu_a + torch.exp(logsigma_a) * noise

        return a, mu_a, logsigma_a

    @torch.no_grad()
    def sample(self, batch_size, x=None, a=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        shape = (batch_size, self.img_channels, self.img_dim, self.img_dim)
        if x is None: 
            x = torch.randn(shape, device=self.device)
        if a is None: 
            a_shape = [batch_size] + self.a_shape
            a = torch.randn(a_shape, device=self.device)
        xs = []

        for t in reversed(range(0, self.timesteps)):
            T = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, T, aux=[a], deterministic=(t==0 or deterministic))
            xs.append(x.cpu().numpy())
        return xs

    def loss_at_step_t(self, x0, t, loss_type="l1", noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        a, mu_a, logsigma_a = self.a_sample(x0)
        predicted_noise = self.model(x_noisy, a, t)
        p_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)

        a_loss = (0.5*(
            -1 + mu_a**2 + torch.exp(2*logsigma_a) - 2*logsigma_a
        ).sum([1,2,3])).mean(0)
        # see (10) in https://arxiv.org/pdf/1312.6114.pdf
        # a_loss = (
        #     0.5*(-1+mu_a**2 + torch.exp(2*logsigma_a) - 2*logsigma_a).mean(0)
        # ).sum()
        # TODO: how should a_loss be weighted?
        loss = p_loss + 1e-3*a_loss

        return loss
