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
        self.gammas = [torch.sqrt(1 - self.alphas[0])]
        for a in self.alphas[1:]:
            self.gammas.append(self.gammas[-1] * torch.sqrt(a) + torch.sqrt(1 - a))
        self.gammas = torch.tensor(self.gammas).to(device)
        self.z_encoder_model = z_encoder_model
        self.z_shape = img_shape

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0).
        """
        if noise is None: noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = get_by_idx(
            self.sqrt_bar_alphas, t, x0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, x0.shape
        )
        gammas_t = get_by_idx(
            self.gammas, t, x0.shape)
        
        # TODO: can be replaced with a function call
        mu_phi, log_sigma_phi = self.z_encoder_model(x0)
        mu_phi = mu_phi.view(x0.shape)
        sigma_phi = log_sigma_phi.exp().view(x0.shape)

        return (sqrt_alphas_cumprod_t * x0
                + gammas_t * mu_phi
                + sqrt_one_minus_alphas_cumprod_t * sigma_phi * noise)

    def z_sample(self, x0, noise=None):
        """Samples an auxiliary latent a."""
        if noise is None: 
            noise_shape = [x0.shape[0]] + self.z_shape
            noise = torch.randn(noise_shape, device=self.device)

        mu_z, logsigma_z = self.z_encoder_model(x0)
        z = mu_z + torch.exp(logsigma_z) * noise

        return z, mu_z, logsigma_z

    @torch.no_grad()
    def p_sample(self, xt, t, aux=None, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Takes a sample from p(x_{t-1}|x_t).
        """
        mu_phi, log_sigma_phi = self.z_encoder_model(xt)
        mu_phi = mu_phi.view(xt.shape)
        sigma_phi = log_sigma_phi.exp().view(xt.shape)

        coefficient_mu_x = get_by_idx(
            1 / torch.sqrt(self.alphas), t, xt.shape)
        coefficient_mu_z = get_by_idx(
            (1 - self.alphas) / torch.sqrt(
                self.alphas * (1 - self.bar_alphas)),
            t, xt.shape)
        coefficient_mu_phi = get_by_idx(
            torch.sqrt(1 - self.alphas), t, xt.shape)
        z = self.noise_model(xt, t)
        xt_prev_mean = (
            coefficient_mu_x * xt
            - coefficient_mu_phi * mu_phi
            - coefficient_mu_z * z)

        if deterministic:
            return xt_prev_mean
        else:
            variance = get_by_idx(
                self.posterior_variance, t, x.shape) * sigma_phi
            print(variance.shape, sigma_phi.shape)
            return xt_prev_mean + variance * torch.randn(x.shape, device=device)

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
        loss = p_loss + z_loss / self.timesteps

        return loss
