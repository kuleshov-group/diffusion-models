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
        self, noise_model, forward_matrix, timesteps, img_shape,
        schedule='linear', device='cpu'
    ):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device
        )
        self.forward_matrix = forward_matrix
        self.z_shape = img_shape

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0).
        """
        # encode x0 into auxiliary encoder variable a
        if noise is None:  noise, _, _ = torch.randn_like(x0)
        original_batch_shape = x0.shape
        batch_size = original_batch_shape[0]
        transormation_matrices = self.forward_matrix(
            self.model.time_mlp(t)).view(batch_size, -1)
        x0 = x0.view(batch_size, -1)
        t = t.view(batch_size, -1)
        noise = noise.view(batch_size, -1)
        x_t = transormation_matrices * (x0 + torch.sqrt(t) * noise)
        return x_t.view(* original_batch_shape)

    def loss_at_step_t(self, x0, t, loss_type="l1", noise=None):
        if noise is not None: raise NotImplementedError()

        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        noise_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)
        m_T = self.forward_matrix(self.model.time_mlp(
            torch.tensor([self.timesteps] * batch_size,
                         device=self.device))).view(batch_size, -1)
        
        trace = self.timesteps * (m_T ** 2).sum(dim=1).mean()
        mu_squared = (
            (m_T * x0.view(batch_size, -1)) ** 2).sum(dim=1).mean()
        log_determinant = torch.log(self.timesteps * m_T ** 2).sum(dim=1).mean()
        kl_divergence = 0.5 * (trace + mu_squared - log_determinant - 784)

        return noise_loss + kl_divergence / self.timesteps
