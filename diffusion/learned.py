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

        Takes a sample from q(xt|x0). t \in {1, \dots, timesteps}
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

    @torch.no_grad()
    def p_sample(self, xt, t_index, aux=None, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device, dtype=torch.long)
        
        coefficient_mu_z = 1 / torch.sqrt(t).view(batch_size, -1)
        m_t = self.forward_matrix(self.model.time_mlp(t)).view(batch_size, -1)
        m_t_minus_1 = self.forward_matrix(
            self.model.time_mlp(t - 1)).view(batch_size, -1)
        if t_index == 1:
            m_t_minus_1 = 1 + 0 * m_t_minus_1  # identity
        z = self.model(xt, t).view(batch_size, -1)
        xt = xt.view(batch_size, -1)
        model_mean = m_t_minus_1 * ((1 / m_t) * xt - coefficient_mu_z * z)
        x_t_minus_1 = model_mean

        if not deterministic:
            x_t_minus_1 += torch.randn_like(
                model_mean) * m_t_minus_1 * torch.sqrt((t - 1) / t).view(batch_size, -1)
        
        return x_t_minus_1.view(* original_batch_shape)

    @torch.no_grad()
    def sample(self, batch_size, x=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        shape = (batch_size, self.img_channels, self.img_dim, self.img_dim)
        if x is None: 
            x = torch.randn(shape, device=self.device)
        xs = []

        for t in reversed(range(1, 1 + self.timesteps)):
            x = self.p_sample(x, t, deterministic=deterministic)
            xs.append(x.cpu().numpy())
        return xs

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1',
                       noise=None):
        if noise is not None: raise NotImplementedError()
        t = t + 1  # t \in {1, \dots, timesteps}
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

        return loss_weights * noise_loss + kl_divergence / self.timesteps, {
            'noise_loss': noise_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }
