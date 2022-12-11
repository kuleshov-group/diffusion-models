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
        schedule='cosine', device='cpu'
    ):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device
        )
        self.forward_matrix = forward_matrix
        self.z_shape = img_shape

    def _forward_sample(self, x0, time):
        return self.forward_matrix(self.model.time_mlp(time))

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0). t \in {1, \dots, timesteps}
        """
        if noise is None:
            noise = torch.randn_like(x0)
        original_batch_shape = x0.shape
        batch_size = original_batch_shape[0]
        transormation_matrices = self._forward_sample(
            x0, t).view(batch_size, -1)
        x_t = transormation_matrices * self._add_noise(
            x0, t, noise).view(batch_size, -1)
        return x_t.view(* original_batch_shape)

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device, dtype=torch.long)
        xt = xt.view(batch_size, -1)
        betas_t = get_by_idx(self.betas, t, xt.shape)
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, xt.shape
        )
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, t, xt.shape)

        # compute m matrices
        m_t_bar = self._forward_sample(None, t).view(batch_size, -1)
        m_t_minus_1_bar = self._forward_sample(None, t - 1).view(batch_size, -1)
        if t_index == 0:
            m_t_minus_1_bar = 1 + 0 * m_t_minus_1_bar  # identity
        m_t_inverse = m_t_minus_1_bar / m_t_bar
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        z = self.model(xt.view(* original_batch_shape), t).view(batch_size, -1)
        xt_prev_mean = m_t_inverse * sqrt_recip_alphas_t * (
            xt - betas_t * m_t_bar * z / sqrt_one_minus_bar_alphas_t
        )

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + m_t_minus_1_bar * torch.sqrt(post_var_t) * noise

        # return x_{t-1}
        return xt_prev.view(* original_batch_shape)

    def _compute_prior_kl_divergence(self, x0, batch_size):
        m_T = self._forward_sample(
            x0, torch.tensor([self.timesteps] * batch_size,
                             device=self.device)).view(batch_size, -1)
        trace = (m_T ** 2).sum(dim=1).mean()
        mu_squared = 0
        log_determinant = torch.log(m_T ** 2).sum(dim=1).mean()
        return 0.5 * (trace + mu_squared - log_determinant - 784)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is not None: raise NotImplementedError()
        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        
        noise_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)

        return loss_weights * noise_loss + kl_divergence / self.timesteps, {
            'noise_loss': noise_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }
