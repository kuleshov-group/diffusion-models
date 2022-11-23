"""Implements the core diffusion algorithms."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx

class GaussianDiffusion(nn.Module):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, model, timesteps, img_shape, schedule='linear', device='cpu'
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.img_dim = img_shape[1]
        self.img_channels = img_shape[0]
        self.schedule = get_schedule(schedule)
        self.device=device

        # initialize the alpha and beta paramters
        self.betas = self.schedule(timesteps)
        self.alphas = 1 - self.betas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.bar_alphas = torch.cumprod(self.alphas, axis=0)
        self.sqrt_bar_alphas = torch.sqrt(self.bar_alphas)
        self.sqrt_one_minus_bar_alphas = torch.sqrt(1. - bar_alphas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        bar_alphas_prev = F.pad(self.bar_alphas[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1. - bar_alphas_prev) / (1. - bar_alphas)
        )

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0).
        """
        if noise is None: noise = torch.randn_like(x0)

        sqrt_bar_alphas_t = get_by_idx(
            self.sqrt_bar_alphas, t, x0.shape
        )
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, x0.shape
        )

        return (
            sqrt_bar_alphas_t * x0 
            + sqrt_one_minus_bar_alphas_t * noise
        )

    @torch.no_grad()
    def p_sample(self, xt, t, aux=None, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Takes a sample from p(x_{t-1}|x_t).
        """
        betas_t = get_by_idx(self.betas, t, xt.shape)
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, xt.shape
        )
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, t, xt.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        inputs = [xt, t] if aux is None else [xt] + aux + [t]
        xt_prev_mean = sqrt_recip_alphas_t * (
            xt - betas_t * self.model(*inputs) / sqrt_one_minus_bar_alphas_t
        )

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + torch.sqrt(post_var_t) * noise

        # return x_{t-1}
        return xt_prev

    @torch.no_grad()
    def sample(self, batch_size, x=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        shape = (batch_size, self.img_channels, self.img_dim, self.img_dim)
        if x is None: 
            x = torch.randn(shape, device=self.device)
        xs = []

        for t in reversed(range(0, self.timesteps)):
            T = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, T, deterministic=(t==0 or deterministic))
            xs.append(x.cpu().numpy())
        return xs

    def p_loss_at_step_t(self, noise, predicted_noise, loss_type="l1"):
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    def loss_at_step_t(self, x0, t, loss_type="l1", noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)

        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, eval=False):
        self.load_state_dict(torch.load(path))
        if eval:
            self.model.eval()
