import torch

def get_schedule(schedule_name):
    if schedule_name == 'linear':
        return linear_beta_schedule
    if schedule_name == 'cosine':
        return cosine_beta_schedule
    if schedule_name == 'quadratic':
        return quadratic_beta_schedule
    if schedule_name == 'sigmoid':
        return sigmoid_beta_schedule
    else:
        raise ValueError()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_by_idx(vals, idx, x_shape):
    batch_size = idx.shape[0]
    retrieved_vals = vals.gather(-1, idx.cpu())
    return retrieved_vals.reshape(
        batch_size, *((1,) * (len(x_shape) - 1))
    ).to(idx.device)    