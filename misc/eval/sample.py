import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image

def sample(model, n_samples, path, deterministic=False):
    samples = model.sample(n_samples, deterministic=deterministic)
    samples = torch.Tensor(samples[-1])
    samples = (samples + 1) * 0.5
    save_image(samples, path, nrow=6)

def viz_latents(model, data_loader, latents, path):
    # see https://avandekleut.github.io/vae/
    for i, batch in enumerate(data_loader):
        x = batch['pixel_values']
        y = batch['label']
        batch_size = x.shape[0]
        # should factor out into an "encode" function
        _, z, _ = model.a_sample(x.to(model.device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i*batch_size > latents:
            plt.colorbar()
            plt.savefig(path)
            break

def interpolate(model, n_samples, path, data_loader=None):
    if data_loader:
        interpolate_from_data(model, n_samples, path, data_loader)
    else:
        interpolate_from_prior(model, n_samples, path)

def interpolate_from_data(model, n_samples, path, data_loader):
    pass

def interpolate_from_prior(model, n_samples, path):
    pass    