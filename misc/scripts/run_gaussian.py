import torch
from models.unet.standard import UNet
from data import get_data_loader
from diffusion.gaussian import GaussianDiffusion
from data.fashion_mnist import FashionMNISTConfig as config
from trainer.gaussian import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
img_shape = [config.img_channels, config.img_dim, config.img_dim]

model = UNet(
    channels=config.unet_channels,
    chan_mults=config.unet_mults,
    img_shape=img_shape,
)
model.to(device)

gaussian_diffusion = GaussianDiffusion(
    model=model,
    img_shape=img_shape,
    timesteps=config.timesteps,
    device=device,
)

trainer = Trainer(gaussian_diffusion)
data_loader = get_data_loader(config.name, config.batch_size)
trainer.fit(data_loader, config.epochs)