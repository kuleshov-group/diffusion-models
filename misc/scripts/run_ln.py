import torch
from models.unet.standard import UNet
from data import get_data_loader
from diffusion.learned import LearnedGaussianDiffusion
from models.modules.encoders import ConvGaussianEncoder
from data.fashion_mnist import FashionMNISTConfig as config
from trainer.gaussian import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
img_shape = [config.img_channels, config.img_dim, config.img_dim]
z_shape = img_shape.copy()

z_encoder = ConvGaussianEncoder(
    img_shape=img_shape,
    a_shape=z_shape,
).to(device)

model = UNet(
    channels=config.unet_channels,
    chan_mults=config.unet_mults,
    img_shape=img_shape,
)
model.to(device)

learned_diffusion = LearnedGaussianDiffusion(
    noise_model=model,
    z_encoder_model=z_encoder,
    img_shape=img_shape,
    timesteps=config.timesteps,
    device=device,
)

trainer = Trainer(learned_diffusion)
data_loader = get_data_loader(config.name, config.batch_size)
trainer.fit(data_loader, config.epochs)