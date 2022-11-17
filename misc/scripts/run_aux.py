import torch
from models.unet.auxiliary import AuxiliaryUNet
from data import get_data_loader
from diffusion.auxiliary import InfoMaxDiffusion
from models.modules.encoders import ConvGaussianEncoder
from data.fashion_mnist import FashionMNISTConfig as config
from trainer.gaussian import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
dim_a = 20
img_shape = [config.img_channels, config.img_dim, config.img_dim]
a_shape = [dim_a, 1, 1]

a_encoder = ConvGaussianEncoder(
	img_shape=img_shape,
	a_shape=a_shape,
).to(device)

model = AuxiliaryUNet(
    channels=config.unet_channels,
    chan_mults=config.unet_mults,
    img_shape=img_shape,
    a_shape=a_shape,
).to(device)

gaussian_diffusion = InfoMaxDiffusion(
    noise_model=model,
    a_encoder_model=a_encoder,
    timesteps=config.timesteps,
    img_shape=img_shape,
    a_shape=a_shape,
    device=device,
)

trainer = Trainer(gaussian_diffusion)
data_loader = get_data_loader(config.name, config.batch_size)
trainer.fit(data_loader, config.epochs)