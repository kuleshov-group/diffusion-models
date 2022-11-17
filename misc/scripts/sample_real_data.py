import torch
from torchvision.utils import save_image
from data import get_data_loader
from data.fashion_mnist import FashionMNISTConfig as config

n_samples = 36
n_per_row = 6

data_loader = get_data_loader(config.name, config.batch_size)
batch = next(iter(data_loader))
samples = batch['pixel_values'][:n_samples]
samples = (samples + 1) * 0.5

path = f'./real-samples.png'
save_image(samples, path, nrow=n_per_row)