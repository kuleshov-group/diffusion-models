"""Implements the core diffusion algorithms."""
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import torchvision
from torchvision.utils import save_image
from torchmetrics.image.inception import InceptionScore
import PIL.Image as Image


def process_samples_for_fid(images):
    processed_images = []
    for image in images:
        image = 255 * (image + 1) * 0.5
        if image.shape[0] == 1:
          image = np.array([image[0]] * 3)
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        pil_img = torchvision.transforms.ToPILImage()(image)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        processed_images.append(np.transpose(np.array(resized_img),
                                             (2, 0, 1)))
    return torch.tensor(processed_images, dtype=torch.uint8)


class Trainer():
    def __init__(
        self, diffusion_model, lr=1e-3, optimizer='adam', 
        folder='.', n_samples=36, from_checkpoint=None):
        self.model = diffusion_model
        if optimizer=='adam':
            optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(optimizer)
        self.optimizer = optimizer
        self.folder = folder
        self.n_samples = n_samples
        self.inception_metric = InceptionScore()
        self.metrics = collections.defaultdict(list)
        if from_checkpoint is not None:
            self.load_model(from_checkpoint)

    def fit(self, data_loader, epochs):
        for epoch in range(epochs):
            metrics_per_epoch = collections.defaultdict(list)
            for step, batch in enumerate(data_loader):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(self.model.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(
                    0, self.model.timesteps, (batch_size,), device=self.model.device
                ).long()

                loss, metrics = self.model.loss_at_step_t(
                    batch, t, loss_type='l1')

                if step % 100 == 0:
                    print_line = f'{epoch}:{step}: Loss: {loss.item():.4f}'
                    for key, value in metrics.items():
                        print_line += f' {key}:{value:.4f}'
                        metrics_per_epoch[key].append(value)
                    print(print_line)
                loss.backward()
                self.optimizer.step()

            # save generated images
            self.save_images(epoch, step)
            self.compute_fid_scores()
            self.record_metrics(metrics_per_epoch)
            self.save_model(epoch)
        self.write_metrics()

    def write_metrics(self):
        for key, values in metrics.items():
            with open(f'{key}.txt', 'w') as f:
                for value in values:
                    f.write(value)

    def record_metrics(self, metrics):
        for key, value in metrics.items():
            self.metrics[key].append(np.mean(value))

    def compute_fid_scores(self):
        self.inception_metric.update(
            process_samples_for_fid(
                self.model.sample(batch_size)[-1]))
        fid_mean, fid_std = self.inception_metric.compute()
        self.metrics['fid'].append(fid_mean)
        print('FID score: {} +- {:4f}'.format(fid_mean, fid_std))

    def save_images(self, epoch, step):
        samples = torch.Tensor(self.model.sample(self.n_samples)[-1])
        samples = (samples + 1) * 0.5
        print(samples.min(), samples.max())
        # samples = (samples - samples.min()) / (samples.max() - samples.min())
        path = f'{self.folder}/sample-{epoch}-{step}.png'
        save_image(samples, path, nrow=6)

    def save_model(self, epoch):
        path = f'{self.folder}/model-{epoch}.pth'
        self.model.save(path)
        print(f'Saved PyTorch model state to {path}')

    def load_model(self, path):
        self.model.load(path)
        print(f'Loaded PyTorch model state from {path}')