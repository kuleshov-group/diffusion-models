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

import metrics


def process_images(images, return_type='float'):
    processed_images = []
    for image in images:
        #TODO: This is for fashion mnist.
        image = 255 * (image + 1) * 0.5
        if image.shape[0] == 1:
          image = np.array([image[0]] * 3)
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        pil_img = torchvision.transforms.ToPILImage()(image)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        processed_images.append(np.transpose(np.array(resized_img),
                                             (2, 0, 1)))
    processed_images = np.array(processed_images)
    if return_type == 'uint':
        return torch.tensor(processed_images, dtype=torch.uint8)
    elif return_type == 'float':
        return torch.tensor(processed_images, dtype=torch.float) / 255.0   


class Trainer():
    def __init__(
        self, diffusion_model, lr=1e-3, optimizer='adam', 
        folder='.', n_samples=36, from_checkpoint=None,
        weighted_time_sample=False):
        self.model = diffusion_model
        if optimizer=='adam':
            optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(optimizer)
        self.optimizer = optimizer
        self.folder = folder
        self.n_samples = n_samples
        self.inception_score = metrics.InceptionMetric()
        self.fid_score = metrics.FID()
        self.weighted_time_sample = weighted_time_sample
        if self.weighted_time_sample:
            print('Using weighted time samples.')
            self.time_weights = 1 / torch.arange(
                1, 1 + self.model.timesteps, device=self.model.device)
            self.loss_weights=self.time_weights.sum()
        else:
            self.loss_weights = 1.0
            print('Using uniform time sampling.')
        self.metrics = collections.defaultdict(list)
        if from_checkpoint is not None:
            self.load_model(from_checkpoint)

    def fit(self, data_loader, epochs):
        for epoch in range(epochs):
            metrics_per_epoch = collections.defaultdict(list)
            for step, batch in enumerate(data_loader):
                self.optimizer.zero_grad()

                batch_size = batch['pixel_values'].shape[0]
                batch = batch['pixel_values'].to(self.model.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                if self.weighted_time_sample:
                    t = torch.multinomial(
                        self.time_weights, batch_size,
                        replacement=True).long()
                else:
                    t = torch.randint(
                        0, self.model.timesteps, (batch_size,),
                        device=self.model.device).long()
                loss, metrics = self.model.loss_at_step_t(
                    x0=batch,
                    t=t,
                    loss_weights=self.loss_weights,
                    loss_type='l1')

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
            self.compute_fid_scores(batch, epoch)
            self.compute_inception_scores(epoch)
            self.record_metrics(epoch, metrics_per_epoch)
            self.save_model(epoch)
        self.write_metrics()

    def write_metrics(self):
        for key, values in self.metrics.items():
            with open(f'{self.folder}/{key}.txt', 'w') as f:
                for value in values:
                    f.write(value)

    def record_metrics(self, epoch, metrics):
        for key, value in metrics.items():
            self.metrics[key].append(f'{epoch} {np.mean(value)}\n')


    def compute_fid_scores(self, real_images, epoch):
        samples = process_images(
            self.model.sample(real_images.shape[0])[-1])
        real_images = process_images(real_images.detach().cpu().numpy())
        fid_mean = self.fid_score.calculate_frechet_distance(
            real_images, samples)
        self.metrics['fid_score'].append(f'{epoch} {fid_mean}\n')
        print('FID score: {:.2f}'.format(fid_mean))

    def compute_inception_scores(self, epoch):
        fid_mean, fid_std = self.inception_score.compute_inception_scores(
            process_images(self.model.sample(128)[-1],
                           return_type='uint'))
        self.metrics['is_score'].append(f'{epoch} {fid_mean}\n')
        print('IS score: {:.2f} +- {:.2f}'.format(fid_mean, fid_std))

    def save_images(self, epoch, step):
        samples = torch.Tensor(self.model.sample(self.n_samples)[-1])
        samples = (samples + 1) * 0.5
        print(samples.min(), samples.max())
        path = f'{self.folder}/sample-{epoch}-{step}.png'
        save_image(samples, path, nrow=6)

    def save_model(self, epoch):
        path = f'{self.folder}/model-{epoch}.pth'
        self.model.save(path)
        print(f'Saved PyTorch model state to {path}')

    def load_model(self, path):
        self.model.load(path)
        print(f'Loaded PyTorch model state from {path}')