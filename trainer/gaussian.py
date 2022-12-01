"""Implements the core diffusion algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchmetrics.image.inception import InceptionScore
import PIL.Image as Image


def process_samples_for_fid(images):
    arr = []
    for image in images:
        image = 255 * (image - image.min()) / (image.max() - image.min())
        image = np.transpose(np.array([image[0]] * 3), (1, 2, 0)).astype(np.uint8)
        pil_img = transforms.ToPILImage()(image)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr).type(torch.uint8)


class Trainer():
    def __init__(
        self, diffusion_model, lr=1e-3, optimizer='adam', 
        folder='.', n_samples=36, from_checkpoint=None
    ):
        self.model = diffusion_model
        if optimizer=='adam':
            optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(optimizer)
        self.optimizer = optimizer
        self.folder = folder
        self.n_samples = n_samples
        self.inception_metric = InceptionScore()

        if from_checkpoint is not None:
            self.load_model(from_checkpoint)

    def fit(self, data_loader, epochs):
        for epoch in range(epochs):
            for step, batch in enumerate(data_loader):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(self.model.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(
                    0, self.model.timesteps, (batch_size,), device=self.model.device
                ).long()

                loss = self.model.loss_at_step_t(batch, t, loss_type="huber")

                if step % 100 == 0:
                    print(f"{epoch}:{step}: Loss: {loss.item()}")

                loss.backward()
                self.optimizer.step()

                # save generated images
                if step % 1000 == 0:
                    self.save_images(epoch, step)
                    samples = process_samples_for_fid(self.model.sample(batch_size)[-1])
                    self.inception_metric.update(samples)
                    print('FID score:', self.inception_metric.compute())
            self.save_images(epoch, step)
            self.save_model(epoch)

    
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
        print(f"Saved PyTorch model state to {path}")

    def load_model(self, path):
        self.model.load(path)
        print(f"Loaded PyTorch model state from {path}")