import os

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torchvision
from pytorch_fid.inception import InceptionV3
from torchmetrics.image.inception import InceptionScore


class FID():
    def __init__(self, device='cpu'):
        self.model = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        self.model.eval()

    def _get_activations(self, images):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- images      : Images
        -- model       : Instance of inception model
        Returns:
        -- A numpy array of dimension (num images, 2048) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        with torch.no_grad():
            pred = self.model(images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(
                pred, output_size=(1, 1))

        return pred.squeeze(3).squeeze(2).cpu().numpy()


    def _calculate_activation_statistics(self, images):
        """Calculation of the statistics used by the FID.
        Params:
        -- images      : Images
        -- model       : Instance of inception model
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                the inception model.
        """
        activations = self._get_activations(images)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma


    def calculate_frechet_distance(self, images_1, images_2, eps=1e-6):
        """Calculates the FID of two paths"""
        mu1, sigma1 = self._calculate_activation_statistics(images_1)
        mu2, sigma2 = self._calculate_activation_statistics(images_2)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


class InceptionMetric():
    def __init__(self):
        self.inception_metric = InceptionScore()

    def compute_inception_scores(self, images):
        self.inception_metric.update(images)
        is_score_mean, is_score_std = self.inception_metric.compute()
        return is_score_mean.cpu().numpy(), is_score_std.cpu().numpy()