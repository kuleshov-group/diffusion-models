import argparse
import os

import numpy as np
import torch

from models.unet.standard import UNet
from models.unet.biheaded import BiheadedUNet
from models.modules import feedforward
from models.unet.auxiliary import AuxiliaryUNet, TimeEmbeddingAuxiliaryUNet
from data import get_data_loader
from diffusion.gaussian import GaussianDiffusion
from diffusion.auxiliary import InfoMaxDiffusion
from diffusion.learned import LearnedGaussianDiffusion
from diffusion.learned_input_and_time import LearnedGaussianDiffusionInputTime
from diffusion.learned_input_and_time import InputTimeReparam2
from diffusion.learned_input_and_time import InputTimeReparam3
from diffusion.learned_input_and_time import InputTimeReparam4
from diffusion.learned_input_and_time import InputTimeReparam5
from diffusion.learned_input_and_time import InputTimeReparam6
from models.modules.encoders import ConvGaussianEncoder
from data.fashion_mnist import FashionMNISTConfig
from trainer.gaussian import Trainer, process_images
from misc.eval.sample import sample, viz_latents

import metrics
# ----------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands')

    # train

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)

    train_parser.add_argument('--model', default='gaussian',
        choices=['gaussian', 'infomax', 'learned', 'learned_input_time'], 
        help='type of ddpm model to run')
    train_parser.add_argument('--schedule', default='cosine',
        choices=['linear', 'cosine'], 
        help='constants scheduler for the diffusion model.')
    train_parser.add_argument('--timesteps', type=int, default=200,
        help='total number of timesteps in the diffusion model')
    train_parser.add_argument('--reparam', type=int, default=1,
        choices=[1, 2, 3, 4, 5, 6], 
        help='reparameterization type for input time diffusion model.')
    train_parser.add_argument('--weighted_time_sample', type=bool, default=False,
        help='total number of timesteps in the diffusion model')
    train_parser.add_argument('--dataset', default='fashion-mnist',
        choices=['fashion-mnist', 'mnist'], help='training dataset')
    train_parser.add_argument('--checkpoint', default=None,
        help='path to training checkpoint')
    train_parser.add_argument('-e', '--epochs', type=int, default=50,
        help='number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=None,
        help='training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.0001,
        help='learning rate')
    train_parser.add_argument('--optimizer', default='adam', choices=['adam'],
        help='optimization algorithm')
    train_parser.add_argument('--folder', default='.',
        help='folder where logs will be stored')

    # eval

    eval_parser = subparsers.add_parser('eval')
    eval_parser.set_defaults(func=eval)

    eval_parser.add_argument('--model', default='gaussian',
        choices=['gaussian', 'infomax', 'learned', 'learned_input_time'], 
        help='type of ddpm model to run')
    eval_parser.add_argument('--dataset', default='fashion-mnist',
        choices=['fashion-mnist', 'mnist'], help='training dataset')
    eval_parser.add_argument('--checkpoint', required=True,
        help='path to training checkpoint')
    eval_parser.add_argument('--deterministic', action='store_true', 
        default=False, help='run in deterministic mode')
    eval_parser.add_argument('--sample', type=int, default=None,
        help='how many samples to draw')
    eval_parser.add_argument('--interpolate', type=int, default=None,
        help='how many samples to interpolate')
    eval_parser.add_argument('--latents', type=int, default=None,
        help='how many points to visualize in latent space')
    eval_parser.add_argument('--folder', default='.',
        help='folder where output will be stored')
    eval_parser.add_argument('--name', default='test-run',
        help='name of the files that will be saved')

    return parser

# ----------------------------------------------------------------------------

def train(args):
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config(args)
    config.epochs = args.epochs or config.epochs
    config.batch_size = args.batch_size or config.batch_size
    config.learning_rate = args.learning_rate or config.learning_rate
    config.optimizer = args.optimizer or config.optimizer
    config.timesteps = args.timesteps or config.timesteps
    
    model = get_model(config, device)

    if args.checkpoint:
        model.load(args.checkpoint)
    trainer = Trainer(
        model,
        weighted_time_sample=args.weighted_time_sample,
        lr=config.learning_rate,
        optimizer=config.optimizer,
        folder=args.folder,
        from_checkpoint=args.checkpoint,
    )
    data_loader = get_data_loader(config.name, config.batch_size)
    trainer.fit(data_loader, config.epochs)

def eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config(args)
    model = get_model(config, device)
    model.load(args.checkpoint, eval=True)
    data_loader = get_data_loader(
        config.name, 128, train=False, labels=True)

    if args.sample:
        path = f'{args.folder}/{args.name}-samples.png'
        sample(model, args.sample, path, args.deterministic)

    if args.latents:
        path = f'{args.folder}/{args.name}-latents.png'
        viz_latents(model, data_loader, args.latents, path)

    scores = {'fid_score': [], 'is_score': []}
    fid_score = metrics.FID()
    inception_score = metrics.InceptionMetric()
    for batch in data_loader:
        real_images = process_images(
            batch['pixel_values'].to(model.device).detach().cpu().numpy())
        samples = process_images(
            model.sample(real_images.shape[0])[-1])
        fid_mean = fid_score.calculate_frechet_distance(
            real_images, samples)
        scores['fid_score'].append(fid_mean)
        is_mean, _ = inception_score.compute_inception_scores(
            (255 * samples).type(torch.uint8))
        scores['is_score'].append(is_mean)
    print('FID score: {:.2f}'.format(np.mean(scores['fid_score'])))
    print('IS score: {:.2f}'.format(np.mean(scores['is_score'])))

# ----------------------------------------------------------------------------

def get_config(args):
    if args.dataset == 'fashion-mnist':
        return FashionMNISTConfig
    else:
        raise ValueError()

def get_model(config, device):
    if args.model == 'gaussian':
        model = create_gaussian(config, device)
    elif args.model == 'infomax':
        model = create_infomax(config, device)
    elif args.model == 'learned':
        model = create_learned(config, device)
    elif args.model == 'learned_input_time':
        model = create_learned_input_time(config, device, args.reparam)
    else:
        raise ValueError(args.model)
    return model

def create_gaussian(config, device):
    img_shape = [config.img_channels, config.img_dim, config.img_dim]
    model = UNet(
        channels=config.unet_channels,
        chan_mults=config.unet_mults,
        img_shape=img_shape,
    )
    model.to(device)

    return GaussianDiffusion(
        model=model,
        schedule=args.schedule,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
    )

def create_infomax(config, device):
    img_shape = [config.img_channels, config.img_dim, config.img_dim]
    a_shape = [config.a_dim, 1, 1]

    a_encoder = ConvGaussianEncoder(
        img_shape=img_shape,
        a_shape=a_shape,
    ).to(device)

    # model = AuxiliaryUNet(
    model = TimeEmbeddingAuxiliaryUNet(
        channels=config.unet_channels,
        chan_mults=config.unet_mults,
        img_shape=img_shape,
        a_shape=a_shape,
    ).to(device)

    return InfoMaxDiffusion(
        noise_model=model,
        a_encoder_model=a_encoder,
        timesteps=config.timesteps,
        img_shape=img_shape,
        a_shape=a_shape,
        device=device,
    )

def create_learned(config, device):
    img_shape = [config.img_channels, config.img_dim, config.img_dim]

    model = UNet(
        channels=config.unet_channels,
        chan_mults=config.unet_mults,
        img_shape=img_shape,
    ).to(device)

    forward_matrix = feedforward.Net(
        input_size=model.time_channels,
        identity=False,
        positive_outputs=True,
    ).to(device)

    return LearnedGaussianDiffusion(
        noise_model=model,
        schedule=args.schedule,
        forward_matrix=forward_matrix,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
    )

def create_learned_input_time(config, device, reparam):
    img_shape = [config.img_channels, config.img_dim, config.img_dim]
    diffusion_models = {
        1: LearnedGaussianDiffusionInputTime,
        2: InputTimeReparam2,
        3: InputTimeReparam3,
        4: InputTimeReparam4,
        5: InputTimeReparam5,
        6: InputTimeReparam6,
    }
    if reparam == 1 or reparam == 2 or reparam == 5 or reparam == 6:
        model = UNet(
            channels=config.unet_channels,
            chan_mults=config.unet_mults,
            img_shape=img_shape,
        ).to(device)
        reverse_model = UNet(
            channels=config.unet_channels,
            chan_mults=config.unet_mults,
            img_shape=img_shape,
        ).to(device)
    else:
        model = BiheadedUNet(
            channels=config.unet_channels,
            chan_mults=config.unet_mults,
            img_shape=img_shape,
        ).to(device)
        reverse_model = None
    forward_matrix = UNet(
        channels=config.unet_channels,
        chan_mults=config.unet_mults,
        img_shape=img_shape,
    ).to(device)

    print('reparam type:', reparam)
    return diffusion_models[reparam](
        noise_model=model,
        forward_matrix=forward_matrix,
        reverse_model=reverse_model,
        schedule=args.schedule,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
    )

# ----------------------------------------------------------------------------

if __name__ == '__main__':
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)