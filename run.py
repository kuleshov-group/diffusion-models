import argparse
import torch
from models.unet.standard import UNet
from models.modules import feedforward
from models.unet.auxiliary import AuxiliaryUNet, TimeEmbeddingAuxiliaryUNet
from data import get_data_loader
from diffusion.gaussian import GaussianDiffusion
from diffusion.auxiliary import InfoMaxDiffusion
from diffusion.learned import LearnedGaussianDiffusion
from models.modules.encoders import ConvGaussianEncoder
from data.fashion_mnist import FashionMNISTConfig
from trainer.gaussian import Trainer
from misc.eval.sample import sample, viz_latents

# ----------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands')

    # train

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)

    train_parser.add_argument('--model', default='gaussian',
        choices=['gaussian', 'infomax', 'learned'], 
        help='type of ddpm model to run')
    train_parser.add_argument('--dataset', default='fashion-mnist',
        choices=['fashion-mnist', 'mnist'], help='training dataset')
    train_parser.add_argument('--checkpoint', default=None,
        help='path to training checkpoint')
    train_parser.add_argument('-e', '--epochs', type=int, default=None,
        help='number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=None,
        help='training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=None,
        help='learning rate')
    train_parser.add_argument('--optimizer', default='adam', choices=['adam'],
        help='optimization algorithm')
    train_parser.add_argument('--folder', default='.',
        help='folder where logs will be stored')

    # eval

    eval_parser = subparsers.add_parser('eval')
    eval_parser.set_defaults(func=eval)

    eval_parser.add_argument('--model', default='gaussian',
        choices=['gaussian', 'infomax', 'learned'], 
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config(args)
    model = get_model(config, device)

    if args.checkpoint:
        model.load(args.checkpoint)

    config.epochs = args.epochs or config.epochs
    config.batch_size = args.batch_size or config.batch_size
    config.learning_rate = args.learning_rate or config.learning_rate
    config.optimizer = args.optimizer or config.optimizer

    trainer = Trainer(
        model,
        lr=config.learning_rate,
        optimizer=config.optimizer,
        folder=args.folder,
        from_checkpoint=args.checkpoint
    )
    data_loader = get_data_loader(config.name, config.batch_size)
    trainer.fit(data_loader, config.epochs)

def eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config(args)
    model = get_model(config, device)
    model.load(args.checkpoint, eval=True)
    data_loader = get_data_loader(
        config.name, 16, train=False, labels=True
    )

    if args.sample:
        path = f'{args.folder}/{args.name}-samples.png'
        sample(model, args.sample, path, args.deterministic)

    if args.latents:
        path = f'{args.folder}/{args.name}-latents.png'
        viz_latents(model, data_loader, args.latents, path)

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
    ).to(device)

    return LearnedGaussianDiffusion(
        noise_model=model,
        forward_matrix=forward_matrix,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
    )

# ----------------------------------------------------------------------------

if __name__ == '__main__':
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)