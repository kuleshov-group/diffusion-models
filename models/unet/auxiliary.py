import torch
from functools import partial
from torch import nn
from ..modules.residual import ResidualBlock
from ..modules.attention import PreNorm, Attention, CrossAttention
from ..modules.embeddings import SineCosinePositionEmbeddings
from ..modules.helpers import (
    exists, default, Residual, Upsample2x, Downsample2x
)
from .standard import UNet

# ---------------------------------------------------------------------------

class ConcatenativeAuxiliaryUNet(UNet):
    """Concatenates auxiliary variables to the input x"""
    def __init__(self, channels, img_shape, a_shape, chan_mults=(1, 2, 4, 8)):
        input_shape = [img_shape[0] + a_shape[0]] + img_shape[1:]
        super().__init__(channels, input_shape, chan_mults)
        self.true_img_channels = img_shape[0]
        self.a_shape = a_shape
        self.img_shape = img_shape

    def forward(self, x, a, time):
        # concatenate x and a
        if self.a_shape[1] == self.a_shape[2] == 1:
            a = a.tile([1, 1, self.img_shape[1], self.img_shape[2]])
        xa = torch.concat([x, a], axis=1)

        # run the usual block
        xa = super().forward(xa, time)

        # only return the x part of xa
        x = xa[:, :self.true_img_channels, :, :]
        return x

class TimeEmbeddingAuxiliaryUNet(UNet):
    """Add auxiliary variables to the time embedding"""
    def __init__(self, channels, img_shape, a_shape, chan_mults=(1, 2, 4, 8)):
        super().__init__(channels, img_shape, chan_mults)
        self.true_img_channels = img_shape[0]
        self.a_shape = a_shape
        self.aux_mlp = nn.Sequential(
            nn.Linear(a_shape[0], self.time_channels),
            # nn.GELU(),
            # nn.Linear(self.time_channels, self.time_channels),
        )

    def forward(self, x, a, time):
        # create embeddings for a
        if self.a_shape[1] == self.a_shape[2] == 1:
            a = a.squeeze()
        a_emb = self.aux_mlp(a)

        # create the time embeddings + add aux embeddings
        t = self.time_mlp(time) + a_emb

        # downsample
        connections = []
        x = self.init_conv(x)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            connections.append(x)
            x = downsample(x)

        # bottleneck
        x = self.bottleneck_layers['block1'](x, t)
        x = self.bottleneck_layers['attention'](x)
        x = self.bottleneck_layers['block2'](x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, connections.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

# ---------------------------------------------------------------------------

class AuxiliaryUNet(nn.Module):
    """Incorporates auxiliary variables via cross-attention"""
    def __init__(
        self, channels, img_shape, a_shape, chan_mults=(1, 2, 4, 8)
    ):
        super().__init__()

        # compute the number of channels in each layers
        chans = [channels] + [m * channels for m in chan_mults]
        chan_in_out = list(zip(chans[:-1], chans[1:]))

        # initialize time embedding layers
        time_channels = channels * 4 # why 4? should comment on this
        self.time_mlp = nn.Sequential(
            SineCosinePositionEmbeddings(channels),
            nn.Linear(channels, time_channels),
            nn.GELU(),
            nn.Linear(time_channels, time_channels),
        )

        # initialize the sequence of downsampling layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.init_conv = nn.Conv2d(img_shape[0], channels, 7, padding=3)
        
        Block = partial(ResidualBlock, groups=8)
        chan_a = a_shape[0]
        for ind, (chan_in, chan_out) in enumerate(chan_in_out):
            is_last = ind >= (len(chan_mults) - 1)
            self.downs.append(
                nn.ModuleList([
                    Block(chan_in, chan_out, time_emb_chan=time_channels),
                    Block(chan_out, chan_out, time_emb_chan=time_channels),
                    Residual(PreNorm(chan_out, Attention(chan_out))),
                    Residual(PreNorm(chan_out, CrossAttention(chan_out, chan_a))),
                    Downsample2x(chan_out) if not is_last else nn.Identity(),
                ])
            )

        # initialize the bottleneck layers
        chan_b = chans[-1]
        self.bottleneck_layers = nn.ModuleDict({
            'block1': Block(chan_b, chan_b, time_emb_chan=time_channels),
            'att': Residual(PreNorm(chan_b, Attention(chan_b))),
            'xatt': Residual(PreNorm(chan_b, CrossAttention(chan_b, chan_a))),
            'block2': Block(chan_b, chan_b, time_emb_chan=time_channels)
        })

        # initialize the upsampling layers
        for ind, (chan_in, chan_out) in enumerate(reversed(chan_in_out[1:])):
            is_last = ind >= (len(chan_mults) - 1)
            self.ups.append(
                nn.ModuleList([
                    Block(chan_out * 2, chan_in, time_emb_chan=time_channels),
                    Block(chan_in, chan_in, time_emb_chan=time_channels),
                    Residual(PreNorm(chan_in, Attention(chan_in))),
                    Residual(PreNorm(chan_in, CrossAttention(chan_in, chan_a))),
                    Upsample2x(chan_in) if not is_last else nn.Identity(),
                ])
            )

        # initialize the last layer
        self.final_conv = nn.Sequential(
            Block(channels, channels), nn.Conv2d(channels, img_shape[0], 1)
        )

    def forward(self, x, a, time):
        # create the time embeddings
        t = self.time_mlp(time)

        # downsample
        connections = []
        x = self.init_conv(x)
        for block1, block2, attn, xattn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = xattn(x, a)
            connections.append(x)
            x = downsample(x)

        # bottleneck
        x = self.bottleneck_layers['block1'](x, t)
        x = self.bottleneck_layers['att'](x)
        x = self.bottleneck_layers['xatt'](x, a)
        x = self.bottleneck_layers['block2'](x, t)

        # upsample
        for block1, block2, attn, xattn, upsample in self.ups:
            x = torch.cat((x, connections.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = xattn(x, a)
            x = upsample(x)

        return self.final_conv(x)