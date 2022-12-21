import torch
from functools import partial
from torch import nn
from ..modules.residual import ResidualBlock
from ..modules.attention import PreNorm, Attention
from ..modules.embeddings import SineCosinePositionEmbeddings
from ..modules.helpers import (
    exists, default, Residual, Upsample2x, Downsample2x
)

# ---------------------------------------------------------------------------

class BiheadedUNet(nn.Module):
    """UNet architecture used to parameterize the denoising model.

    TALK ABOUT PARAMS HERE
    """
    def __init__(self, channels, img_shape, chan_mults=(1, 2, 4, 8)):
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
        self.time_channels = time_channels

        # initialize the sequence of downsampling layers
        self.downs = nn.ModuleList([])
        self.ups_1 = nn.ModuleList([])
        self.ups_2 = nn.ModuleList([])
        self.init_conv = nn.Conv2d(img_shape[0], channels, 7, padding=3)
        
        Block = partial(ResidualBlock, groups=8)
        for ind, (chan_in, chan_out) in enumerate(chan_in_out):
            is_last = ind >= (len(chan_mults) - 1)
            self.downs.append(
                nn.ModuleList([
                    Block(chan_in, chan_out, time_emb_chan=time_channels),
                    Block(chan_out, chan_out, time_emb_chan=time_channels),
                    Residual(PreNorm(chan_out, Attention(chan_out))),
                    Downsample2x(chan_out) if not is_last else nn.Identity(),
                ])
            )

        # initialize the bottleneck layers
        mid_chan = chans[-1]
        self.bottleneck_layers = nn.ModuleDict({
            'block1': Block(mid_chan, mid_chan, time_emb_chan=time_channels),
            'attention': Residual(PreNorm(mid_chan, Attention(mid_chan))),
            'block2': Block(mid_chan, mid_chan, time_emb_chan=time_channels)
        })

        # initialize the upsampling layers
        for ind, (chan_in, chan_out) in enumerate(reversed(chan_in_out[1:])):
            is_last = ind >= (len(chan_mults) - 1)
            self.ups_1.append(
                nn.ModuleList([
                    Block(chan_out * 2, chan_in, time_emb_chan=time_channels),
                    Block(chan_in, chan_in, time_emb_chan=time_channels),
                    Residual(PreNorm(chan_in, Attention(chan_in))),
                    Upsample2x(chan_in) if not is_last else nn.Identity(),
                ])
            )
            self.ups_2.append(
                nn.ModuleList([
                    Block(chan_out * 2, chan_in, time_emb_chan=time_channels),
                    Block(chan_in, chan_in, time_emb_chan=time_channels),
                    Residual(PreNorm(chan_in, Attention(chan_in))),
                    Upsample2x(chan_in) if not is_last else nn.Identity(),
                ])
            )

        # initialize the last layer
        self.final_conv_1 = nn.Sequential(
            Block(channels, channels),
            nn.Conv2d(channels, img_shape[0], 1)
        )
        self.final_conv_2 = nn.Sequential(
            Block(channels, channels),
            nn.Conv2d(channels, img_shape[0], 1)
        )

    def forward(self, x, time):
        # create the time embeddings
        t = self.time_mlp(time)

        # downsample
        connections_1 = []
        connections_2 = []
        x = self.init_conv(x)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            connections_1.append(x)
            connections_2.append(x)
            x = downsample(x)

        # bottleneck
        x = self.bottleneck_layers['block1'](x, t)
        x = self.bottleneck_layers['attention'](x)
        x = self.bottleneck_layers['block2'](x, t)

        x_1 = x
        x_2 = x
        # upsample
        for block1, block2, attn, upsample in self.ups_1:
            x_1 = torch.cat((x_1, connections_1.pop()), dim=1)
            x_1 = block1(x_1, t)
            x_1 = block2(x_1, t)
            x_1 = attn(x_1)
            x_1 = upsample(x_1)
        # upsample
        for block1, block2, attn, upsample in self.ups_2:
            x_2 = torch.cat((x_2, connections_2.pop()), dim=1)
            x_2 = block1(x_2, t)
            x_2 = block2(x_2, t)
            x_2 = attn(x_2)
            x_2 = upsample(x_2)

        return self.final_conv_1(x_1), self.final_conv_2(x_2)