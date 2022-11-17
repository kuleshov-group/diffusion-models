"""Modules that serve as building blocks for the denoising model."""

import torch
from torch import nn
from .helpers import exists

# ---------------------------------------------------------------------------

class ResidualLayer(nn.Module):
    """A ResNet layer, used within a ResNet block.

    A ResNet layer consists of a convolution, normalization (group norm),
    and an activation function.
    """
    def __init__(self, chan_in, chan_out, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(chan_in, chan_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, chan_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    """A ResNet block, used inside the Unet.
    
    A ResidualBlock contains two ResidualLayers and implements the following:
    y = residual_layer2(residual_layer1(x) + time_embeddings(x)) + conv(x)

    is this the correct citation?
    https://arxiv.org/abs/1512.03385
    """
    
    def __init__(self, chan_in, chan_out, *, time_emb_chan=None, groups=8):
        super().__init__()
        self.time_embedding_generator = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_chan, chan_out))
            if exists(time_emb_chan)
            else None
        )

        self.layer1 = ResidualLayer(chan_in, chan_out, groups=groups)
        self.layer2 = ResidualLayer(chan_out, chan_out, groups=groups)
        self.res_conv = nn.Conv2d(chan_in, chan_out, 1)

    def forward(self, x, time_emb=None):
        h = self.layer1(x)

        if exists(self.time_embedding_generator) and exists(time_emb):
            time_emb = self.time_embedding_generator(time_emb)
            h = time_emb[..., None, None] + h
            # h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.layer2(h)
        return h + self.res_conv(x)