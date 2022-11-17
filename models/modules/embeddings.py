"""Modules that serve as building blocks for the denoising model."""

import math
import torch
from torch import nn

# ---------------------------------------------------------------------------

class SineCosinePositionEmbeddings(nn.Module):
    """Used to encode a scalar denoising step t in a UNet"""
    def __init__(self, chan):
        super().__init__()
        self.chan = chan # number of embedding channels

    def forward(self, time):
        # TODO: UNDERSTAND AND SIMPLIFY
        device = time.device # why?
        half_chan = self.chan // 2
        embeddings = math.log(10000) / (half_chan - 1)
        embeddings = torch.exp(torch.arange(half_chan, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
