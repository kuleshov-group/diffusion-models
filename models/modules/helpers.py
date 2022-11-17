"""Helpers used to simplify the code in other modules."""

from torch import nn

# ---------------------------------------------------------------------------
# python helpers

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# ---------------------------------------------------------------------------
# pytorch helpers

class Residual(nn.Module):
    """Implements a residual connection.

    Given a function f and input x, returns x + f(x).
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample2x(channels):
    return nn.ConvTranspose2d(channels, channels, 4, 2, 1)

def Downsample2x(channels):
    return nn.Conv2d(channels, channels, 4, 2, 1)