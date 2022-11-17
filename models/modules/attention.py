"""Modules that serve as building blocks for the denoising model."""

import torch
from torch import nn, einsum
from einops import rearrange

# ---------------------------------------------------------------------------        

# TODO: is this needed?
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_kv = nn.Conv2d(dim_kv, hidden_dim * 2, 1, bias=False)
        self.to_q = nn.Conv2d(dim_q, hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim_q, 1)

    def forward(self, x, a):
        b, c, h, w = x.shape
        ab, ac, ah, aw = a.shape

        k, v = self.to_kv(a).chunk(2, dim=1)
        q = self.to_q(x)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), [q, k, v]
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)  

class AdaptiveCrossAttention(nn.Module):
    def __init__(self, dim, dim_a, heads=4, dim_head=32):
        super().__init__()
        self.attn = CrossAttention(dim, heads, dim_head)
        self.a_conv = nn.Conv2d(dim_a, dim)

    def forward(self, x, a):
        xb, xc, xh, xw = x.shape
        ab, ac, ah, aw = a.shape
        
        if ah == aw == 1:
            a = a.tile([1, 1, xh, xw])
