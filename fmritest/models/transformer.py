"""Masked region-reconstruction transformer (region-as-token).

Each of the R regions is one token whose content is that region's time series over a window of
W timepoints. A random subset of region tokens is masked (content replaced by a learned mask
vector; the region identity embedding is kept), and the model reconstructs the masked regions'
full time series from the unmasked ones. Loss is MSE on masked tokens only.

Why region-as-token: masked-region prediction can only be solved by learning which regions
co-fluctuate, i.e. the functional network structure we want to locate. Time-as-token masking
is largely solvable by temporal interpolation (HRF smoothness), which says little about networks.

The implementation is deliberately explicit (no nn.MultiheadAttention) and every intermediate
passes through a named `HookPoint` so the interp API can read, ablate and patch it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HookPoint(nn.Identity):
    """Identity module; forward hooks registered on it can read or replace the activation."""


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.hook_pattern = HookPoint()   # (B, H, Tq, Tk) attention probabilities
        self.hook_z = HookPoint()         # (B, T, H, dh) per-head outputs before W_O

    def forward(self, x):
        B, T, D = x.shape
        q = self.W_Q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.h, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        pattern = self.hook_pattern(F.softmax(scores, dim=-1))
        z = self.hook_z((pattern @ v).transpose(1, 2))  # (B, T, H, dh)
        return self.W_O(z.reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.hook_post = HookPoint()      # (B, T, d_mlp) neuron activations

    def forward(self, x):
        return self.W_out(self.hook_post(F.gelu(self.W_in(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)
        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_pre(x)
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.ln1(x))))
        return self.hook_resid_post(x + self.hook_mlp_out(self.mlp(self.ln2(x))))


class MaskedRegionTransformer(nn.Module):
    def __init__(self, n_regions: int, window: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_mlp: int = 128):
        super().__init__()
        self.n_regions, self.window = n_regions, window
        self.embed = nn.Linear(window, d_model)
        self.region_emb = nn.Parameter(0.02 * torch.randn(n_regions, d_model))
        self.mask_emb = nn.Parameter(0.02 * torch.randn(d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_mlp) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, window)
        self.hook_embed = HookPoint()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, R, W) region time series; mask: (B, R) bool, True = masked. Returns (B, R, W)."""
        h = self.embed(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_emb.expand_as(h), h)
        h = self.hook_embed(h + self.region_emb)
        for blk in self.blocks:
            h = blk(h)
        return self.unembed(self.ln_final(h))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over masked tokens only (all W timepoints of each masked region)."""
    m = mask[..., None].to(pred.dtype)
    return ((pred - target) ** 2 * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)


def build_model(cfg_model: dict, n_regions: int) -> MaskedRegionTransformer:
    return MaskedRegionTransformer(n_regions, cfg_model["window"], cfg_model["d_model"],
                                   cfg_model["n_heads"], cfg_model["n_layers"], cfg_model["d_mlp"])
