"""Small transformer helper layers used by the DINOTrack neck."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019), used inside Differential Attention.
    [UNSPECIFIED] eps -> 1e-5 (Differential-Transformer convention)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.pow(x, 2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class DropPath(nn.Module):
    """Stochastic depth per sample (timm formulation). Rate taken from
    config (Fig. 3 shows DropPath; [UNSPECIFIED] rate -> default 0.0)."""

    def __init__(self, prob: float = 0.0):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x / keep * mask
